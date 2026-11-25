"""Compare SGLang MoEGate + TopK with DeepGEMM at the same local token count."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import deep_gemm


# Keep the record format and event order in sync with sm90_bf16_mega_gate.cuh.
TRACE_SLOTS = 512
TRACE_PTR_ENV = "DG_SM90_MEGA_GATE_TRACE_PTR"
TRACE_CTA_ENV = "DG_SM90_MEGA_GATE_TRACE_CTA"
TRACE_SPANS = (
    ("Wait for empty stage", "k_block"),
    ("Issue TMA copies", "k_block"),
    ("Wait for full stage", "k_block"),
    ("Issue WGMMA group", "k_block"),
    ("WGMMA wait<1>", "k_block"),
    ("Drain WGMMA", None),
    ("Write scores to scratch", None),
    ("Publish scores and elect router", None),
    ("Route tokens", "first_token"),
)
ROUTE_WINNER_EVENT = 2 * len(TRACE_SPANS) + 1


def time_graphs(calls, repetitions, rounds):
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name, call in calls.items():
        with torch.cuda.stream(stream):
            for _ in range(5):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(repetitions):
                call()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in graphs}
    names = list(graphs)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for round_idx in range(rounds):
        for name in names if round_idx % 2 == 0 else reversed(names):
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / repetitions)
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()}


def capture_baseline_trace(run, directory, tokens, handoff):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.profiler.record_function(f"SGLang router, {handoff} boundary"):
            graph.replay()
            torch.cuda.synchronize()

    directory.mkdir(parents=True, exist_ok=True)
    stem = "sglang-moe-gate-topk" if handoff == "gate" else "sglang-humming-input"
    path = directory / f"{stem}-{tokens}.json"
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
    if not kernels:
        raise RuntimeError(f"No GPU kernels in {path}; check CUPTI CUDA Graph activity support")
    origin = kernels[0]["ts"]
    summary = dict(path=str(path), gpu_span_us=max(event["ts"] + event["dur"] for event in kernels) - origin,
                   kernels=[dict(name=event["name"], start_us=event["ts"] - origin, duration_us=event["dur"])
                            for event in kernels])
    print("SGLANG BASELINE PROFILE", json.dumps(summary), flush=True)
    return summary


def capture_ncu(run):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        run()
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()


def prepare_humming_inputs(runner, x, ids, weights):
    from humming.layer import HummingMethod

    metadata, _ = runner._prepare_indexed_gemm_kwargs(ids.to(torch.int32))
    quantized, scale = HummingMethod.may_quant_input(layer=runner.layer, inputs=x, sublayer_name="w13")
    return quantized, scale, weights, metadata


def format_kernel_trace(host, ctas, tokens, config):
    events, intervals, route_winners = [], [], []
    producers = config["num_expert_groups"] * config["num_split_k"]
    warps = config["num_gate_warpgroups"] * 4
    roles = ["TMA", "WGMMA", "CTA publish"] + [f"Route warp {warp}" for warp in range(warps)]
    attempted = dropped = unpaired = 0
    origin = None
    for row, cta in enumerate(ctas):
        token_tile, producer = divmod(cta, producers)
        split, expert_group = divmod(producer, config["num_expert_groups"])
        events.append(dict(name="process_name", ph="M", pid=cta, tid=0,
                           args=dict(name=f"CTA {cta} (tile {token_tile}, expert group {expert_group}, split {split})")))
        for track, role in enumerate(roles):
            count = host[row, track, 0, 0].item()
            if count == 0:
                continue
            attempted += count
            dropped += max(0, count - TRACE_SLOTS)
            events.append(dict(name="thread_name", ph="M", pid=cta, tid=track, args=dict(name=role)))
            begin = None
            for value, timestamp in host[row, track, 1:1 + min(count, TRACE_SLOTS)].tolist():
                event, aux = value >> 56, value & ((1 << 56) - 1)
                origin = timestamp if origin is None else min(origin, timestamp)
                if event == ROUTE_WINNER_EVENT:
                    winner = dict(cta=cta, token_tile=token_tile)
                    route_winners.append(winner)
                    intervals.append(dict(name="Router elected", cat=role, ph="i", s="t",
                                          pid=cta, tid=track, ts=timestamp, args=winner))
                elif 1 <= event < ROUTE_WINNER_EVENT and event % 2:
                    if begin is not None:
                        raise RuntimeError(f"Overlapping MegaGate trace markers on CTA {cta}, {role}")
                    begin = event, timestamp, aux
                elif begin is not None and event == begin[0] + 1 and aux == begin[2]:
                    label, detail = TRACE_SPANS[(event - 2) // 2]
                    begin_ns = begin[1]
                    intervals.append(dict(name=label, cat=role, ph="X", pid=cta, tid=track,
                                          ts=begin_ns, dur=(timestamp - begin_ns) / 1000,
                                          args={detail: aux} if detail else {}))
                    begin = None
                else:
                    raise RuntimeError(f"Unexpected MegaGate trace marker {event} on CTA {cta}, {role}")
            unpaired += begin is not None
    if origin is None:
        raise RuntimeError("MegaGate trace is empty. Build DeepGEMM with the trace-enabled launcher and kernel.")
    for interval in intervals:
        interval["ts"] = (interval["ts"] - origin) / 1000
    metadata = dict(trace_schema=1, tokens=tokens, grid=math.ceil(tokens / config["block_tokens"]) * producers,
                    threads=warps * 32, captured_ctas=len(ctas), slots_per_track=TRACE_SLOTS,
                    attempted_events=attempted, dropped_events=dropped, unpaired_intervals=unpaired,
                    route_winners=route_winners, config=config)
    return dict(traceEvents=events + intervals, displayTimeUnit="us", metadata=metadata)


def capture_kernel_trace(run, out, args, tokens, config):
    producers = config["num_expert_groups"] * config["num_split_k"]
    grid = math.ceil(tokens / config["block_tokens"]) * producers
    cta = args.kernel_profile_cta
    ctas = range(grid) if cta is None else [cta]
    tracks = 3 + config["num_gate_warpgroups"] * 4
    trace = torch.zeros((len(ctas), tracks, TRACE_SLOTS + 1, 2), dtype=torch.int64, device="cuda")
    expected = tuple(tensor.clone() for tensor in out)
    trace_env = {TRACE_PTR_ENV: str(trace.data_ptr()), TRACE_CTA_ENV: str(-1 if cta is None else cta)}
    previous = {name: os.environ.get(name) for name in trace_env}
    os.environ.update(trace_env)
    try:
        run()  # Warm the instrumented JIT specialization.
        trace.zero_()
        run()
        torch.cuda.synchronize()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    for actual, reference in zip(out, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    profile = format_kernel_trace(trace.cpu(), ctas, tokens, config)
    profile["metadata"].update(hidden=args.hidden, experts=args.experts, topk=args.topk)
    args.kernel_profile_dir.mkdir(parents=True, exist_ok=True)
    path = args.kernel_profile_dir / f"sm90-megagate-{tokens}.json"
    path.write_text(json.dumps(profile))
    summary = {key: value for key, value in profile["metadata"].items() if key != "route_winners"}
    summary.update(path=str(path), routing_ctas=len(profile["metadata"]["route_winners"]))
    print("PROFILE", json.dumps(summary), flush=True)
    return summary


def revision(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192],
                        help="Local input rows passed to MoEGate, including any graph padding in the target shape")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--handoff", choices=("gate", "humming-indexed"), default="gate",
                        help="Include Humming W4A8 indexed routing layout and gate-up input quantization on both sides")
    parser.add_argument("--ncu", choices=("sglang", "mega"), help="Profile one warmed eager call with Nsight Compute")
    parser.add_argument("--kernel-profile-dir", type=Path, help="Write a Perfetto trace from a separate eager launch")
    parser.add_argument("--kernel-profile-cta", type=int, help="Record only this CTA index for each shape")
    parser.add_argument("--baseline-profile-dir", type=Path, help="Write a CUPTI trace of one SGLang baseline replay per shape")
    args = parser.parse_args()
    if args.ncu:
        if len(args.tokens) != 1 or args.kernel_profile_dir or args.baseline_profile_dir:
            parser.error("--ncu requires one token count and a separate run from timeline collection")
    elif args.output is None:
        parser.error("--output is required for benchmark timing")
    if args.kernel_profile_cta is not None and args.kernel_profile_dir is None:
        parser.error("--kernel-profile-cta requires --kernel-profile-dir")
    if os.environ.get(TRACE_PTR_ENV) not in (None, "", "0"):
        parser.error(f"Unset {TRACE_PTR_ENV} before running the benchmark")
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(20261124)

    from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
    get_parallel().override_permanently(attn_cp_size=1, moe_ep_size=1, tp_group=SimpleNamespace(world_size=1))
    config = SimpleNamespace(hidden_size=args.hidden, n_routed_experts=args.experts, topk_method="noaux_tc")
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            gate = MoEGate(config, quant_config=None, is_deepseek_v4=True)
    finally:
        torch.set_default_dtype(original_dtype)
    gate.weight.data.normal_()
    bias = gate.e_score_correction_bias
    bias.data.normal_(std=0.1)
    topk = TopK(args.topk, layer_id=0, scoring_func="sqrtsoftplus", renormalize=True,
                correction_bias=bias, sqrtsoftplus_log1p=True, output_format=TopKOutputFormat.STANDARD,
                routed_scaling_factor=1.0, allow_routed_experts_capture=False)
    humming = None
    if args.handoff == "humming-indexed":
        from bench_sm90_w4a8_megamoe import EXPERTS, HIDDEN, TOPK, prepare_humming

        if (args.hidden, args.experts, args.topk) != (HIDDEN, EXPERTS, TOPK):
            parser.error("--handoff humming-indexed requires the V4.1 Flash routing shape")
        humming = prepare_humming(rank=0)

    sources = [Path(inspect.getfile(module)).resolve() for module in (MoEGate, TopK)]
    result = dict(metadata=dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                               deep_gemm=deep_gemm.__file__, revision=revision(Path(__file__).resolve().parent),
                               sglang_revision=revision(sources[0].parent),
                               sglang_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                               hidden=args.hidden, experts=args.experts, topk=args.topk, seed=20261124,
                               sglang_baseline="single-rank text MoEGate + TopK",
                               handoff=args.handoff,
                               sglang_router_logits_dtype="float32",
                               sglang_uses_padding_bound=False, sglang_uses_expert_remap=False,
                               tiny_gemm_max_tokens=gate.tiny_router_gemm_max_tokens,
                               repetitions=args.repetitions, rounds=args.rounds, pdl=deep_gemm.get_pdl()), cases=[])
    print(json.dumps(result["metadata"]), flush=True)
    for tokens in args.tokens:
        launch = deep_gemm.get_bf16_mega_gate_config(tokens, args.hidden, args.experts, args.topk)
        if args.kernel_profile_cta is not None:
            grid = math.ceil(tokens / launch["block_tokens"]) * launch["num_expert_groups"] * launch["num_split_k"]
            if not 0 <= args.kernel_profile_cta < grid:
                parser.error(f"--kernel-profile-cta outside grid [0, {grid}) for {tokens} tokens")
        x = (torch.randn(tokens, args.hidden, device="cuda") / math.sqrt(args.hidden)).bfloat16()
        out = (torch.empty((tokens, args.topk), dtype=torch.int64, device="cuda"),
               torch.empty((tokens, args.topk), dtype=torch.float32, device="cuda"))

        def baseline():
            return topk(x, gate(x))

        def mega():
            return deep_gemm.bf16_mega_gate(x, gate.weight, args.topk, False, 0, 1.0, 0,
                                            "sqrtsoftplus", bias=bias, out=out)

        runs = dict(sglang=baseline, mega=mega)
        if humming is not None:
            def baseline_handoff():
                routing = baseline()
                return prepare_humming_inputs(humming, x, routing.topk_ids, routing.topk_weights)

            def mega_handoff():
                return prepare_humming_inputs(humming, x, *mega())

            runs = dict(sglang=baseline_handoff, mega=mega_handoff)
        if args.ncu:
            capture_ncu(runs[args.ncu])
            return

        reference = baseline()
        ids, weights = mega()
        scores = F.softplus(gate(x)).sqrt()
        rankings = scores + bias
        torch.testing.assert_close(rankings.gather(1, ids), rankings.topk(args.topk, dim=1).values,
                                   rtol=2e-5, atol=3e-6)
        selected = scores.gather(1, ids)
        torch.testing.assert_close(weights, selected / selected.sum(1, keepdim=True), rtol=2e-5, atol=2e-6)
        # The SGLang top-k implementation can return slots in a different order.
        reference_ids = reference.topk_ids.sort(dim=1).values
        agreement = (ids.sort(dim=1).values == reference_ids).all(dim=1).float().mean().item()
        if humming is not None:
            sglang_quant, sglang_scale, _, sglang_layout = baseline_handoff()
            mega_quant, mega_scale, _, mega_layout = mega_handoff()
            torch.testing.assert_close(mega_quant.view(torch.uint8), sglang_quant.view(torch.uint8))
            torch.testing.assert_close(mega_scale, sglang_scale)
            torch.testing.assert_close(mega_layout["num_tokens_padded"], sglang_layout["num_tokens_padded"])
        timing = time_graphs(runs, args.repetitions, args.rounds)
        entry = dict(tokens=tokens, row_expert_set_agreement=agreement, config=launch,
                     timing=timing, speedup=timing["sglang"]["median_us"] / timing["mega"]["median_us"])
        result["cases"].append(entry)
        print(f"{tokens:5d}  SGLang {args.handoff} {timing['sglang']['median_us']:8.3f} us  "
              f"MegaGate {timing['mega']['median_us']:8.3f} us  {entry['speedup']:.3f}x  agreement={agreement:.6f}",
              flush=True)
        if args.kernel_profile_dir is not None:
            entry["kernel_profile"] = capture_kernel_trace(mega, out, args, tokens, launch)
        if args.baseline_profile_dir is not None:
            entry["baseline_profile"] = capture_baseline_trace(runs["sglang"], args.baseline_profile_dir, tokens, args.handoff)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
