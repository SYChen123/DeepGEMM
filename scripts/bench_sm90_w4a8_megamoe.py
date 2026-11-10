#!/usr/bin/env python3
r"""DSV4.1 Flash routed MoE on four Hopper GPUs, using synthetic MXFP4 weights.

Run with the TVM-FFI DeepGEMM build, PyTorch, SGLang and Humming installed:
    torchrun --standalone --nproc-per-node=4 scripts/bench_sm90_w4a8_megamoe.py

Humming TP4 shards the intermediate dimension, gathers BF16 activations,
executes the indexed W4A8 expert chain, all-reduces partial BF16 outputs and
returns the local token slice. Global routes are prepared before timing.
The report also includes a clean Humming TP total with replicated inputs.

Clean totals default to CUDA Graph replay. Eager diagnostic passes measure
launch boundaries and instrument the fused kernel on device. The instrumented
variant adds timestamp writes and synchronization. Its phase windows include
waits and may overlap, so use the clean total for latency comparisons.

For NCU 2025.3, profile rank 0 only and keep all four ranks in the workload:
    DG_JIT_WITH_LINEINFO=1 torchrun --standalone --nproc-per-node=4 --no-python \
        bash -c 'if test "$LOCAL_RANK" = 0; then
            exec ncu --profile-from-start off --clock-control none --cache-control none \
                --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
                --nvtx --nvtx-include "$1/" --import-source yes -o "$2" python "${@:3}"
        else exec python "${@:3}"; fi' bash mega mega-decode-rank0 \
        scripts/bench_sm90_w4a8_megamoe.py --shape decode --backend mega --ncu

Keep this rank-0 capture to one metric pass per kernel: replaying an NVLink
or NCCL kernel by itself can leave it waiting for the other ranks. For
Humming, replace `mega` with `humming` and select `--backend humming`.

With an NCU version that supports the communicator and lockstep options,
use coordinated application replay for larger metric sets on all four ranks:
    DG_JIT_WITH_LINEINFO=1 torchrun --standalone --nproc-per-node=4 --no-python ncu \
        --communicator tcp --communicator-tcp-hostname 127.0.0.1 \
        --communicator-tcp-num-peers 4 --lockstep-kernel-launch \
        --replay-mode application --profile-from-start off --nvtx \
        --nvtx-include 'fused/' --import-source yes --section SourceCounters \
        -o 'mega-decode-rank%q{RANK}' python \
        scripts/bench_sm90_w4a8_megamoe.py --shape decode --backend mega --ncu

You can narrow the capture to an NVTX stage such as 'fused/', 'all_reduce/',
or 'l1_gemm/'.

脚本运行方式（参考以上方式采集ncu profile）
torchrun --standalone --nproc-per-node=4 \
  scripts/bench_sm90_w4a8_megamoe.py --shape both --output bench_moe.json
"""

import argparse
import json
import os
import statistics
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist


WORLD = 4
HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS = 384
TOPK = 6
SHAPES = {"prefill": 4096, "decode": 32}
# Matches SM90MegaMoEPhase in sm90_mega_moe_profile.cuh.
MEGA_PHASES = (
    "dispatch_metadata",
    "dispatch_pull",
    "l1_gemm",
    "l1_epilogue",
    "l2_gemm",
    "l2_scatter",
    "combine_barrier",
    "combine_reduce",
    "workspace_cleanup",
)


def parse_args():
    description, _, usage = __doc__.partition("\n\n")
    parser = argparse.ArgumentParser(
        description=description, epilog=usage, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shape", choices=(*SHAPES, "both"), default="both")
    parser.add_argument("--backend", choices=("mega", "humming", "both"), default="both")
    parser.add_argument("--mega-layout", choices=("standard", "register"), default="standard")
    parser.add_argument("--sms", type=int, help="DeepGEMM SM budget (default: device default)")
    parser.add_argument("--timing", choices=("graph", "eager"), default="graph")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--breakdown-iters", type=int, default=5,
                        help="Eager diagnostic samples per shape; 0 skips breakdown")
    parser.add_argument("--output", type=Path, help="Write all timing results as rank-0 JSON")
    parser.add_argument("--ncu", action="store_true",
                        help="Warm up, then capture one eager iteration on all four ranks")
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iters) < 1 or args.breakdown_iters < 0:
        parser.error("warmup, rounds and iters must be positive; breakdown-iters must be nonnegative")
    if args.sms is not None and (args.sms < 2 or args.sms % 2):
        parser.error("--sms must be a positive even MegaMoE SM budget")
    if args.ncu and (args.shape == "both" or args.backend == "both"):
        parser.error("--ncu requires one --shape and one --backend")
    if args.mega_layout == "register" and (args.shape != "decode" or args.backend == "humming"):
        parser.error("--mega-layout register requires --shape decode and the MegaMoE backend")
    return args


def make_weights(rank, tensor_parallel):
    """Generate the same full expert before taking its EP or TP slice."""
    local_experts = EXPERTS // WORLD
    experts = range(EXPERTS) if tensor_parallel else range(
        rank * local_experts, (rank + 1) * local_experts
    )
    width = INTERMEDIATE // WORLD if tensor_parallel else INTERMEDIATE
    begin, end = rank * width, (rank + 1) * width
    result = []
    for layer, (n, k) in enumerate(((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))):
        shape = (2 * width, HIDDEN) if layer == 0 else (HIDDEN, width)
        weights = torch.empty((len(experts), shape[0], shape[1] // 2),
                              dtype=torch.uint8, device="cuda")
        scales = torch.full((len(experts), shape[0], shape[1] // 32), 120,
                            dtype=torch.uint8, device="cuda")
        for slot, expert in enumerate(experts):
            generator = torch.Generator(device="cuda").manual_seed(6318 + 2 * expert + layer)
            full = torch.randint(256, (n, k // 2), generator=generator,
                                 dtype=torch.uint8, device="cuda")
            if not tensor_parallel:
                weights[slot].copy_(full)
            elif layer == 0:
                weights[slot, :width].copy_(full[begin:end])
                weights[slot, width:].copy_(full[INTERMEDIATE + begin:INTERMEDIATE + end])
            else:
                weights[slot].copy_(full[:, begin // 2:end // 2])
        result.append((weights, scales))
    return result


def prepare_humming(rank):
    from humming.layer import HummingMethod
    from humming.schema import BaseWeightSchema, HummingInputSchema
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore

    width = INTERMEDIATE // WORLD
    layer = torch.nn.Module()
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = width
    layer.params_dtype = torch.bfloat16
    schema = BaseWeightSchema.from_config({"quant_method": "mxfp4"})
    for name, (weight, scale), n, k in zip(
        ("w13", "w2"), make_weights(rank, tensor_parallel=True),
        (2 * width, HIDDEN), (HIDDEN, width),
    ):
        weight_schema, tensors = schema.convert_humming(
            tensors={"weight": weight, "weight_scale": scale},
            shape_n_stacks=[width, width] if name == "w13" else [n],
            shape_k_stacks=[k],
            param_dtype=torch.bfloat16,
            num_experts=EXPERTS,
        )
        for key, tensor in tensors.items():
            layer.register_parameter(f"{name}_{key}", torch.nn.Parameter(tensor, requires_grad=False))
        HummingMethod.prepare_layer_meta(
            layer, shape_n=n, shape_k=k, weight_schema=weight_schema,
            input_schema=HummingInputSchema.from_config({"dtype": "float8e4m3"}),
            num_experts=EXPERTS, pad_n_to_multiple=256, pad_k_to_multiple=128,
            has_bias=False, torch_dtype=torch.bfloat16, sublayer_name=name,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=name)
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device="cuda"))
    runner = HummingRunnerCore(MoeRunnerConfig(
        num_experts=EXPERTS, num_local_experts=EXPERTS, hidden_size=HIDDEN,
        intermediate_size_per_partition=width, top_k=TOPK, params_dtype=torch.bfloat16,
        activation="silu", is_gated=True, swiglu_limit=10.0, routed_scaling_factor=1.0,
        gate_up_interleaved=False,
    ))
    runner.layer = layer
    return runner


def prepare_mega(rank, register):
    import deep_gemm

    raw = make_weights(rank, tensor_parallel=False)
    weights = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(*(
        (w.view(torch.int8), s.view(torch.float8_e8m0fnu).float()) for w, s in raw
    ))
    if register:
        weights = tuple((deep_gemm.pack_sm90_fp4_register_weights(w), s) for w, s in weights)
    return weights


class Stages:
    def __init__(self, nvtx=False):
        self.nvtx = nvtx
        self.events = {}

    @contextmanager
    def __call__(self, name):
        if self.nvtx:
            with torch.cuda.nvtx.range(name):
                yield
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events[name] = (start, end)

    def elapsed_us(self):
        return {name: start.elapsed_time(end) * 1000 for name, (start, end) in self.events.items()}


def untimed(_):
    return nullcontext()


@contextmanager
def humming_stages(runner, stage):
    """Measure the actual SGLang indexed path without copying the runner."""
    from sglang.srt.layers.moe.moe_runner import humming

    method = humming.HummingMethod

    def wrap(fn, name):
        def measured(*args, **kwargs):
            with stage(name(kwargs)):
                return fn(*args, **kwargs)
        return measured

    with ExitStack() as stack:
        for obj, attr, name in (
            (runner, "_prepare_indexed_gemm_kwargs", lambda _: "route_align"),
            (runner, "apply_activation", lambda _: "activation"),
            (humming, "moe_fused_mul_sum", lambda _: "local_topk_combine"),
        ):
            stack.enter_context(patch.object(obj, attr, wrap(getattr(obj, attr), name)))
        for attr, suffix in (("may_quant_input", "quant"), ("forward_layer", "gemm")):
            def name(kwargs, suffix=suffix):
                linear = "l1" if kwargs["sublayer_name"] == "w13" else "l2"
                return f"{linear}_{suffix}"
            stack.enter_context(patch.object(method, attr, staticmethod(wrap(getattr(method, attr), name))))
        yield


def make_humming_call(runner, x, ids, weights, rank):
    rows = x.size(0)
    all_x = x.new_empty((rows * WORLD, HIDDEN))
    all_ids = ids.new_empty((rows * WORLD, TOPK))
    all_weights = weights.new_empty((rows * WORLD, TOPK))
    dist.all_gather_into_tensor(all_ids, ids)
    dist.all_gather_into_tensor(all_weights, weights)

    def run(stage=untimed, gather=True):
        with stage("humming"):
            if gather:
                with stage("all_gather_activations"):
                    dist.all_gather_into_tensor(all_x, x)
            with stage("tp_moe"):
                with humming_stages(runner, stage) if stage is not untimed else nullcontext():
                    partial = runner._run_indexed_gemm(all_x, all_weights, all_ids)
                with stage("all_reduce"):
                    dist.all_reduce(partial)
                return partial.narrow(0, rank * rows, rows)

    return run


def make_mega_call(weights, x, ids, route_weights):
    import deep_gemm

    rows = x.size(0)
    result = torch.empty_like(x)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, EXPERTS, rows, TOPK, HIDDEN, INTERMEDIATE, mma_type="fp8xfp4",
    )

    def run(stage=untimed, phase_timings=None):
        with stage("mega"):
            with stage("pre_dispatch"):
                deep_gemm.mega_moe_pre_dispatch_sm90(
                    x, ids, route_weights, buffer.x, buffer.x_sf,
                    buffer.topk_idx, buffer.topk_weights, num_tokens=rows, group_size=128,
                )
            with stage("fused"):
                if phase_timings is None:
                    deep_gemm.fp8_fp4_mega_moe(result, *weights, buffer, activation_clamp=10.0)
                else:
                    deep_gemm.fp8_fp4_mega_moe(
                        result, *weights, buffer, activation_clamp=10.0, phase_timings=phase_timings,
                    )
        return result

    return run


def summarize(samples):
    names = list(samples)
    values = torch.tensor(list(samples.values()), dtype=torch.float64, device="cuda")
    ranks = values.new_empty((WORLD, *values.shape))
    dist.all_gather_into_tensor(ranks.flatten(), values.flatten())
    by_rank = ranks.tolist()
    result = {}
    for i, name in enumerate(names):
        slowest = [max(round_times) for round_times in zip(*(rank[i] for rank in by_rank))]
        result[name] = {
            "median_us": statistics.median(slowest),
            "rank_max_samples_us": slowest,
            "rank_medians_us": [statistics.median(rank[i]) for rank in by_rank],
        }
    return result


def warmup(run, count):
    for _ in range(count):
        run()
    torch.cuda.synchronize()


def benchmark(run, args):
    warmup(run, args.warmup)
    replay = run
    if args.timing == "graph":
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        replay = graph.replay

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.rounds):
        dist.barrier()
        start.record()
        for _ in range(args.iters):
            replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.iters)
    if args.timing == "graph":
        graph.reset()
    return samples


def device_phases(timings):
    rows = timings.tolist()
    result = {}
    for i, name in enumerate(MEGA_PHASES):
        active = [cta[i] for cta in rows if cta[i][0] != 0]
        result[f"{name}.window"] = (
            (max(row[1] for row in active) - min(row[0] for row in active)) / 1000 if active else 0.0
        )
        result[f"{name}.max_cta"] = max((row[2] for row in active), default=0) / 1000
    return result


def breakdown(run, count, timings=None):
    samples = {}
    kwargs = {} if timings is None else {"phase_timings": timings}
    for _ in range(count):
        if timings is not None:
            timings.zero_()
        dist.barrier()
        stages = Stages()
        run(stages, **kwargs)
        torch.cuda.synchronize()
        values = stages.elapsed_us()
        if timings is not None:
            values.update(device_phases(timings.cpu()))
        for name, value in values.items():
            samples.setdefault(name, []).append(value)
    return summarize(samples)


def capture_ncu(run, warmup_count):
    warmup(run, warmup_count)
    dist.barrier()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        run(Stages(nvtx=True))
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()


def run_shape(args, shape, rank, mega_weights, humming_runner, sms):
    rows = SHAPES[shape]
    generator = torch.Generator(device="cuda").manual_seed(9000 + rank + rows)
    x = torch.randn((rows, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
    ids = torch.rand((rows, EXPERTS), device="cuda", generator=generator).topk(TOPK).indices.int()
    weights = torch.rand((rows, TOPK), device="cuda", generator=generator)
    weights.div_(weights.sum(-1, keepdim=True))
    result = {
        "shape": shape, "tokens_per_rank": rows, "tokens_total": rows * WORLD,
        "ranks": WORLD, "hidden": HIDDEN, "intermediate": INTERMEDIATE,
        "experts": EXPERTS, "topk": TOPK, "mega_sms": sms,
        "mega_layout": args.mega_layout, "timing_mode": args.timing,
        "mega_sparse_dispatch_counts": os.environ.get("DG_SM90_FP4_SPARSE_DISPATCH_COUNTS", "0"),
        "humming_output_collective": "all_reduce",
        "totals": {}, "eager_breakdown": {},
    }
    runs = {}
    if mega_weights is not None:
        runs["mega"] = make_mega_call(mega_weights, x, ids, weights)
    if humming_runner is not None:
        runs["humming"] = make_humming_call(humming_runner, x, ids, weights, rank)
    if args.ncu:
        capture_ncu(runs[args.backend], args.warmup)
        return None

    totals = {}
    for name, run in runs.items():
        totals[name] = benchmark(run, args)
        if name == "humming":
            totals["humming_tp_replicated"] = benchmark(lambda: run(gather=False), args)
    result["totals"] = summarize(totals)
    if "mega" in totals and "humming" in totals:
        mega_out = runs["mega"]().clone()
        humming_out = runs["humming"]()
        relative_l2 = (mega_out.float() - humming_out.float()).norm() / humming_out.float().norm()
        relative_l2 = torch.nan_to_num(relative_l2, nan=float("inf"), posinf=float("inf"))
        dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
        if not bool(torch.isfinite(relative_l2)):
            raise RuntimeError("MegaMoE/Humming comparison produced a nonfinite relative L2 error")
        result["relative_l2_rank_max"] = relative_l2.item()
        result["speedup_vs_humming"] = {
            name: result["totals"][name]["median_us"] / result["totals"]["mega"]["median_us"]
            for name in ("humming", "humming_tp_replicated")
        }

    if args.breakdown_iters:
        for name, run in runs.items():
            result["eager_breakdown"][name] = breakdown(run, args.breakdown_iters)
            if name == "mega":
                grid = sms * (2 if args.mega_layout == "register" else 1)
                timings = torch.zeros((grid, len(MEGA_PHASES), 3), dtype=torch.int64, device="cuda")
                # Compile and warm the instrumented variant outside the diagnostic samples.
                run(phase_timings=timings)
                torch.cuda.synchronize()
                result["mega_device_breakdown"] = breakdown(run, args.breakdown_iters, timings)
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != WORLD:
        raise SystemExit("Run with torchrun --standalone --nproc-per-node=4 on one Hopper node")
    if not torch.cuda.is_available():
        raise SystemExit("Four Hopper CUDA GPUs are required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()[0] != 9:
        raise SystemExit("This benchmark requires SM90 (Hopper)")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        mega_weights = humming_runner = sms = None
        if args.backend != "humming":
            import deep_gemm

            if args.sms is not None:
                deep_gemm.set_num_sms(args.sms)
            sms = deep_gemm.get_num_sms()
            mega_weights = prepare_mega(rank, args.mega_layout == "register")
        if args.backend != "mega":
            humming_runner = prepare_humming(rank)
        results = []
        for shape in SHAPES if args.shape == "both" else (args.shape,):
            result = run_shape(args, shape, rank, mega_weights, humming_runner, sms)
            if result is not None and rank == 0:
                results.append(result)
                print("RESULT_JSON", json.dumps(result), flush=True)
        if args.output and rank == 0 and results:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        # All ranks must finish GPU work before releasing NCCL communicators.
        dist.barrier()
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
