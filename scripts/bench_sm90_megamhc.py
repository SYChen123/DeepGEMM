"""Single-GPU DSv4.1 shifted Mega mHC correctness, latency and kernel timeline.

Build with scripts/build_sm90_mhc_dev.py first. Example (physical GPU 1):
  CUDA_VISIBLE_DEVICES=1 python /mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/code/DeepGEMM/scripts/bench_sm90_megamhc.py \
       --model /mnt/a6intlalipayplus-a6/common/models/DeepSeek-V4.1-Flash/ --tokens 4096 32 --output results.json

Collect %globaltimer traces in a SEPARATE launch after uninstrumented timing:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/build-sm90-mhc \
    /path/to/csy/bin/python scripts/bench_sm90_megamhc.py \
    --model /path/to/DeepSeek-V4.1-Flash --tokens 4096 32 \
    --kernel-profile-dir profiles --output results-profile.json
Open profiles/*.json in Perfetto. --kernel-profile-cta 0 selects one CTA.
The baseline calls the current SGLang post/mix_stats/combine dispatchers. Decode
uses its stats side stream; both branches join before the boundary is timed.
Inputs are synthetic activations with real mHC and RMSNorm checkpoint weights.
No model server or distributed collective is included in this operator test.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import torch
import deep_gemm


TRACE_SLOTS = 4096
EVENTS = {1: (2, "role"), 3: (4, "projection task"), 5: (6, "wait"),
          7: (8, "Post"), 9: (10, "WGMMA completion gate")}


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "source snapshot"


def load_material(args):
    from safetensors import safe_open
    config = json.loads((args.model / "config.json").read_text())
    config = config.get("text_config", config)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{args.layer}."
    names = {"fn": prefix + f"hc_{args.sublayer}_fn",
             "scale": prefix + f"hc_{args.sublayer}_scale",
             "base": prefix + f"hc_{args.sublayer}_base",
             "weight": prefix + f"{args.sublayer}_norm.weight"}
    values, hashes = {}, {}
    for name, key in names.items():
        with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        hashes[key] = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        values[name] = value.to(device="cuda", dtype=torch.bfloat16 if name == "weight" else torch.float32).contiguous().squeeze()
    values["fn"] = values["fn"].view(24, -1)
    return config, values, hashes


def accuracy(actual, expected):
    metrics = {}
    for name in actual:
        a, b = actual[name].float(), expected[name].float()
        if not torch.isfinite(a).all():
            raise AssertionError(f"Nonfinite {name}")
        error = a - b
        norm = b.square().sum().clamp_min(1e-30)
        diff = error.square().sum() / (a.square().sum() + norm).clamp_min(1e-30)
        metrics[name] = {"max_abs": error.abs().max().item(),
                         "relative_l2": (error.square().sum() / norm).sqrt().item(),
                         "diff": diff.item()}
        if diff > 5e-5:
            raise AssertionError(f"{name}: {metrics[name]}")
    return metrics


def reference(inputs):
    x, r, pre, post, comb, fn, scale, base, w, eps, rms_eps, iters = inputs
    # Match the shared BF16 boundaries; retain an independent FP32 projection.
    updated = x.float().unsqueeze(1) * post.unsqueeze(-1)
    for j in range(4):
        updated = updated + r[:, j:j + 1, :].float() * comb[:, j, :].unsqueeze(-1)
    updated = updated.bfloat16()
    collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16().float()
    normalized = (collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + rms_eps) * w.float()).bfloat16()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mix = updated.flatten(1).float() @ fn.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    mix *= torch.rsqrt(updated.float().square().mean((1, 2)) + rms_eps).unsqueeze(-1)
    next_pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + eps
    next_post = torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8]) * 2
    next_comb = (mix[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + eps
    next_comb /= next_comb.sum(-2, keepdim=True) + eps
    for _ in range(1, iters):
        next_comb /= next_comb.sum(-1, keepdim=True) + eps
        next_comb /= next_comb.sum(-2, keepdim=True) + eps
    return dict(new_residual=updated, y_bf16=normalized, new_prev_mix=next_pre,
                new_post_mix=next_post, new_comb_res_mix=next_comb)


def time_graphs(runs, repetitions, rounds):
    graphs = {}
    for name, run in runs.items():
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                run()
        graphs[name] = graph
        for _ in range(2):
            graph.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in runs}
    for i in range(rounds):
        order = list(runs) if i % 2 == 0 else list(reversed(runs))
        for name in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in times.items()}


def capture_profile(run, args, tokens):
    sms = deep_gemm.get_num_sms()
    trace = torch.zeros((sms, 24, TRACE_SLOTS + 1, 2), dtype=torch.int64, device="cuda")
    if os.environ.get("DG_SM90_MHC_TRACE_PTR") not in (None, "", "0"):
        raise RuntimeError("Unset DG_SM90_MHC_TRACE_PTR; use --kernel-profile-dir")
    os.environ["DG_SM90_MHC_TRACE_PTR"] = str(trace.data_ptr())
    try:
        run()  # JIT and warmup instrumented specialization.
        torch.cuda.synchronize()
        trace.zero_()
        run()
        torch.cuda.synchronize()
    finally:
        os.environ.pop("DG_SM90_MHC_TRACE_PTR")
    host = trace.cpu()
    counts = host[:, :, 0, 0]
    if args.kernel_profile_cta is not None and not 0 <= args.kernel_profile_cta < sms:
        raise ValueError("--kernel-profile-cta outside grid")
    minimum = host[:, :, 1:, 1]
    origin = minimum[minimum > 0].min().item()
    events, totals = [], {}
    ctas = range(sms) if args.kernel_profile_cta is None else [args.kernel_profile_cta]
    for cta in ctas:
        for warp in range(24):
            role = ("IO", "Fn", "idle", "store")[warp] if warp < 4 else (
                "Post0" if warp < 8 else "Post1" if warp < 12 else
                "WGMMA" if warp < 16 else "Mix" if warp < 20 else "Norm")
            stack = {}
            for value, timestamp in host[cta, warp, 1:1 + min(TRACE_SLOTS, counts[cta, warp].item())].tolist():
                event, aux = value >> 56, value & ((1 << 56) - 1)
                if event in EVENTS:
                    stack[EVENTS[event][0]] = (timestamp, EVENTS[event][1], aux)
                elif event in stack:
                    start, label, detail = stack.pop(event)
                    duration = (timestamp - start) / 1000
                    events.append(dict(name=label, cat=role, ph="X", pid=cta, tid=warp,
                                       ts=(start - origin) / 1000, dur=duration, args={"aux": detail}))
                    key = f"{role}:{label}"
                    totals[key] = totals.get(key, 0) + duration
    metadata = dict(tokens=tokens, grid=sms, threads=768, slots=TRACE_SLOTS,
                    attempted_events=counts.sum().item(),
                    dropped_events=(counts - TRACE_SLOTS).clamp_min(0).sum().item(),
                    sum_warp_intervals_us=totals,
                    note="Instrumented launch; interval sums across warps are not kernel latency.")
    args.kernel_profile_dir.mkdir(parents=True, exist_ok=True)
    path = args.kernel_profile_dir / f"sm90-megamhc-{tokens}.json"
    path.write_text(json.dumps(dict(traceEvents=events, displayTimeUnit="us", metadata=metadata)))
    print("PROFILE", path, json.dumps(metadata), flush=True)
    return metadata


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sublayer", choices=("attn", "ffn"), default="ffn")
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 32])
    parser.add_argument("--sms", type=int)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("sm90-megamhc-results.json"))
    parser.add_argument("--kernel-profile-dir", type=Path)
    parser.add_argument("--kernel-profile-cta", type=int)
    args = parser.parse_args()
    torch.manual_seed(20261118)
    assert torch.cuda.get_device_capability() == (9, 0)
    if args.sms is not None:
        deep_gemm.set_num_sms(args.sms)
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight
    from sglang.srt.runtime_context import get_parallel
    # A standalone operator process has no server configuration publication.
    # State the actual single-rank topology so the unmodified post dispatcher
    # can evaluate its CP gate.
    get_parallel().override_permanently(attn_cp_size=1)
    import sglang
    config, material, hashes = load_material(args)
    hidden = config["hidden_size"]
    assert hidden == 5120 and config["hc_mult"] == 4
    eps, rms_eps, iters = config["hc_eps"], config["rms_norm_eps"], config["hc_sinkhorn_iters"]
    fn, scale, base, weight = [material[k] for k in ("fn", "scale", "base", "weight")]
    cfg = mhc.HcConfig(4, iters, eps, rms_eps, hidden, True, False)
    norm = SimpleNamespace(weight=weight, variance_epsilon=rms_eps,
                           cast_x_before_out_mul=False, variance_size_override=None)
    hc = mhc.HcSubLayer(cfg, fn, scale, base, norm, bf16_parts=split_bf16_hc_weight(fn))
    metadata = dict(device=torch.cuda.get_device_name(), sms=deep_gemm.get_num_sms(),
                    torch=torch.__version__, cuda=torch.version.cuda, deep_gemm=deep_gemm.__file__,
                    sglang=sglang.__file__, sglang_revision=revision(Path(sglang.__file__).parents[2]),
                    checkpoint=str(args.model), weight_sha256=hashes, layer=args.layer,
                    sublayer=args.sublayer, hidden=hidden, hc=4, sinkhorn_iters=iters,
                    hc_eps=eps, rms_eps=rms_eps, seed=20261118,
                    repetitions=args.repetitions, rounds=args.rounds)
    kernel_path = Path(deep_gemm.__file__).parent / "include/deep_gemm/impls/sm90_mega_mhc.cuh"
    metadata["kernel_sha256"] = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    metadata["pdl"] = deep_gemm.get_pdl()
    metadata["deterministic"] = deep_gemm.get_deterministic_algorithms()
    print("SETUP", json.dumps(metadata), flush=True)
    result = dict(metadata=metadata, cases=[])
    for tokens in args.tokens:
        x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
        residual = torch.randn((tokens, 4, hidden), device="cuda", dtype=torch.bfloat16)
        pre, post, comb = mhc.mix_stats(hc, residual)
        stream = torch.cuda.Stream() if tokens == 32 else None

        def baseline():
            updated = mhc.post(cfg, x, residual, post, comb)
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
            next_pre, next_post, next_comb = mhc.mix_stats(hc, updated, stream)
            normalized = mhc.combine(hc, mhc.HcState(updated, pre))
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            return dict(new_residual=updated, y_bf16=normalized, new_prev_mix=next_pre,
                        new_post_mix=next_post, new_comb_res_mix=next_comb)

        outputs = dict(new_residual=torch.empty_like(residual), y_bf16=torch.empty_like(x),
                       new_prev_mix=torch.empty_like(pre).unsqueeze(-1),
                       new_post_mix=torch.empty_like(post).unsqueeze(-1), new_comb_res_mix=torch.empty_like(comb))

        def mega():
            deep_gemm.mega_mhc(
                x=x, residual=residual, shifted_prev_mix=pre.unsqueeze(-1),
                post_mix=post.unsqueeze(-1), comb_res_mix=comb, fn=fn,
                mix_scales=scale, mix_bases=base, hc_mult=4, hc_norm_eps=rms_eps,
                hc_pre_eps=eps, hc_post_scale=2., sinkhorn_eps=eps, num_sinkhorn_iters=iters,
                rmsnorm_weight=weight, rmsnorm_eps=rms_eps, rmsnorm_scale=1., **outputs)
            return {k: v.squeeze(-1) if k in ("new_prev_mix", "new_post_mix") else v for k, v in outputs.items()}

        base_out, actual = baseline(), mega()
        torch.cuda.synchronize()
        ref = reference((x, residual, pre, post, comb, fn, scale, base, weight, eps, rms_eps, iters))
        checks = {"mega_vs_baseline": accuracy(actual, base_out),
                  "mega_vs_fp32": accuracy(actual, ref), "baseline_vs_fp32": accuracy(base_out, ref)}
        # Repeated graph replay must consume changed inputs, not stale split state.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mega()
        for _ in range(5):
            x.normal_()
            graph.replay()
            torch.cuda.synchronize()
            accuracy(actual, baseline())
        timing = time_graphs(dict(baseline=baseline, mega=mega), args.repetitions, args.rounds)
        case = dict(tokens=tokens, stats_side_stream=stream is not None, accuracy=checks, timing=timing,
                    speedup=timing["baseline"]["median_us"] / timing["mega"]["median_us"])
        if args.kernel_profile_dir is not None:
            case["profile"] = capture_profile(mega, args, tokens)
        result["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
