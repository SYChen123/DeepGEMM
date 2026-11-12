"""Compare serial and pipelined SM90 MegaMoE RS on V4.1 Flash EP4 shapes."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from bench_sm90_w4a8_megamoe import (
    EXPERTS, HIDDEN, INTERMEDIATE, MEGA_PHASES, TOPK, WORLD,
    device_phases, prepare_mega, summarize, warmup,
)

# Matches SM90MegaMoERSStage in sm90_mega_moe_profile.cuh.
RS_STAGES = ("full_wait", "weight_read", "weight_decode", "wgmma_issue", "wgmma_wait", "promote")


def make_inputs(rows, rank, hot_expert=False):
    rng = torch.Generator(device="cuda").manual_seed(9000 + rank + rows)
    x = torch.randn((rows, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=rng)
    x.mul_(0.1)
    scores = torch.rand((rows, EXPERTS), device="cuda", generator=rng)
    if hot_expert:
        scores[:, 0] = 2.0
    ids = scores.topk(TOPK).indices.int()
    probs = torch.rand((rows, TOPK), device="cuda", generator=rng)
    probs.div_(probs.sum(-1, keepdim=True))
    return x, ids, probs


def make_run(weights, inputs, buffer, pipeline, phase_timings=None):
    import deep_gemm

    os.environ["DG_SM90_FP4_RS_PIPELINE"] = "1" if pipeline else "0"
    x, ids, probs = inputs
    output = torch.empty_like(x)

    def run():
        deep_gemm.mega_moe_pre_dispatch_sm90(
            x, ids, probs, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
            num_tokens=x.size(0), group_size=128,
        )
        deep_gemm.fp8_fp4_mega_moe(
            output, *weights, buffer, activation_clamp=10.0, use_sm90_register_weights=True,
            phase_timings=phase_timings,
        )

    return run, output


def capture(run):
    warmup(run, 3)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def compare_outputs(serial, pipeline):
    delta = serial.float() - pipeline.float()
    finite = torch.isfinite(serial).all() & torch.isfinite(pipeline).all()
    error = delta.norm() / serial.float().norm().clamp_min(1e-20)
    values = torch.stack([(~finite).float(), error, delta.abs().amax()])
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    nonfinite, relative_l2, max_abs = values.tolist()
    return {
        "finite": nonfinite == 0,
        "max_rank_relative_l2": relative_l2,
        "max_rank_abs_error": max_abs,
        "bitwise_equal": max_abs == 0,
        "passed": nonfinite == 0 and relative_l2 <= 0.01,
    }


def check_rows(weights, inputs, buffer, rounds, iters):
    variants = ("serial", "pipeline")
    graphs, outputs = {}, {}
    for variant in variants:
        run, outputs[variant] = make_run(weights, inputs, buffer, variant == "pipeline")
        graphs[variant] = capture(run)
    accuracy = compare_outputs(*outputs.values())
    samples = {variant: [] for variant in variants}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(rounds):
        order = variants if i % 2 == 0 else variants[::-1]
        for variant in order:
            dist.barrier()
            start.record()
            for _ in range(iters):
                graphs[variant].replay()
            end.record()
            end.synchronize()
            samples[variant].append(1000 * start.elapsed_time(end) / iters)
    dist.barrier()
    for graph in graphs.values():
        graph.reset()
    return accuracy, summarize(samples) if rounds else None


def profile_rs(weights, inputs, buffer, pipeline, iters):
    import deep_gemm

    coarse_phases = len(MEGA_PHASES)
    timings = torch.zeros(
        (2 * deep_gemm.get_num_sms(), coarse_phases + 2 * len(RS_STAGES), 3),
        device="cuda", dtype=torch.int64,
    )
    run, _ = make_run(weights, inputs, buffer, pipeline, phase_timings=timings)
    warmup(run, 3)
    totals = torch.zeros_like(timings[0, coarse_phases:])
    coarse_samples = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        timings.zero_()
        dist.barrier()
        start.record()
        run()
        end.record()
        end.synchronize()
        totals += timings[:, coarse_phases:].sum(0)
        phases = device_phases(timings[:, :coarse_phases].cpu())
        phases["instrumented_eager"] = 1000 * start.elapsed_time(end)
        for name, value in phases.items():
            coarse_samples.setdefault(name, []).append(value)
    dist.all_reduce(totals)
    stages = {}
    for linear, layer in enumerate(("w13", "w2")):
        stage_totals = totals[linear * len(RS_STAGES):(linear + 1) * len(RS_STAGES)].tolist()
        stages[layer] = {
            name: {"sampled_k128": count, "ns_per_k128": elapsed / count}
            for name, (count, _, elapsed) in zip(RS_STAGES, stage_totals)
        }
    return {"per_k128": stages, "coarse": summarize(coarse_samples)}


@torch.inference_mode()
def main():
    import deep_gemm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--profile-iters", type=int, default=0,
                        help="Collect detailed RS stage diagnostics after the Graph A/B")
    args = parser.parse_args()
    if min(*args.rows, args.rounds, args.iters) <= 0:
        parser.error("rows, rounds, and iters must be positive")
    if args.profile_iters < 0:
        parser.error("profile-iters must be nonnegative")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if max(args.rows) > deep_gemm.get_sm90_fp4_row_major_rs_max_tokens():
        parser.error("rows exceeds the SM90 RS limit")
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        assert dist.get_world_size() == WORLD
        rank = dist.get_rank()
        weights = prepare_mega(rank, register=False)
        rng = torch.Generator(device="cuda").manual_seed(15000 + rank)
        for _, scales in weights:
            scales.view(torch.uint8).random_(119, 123, generator=rng)
        report = {
            "device": torch.cuda.get_device_name(),
            "world_size": WORLD,
            "num_sms": deep_gemm.get_num_sms(),
            "rounds": args.rounds,
            "iterations_per_round": args.iters,
            "rows": {},
        }
        for rows in args.rows:
            buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                dist.group.WORLD, EXPERTS, rows, TOPK, HIDDEN, INTERMEDIATE,
                mma_type="fp8xfp4",
            )
            uniform, timings = check_rows(
                weights, make_inputs(rows, rank), buffer, args.rounds, args.iters,
            )
            hot, _ = check_rows(weights, make_inputs(rows, rank, hot_expert=True), buffer, 0, args.iters)
            result = {"uniform_check": uniform, "hot_expert_check": hot, "timings": timings}
            if args.profile_iters:
                inputs = make_inputs(rows, rank)
                result["profile"] = {
                    variant: profile_rs(weights, inputs, buffer, variant == "pipeline", args.profile_iters)
                    for variant in ("serial", "pipeline")
                }
            report["rows"][str(rows)] = result
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{rows} rows: {json.dumps(result)}", flush=True)
            if not (uniform["passed"] and hot["passed"]):
                raise RuntimeError(f"RS pipeline output mismatch for {rows} rows")
            buffer.destroy()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
