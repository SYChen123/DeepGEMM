"""Check the patched SGLang gate through Hopper MegaMoE pre-dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import deep_gemm


@torch.no_grad()
def main():
    from sglang.srt.environ import envs
    from sglang.srt.eplb.expert_location import set_global_expert_location_metadata
    from sglang.srt.layers.moe import mega_moe
    from sglang.srt.layers.moe.hash_topk import HashTopK
    from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.srt.runtime_context import get_exec, get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(846)
    publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
    get_parallel().override_permanently(attn_cp_size=1, moe_ep_size=1, tp_group=SimpleNamespace(world_size=1))

    rows, hidden, experts, k, valid_rows, capacity = 17, 5120, 384, 6, 13, 32
    physical_ids = torch.arange(experts - 1, -1, -1, device="cuda", dtype=torch.int32)[None, :]
    set_global_expert_location_metadata(SimpleNamespace(
        logical_to_rank_dispatch_physical_map=physical_ids,
        logical_to_all_physical_map=physical_ids[..., None],
        logical_to_all_physical_map_num_valid=torch.ones_like(physical_ids),
        num_physical_experts=experts,
    ))
    scale = 1.5
    config = SimpleNamespace(
        hidden_size=hidden, n_routed_experts=experts, num_experts_per_tok=k,
        topk_method="noaux_tc", model_type="deepseek_v41", moe_intermediate_size=2304, image_token_id=66,
    )
    x = (torch.randn(rows, hidden, device="cuda") / hidden**0.5).bfloat16()
    input_ids = torch.full((rows,), 3, device="cuda", dtype=torch.int64)
    input_ids[1::2] = config.image_token_id
    bound = torch.tensor([valid_rows], device="cuda", dtype=torch.int32)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, moe_num_token_non_padded=lambda: bound)
    buf = SimpleNamespace(
        x=torch.empty((capacity, hidden), device="cuda", dtype=torch.float8_e4m3fn),
        x_sf=torch.empty((capacity, hidden // 128), device="cuda", dtype=torch.float32),
        topk_idx=torch.empty((capacity, k), device="cuda", dtype=torch.int64),
        topk_weights=torch.empty((capacity, k), device="cuda", dtype=torch.float32),
    )

    # Exercise the real SGLang Hopper staging routine and DeepGEMM pre-dispatch.
    # The expert GEMM and cross-rank execution are outside this test's boundary.
    def stage(expert_layer, hidden_states, ids, weights, **kwargs):
        return mega_moe.run_sm90_mega_routed(
            expert_layer, hidden_states, ids.to(torch.int32), weights, buf,
            kwargs["num_tokens"], hidden_size=kwargs["hidden_size"],
            routed_scaling_factor=kwargs["routed_scaling_factor"],
        )

    def skip_expert_gemm(y, *args, **kwargs):
        y.zero_()

    for router in ("text", "vision", "hash", "text with static EP mapping"):
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                gate = MoEGate(config, None, is_hash_moe=router == "hash",
                               is_deepseek_v4=True, vl_correction_bias=router == "vision")
                if router == "hash":
                    topk = HashTopK(k, experts, 0, vocab_size=128, routed_scaling_factor=scale, layer_id=0)
        finally:
            torch.set_default_dtype(original_dtype)
        gate.weight.normal_()
        if router != "hash":
            gate.e_score_correction_bias.normal_(std=0.1)
            topk = TopK(
                k, layer_id=0, scoring_func="sqrtsoftplus", renormalize=True,
                correction_bias=gate.e_score_correction_bias, sqrtsoftplus_log1p=True,
                output_format=TopKOutputFormat.STANDARD, routed_scaling_factor=scale,
                allow_routed_experts_capture=False,
            )
        if router == "vision":
            gate.e_score_correction_bias_vl.normal_(std=0.12)

        expert_layer = SimpleNamespace(
            should_fuse_routed_scaling_factor_in_topk=False, moe_runner_config=SimpleNamespace(swiglu_limit=None),
            mega_l1_weights=None, mega_l2_weights=None,
        )
        moe = SimpleNamespace(config=config, is_hash=router == "hash", gate=gate, topk=topk, experts=expert_layer,
                              layer_id=0, num_fused_shared_experts=0, routed_scaling_factor=scale,
                              mega_shared_l1_weights=None, mega_shared_l2_weights=None)
        outputs = []
        for enabled in (False, True):
            with (
                get_exec().moe.override(ep_dispatch_algorithm="static" if router.endswith("mapping") else None),
                envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(enabled),
                patch.object(mega_moe, "run_mega_routed_experts", side_effect=stage),
                patch.object(deep_gemm, "fp8_mega_moe", side_effect=skip_expert_gemm),
                patch.object(deep_gemm, "bf16_mega_gate", wraps=deep_gemm.bf16_mega_gate) as fused,
            ):
                mega_moe._run_mega_routed(moe, x, batch, input_ids, rows)
                assert fused.call_count == int(enabled), f"{router} did not select the expected routing path"
            assert (buf.topk_idx[valid_rows:] == -1).all()
            if enabled:
                assert (buf.topk_weights[valid_rows:] == 0).all()
            ids, order = buf.topk_idx[:valid_rows].sort(dim=1)
            outputs.append((ids, buf.topk_weights[:valid_rows].gather(1, order),
                            buf.x[:rows].view(torch.uint8).clone(), buf.x_sf[:rows].clone()))
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        print(f"{router} router and SM90 MegaMoE pre-dispatch passed")


if __name__ == "__main__":
    main()
