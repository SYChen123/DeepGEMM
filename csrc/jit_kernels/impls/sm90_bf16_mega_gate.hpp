#pragma once

#include <algorithm>
#include <format>

#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <deep_gemm/layout/mega_gate.cuh>

#include "../../runtime/jit.hpp"
#include "../../runtime/runtime.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

struct SM90BF16MegaGateConfig {
    int block_tokens;
    int block_experts;
    int num_expert_groups;
    int num_split_k;
    int num_stages;
    int stages_per_merge;
    int num_threads;
    int num_ctas;
};

static SM90BF16MegaGateConfig get_sm90_bf16_mega_gate_config(
        const int num_tokens, const int hidden, const int num_routed_experts, const int num_sms) {
    constexpr int kAlignment = layout::mega_gate::kExpertAlignment;
    constexpr int kBlockK = layout::mega_gate::BLOCK_K;
    const int aligned_experts = align(num_routed_experts, kAlignment);
    constexpr int block_tokens = layout::mega_gate::kSm90BlockTokens;
    int block_experts = 32;
    int max_splits = 8;
    int num_threads = 256;
    if (num_tokens >= 16) {
        block_experts = num_tokens <= 64 ? 16 : num_tokens < 1024 ? 64 : 128;
        max_splits = num_tokens <= 1024 ? 2 : 1;
        if (num_tokens <= 1024)
            num_threads = num_tokens < 1024 ? 512 : 384;
    }
    const int num_expert_groups = aligned_experts / block_experts;
    const int num_token_blocks = ceil_div(num_tokens, block_tokens);
    const int num_tiles = num_token_blocks * num_expert_groups;
    const bool is_v41_flash_gate = hidden == 5120 and num_routed_experts == 384;
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms();
    if (is_v41_flash_gate and not deterministic and num_tokens > 16 and num_tokens <= 32 and
        num_tiles * 4 <= num_sms) {
        // Cover more SMs during projection; the last producer has one warp per routed row.
        max_splits = 4;
        num_threads = 1024;
    }
    int num_split_k = 1;
    if (not deterministic) {
        while (num_split_k < max_splits and num_tiles * num_split_k * 2 <= num_sms and
               hidden % (kBlockK * num_split_k * 2) == 0)
            num_split_k *= 2;
    }
    // Four K64 stages let the TMA producer queue more independently ready stages
    // at the same shared-memory cost as two K128 stages.
    const bool use_four_stages = is_v41_flash_gate and num_tokens >= 4096;
    const int stages_per_merge = not use_four_stages and hidden % (2 * kBlockK * num_split_k) == 0 ? 2 : 1;
    const int num_stages = stages_per_merge == 1 ? 4 : block_experts <= 64 ? 3 : 2;
    return {block_tokens, block_experts, num_expert_groups, num_split_k, num_stages,
            stages_per_merge, num_threads, num_tiles * num_split_k};
}

static void sm90_bf16_mega_gate(
        const torch::Tensor& x, const torch::Tensor& weight,
        const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
        const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& mask,
        const std::optional<torch::Tensor>& fix_routing_mask, const std::optional<torch::Tensor>& to_physical_map,
        const std::optional<torch::Tensor>& logical_count, const torch::Tensor& topk_idx,
        const std::optional<torch::Tensor>& unmapped_topk_idx, const torch::Tensor& topk_weights,
        const std::optional<torch::Tensor>& force_random, const int num_tokens, const int hidden,
        const int num_routed_experts, const int num_topk, const int num_shared_experts,
        const float routed_scaling_factor, const int ep_rank, const int scoring_type,
        const SM90BF16MegaGateConfig& config, const torch::Tensor& scratch, const torch::Tensor& score_barriers) {
    constexpr int kBlockK = layout::mega_gate::BLOCK_K;
    constexpr int kAlignment = layout::mega_gate::kExpertAlignment;
    const int aligned_experts = align(num_routed_experts, kAlignment);
    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                               config.block_tokens, kBlockK, hidden, 1, 128);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                    config.block_experts, kBlockK, hidden, 1, 128);
    const bool score_on_store = config.num_split_k == 1 and (bias.has_value() or image_bias.has_value());
    // The benchmark owns this buffer and keeps it alive until the eager launch completes.
    uint64_t* trace_buffer = nullptr;
    int trace_cta = -1;
    if (const auto value = deep_jit::get_env<std::string>("DG_SM90_MEGA_GATE_TRACE_PTR"); not value.empty())
        trace_buffer = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(std::stoull(value)));
    if (trace_buffer != nullptr) {
        trace_cta = deep_jit::get_env<int>("DG_SM90_MEGA_GATE_TRACE_CTA", -1);
        DG_HOST_ASSERT(trace_cta >= -1 and trace_cta < config.num_ctas);
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
    }
    const auto kernel = jit->compile("sm90_bf16_mega_gate", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_gate_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}};
)", aligned_experts, config.block_tokens, config.block_experts, config.num_split_k,
        config.num_stages, config.stages_per_merge, num_topk, scoring_type, config.num_threads, score_on_store,
        unmapped_topk_idx.has_value(), to_physical_map.has_value(), trace_buffer != nullptr));

    jit->launch(
        kernel, {
            .num_smem_bytes = config.num_stages * (config.block_tokens + config.block_experts) * config.stages_per_merge *
                kBlockK * sizeof(cutlass::bfloat16_t),
            .grid_dim = dim3(config.num_ctas, 1, 1),
            .block_dim = dim3(config.num_threads, 1, 1),
        },
        tensor_map_x, tensor_map_weight, bias ? bias->data_ptr() : nullptr,
        image_bias ? image_bias->data_ptr() : nullptr, image_token_mask ? image_token_mask->data_ptr() : nullptr,
        mask ? mask->data_ptr() : nullptr, to_physical_map ? to_physical_map->data_ptr() : nullptr,
        logical_count ? logical_count->data_ptr() : nullptr, topk_idx.data_ptr(),
        unmapped_topk_idx ? unmapped_topk_idx->data_ptr() : nullptr, topk_weights.data_ptr(),
        scratch.data_ptr(), score_barriers.data_ptr(), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(hidden), static_cast<uint32_t>(num_routed_experts), static_cast<uint32_t>(num_shared_experts),
        to_physical_map ? static_cast<uint32_t>(to_physical_map->size(1)) : 0u,
        routed_scaling_factor, ep_rank, unmapped_topk_idx ? unmapped_topk_idx->stride(0) : int64_t{0},
        fix_routing_mask ? fix_routing_mask->data_ptr() : nullptr, force_random ? force_random->data_ptr() : nullptr,
        trace_buffer, trace_cta);
}

} // namespace deep_gemm
