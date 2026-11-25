#pragma once

#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/epilogue/mega_gate.cuh>

namespace deep_gemm::epilogue::mega_gate {

template <uint32_t kStoreScoringType, uint32_t kNumRoutedExperts,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
CUTLASS_DEVICE bool sm100_store_tmem_scores_to_global(float* scores, const uint32_t& tmem_base_addr,
                                                      const uint32_t& effective_umma_n,
                                                      const uint32_t& expert_base_idx,
                                                      const uint32_t& mma_cta_rank,
                                                      const uint32_t& gate_warp_idx) {
    constexpr uint32_t kNumTmemFragmentRows = 8;
    constexpr bool kPartitionedTokenPaths = UMMA_M == 128 and kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const auto subpartition_idx = gate_warp_idx % 4;
    const auto warp_idx_in_subpartition = gate_warp_idx / 4;
    constexpr uint32_t kTokenStride = kNumGateWarps / 4 * kNumTmemFragmentRows;
    const auto num_token_cols = kPartitionedTokenPaths ? effective_umma_n / 2 : effective_umma_n;
    const auto token_base_idx = kPartitionedTokenPaths ? subpartition_idx / 2 * num_token_cols : 0u;
    const auto expert_atom_idx = kPartitionedTokenPaths ? subpartition_idx % 2 : subpartition_idx;
    const auto global_expert_idx = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                   expert_atom_idx * 32 + ptx::get_lane_idx();
    const auto tmem_addr = tmem_base_addr + (subpartition_idx * 32 << 16);

    auto scores_are_finite = true;
    for (auto token_idx = warp_idx_in_subpartition * kNumTmemFragmentRows;
         token_idx < num_token_cols; token_idx += kTokenStride) {
        uint32_t values[kNumTmemFragmentRows];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr + token_idx,
                                              values[0], values[1], values[2], values[3],
                                              values[4], values[5], values[6], values[7]);
        cutlass::arch::fence_view_async_tmem_load();
        #pragma unroll
        for (uint32_t row_idx = 0; row_idx < kNumTmemFragmentRows; ++ row_idx) {
            const auto dst_token_idx = token_base_idx + token_idx + row_idx;
            const auto score = apply_scoring<kStoreScoringType>(__uint_as_float(values[row_idx]));
            scores_are_finite = scores_are_finite and isfinite(score);
            ptx::st_global(scores + static_cast<uint64_t>(dst_token_idx) * kNumRoutedExperts +
                           global_expert_idx, score);
        }
    }
    return scores_are_finite;
}

} // namespace deep_gemm::epilogue::mega_gate
