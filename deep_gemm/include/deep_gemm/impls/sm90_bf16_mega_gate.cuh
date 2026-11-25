#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/mega_gate.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

namespace sm90_mega_gate_trace {

// Wire format shared with scripts/bench_sm90_megagate.py. Each track has one writer.
constexpr uint32_t kSlots = 512;
enum Event : uint32_t {
    EmptyWaitBegin = 1, EmptyWaitEnd,
    TMAIssueBegin, TMAIssueEnd,
    FullWaitBegin, FullWaitEnd,
    WGMMAIssueBegin, WGMMAIssueEnd,
    WGMMAWaitBegin, WGMMAWaitEnd,
    DrainBegin, DrainEnd,
    ScratchBegin, ScratchEnd,
    PublishBegin, PublishEnd,
    RouteBegin, RouteEnd,
    RouteWinner,
};

template <bool kTrace, uint32_t kTracks>
struct Track {
    uint64_t* records = nullptr;
    uint32_t count = 0;

    CUTLASS_DEVICE Track(uint64_t* buffer, const int trace_cta, const uint32_t track, const bool writer) {
        if constexpr (kTrace) {
            if (writer and (trace_cta < 0 or static_cast<uint32_t>(trace_cta) == blockIdx.x)) {
                const uint64_t cta = trace_cta < 0 ? blockIdx.x : 0;
                records = buffer + (cta * kTracks + track) * (kSlots + 1) * 2;
            }
        }
    }

    CUTLASS_DEVICE void record(const Event event, const uint32_t aux = 0) {
        if constexpr (kTrace) {
            if (records != nullptr and ++count <= kSlots) {
                uint64_t timer;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer) :: "memory");
                records[count * 2] = (static_cast<uint64_t>(event) << 56) | aux;
                records[count * 2 + 1] = timer;
            }
        }
    }

    CUTLASS_DEVICE void finish() const {
        if constexpr (kTrace) {
            if (records != nullptr)
                records[0] = count;
        }
    }
};

} // namespace sm90_mega_gate_trace

template <uint32_t kAlignedExperts, uint32_t kBlockTokens, uint32_t kBlockExperts, uint32_t kNumSplitK,
          uint32_t kNumStages, uint32_t kStagesPerMerge, uint32_t kTopK, uint32_t kScoringType,
          uint32_t kNumThreads, bool kScoreOnStore, bool kHasUnmappedIds, bool kHasPhysicalMap, bool kTrace = false>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
sm90_bf16_mega_gate_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
                         const float* bias, const float* image_bias, const bool* image_token_mask,
                         const bool* mask, const int* to_physical_map, const int* logical_count,
                         int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
                         void* gmem_scratch, void* gmem_score_barriers,
                         uint32_t num_tokens, uint32_t hidden, uint32_t num_routed_experts,
                         uint32_t num_shared_experts, uint32_t num_duplicate_experts,
                         float routed_scaling_factor, int ep_rank, int64_t unmapped_topk_idx_stride,
                         const bool* fix_routing_mask, const bool* force_random,
                         uint64_t* trace_buffer, int trace_cta) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900 and __CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    namespace gate = epilogue::mega_gate;
    namespace trace = sm90_mega_gate_trace;
    constexpr auto kMajor = cute::UMMA::Major::K;
    constexpr uint32_t kAtomK = layout::mega_gate::BLOCK_K;
    constexpr uint32_t kBlockK = kAtomK * kStagesPerMerge;
    constexpr uint32_t kSwizzle = 128;
    constexpr uint32_t kExpertTiles = kAlignedExperts / kBlockExperts;
    constexpr uint32_t kProducers = kExpertTiles * kNumSplitK;
    constexpr uint32_t kWarps = kNumThreads / 32;
    constexpr uint32_t kExpertWaves = kAlignedExperts / 128;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using TraceTrack = trace::Track<kTrace, 3 + kWarps>;

    __shared__ bool route_here;

    const uint32_t token_block = blockIdx.x / kProducers;
    const uint32_t producer = blockIdx.x % kProducers;
    const uint32_t expert_base = (producer % kExpertTiles) * kBlockExperts;
    const uint32_t split = producer / kExpertTiles;
    const uint32_t token_base = token_block * kBlockTokens;
    const uint32_t valid_tokens = cute::min(kBlockTokens, num_tokens - token_base);
    const uint32_t k_blocks = hidden / (kBlockK * kNumSplitK);
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = ptx::get_lane_idx();
    const layout::mega_gate::Workspace<kNumSplitK, kBlockTokens, kAlignedExperts> workspace(
        gmem_scratch, gmem_score_barriers);
    auto* partial = workspace.get_score_ptr(token_block, split) + expert_base;

    cudaGridDependencySynchronize();

    using WGMMA = typename mma::sm90::BF16MMASelector<kBlockExperts>::type;
    constexpr uint32_t kXElements = kBlockTokens * kBlockK;
    constexpr uint32_t kWeightElements = kBlockExperts * kBlockK;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_x = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto* smem_weight = smem_x + kNumStages * kXElements;
    DG_STATIC_ASSERT(sizeof(Barrier) == sizeof(uint64_t), "Unexpected SM90 barrier size");
    __shared__ uint64_t barrier_storage[2 * kNumStages];
    auto* full = reinterpret_cast<Barrier*>(barrier_storage);
    auto* empty = full + kNumStages;
    if (threadIdx.x < kNumStages) {
        full[threadIdx.x].init(1);
        empty[threadIdx.x].init(4);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (threadIdx.x == 128) {
        TraceTrack tma_trace(trace_buffer, trace_cta, 0, true);
        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t stage = block % kNumStages;
            const uint32_t phase = (block / kNumStages) & 1;
            const uint32_t k = (split * k_blocks + block) * kBlockK;
            tma_trace.record(trace::EmptyWaitBegin, block);
            empty[stage].wait(phase ^ 1);
            tma_trace.record(trace::EmptyWaitEnd, block);
            tma_trace.record(trace::TMAIssueBegin, block);
            tma::copy<kBlockK, kBlockTokens, kSwizzle, cutlass::bfloat16_t>(
                &tensor_map_x, &full[stage], smem_x + stage * kXElements, k, token_base, 1);
            tma::copy<kBlockK, kBlockExperts, kSwizzle, cutlass::bfloat16_t>(
                &tensor_map_weight, &full[stage], smem_weight + stage * kWeightElements, k, expert_base, 1);
            full[stage].arrive_and_expect_tx((kXElements + kWeightElements) * sizeof(cutlass::bfloat16_t));
            tma_trace.record(trace::TMAIssueEnd, block);
        }
        tma_trace.finish();
    } else if (warp < 4) {
        TraceTrack mma_trace(trace_buffer, trace_cta, 1, threadIdx.x == 0);
        auto x_desc = mma::sm90::make_gmma_desc<kMajor, kBlockTokens, kAtomK, kSwizzle>(smem_x, 0, 0);
        auto weight_desc = mma::sm90::make_gmma_desc<kMajor, kBlockExperts, kAtomK, kSwizzle>(smem_weight, 0, 0);
        const uint32_t x_desc_lo = x_desc.reg32_[0];
        const uint32_t weight_desc_lo = weight_desc.reg32_[0];
        float accum[WGMMA::kNumAccum] = {};

        for (uint32_t block = 0; block < k_blocks; ++block) {
            const uint32_t stage = block % kNumStages;
            const uint32_t phase = (block / kNumStages) & 1;
            mma_trace.record(trace::FullWaitBegin, block);
            full[stage].wait(phase);
            mma_trace.record(trace::FullWaitEnd, block);
            mma_trace.record(trace::WGMMAIssueBegin, block);
            #pragma unroll
            for (auto& value : accum)
                ptx::warpgroup_fence_operand(value);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kBlockK; k += WGMMA::K) {
                const uint32_t atom = k / kAtomK;
                x_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajor, kBlockTokens, kAtomK, kSwizzle, nv_bfloat16>(
                    x_desc_lo + stage * kXElements * sizeof(cutlass::bfloat16_t) / 16,
                    0, k % kAtomK, atom * kBlockTokens * kAtomK);
                weight_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajor, kBlockExperts, kAtomK, kSwizzle, nv_bfloat16>(
                    weight_desc_lo + stage * kWeightElements * sizeof(cutlass::bfloat16_t) / 16,
                    0, k % kAtomK, atom * kBlockExperts * kAtomK);
                WGMMA::wgmma(x_desc, weight_desc, accum, block != 0 or k != 0);
            }
            ptx::warpgroup_commit_batch();
            mma_trace.record(trace::WGMMAIssueEnd, block);
            mma_trace.record(trace::WGMMAWaitBegin, block);
            ptx::warpgroup_wait<1>();
            mma_trace.record(trace::WGMMAWaitEnd, block);
            if (lane == 0 and block > 0)
                empty[(block - 1) % kNumStages].arrive();
        }
        mma_trace.record(trace::DrainBegin);
        ptx::warpgroup_wait<0>();
        mma_trace.record(trace::DrainEnd);
        #pragma unroll
        for (auto& value : accum)
            ptx::warpgroup_fence_operand(value);
        if (lane == 0)
            empty[(k_blocks - 1) % kNumStages].arrive();

        const uint32_t row = warp * 16 + lane / 4;
        const uint32_t col = (lane % 4) * 2;
        constexpr uint32_t kStoreScoring = kScoreOnStore ? kScoringType :
            static_cast<uint32_t>(layout::mega_gate::ScoringType::Identity);
        const auto score_pair = [](const float first, const float second) {
            return make_float2(gate::apply_scoring<kStoreScoring>(first), gate::apply_scoring<kStoreScoring>(second));
        };
        mma_trace.record(trace::ScratchBegin);
        #pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            if (row < valid_tokens)
                reinterpret_cast<float2*>(partial + row * kAlignedExperts + col)[i * 4] =
                    score_pair(accum[i * 4], accum[i * 4 + 1]);
            if (row + 8 < valid_tokens)
                reinterpret_cast<float2*>(partial + (row + 8) * kAlignedExperts + col)[i * 4] =
                    score_pair(accum[i * 4 + 2], accum[i * 4 + 3]);
        }
        mma_trace.record(trace::ScratchEnd);
        mma_trace.finish();
    }

    // The last producer routes this tile. Other CTAs never wait for work that
    // may still be queued, so the grid does not depend on CTA co-residency.
    __syncthreads();
    if (threadIdx.x == 0) {
        TraceTrack publish_trace(trace_buffer, trace_cta, 2, true);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> arrivals(*workspace.get_score_barrier_ptr(token_block));
        publish_trace.record(trace::PublishBegin);
        route_here = arrivals.fetch_add(1, cuda::memory_order_acq_rel) == kProducers - 1;
        if (route_here)
            arrivals.store(0, cuda::memory_order_relaxed);
        publish_trace.record(trace::PublishEnd);
        if (route_here)
            publish_trace.record(trace::RouteWinner);
        publish_trace.finish();
    }
    __syncthreads();
    if (not route_here)
        return;

    const auto load_reduced_score = [&](const uint32_t row, const uint32_t expert) {
        float value = 0.0f;
        #pragma unroll
        for (uint32_t part = 0; part < kNumSplitK; ++part)
            value += workspace.get_score_ptr(token_block, part)[row * kAlignedExperts + expert];
        return value;
    };

    TraceTrack route_trace(trace_buffer, trace_cta, 3 + warp, lane == 0 and warp < valid_tokens);
    route_trace.record(trace::RouteBegin, token_base + warp);
    for (uint32_t row = warp; row < valid_tokens; row += kWarps) {
        const uint32_t token = token_base + row;
        const uint32_t output_topk = kTopK + num_shared_experts;
        if (mask != nullptr and not mask[token]) {
            if (lane < output_topk) {
                topk_idx[token * output_topk + lane] = -1;
                topk_weights[token * output_topk + lane] = 0.0f;
            }
            if constexpr (kHasUnmappedIds) {
                if (lane < kTopK)
                    unmapped_topk_idx[token * unmapped_topk_idx_stride + lane] = -1;
            }
            continue;
        }
        if (force_random != nullptr and force_random[token]) {
            gate::store_random_topk_token<kTopK, kHasUnmappedIds, kHasPhysicalMap>(
                token, lane, num_routed_experts, num_shared_experts, logical_count, ep_rank,
                unmapped_topk_idx_stride, topk_idx, unmapped_topk_idx, topk_weights);
            continue;
        }

        int selected = -1;
        float selected_score = 0.0f;
        if (fix_routing_mask != nullptr and fix_routing_mask[token]) {
            if (lane < kTopK) {
                selected = static_cast<int>(unmapped_topk_idx[token * unmapped_topk_idx_stride + lane]);
                DG_TRAP_ONLY_DEVICE_ASSERT(selected >= 0 and selected < static_cast<int>(num_routed_experts));
                selected_score = load_reduced_score(row, selected);
            }
        } else {
            const auto* selection_bias = image_token_mask != nullptr and image_token_mask[token] ? image_bias : bias;
            float scores[kExpertWaves * 4];
            float unbiased[kExpertWaves * 4];
            bool scores_are_finite = true;
            #pragma unroll
            for (uint32_t wave = 0; wave < kExpertWaves; ++wave) {
                const uint32_t expert = wave * 128 + lane * 4;
                float4 values = {};
                #pragma unroll
                for (uint32_t part = 0; part < kNumSplitK; ++part) {
                    const auto* source = workspace.get_score_ptr(token_block, part) + row * kAlignedExperts + expert;
                    const auto partial_scores = ptx::ld_global(reinterpret_cast<const float4*>(source));
                    values.x += partial_scores.x;
                    values.y += partial_scores.y;
                    values.z += partial_scores.z;
                    values.w += partial_scores.w;
                }
                const float* raw = reinterpret_cast<const float*>(&values);
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    float rank_score = -cute::numeric_limits<float>::infinity();
                    if (expert + i < num_routed_experts) {
                        scores_are_finite = scores_are_finite and isfinite(raw[i]);
                        rank_score = raw[i];
                        if (selection_bias != nullptr) {
                            if constexpr (not kScoreOnStore)
                                rank_score = gate::apply_scoring<kScoringType>(rank_score);
                            rank_score += selection_bias[expert + i];
                        }
                    }
                    if constexpr (kNumSplitK > 1)
                        unbiased[wave * 4 + i] = raw[i];
                    scores[wave * 4 + i] = rank_score;
                }
            }
            DG_TRAP_ONLY_DEVICE_ASSERT(scores_are_finite);
            gate::select_warp_topk<kExpertWaves, kTopK>(scores, 0, selected);
            DG_TRAP_ONLY_DEVICE_ASSERT(lane >= kTopK or (selected >= 0 and selected < static_cast<int>(num_routed_experts)));
            if constexpr (kNumSplitK > 1) {
                selected_score = gate::select_warp_expert_value<kExpertWaves>(unbiased, 0, selected);
            } else if (lane < kTopK) {
                selected_score = load_reduced_score(row, selected);
            }
        }
        if (lane < kTopK) {
            if constexpr (not kScoreOnStore)
                selected_score = gate::apply_scoring<kScoringType>(selected_score);
        } else {
            selected_score = 0.0f;
        }
        gate::store_topk_token<kTopK, kHasUnmappedIds, kHasPhysicalMap>(
            token, lane, num_routed_experts, num_shared_experts, num_duplicate_experts, routed_scaling_factor,
            ep_rank, unmapped_topk_idx_stride, to_physical_map, logical_count, logical_count,
            topk_idx, unmapped_topk_idx, topk_weights, selected, selected_score);
    }
    route_trace.record(trace::RouteEnd, token_base + warp);
    route_trace.finish();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports SM90-family GPUs");
#endif
}

} // namespace deep_gemm
