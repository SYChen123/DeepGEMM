#pragma once

#include <cstdint>

namespace deep_gemm {

// Per CTA and phase: first entry, last exit and sum of intervals, in device ns.
// The caller clears the buffer before each launch. Phases on different warp
// roles can overlap, and a GEMM phase includes waits in the math warpgroup.
enum class SM90MegaMoEPhase {
    DispatchMetadata,
    DispatchPull,
    Linear1Gemm,
    Linear1Epilogue,
    Linear2Gemm,
    Linear2Scatter,
    CombineBarrier,
    CombineReduce,
    WorkspaceCleanup,
    Count
};

constexpr int kSM90MegaMoEProfilePhases = static_cast<int>(SM90MegaMoEPhase::Count);
constexpr int kSM90MegaMoEProfileFields = 3;

// Extended RS buffers append these counters for W13 and W2, in that order.
// Each entry contains the sampled K128 count, zero, and cumulative device ns.
// Sample one K128 out of four to limit the effect of the probe on scheduling.
enum class SM90MegaMoERSStage {
    FullWait,  // Sum of the weight and activation waits when split.
    WeightRead,
    WeightDecode,
    WGMMAIssue,
    WGMMAWait,
    Promote,
    Count
};

constexpr int kSM90MegaMoERSStages = static_cast<int>(SM90MegaMoERSStage::Count);
constexpr int kSM90MegaMoERSProfilePhases = kSM90MegaMoEProfilePhases + 2 * kSM90MegaMoERSStages;
constexpr int kSM90MegaMoERSProfilePeriod = 4;

}  // namespace deep_gemm
