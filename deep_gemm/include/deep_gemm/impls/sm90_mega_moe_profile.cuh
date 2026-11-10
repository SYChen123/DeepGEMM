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

}  // namespace deep_gemm
