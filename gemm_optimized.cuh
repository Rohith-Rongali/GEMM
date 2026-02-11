#ifndef GEMM_OPTIMIZED_CUH
#define GEMM_OPTIMIZED_CUH

#include <cuda_pipeline.h>

// Optimizations:
//   1. Warp tiling:     Block(128×128) → Warp(64×32) → Thread(8×8)
//   2. Transposed A:    As[k][m] — enables float4 smem loads
//   3. float4 loads:    global and shared memory
//   4. cp.async:        B loaded directly global → shared, bypasses registers
//   5. __launch_bounds__/__restrict__: compiler register allocation hints
//
// Config (autotuned, best of 154 on L40S @ N=4096):
//   BM=128, BN=128, BK=32, WM=64, WN=32, TM=8, TN=8
//   8 warps (2×4), 256 threads, 32 KB smem

#define BM_W 128
#define BN_W 128
#define BK_W 32
#define WM_W 64
#define WN_W 32
#define TM_W 8
#define TN_W 8

__global__ __launch_bounds__(256)
void gemm_optimized(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, int N) {
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    const uint tid = threadIdx.x;

    constexpr uint WARP_COLS = BN_W / WN_W;
    const uint warpIdx = tid / 32;
    const uint warpRow = warpIdx / WARP_COLS;
    const uint warpCol = warpIdx % WARP_COLS;

    const uint tidInWarp = tid % 32;
    const uint threadRowInWarp = (tidInWarp / (WN_W / TN_W)) * TM_W;
    const uint threadColInWarp = (tidInWarp % (WN_W / TN_W)) * TN_W;

    // As transposed: As[k][m] so compute phase can float4-load along m
    __shared__ float As[BK_W][BM_W];
    __shared__ float Bs[BK_W][BN_W];

    float threadResults[TM_W][TN_W] = {0.0f};
    float regA[TM_W];
    float regB[TN_W];

    const uint ARowStart = by * BM_W;
    const uint BColStart = bx * BN_W;
    constexpr uint numThreads = 256;

    constexpr uint f4PerRowA    = BK_W / 4;
    constexpr uint f4PerRowB    = BN_W / 4;
    constexpr uint f4PerThreadA = (BM_W * f4PerRowA) / numThreads;
    constexpr uint f4PerThreadB = (BK_W * f4PerRowB) / numThreads;

    const uint numTiles = ((uint)N + BK_W - 1) / BK_W;

    for (uint tile = 0; tile < numTiles; ++tile) {
        const uint kOffset = tile * BK_W;

        // Load B via cp.async — no transpose needed, overlaps with A loads below
        #pragma unroll
        for (uint i = 0; i < f4PerThreadB; ++i) {
            uint f4Idx     = i * numThreads + tid;
            uint rowB      = f4Idx / f4PerRowB;
            uint f4ColB    = f4Idx % f4PerRowB;
            uint globalRow = kOffset + rowB;
            uint globalCol = BColStart + f4ColB * 4;

            if (globalRow < (uint)N && globalCol + 3 < (uint)N) {
                __pipeline_memcpy_async(&Bs[rowB][f4ColB * 4],
                                        &B[globalRow * N + globalCol],
                                        sizeof(float4));
            } else {
                for (uint e = 0; e < 4; ++e) {
                    uint gc = globalCol + e;
                    Bs[rowB][f4ColB * 4 + e] = (globalRow < (uint)N && gc < (uint)N)
                                                ? B[globalRow * N + gc] : 0.0f;
                }
            }
        }
        __pipeline_commit();

        // Load A via registers — transpose scatter: A[m][k] → As[k][m]
        #pragma unroll
        for (uint i = 0; i < f4PerThreadA; ++i) {
            uint f4Idx     = i * numThreads + tid;
            uint rowA      = f4Idx / f4PerRowA;
            uint f4ColA    = f4Idx % f4PerRowA;
            uint globalRow = ARowStart + rowA;
            uint globalCol = kOffset + f4ColA * 4;

            if (globalRow < (uint)N && globalCol + 3 < (uint)N) {
                float4 val = reinterpret_cast<const float4*>(&A[globalRow * N + globalCol])[0];
                As[f4ColA * 4 + 0][rowA] = val.x;
                As[f4ColA * 4 + 1][rowA] = val.y;
                As[f4ColA * 4 + 2][rowA] = val.z;
                As[f4ColA * 4 + 3][rowA] = val.w;
            } else {
                for (uint e = 0; e < 4; ++e) {
                    uint gc = globalCol + e;
                    As[f4ColA * 4 + e][rowA] = (globalRow < (uint)N && gc < (uint)N)
                                                ? A[globalRow * N + gc] : 0.0f;
                }
            }
        }

        __pipeline_wait_prior(0);
        __syncthreads();

        // Compute: float4 smem loads into registers, 8×8 outer product
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < BK_W; ++dotIdx) {
            #pragma unroll
            for (uint i = 0; i < TM_W; i += 4) {
                float4 a4 = reinterpret_cast<const float4*>(
                    &As[dotIdx][warpRow * WM_W + threadRowInWarp + i])[0];
                regA[i + 0] = a4.x; regA[i + 1] = a4.y;
                regA[i + 2] = a4.z; regA[i + 3] = a4.w;
            }

            #pragma unroll
            for (uint i = 0; i < TN_W; i += 4) {
                float4 b4 = reinterpret_cast<const float4*>(
                    &Bs[dotIdx][warpCol * WN_W + threadColInWarp + i])[0];
                regB[i + 0] = b4.x; regB[i + 1] = b4.y;
                regB[i + 2] = b4.z; regB[i + 3] = b4.w;
            }

            #pragma unroll
            for (uint m = 0; m < TM_W; ++m)
                #pragma unroll
                for (uint n = 0; n < TN_W; ++n)
                    threadResults[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (uint m = 0; m < TM_W; ++m) {
        uint row = ARowStart + warpRow * WM_W + threadRowInWarp + m;
        if (row < (uint)N) {
            #pragma unroll
            for (uint n = 0; n < TN_W; ++n) {
                uint col = BColStart + warpCol * WN_W + threadColInWarp + n;
                if (col < (uint)N)
                    C[row * N + col] = threadResults[m][n];
            }
        }
    }
}

#endif  // GEMM_OPTIMIZED_CUH
