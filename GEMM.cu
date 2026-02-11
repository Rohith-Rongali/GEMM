// GEMM: C = A * B  (square N×N, row-major, single precision)
//
// Benchmarks CUDA kernel against cuBLAS FP32.
// Kernel: gemm_optimized (warp-tiled, transposed smem, float4, autotuned)
// Target: NVIDIA L40S (sm_89)
//
// Usage: make && ./gemm

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iomanip>
#include <sys/time.h>

#include "gemm_optimized.cuh"

using namespace std;

const int WARMUP_RUNS   = 10;
const int BENCHMARK_RUNS = 100;
vector<int> MATRIX_SIZES = {4096};

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

void init_matrices(float* A, float* B, float* C, int N) {
    struct timeval t;
    gettimeofday(&t, nullptr);
    srand(t.tv_usec);
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = 0.0f;
    }
}

void reset_matrix(float* C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
}

void verify_result(const float* ref, const float* res, int N, const char* name) {
    double max_err = 0.0, avg_err = 0.0;
    int bad = 0;
    for (int i = 0; i < N * N; i++) {
        double diff = fabs(ref[i] - res[i]);
        double rel  = diff / (fabs(ref[i]) + 1e-12);
        avg_err += rel;
        if (diff > max_err) max_err = diff;
        if (diff > 1e-3 && rel > 1e-5) bad++;
    }
    avg_err /= (N * N);
    if (bad > 0)
        printf("  FAILED %s: %d errors, max=%.2e avg=%.2e\n", name, bad, max_err, avg_err);
    else
        printf("  PASSED %s (max=%.2e avg=%.2e)\n", name, max_err, avg_err);
}

double gflops(int N, double ms) {
    return (2.0 * N * N * N / ms) / 1e6;
}

double benchmark_cublas(cublasHandle_t handle, const float* d_A, const float* d_B,
                        float* d_C, int N, int runs) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; i++)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

template<typename KernelFunc>
double benchmark_kernel(KernelFunc kernel, const float* d_A, const float* d_B,
                        float* d_C, int N, dim3 grid, dim3 block, int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; i++)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Device: " << prop.name
         << " (sm_" << prop.major << prop.minor << ", "
         << prop.multiProcessorCount << " SMs)" << endl;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cout << left  << setw(8)  << "Size"
         << right << setw(14) << "cuBLAS GFLOPS"
         << setw(14) << "Opt GFLOPS"
         << setw(10) << "% cuBLAS" << endl;
    cout << string(46, '-') << endl;

    for (int N : MATRIX_SIZES) {
        size_t bytes = N * N * sizeof(float);

        float* h_A   = new float[N * N];
        float* h_B   = new float[N * N];
        float* h_C   = new float[N * N];
        float* h_ref = new float[N * N];

        init_matrices(h_A, h_B, h_C, N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        double cublas_ms = benchmark_cublas(cublas_handle, d_A, d_B, d_C, N, BENCHMARK_RUNS);
        double cublas_gf = gflops(N, cublas_ms);
        CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes, cudaMemcpyDeviceToHost));

        constexpr int WARP_THREADS = (BM_W / TM_W) * (BN_W / TN_W);
        dim3 block_warp(WARP_THREADS, 1);
        dim3 grid_warp((N + BN_W - 1) / BN_W, (N + BM_W - 1) / BM_W);

        reset_matrix(h_C, N);
        CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

        double warp_ms = benchmark_kernel(gemm_optimized, d_A, d_B, d_C, N,
                                          grid_warp, block_warp, BENCHMARK_RUNS);
        double warp_gf = gflops(N, warp_ms);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        verify_result(h_ref, h_C, N, "Opt");
        double pct = (warp_gf / cublas_gf) * 100.0;

        cout << left  << setw(8)  << N
             << right << setw(14) << fixed << setprecision(1) << cublas_gf
             << setw(14) << warp_gf
             << setw(9)  << setprecision(0) << pct << "%" << endl;

        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_ref;
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
