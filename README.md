## GEMM
Hand-written CUDA SGEMM kernel achieving ~90% of cuBLAS FP32 performance on NVIDIA L40S.

## Usage
make && ./gemm

## Files
- `gemm_optimized.cuh` — kernel implementation
- `GEMM.cu` — benchmark harness
- `Makefile` — build system (targets: `all`, `run`, `clean`)
