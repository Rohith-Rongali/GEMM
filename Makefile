NVCC   = /usr/local/cuda-12.4/bin/nvcc
TARGET = gemm
SOURCE = GEMM.cu
ARCH   = -arch=sm_89
FLAGS  = -O3 -use_fast_math -Xcompiler -Wall
LIBS   = -lcublas

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCE) gemm_optimized.cuh
	$(NVCC) $(ARCH) $(FLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
