NVCC ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra
TARGET := qr8_benchmark
SOURCE := src/qr8_benchmark.cu
INCLUDES := -Iinclude
LIBS := -lcublas -lcusolver -lcurand

ifeq ($(CUDA_ARCH),native)
ARCH_FLAGS := -arch=native
else
ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

.PHONY: all clean smoke

all: $(TARGET)

$(TARGET): $(SOURCE) include/benchmark_types.hpp
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(INCLUDES) $(SOURCE) -o $@ $(LIBS)

smoke: $(TARGET)
	mkdir -p results
	./$(TARGET) --m 512 --n 32 --warmups 1 --trials 2 --maximum-refinements 2 --raw-csv results/smoke.csv
	python3 tests/validate_results.py results/smoke.csv

.PHONY: suite-smoke
suite-smoke: $(TARGET)
	mkdir -p results/suite-smoke
	python3 scripts/run_suite.py --synthetic-only --synthetic-grid config/synthetic_grid.json --warmups 1 --trials 2 --output both --results results/suite-smoke

clean:
	rm -f $(TARGET)
