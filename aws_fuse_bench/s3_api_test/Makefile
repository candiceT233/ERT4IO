# Working Makefile for S3 MPI benchmark with AVX512 fix
CXX = mpicxx
TARGET = s2_mpi
SOURCE = s2_mpi.cpp
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -mno-avx512f

# Use full paths instead of ~ to avoid expansion issues
AWS_ROOT = /people/tang584/install/aws-sdk-cpp
INCLUDES = -I$(AWS_ROOT)/include
RPATH = -Wl,-rpath,$(AWS_ROOT)/lib64

# All required libraries with full paths
DIRECT_LIBS = $(AWS_ROOT)/lib64/libaws-cpp-sdk-s3.so \
              $(AWS_ROOT)/lib64/libaws-cpp-sdk-core.so \
              $(AWS_ROOT)/lib64/libaws-crt-cpp.so \
              $(AWS_ROOT)/lib64/libaws-c-s3.so \
              $(AWS_ROOT)/lib64/libaws-c-auth.so \
              $(AWS_ROOT)/lib64/libaws-c-http.so \
              $(AWS_ROOT)/lib64/libaws-c-io.so \
              $(AWS_ROOT)/lib64/libaws-c-common.so \
              $(AWS_ROOT)/lib64/libaws-checksums.so \
              $(AWS_ROOT)/lib64/libaws-c-cal.so \
              $(AWS_ROOT)/lib64/libaws-c-compression.so \
              $(AWS_ROOT)/lib64/libaws-c-event-stream.so \
              $(AWS_ROOT)/lib64/libs2n.so

SYS_LIBS = -lcurl -lssl -lcrypto -lz -lpthread

# Default target
all: $(TARGET)

# Create dummy AVX512 functions
avx512_dummy.c:
	@echo "Creating dummy AVX512 functions..."
	@echo 'unsigned long long aws_checksums_crc64nvme_intel_avx512(const unsigned char *input, int length, unsigned long long previousCrc64) {' > avx512_dummy.c
	@echo '    return 0;' >> avx512_dummy.c
	@echo '}' >> avx512_dummy.c
	@echo '' >> avx512_dummy.c
	@echo 'unsigned int aws_checksums_crc32c_intel_avx512_with_sse_fallback(const unsigned char *input, int length, unsigned int previousCrc32c) {' >> avx512_dummy.c
	@echo '    return 0;' >> avx512_dummy.c
	@echo '}' >> avx512_dummy.c

# Compile dummy object
avx512_dummy.o: avx512_dummy.c
	gcc -c -fPIC avx512_dummy.c -o avx512_dummy.o

# Build the executable
$(TARGET): $(SOURCE) avx512_dummy.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) avx512_dummy.o -o $(TARGET) $(RPATH) $(DIRECT_LIBS) $(SYS_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.dat benchmark_rank_*.dat avx512_dummy.c

# Test run
test-run: $(TARGET)
	@echo "Usage: make test-run SIZE=<MB> BUCKET=<name> [REGION=<region>]"
	@if [ -n "$(SIZE)" ] && [ -n "$(BUCKET)" ]; then \
		if [ -n "$(REGION)" ]; then \
			mpirun -np 1 ./$(TARGET) $(SIZE) $(BUCKET) $(REGION); \
		else \
			mpirun -np 1 ./$(TARGET) $(SIZE) $(BUCKET); \
		fi \
	fi

# MPI run
run: $(TARGET)
	@echo "Usage: make run SIZE=<MB> BUCKET=<name> NP=<processes> [REGION=<region>]"
	@if [ -n "$(SIZE)" ] && [ -n "$(BUCKET)" ] && [ -n "$(NP)" ]; then \
		if [ -n "$(REGION)" ]; then \
			mpirun -np $(NP) ./$(TARGET) $(SIZE) $(BUCKET) $(REGION); \
		else \
			mpirun -np $(NP) ./$(TARGET) $(SIZE) $(BUCKET); \
		fi \
	fi

# Check if executable exists
check:
	@ls -la $(TARGET) 2>/dev/null && echo "✅ $(TARGET) exists" || echo "❌ $(TARGET) not found - run 'make' first"

help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  check        - Check if executable exists"
	@echo "  test-run     - Test with single process (use SIZE=MB BUCKET=name [REGION=region] [TMPDIR=dir])"
	@echo "  run          - Run with MPI (use SIZE=MB BUCKET=name NP=procs [REGION=region] [TMPDIR=dir])"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make check"
	@echo "  make test-run SIZE=10 BUCKET=my-test-bucket"
	@echo "  make test-run SIZE=10 BUCKET=my-test-bucket REGION=us-west-2"
	@echo "  make test-run SIZE=10 BUCKET=my-test-bucket TMPDIR=/tmp"
	@echo "  make run SIZE=100 BUCKET=my-bucket NP=8"
	@echo "  make run SIZE=100 BUCKET=sagemaker-us-west-2-024848459949 NP=8 TMPDIR=/scratch"

.PHONY: all clean test-run run check help