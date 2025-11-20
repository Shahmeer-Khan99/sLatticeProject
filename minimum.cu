#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cuda.h>
#include <climits>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : " \
              << cudaGetErrorString(err) << " (" << err << ")\n"; \
    exit(EXIT_FAILURE); \
  } \
} while(0)

int sequential_minimum(const std::vector<int>& array) {
  assert(!array.empty());
  int minv = array[0];
  for (size_t i = 1; i < array.size(); ++i) {
    if (array[i] < minv) minv = array[i];
  }
  return minv;
}

__global__ void kernel_global_min_atomic(const int* data, size_t N, int* d_global_min) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicMin(d_global_min, data[idx]);
  }
}

int cuda_global_min_atomic(const std::vector<int>& host_array, int blocks, int threads_per_block, float &milliseconds_out) {
  size_t N = host_array.size();
  int *d_data = nullptr;
  int *d_global_min = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_data, host_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  int h_init = INT_MAX;
  CUDA_CHECK(cudaMalloc(&d_global_min, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_global_min, &h_init, sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  kernel_global_min_atomic<<<blocks, threads_per_block>>>(d_data, N, d_global_min);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds_out, start, stop));

  int h_min;
  CUDA_CHECK(cudaMemcpy(&h_min, d_global_min, sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_global_min));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return h_min;
}

template <unsigned int BLOCK_SIZE>
__global__ void kernel_block_reduce_min(const int* data, size_t N, int* d_block_mins) {
  extern __shared__ int sdata[]; // size = BLOCK_SIZE * sizeof(int)
  unsigned int tid = threadIdx.x;
  unsigned int blockStart = (blockIdx.x * (BLOCK_SIZE * 2));
  unsigned int idx1 = blockStart + tid;
  unsigned int idx2 = blockStart + tid + BLOCK_SIZE;

  int myVal = INT_MAX;

  if (idx1 < N) myVal = data[idx1];
  if (idx2 < N) {
    int v2 = data[idx2];
    if (v2 < myVal) myVal = v2;
  }
  sdata[tid] = myVal;
  __syncthreads();

  if (BLOCK_SIZE >= 1024) { if (tid < 512) { int v = sdata[tid + 512]; if (v < sdata[tid]) sdata[tid] = v; } __syncthreads(); }
  if (BLOCK_SIZE >= 512)  { if (tid < 256) { int v = sdata[tid + 256]; if (v < sdata[tid]) sdata[tid] = v; } __syncthreads(); }
  if (BLOCK_SIZE >= 256)  { if (tid < 128) { int v = sdata[tid + 128]; if (v < sdata[tid]) sdata[tid] = v; } __syncthreads(); }
  if (BLOCK_SIZE >= 128)  { if (tid < 64)  { int v = sdata[tid + 64];  if (v < sdata[tid]) sdata[tid] = v; } __syncthreads(); }

  if (tid < 32) {
    volatile int* smem = sdata;
    if (BLOCK_SIZE >= 64) { int v = smem[tid + 32]; if (v < smem[tid]) smem[tid] = v; }
    if (BLOCK_SIZE >= 32) { int v = smem[tid + 16]; if (v < smem[tid]) smem[tid] = v; }
    if (BLOCK_SIZE >= 16) { int v = smem[tid + 8];  if (v < smem[tid]) smem[tid] = v; }
    if (BLOCK_SIZE >= 8)  { int v = smem[tid + 4];  if (v < smem[tid]) smem[tid] = v; }
    if (BLOCK_SIZE >= 4)  { int v = smem[tid + 2];  if (v < smem[tid]) smem[tid] = v; }
    if (BLOCK_SIZE >= 2)  { int v = smem[tid + 1];  if (v < smem[tid]) smem[tid] = v; }
  }

  if (tid == 0) {
    d_block_mins[blockIdx.x] = sdata[0];
  }
}

int cuda_reduction_min(const std::vector<int>& host_array, int blocks, int threads_per_block, float &milliseconds_out) {
  size_t N = host_array.size();
  int *d_data = nullptr;

  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_data, host_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  const unsigned int BLOCK = 1024;
  size_t elemsPerBlock = BLOCK * 2ULL;
  int computed_blocks = (N + elemsPerBlock - 1) / elemsPerBlock;
  if (computed_blocks <= 0) computed_blocks = 1;

  int *d_block_mins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_block_mins, computed_blocks * sizeof(int)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  if (BLOCK != (unsigned int)threads_per_block) {}
  size_t sharedBytes = BLOCK * sizeof(int);

  kernel_block_reduce_min<BLOCK><<<computed_blocks, BLOCK, sharedBytes>>>(d_data, N, d_block_mins);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h_block_mins(computed_blocks);
  CUDA_CHECK(cudaMemcpy(h_block_mins.data(), d_block_mins, computed_blocks * sizeof(int), cudaMemcpyDeviceToHost));

  int final_min = INT_MAX;
  for (int v : h_block_mins) if (v < final_min) final_min = v;

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds_out, start, stop));

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_block_mins));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return final_min;
}

std::vector<int> generate_random_vector(size_t N, int seed = 42, bool put_known_min = true) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, INT_MAX/4);

  std::vector<int> a;
  a.reserve(N);
  for (size_t i = 0; i < N; ++i) a.push_back(dist(rng));

  if (put_known_min && N > 0) {
    a[N/3] = -1000000;
    a[N/2] = -500000;
    a[N-1] = -1;
  }
  return a;
}

void print_usage(const char* exe) {
  std::cout << "Usage: " << exe << " [N] [mode]\n";
  std::cout << "  N     : number of elements (default 1<<20)\n";
  std::cout << "  mode  : 0=sequential, 1=global_atomic, 2=reduction, 3=all (default 3)\n";
  std::cout << "Example: ./min_bench 1048576 3\n";
}

int main(int argc, char** argv) {
  size_t N = (1 << 20); // 1M default
  int mode = 3;
  if (argc >= 2) N = std::stoull(argv[1]);
  if (argc >= 3) mode = std::stoi(argv[2]);
  if (N == 0) { print_usage(argv[0]); return 1; }

  int threads_per_block = 1024;
  int preferred_blocks = 1024;
  int atomic_blocks = (N + threads_per_block - 1) / threads_per_block;
  if (atomic_blocks < 1) atomic_blocks = 1;

  std::vector<int> a = generate_random_vector(N, 12345, true);

  std::cout << "impl,N,blocks,threads_ms,time_ms,result\n";

  if (mode == 0 || mode == 3) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int seqmin = sequential_minimum(a);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "sequential," << N << ",-,-," << ms << "," << seqmin << "\n";
  }

  if (mode == 1 || mode == 3) {
    int blocks_to_launch = preferred_blocks;
    if ((size_t)blocks_to_launch * threads_per_block < N) {
      blocks_to_launch = atomic_blocks;
    }
    float gpu_ms = 0.0f;
    int gmin = cuda_global_min_atomic(a, blocks_to_launch, threads_per_block, gpu_ms);
    std::cout << "global_atomic," << N << "," << blocks_to_launch << "," << threads_per_block
              << "," << gpu_ms << "," << gmin << "\n";
  }

  if (mode == 2 || mode == 3) {
    float gpu_ms = 0.0f;
    int rmin = cuda_reduction_min(a, 0, threads_per_block, gpu_ms);
    std::cout << "reduction," << N << ",auto," << threads_per_block << "," << gpu_ms << "," << rmin << "\n";
  }

  return 0;
}
