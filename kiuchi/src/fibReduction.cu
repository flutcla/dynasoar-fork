#include "fibReduction.h"
#define POW2(n) (1 << (n))
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

__device__ void initMem(int* data, int loopCount, int idx)
{
  if (idx >= POW2(loopCount - 1)) return;

  int currentOffset = POW2(loopCount - 1) - 1;
  int nextOffset = POW2(loopCount) - 1;
  int n = data[currentOffset + idx];
  if (n <= 1) return;
  data[currentOffset + idx] = -1;
  data[nextOffset + 2 * idx] = n - 1;
  data[nextOffset + 2 * idx + 1] = n - 2;
}

__device__ void addUp(int* data, int loopCount, int idx)
{
  if (idx >= POW2(loopCount - 1)) return;

  int currentOffset = POW2(loopCount - 1) - 1;
  int nextOffset = POW2(loopCount) - 1;
  int n = data[currentOffset + idx];
  if (n != -1) return;
  data[currentOffset + idx] = data[nextOffset + 2 * idx] + data[nextOffset + 2 * idx + 1];
}

__global__ void fibReductionCUDA(int* data)
{
  int n = data[0];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 1; i <= n; i++) {
    initMem(data, i, idx);
    __syncthreads();
  }
  for (int i = n; i > 0; i--) {
    addUp(data, i, idx);
    __syncthreads();
  }
}

int fibReduction(int n)
{
  int size = (POW2(n) - 1) * sizeof(int);

  int* h_data = (int*)malloc(size);
  h_data[0] = n;

  int* d_data = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, size));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice));

  int blockCount = 1024;
  int blockNeeded = POW2(n - 2);
  int gridCount = (blockNeeded + blockCount - 1) / blockCount;

  // dim3 grid(65535, 65535);
  // dim3 block(1024);
  fibReductionCUDA << <gridCount, blockCount >> > (d_data);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "kernel launch failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

  int result = h_data[0];

  for (int loop = 1; loop <= n; loop++)
  {
    for (int i = POW2(loop - 1) - 1; i < POW2(loop) - 1; i++)
    {
      printf("%d ", h_data[i]);
    }
    printf("\n");
  }

  free(h_data);
  cudaFree(d_data);

  return result;
}
