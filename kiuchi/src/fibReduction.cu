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

__device__ void initMem(int* sdata, int loopCount, int idx)
{
  if (idx >= POW2(loopCount - 1)) return;

  int currentOffset = POW2(loopCount - 1) - 1;
  int nextOffset = POW2(loopCount) - 1;
  int n = sdata[currentOffset + idx];
  if (n <= 1) return;
  sdata[currentOffset + idx] = -1;
  sdata[nextOffset + 2 * idx] = n - 1;
  sdata[nextOffset + 2 * idx + 1] = n - 2;
}

__device__ void addUp(int* sdata, int loopCount, int idx)
{
  if (idx >= POW2(loopCount - 1)) return;

  int currentOffset = POW2(loopCount - 1) - 1;
  int nextOffset = POW2(loopCount) - 1;
  int n = sdata[currentOffset + idx];
  if (n != -1) return;
  sdata[currentOffset + idx] = sdata[nextOffset + 2 * idx] + sdata[nextOffset + 2 * idx + 1];
}

__global__ void fibReductionCUDA(int* g_idata, int* g_odata)
{
  extern __shared__ int sdata[];
  int n = g_idata[0];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0)
  {
    sdata[0] = n;
  }
  __syncthreads();

  for (int i = 1; i <= n; i++) {
    initMem(sdata, i, idx);
    __syncthreads();
  }
  for (int i = n; i > 0; i--) {
    addUp(sdata, i, idx);
    __syncthreads();
  }

  if (idx == 0)
  {
    for (int i = 0; i < POW2(n) - 1; i++)
    {
      g_odata[i] = sdata[i];
    }
  }
}

int fibReduction(int n)
{
  int size = (POW2(n) - 1) * sizeof(int);

  int* h_idata = (int*)malloc(sizeof(int));
  int* h_odata = (int*)malloc(size);
  h_idata[0] = n;

  int* d_idata = NULL;
  int* d_odata = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, size));
  CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, sizeof(int), cudaMemcpyHostToDevice));

  int threadCount = 1024;
  int threadNeeded = POW2(n - 2);
  int blockCount = threadNeeded / threadCount;

  fibReductionCUDA << <blockCount, threadCount, size >> > (d_idata, d_odata);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "kernel launch failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));

  int result = h_odata[0];

  // for (int loop = 1; loop <= n; loop++)
  // {
  //   for (int i = POW2(loop - 1) - 1; i < POW2(loop) - 1; i++)
  //   {
  //     printf("%d ", h_odata[i]);
  //   }
  //   printf("\n");
  // }

  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);

  return result;
}
