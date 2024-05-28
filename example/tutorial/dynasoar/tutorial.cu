#include "tutorial.h"
#include "dynasoar.h"

__device__ AllocatorT* device_allocator;        // device side
AllocatorHandle<AllocatorT>* allocator_handle;  // host side

int main(int argc, char** argv)
{
  // Some boilerplate code.... Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    cudaMemcpyHostToDevice);

  int result;
  bool isCalculated = false;

  do_calc << <1, 1 >> > (9, &result, &isCalculated);
  cudaDeviceSynchronize();

  for (int i = 1; i < 20; i++)
  {
    if (isCalculated) {
      printf("-- Result: %i --\n", result);
      break;
    }
    printf("====== Iteration: %i ======\n", i);
    allocator_handle->parallel_do<Fib, &Fib::calc>();
    allocator_handle->parallel_do<Fib, &Fib::printInfo>();
    allocator_handle->parallel_do<Sum, &Sum::calc>();
    allocator_handle->parallel_do<Sum, &Sum::printInfo>();
  }
}

__global__ void do_calc(int n, int* result, bool* isCalculated)
{
  new(device_allocator) Fib(result, isCalculated, n);
}

__device__ void Fib::calc()
{
  if (n <= 1) {
    *result = n;
    *isCalculated = true;
    destroy(device_allocator, this);
    return;
  }
  Sum* sum = new(device_allocator) Sum(result, isCalculated);
  new(device_allocator) Fib(&sum->x, &sum->isXCalculated, n - 1);
  new(device_allocator) Fib(&sum->y, &sum->isYCalculated, n - 2);
  destroy(device_allocator, this);
}

__device__ void Fib::printInfo()
{
  printf("N: %i\n", (int)n);
}

__device__ void Sum::calc()
{
  if (isXCalculated && isYCalculated)
  {
    printf("x + y = %i\n", (int)x + (int)y);
    *result = x + y;
    *isCalculated = true;
    destroy(device_allocator, this);
  }
}

__device__ void Sum::printInfo()
{
  if (isXCalculated && isYCalculated)
  {
    printf("X: %i, Y: %i\n", (int)x, (int)y);
  }
  else if (isXCalculated)
  {
    printf("X: %i\n", (int)x);
  }
  else if (isYCalculated)
  {
    printf("Y: %i\n", (int)y);
  }
}
