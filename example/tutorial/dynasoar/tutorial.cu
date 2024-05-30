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

  int result = -1;

  do_calc << <1, 1 >> > (20, &result);
  cudaDeviceSynchronize();

  for (int i = 1; i < 50; i++)
  {
    if (result != -1) {
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

__global__ void do_calc(int n, int* result)
{
  new(device_allocator) Fib(result, n);
}

__device__ void Fib::calc()
{
  if (n <= 1) {
    *result = n;
    destroy(device_allocator, this);
    return;
  }
  Sum* sum = new(device_allocator) Sum(result);
  new(device_allocator) Fib(&sum->x, n - 1);
  new(device_allocator) Fib(&sum->y, n - 2);
  destroy(device_allocator, this);
}

__device__ void Fib::printInfo()
{
  printf("N: %i\n", (int)n);
}

__device__ void Sum::calc()
{
  if (x != -1 && y != -1)
  {
    printf("x + y = %i\n", (int)x + (int)y);
    *result = x + y;
    destroy(device_allocator, this);
  }
}

__device__ void Sum::printInfo()
{
  printf("X: %i, Y: %i\n", (int)x, (int)y);
}
