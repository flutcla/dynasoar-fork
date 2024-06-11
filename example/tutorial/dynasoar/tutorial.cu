#include "tutorial.h"
#include "dynasoar.h"

__device__ AllocatorT* device_allocator;        // device side
AllocatorHandle<AllocatorT>* allocator_handle;  // host side

int DEFAULT = 36;

int main(int argc, char** argv)
{
  int n = DEFAULT;
  if (argc == 2) {
    n = atoi(argv[1]);
    if (n == 0) n = DEFAULT;
  }

  // Some boilerplate code.... Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    cudaMemcpyHostToDevice);

  int h_result;
  int* d_result;

  cudaMalloc(&d_result, sizeof(int));

  do_calc << <1, 1 >> > (n, d_result);
  cudaDeviceSynchronize();

  struct timespec cpu_time_start, cpu_time_end;
  double cpu_time;
  timespec_get(&cpu_time_start, TIME_UTC);
  for (int i = 1; i < 200; i++)
  {
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_result != -1) {
      printf("-- Result: Fib(%i) = %i --\n", n, h_result);
      break;
    }
    // printf("====== Iteration: %i ======\n", i);
    allocator_handle->parallel_do<Fib, &Fib::calc>();
    // allocator_handle->parallel_do<Fib, &Fib::printInfo>();
    allocator_handle->parallel_do<Sum, &Sum::calc>();
    // allocator_handle->parallel_do<Sum, &Sum::calc>();
    // allocator_handle->parallel_do<Sum, &Sum::printInfo>();
    // allocator_handle->DBG_print_state_stats();
  }
  timespec_get(&cpu_time_end, TIME_UTC);
  cpu_time = (cpu_time_end.tv_sec - cpu_time_start.tv_sec) +
    (cpu_time_end.tv_nsec - cpu_time_start.tv_nsec) / 1e9;
  printf("fib_single(%d) = %d (%f sec)\n", n, h_result, cpu_time);
}

__global__ void do_calc(int n, int* result)
{
  *result = -1;
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
  result = &sum->y;
  n = n - 2;
  // new(device_allocator) Fib(&sum->y, n - 2);
  // destroy(device_allocator, this);
}

#ifdef PRINT_INFO
__device__ void Fib::printInfo()
{
  printf("N: %i\n", (int)n);
}
#endif

__device__ void Sum::calc()
{
  if (x != -1 && y != -1)
  {
    *result = x + y;
    destroy(device_allocator, this);
  }
}

#ifdef PRINT_INFO
__device__ void Sum::printInfo()
{
  printf("X: %i, Y: %i\n", (int)x, (int)y);
}
#endif
