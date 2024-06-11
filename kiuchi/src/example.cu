#include "example.h"

void exec_example() {
  // Some boilerplate code.... Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>();
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    cudaMemcpyHostToDevice);

  // Allocate a few objects.
  create_objs << <5, 10 >> > ();
  cudaDeviceSynchronize();

  // Run a do-all operations in parallel.
  allocator_handle->parallel_do<Bar, &Bar::increment_by_one>();

  // If a member function takes an argument, we have to specify its type here.
  allocator_handle->parallel_do<Bar, int, &Bar::increment_by_n>(/*n=*/ 10);

  // Now print some stuff.
  allocator_handle->parallel_do<Bar, &Bar::print_second>();
}