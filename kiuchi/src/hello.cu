#include <hello.h>

#include <stdio.h>

__global__ void helloCUDA()
{
  printf("Hello, CUDA!\n");
}

void hello()
{
  helloCUDA << <1, 1 >> > ();
  cudaDeviceSynchronize();
}