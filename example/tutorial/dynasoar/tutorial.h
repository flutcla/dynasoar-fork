#pragma once

#include "dynasoar.h"

class Fib;
class Sum;

using AllocatorT = SoaAllocator</*num_objs=*/ 262144, Fib, Sum>;

__global__ void do_calc(int n, int* result);

class Fib : public AllocatorT::Base
{
public:
  declare_field_types(Fib, int*, int)

public:
  Field<Fib, 0> result;
  Field<Fib, 1> n;

public:
  __device__ Fib(int* result, int n)
    : result(result), n(n) {}

  __device__ void calc();

  __device__ void printInfo();
};

class Sum : public AllocatorT::Base
{
public:
  declare_field_types(Sum, int*, int, int)

public:
  Field<Sum, 0> result;
  Field<Sum, 1> x;
  Field<Sum, 2> y;

public:
  __device__ Sum(int* result)
    : result(result) {
    x = -1;
    y = -1;
  }

  __device__ void calc();

  __device__ void printInfo();
};
