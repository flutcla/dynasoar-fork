#pragma once

#include "dynasoar.h"

class Foo;
class Bar;
class Fib;
class Sum;

using AllocatorT = SoaAllocator</*num_objs=*/ 262144, Foo, Bar, Fib, Sum>;

__global__ void do_calc(int n);

class Fib : public AllocatorT::Base
{
public:
  declare_field_types(Fib, int)

private:
  Field<Fib, 0> n;

public:
  __device__ Fib(int n) : n(n) {}

  __device__ void calc();

  __device__ void print_n() {
    printf("N: %i\n", (int)n);
  }
};

class Sum : public AllocatorT::Base
{
public:
  declare_field_types(Sum, int, int, int)

private:
  Field<Sum, 0> x;
  Field<Sum, 1> y;
  Field<Sum, 1> counter;

public:
  __device__ Sum() {}

  __device__ void calc();
};
