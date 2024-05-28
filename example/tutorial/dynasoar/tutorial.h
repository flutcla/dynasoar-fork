#pragma once

#include "dynasoar.h"

class Fib;
class Sum;

using AllocatorT = SoaAllocator</*num_objs=*/ 262144, Fib, Sum>;

__global__ void do_calc(int n, int* result, bool* isCalculated);

class Fib : public AllocatorT::Base
{
public:
  declare_field_types(Fib, int*, bool*, int)

private:
  Field<Fib, 0> result;
  Field<Fib, 1> isCalculated;
  Field<Fib, 2> n;

public:
  __device__ Fib(int* result, bool* isCalculated, int n)
    : result(result), isCalculated(isCalculated), n(n) {}

  __device__ void calc();

  __device__ void printInfo();
};

class Sum : public AllocatorT::Base
{
public:
  declare_field_types(Sum, int*, bool*, int, int, bool, bool)

public:
  Field<Sum, 0> result;
  Field<Sum, 1> isCalculated;
  Field<Sum, 2> x;
  Field<Sum, 3> y;
  Field<Sum, 4> isXCalculated;
  Field<Sum, 5> isYCalculated;

public:
  __device__ Sum(int* result, bool* isCalculated)
    : result(result), isCalculated(isCalculated) {}

  __device__ void calc();

  __device__ void printInfo();
};
