#pragma once

#include "dynasoar.h"
// #define PRINT_INFO

class Result;
class Fib;
class Sum;

using AllocatorT = SoaAllocator</*num_objs=*/ 262144000, Result, Fib, Sum>;

__global__ void do_calc(int n, intptr_t* result);

class Result : public AllocatorT::Base
{
public:
  declare_field_types(Result, intptr_t*)

public:
  Field<Result, 0> result;

public:
  __device__ Result(intptr_t* result)
    : result(result) {}
};

class Fib : public AllocatorT::Base
{
public:
  declare_field_types(Fib, intptr_t*, int)

public:
  Field<Fib, 0> result;
  Field<Fib, 1> n;

public:
  __device__ Fib(intptr_t* result, int n)
    : result(result), n(n) {}

  __device__ void calc();

#ifdef PRINT_INFO
  __device__ void printInfo();
#endif
};

class Sum : public AllocatorT::Base
{
public:
  declare_field_types(Sum, intptr_t*, intptr_t, intptr_t)

public:
  Field<Sum, 0> result;
  Field<Sum, 1> x;
  Field<Sum, 2> y;

public:
  __device__ Sum(intptr_t* result)
    : result(result) {
    x = -1;
    y = -1;
  }

  __device__ void calc();

#ifdef PRINT_INFO
  __device__ void printInfo();
#endif
};
