#include "hello.h"
// #include "example.h"
#include "fibReduction.h"

int main(int argc, char** argv) {
  hello();
  // exec_example();
  int n = (argc == 2) ? atoi(argv[1]) : 15;
  printf("fib(%d) = %d\n", n, fibReduction(n));
}
