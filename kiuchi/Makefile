OPTIMIZATIONS = -O3 -DNDEBUG
BIN_DIR = bin/
OBJ_DIR = obj/
SRC_DIR = src/
CUDA_SRC = $(wildcard $(SRC_DIR)*.cu)
C_SRC = $(wildcard $(SRC_DIR)*.c)
CPP_SRC = $(wildcard $(SRC_DIR)*.cpp)
OBJ = $(patsubst $(SRC_DIR)%.cu,$(OBJ_DIR)%.o,$(CUDA_SRC)) \
      $(patsubst $(SRC_DIR)%.c,$(OBJ_DIR)%.o,$(C_SRC)) \
      $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(CPP_SRC))
OUT = $(BIN_DIR)main

NVCC = nvcc
CXX = g++
CC = gcc
CUDA_CXXFLAGS = -rdc=true -Xcudafe "--diag_suppress=1427" -std=c++11 -lineinfo --expt-extended-lambda -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -maxrregcount=64
CXXFLAGS = -std=c++11
CFLAGS = -std=c11
INCLUDES = -I./src -I../lib/cub -I../ -I../example/configuration/dynasoar

.PHONY: all clean
all: $(BIN_DIR) $(OBJ_DIR) $(OUT)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.cu
	$(NVCC) $(OPTIMIZATIONS) $(CUDA_CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.c
	$(CC) $(OPTIMIZATIONS) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) $(OPTIMIZATIONS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUT): $(OBJ)
	$(NVCC) $(OPTIMIZATIONS) $(CUDA_CXXFLAGS) $(INCLUDES) $(OBJ) -o $@

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)
