#!/bin/bash

g++ -O3 -march=native -o ../build/02-cpu-baseline ../src/02-cpu-baseline/stencil-2d.cpp
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/03-gpu-baseline ../src/03-gpu-baseline/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/04-work-partitioning ../src/04-work-partitioning/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/05-streams ../src/05-streams/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/06-mgpu ../src/06-mgpu/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/07-halos ../src/07-halos/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/07-halos-2 ../src/07-halos-2/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/08-p2p ../src/08-p2p/stencil-2d-solution.cu
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/09-overlap ../src/09-overlap/stencil-2d-solution.cu
nvcc -ccbin=mpic++ -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/11-mpi ../src/11-mpi/stencil-2d-solution.cu
nvcc -ccbin=mpic++ -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -o ../build/12-overlap ../src/12-overlap/stencil-2d-solution.cu
nvcc -ccbin=mpic++ -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -rdc=true -o ../build/13-nvshmem ../src/13-nvshmem/stencil-2d-solution.cu -lnvshmem
