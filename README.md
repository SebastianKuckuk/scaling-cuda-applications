# Scaling CUDA-Accelerated Applications

This repository collects material for the two-day workshop *Scaling CUDA-Accelerated Applications*.

## Sections

The tutorial starts with [01-introduction](./material/01-introduction.ipynb) which introduces the test application to be scaled in this course.
It is followed by incremental revisions of the application to discuss and demonstrate different aspects of multi-GPU and multi-node programming.

* [02-cpu-baseline](./material/02-cpu-baseline.ipynb) presents a serial CPU-only baseline implementation.
* [03-gpu-baseline](./material/03-gpu-baseline.ipynb) ports the baseline to a single GPU with CUDA.
* [04-work-partitioning](./material/04-work-partitioning.ipynb) explores work partitioning as preparatory step.
* [05-streams](./material/05-streams.ipynb) (re-)introduces CUDA streams and uses them to accelerate the partitioned workload.
* [06-mgpu](./material/06-mgpu.ipynb) scales execution to multiple GPUs on one node.
* [07-halos](./material/07-halos.ipynb) introduces partitioned allocations and adds halo exchanges for domain decomposition across GPUs.
* [07-halos-2](./material/07-halos-2.ipynb) de-serializes execution across GPUs.
* [08-p2p](./material/08-p2p.ipynb) uses GPU peer-to-peer transfers for inter-GPU communication.
* [09-overlap](./material/09-overlap.ipynb) overlaps communication and computation.
* [10-mpi-hw](./material/10-mpi-hw.ipynb) introduces MPI with a Hello World application.
* [11-mpi](./material/11-mpi.ipynb) applies MPI to decompose the stencil across GPU-equipped nodes.
* [12-overlap](./material/12-overlap.ipynb) overlaps MPI communication with GPU computation.
* [13-nccl](./material/13-nccl.ipynb) implements communication using NCCL.
* [14-nvshmem](./material/14-nvshmem.ipynb) implements communication using NVSHMEM.
* [15-outlook](./material/15-outlook.ipynb) outlines next steps.

## Further Learning

* TBA
