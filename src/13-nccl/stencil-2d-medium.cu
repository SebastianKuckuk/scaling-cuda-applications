#include "../stencil-2d-util.h"

#include "../cuda-util.h"
#include "../nccl-util.h"

#include <mpi.h>


__global__ void stencil2D(const double *__restrict__ u, double *__restrict__ uNew,
                          size_t localBeginInnerX, size_t localEndInnerX,
                          size_t localBeginInnerY, size_t localEndInnerY,
                          size_t localNumCellsX) {

    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tidy = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t i0 = localBeginInnerX + tidx;
    const size_t i1 = localBeginInnerY + tidy;

    if (i0 < localEndInnerX && i1 < localEndInnerY) {
        uNew[i0 + i1 * localNumCellsX] = u[i0 + i1 * localNumCellsX]
            + alpha * (
                      u[(i0 - 1) +  i1      * localNumCellsX]
                +     u[(i0 + 1) +  i1      * localNumCellsX]
                +     u[ i0      + (i1 + 1) * localNumCellsX]
                +     u[ i0      + (i1 - 1) * localNumCellsX]
                - 4 * u[ i0      +  i1      * localNumCellsX]);
    }
}


struct Patch {
    // patch boundaries in global coordinates excluding boundaries
    size_t globalInnerBeginX;
    size_t globalInnerEndX;
    size_t globalInnerBeginY;
    size_t globalInnerEndY;

    // patch local extents including halos/ boundaries
    size_t localNumCellsX;
    size_t localNumCellsY;
    size_t localSize;          // in bytes

    // pointers to CPU allocation
    double* localU;

    // pointers to the GPU allocation
    double* d_localU;
    double* d_localUNew;

    // execution configuration
    dim3 blockSize;
    dim3 gridSize;

    // patch streams
    // TODO: which streams does the NCCL version need? note that NCCL operations on a
    //       single communicator issued from concurrent streams may deadlock
    TODO
};


int main(int argc, char *argv[]) {
    // initialize MPI
    MPI_Init(&argc, &argv);

    int numRanks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // choose GPU
    // Note: NCCL requires a distinct device per rank of a communicator
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = rank % numDevicesPerNode;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "Rank " << rank << " using device " << deviceId << std::endl;

    // initialize NCCL
    // TODO: create a unique ID on one rank, distribute it to all ranks, and create a
    //       communicator from it
    //       - NCCL has no rendezvous of its own, so use MPI to distribute the ID
    //       - the communicator binds to the current device, so this has to follow cudaSetDevice
    TODO

    // initialize patches
    int numPatches = numRanks;  // one patch per rank
    checkCudaError(cudaSetDevice(deviceId));

    Patch patch;                // each rank has only one patch
    int patchIdx = rank;

    size_t patchHeight = ceilingDivide(globalNumCellsY - 2, numPatches);

    // no partitioning in the x-dimension
    patch.globalInnerBeginX = 1;
    patch.globalInnerEndX = globalNumCellsX - 1;

    patch.globalInnerBeginY = 1 + patchIdx * patchHeight;
    patch.globalInnerEndY   = std::min(     // the end is either
        1 + (patchIdx + 1) * patchHeight,   // the beginning of the next patch, or
        globalNumCellsY - 1);               // the end of the global domain

    // local extents including halos
    patch.localNumCellsX = patch.globalInnerEndX - patch.globalInnerBeginX + 2;   // two halo layers of size one each
    patch.localNumCellsY = patch.globalInnerEndY - patch.globalInnerBeginY + 2;
    patch.localSize = patch.localNumCellsX * patch.localNumCellsY * sizeof(double);

    // execution configuration
    auto numInnerCellsX = patch.globalInnerEndX - patch.globalInnerBeginX;
    auto numInnerCellsY = patch.globalInnerEndY - patch.globalInnerBeginY;

    patch.blockSize = dim3(16, 16);
    patch.gridSize  = dim3(
        ceilingDivide(numInnerCellsX, patch.blockSize.x),
        ceilingDivide(numInnerCellsY, patch.blockSize.y));

    // create streams
    TODO

    // allocate CPU
    checkCudaError(cudaMallocHost(&patch.localU, patch.localSize));

    // allocate GPU
    // Note: NCCL operates on ordinary device allocations - no symmetric heap required
    checkCudaError(cudaMalloc((void **)&patch.d_localU, patch.localSize));
    checkCudaError(cudaMalloc((void **)&patch.d_localUNew, patch.localSize));

    // init temperature fields including their boundaries
    initTemperaturePatch(patch.localU,
        globalNumCellsX, globalNumCellsY,                   // global index space
        patch.localNumCellsX, patch.localNumCellsY,         // local index space
        patch.globalInnerBeginX, patch.globalInnerBeginY    // global offset for this patch
    );

    // copy data to GPU
    checkCudaError(cudaMemcpy(patch.d_localU, patch.localU, patch.localSize, cudaMemcpyHostToDevice));

    // initialize uNew by copying from u
    checkCudaError(cudaMemcpy(patch.d_localUNew, patch.d_localU, patch.localSize, cudaMemcpyDeviceToDevice));

    // define print and work
    auto print = [&](size_t it) {
        if (0 == rank)
            std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        // all ranks copy data back to CPU in parallel
        checkCudaError(cudaMemcpy(patch.localU, patch.d_localU, patch.localSize, cudaMemcpyDeviceToHost));

        // Note: brute-force full-synchronize approach for simplicity - optimize with point-to-point messages triggering next write
        for (int printRank = 0; printRank < numRanks; ++printRank) {
            MPI_Barrier(MPI_COMM_WORLD);    // make sure all ranks are in the same iteration

            if (rank == printRank) {        // only one rank per loop iteration is allowed to write
                writeTemperaturePatchNpy("../output/temperature_" + idx + ".npy",
                    patch.localU,
                    globalNumCellsX, globalNumCellsY, patch.localNumCellsX, patch.localNumCellsY,
                    numPatches, rank);
            }
        }
    };

    auto work = [&](size_t it) {
        // compute layers to be communicated
        // TODO: compute the first and the last inner row, each on the stream that will
        //       carry the matching send
        stencil2D<<<TODO>>>(TODO);
        stencil2D<<<TODO>>>(TODO);

        // start bulk compute
        stencil2D<<<TODO>>>(TODO);

        // exchange halos
        // TODO: exchange the halo layers with NCCL
        //       - group the operations so that matching sends and receives are resolved
        //         together instead of deadlocking on each other
        //       - no host synchronization is needed before sending: why?
        TODO

        // synchronize
        // TODO: wait for both streams - note that these waits are only required because
        //       the buffer swap below happens on the host
        TODO

        std::swap(patch.d_localU, patch.d_localUNew);

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    checkCudaError(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePushA("work");
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    checkCudaError(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    nvtxRangePop();

    // print stats and diagnostic result
    if (0 == rank)
        printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    checkCudaError(cudaMemcpy(patch.localU, patch.d_localU, patch.localSize, cudaMemcpyDeviceToHost));

    auto rankTotalTemperature = accumulateTemperature(patch.localU, patch.localNumCellsX, patch.localNumCellsY);
    std::cout << "  Total temperature on rank " << rank << " is " << rankTotalTemperature << std::endl;
    // reduce the total temperature across GPUs
    // TODO: reduce rankTotalTemperature into totalTemperature on rank 0, using NCCL
    //       instead of the MPI_Reduce of the previous version
    //       - NCCL collectives operate on device buffers, so the value has to be staged
    //         on the GPU first
    //       - the reduction is stream-ordered, as every other NCCL operation
    double totalTemperature = 0.0;
    TODO

    if (0 == rank)
        std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    TODO    // destroy the streams

    checkCudaError(cudaFree(patch.d_localU));
    checkCudaError(cudaFree(patch.d_localUNew));

    checkCudaError(cudaFreeHost(patch.localU));

    TODO    // destroy the NCCL communicator

    MPI_Finalize();

    return 0;
}
