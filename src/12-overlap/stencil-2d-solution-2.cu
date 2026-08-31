#include "../stencil-2d-util.h"

#include "../cuda-util.h"

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


// number of threads per block used for the temperature accumulation
// Note: fixed at compile time because the block-level reduction below uses a
//       statically sized shared-memory buffer
constexpr int accumulateBlockSize = 256;


__global__ void accumulateTemperature2D(const double *__restrict__ u,
                                        size_t localNumCellsX, size_t localNumCellsY,
                                        double *__restrict__ acc) {

    // sum the inner cells of this patch only - the surrounding ring is either a halo layer
    // owned by the neighbouring patch or a global boundary, exactly like the host-side
    // accumulateTemperature, so no cell is counted twice
    const size_t numInnerX = localNumCellsX - 2;
    const size_t numInnerCells = numInnerX * (localNumCellsY - 2);

    // grid-stride loop - one partial sum per thread, independent of the field size
    double sum = 0.0;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numInnerCells;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {

        const size_t i0 = 1 + idx % numInnerX;
        const size_t i1 = 1 + idx / numInnerX;
        sum += u[i0 + i1 * localNumCellsX];
    }

    // tree reduction inside the block
    __shared__ double blockSums[accumulateBlockSize];
    blockSums[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = accumulateBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(stride))
            blockSums[threadIdx.x] += blockSums[threadIdx.x + stride];
        __syncthreads();
    }

    // combine the per-block results
    // Note: double atomics require sm_60 or newer
    if (0 == threadIdx.x)
        atomicAdd(acc, blockSums[0]);
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
    cudaStream_t topStream;
    cudaStream_t bottomStream;
    cudaStream_t bulkStream;
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
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = rank % numDevicesPerNode;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "Rank " << rank << " using device " << deviceId << std::endl;

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
    checkCudaError(cudaStreamCreate(&patch.topStream));
    checkCudaError(cudaStreamCreate(&patch.bottomStream));
    checkCudaError(cudaStreamCreate(&patch.bulkStream));

    // allocate CPU
    checkCudaError(cudaMallocHost(&patch.localU, patch.localSize));

    // allocate GPU
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
        MPI_Request requests[4] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

        // start with receiving halos
        if (rank > 0)
            MPI_Irecv(&patch.d_localUNew[0 * patch.localNumCellsX], patch.localNumCellsX, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
        if (rank < numRanks - 1)
            MPI_Irecv(&patch.d_localUNew[(patch.localNumCellsY - 1) * patch.localNumCellsX], patch.localNumCellsX, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[2]);

        // compute layers to be communicated
        stencil2D<<<ceilingDivide(patch.localNumCellsX - 2, 256), 256, 0, patch.bottomStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, 1, 2, patch.localNumCellsX);
        stencil2D<<<ceilingDivide(patch.localNumCellsX - 2, 256), 256, 0, patch.topStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, patch.localNumCellsY - 2, patch.localNumCellsY - 1, patch.localNumCellsX);

        // start bulk compute
        stencil2D<<<patch.gridSize, patch.blockSize, 0, patch.bulkStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, 2, patch.localNumCellsY - 2, patch.localNumCellsX);
        
        // synchronize with lower and upper stream and send halos
        checkCudaError(cudaStreamSynchronize(patch.bottomStream));
        if (rank > 0)
            MPI_Isend(&patch.d_localUNew[1 * patch.localNumCellsX], patch.localNumCellsX, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);

        checkCudaError(cudaStreamSynchronize(patch.topStream));
        if (rank < numRanks - 1)
            MPI_Isend(&patch.d_localUNew[(patch.localNumCellsY - 2) * patch.localNumCellsX], patch.localNumCellsX, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);

        // wait for halo exchanges to complete
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        // synchronize bulk compute
        checkCudaError(cudaStreamSynchronize(patch.bulkStream));

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

    // accumulate the local temperature on the device
    // Note: a CUDA-aware MPI takes device pointers in collectives just as in the halo
    //       exchange above, so the field never has to be copied back to accumulate it
    // Note: slot 0 holds this rank's contribution, slot 1 the global sum on the root
    double *d_temperature;
    checkCudaError(cudaMalloc((void **)&d_temperature, 2 * sizeof(double)));
    checkCudaError(cudaMemsetAsync(d_temperature, 0, 2 * sizeof(double), patch.bulkStream));

    const size_t numInnerCells = (patch.localNumCellsX - 2) * (patch.localNumCellsY - 2);
    const int accumulateGridSize = static_cast<int>(std::max<size_t>(1, std::min<size_t>(
        1024, ceilingDivide(numInnerCells, accumulateBlockSize))));

    accumulateTemperature2D<<<accumulateGridSize, accumulateBlockSize, 0, patch.bulkStream>>>(
        patch.d_localU, patch.localNumCellsX, patch.localNumCellsY, d_temperature);

    // Note: unlike NCCL, MPI is not stream aware - the kernel producing the send buffer has
    //       to be waited for explicitly before the collective may read it
    checkCudaError(cudaStreamSynchronize(patch.bulkStream));

    // reduce the total temperature across GPUs
    MPI_Reduce(
        d_temperature,          // send buffer, on the device
        d_temperature + 1,      // receive buffer, on the device, only written on the root
        1,                      // count
        MPI_DOUBLE,             // datatype
        MPI_SUM,                // operation
        0,                      // root
        MPI_COMM_WORLD);        // communicator

    // a single 16-byte transfer replaces the full-field copy back
    double temperature[2] = { 0.0, 0.0 };
    checkCudaError(cudaMemcpy(temperature, d_temperature, 2 * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "  Total temperature on rank " << rank << " is " << temperature[0] << std::endl;
    if (0 == rank)
        std::cout << "  Total temperature is " << temperature[1] << std::endl;

    checkCudaError(cudaFree(d_temperature));

    // clean up
    checkCudaError(cudaStreamDestroy(patch.topStream));
    checkCudaError(cudaStreamDestroy(patch.bottomStream));
    checkCudaError(cudaStreamDestroy(patch.bulkStream));

    checkCudaError(cudaFree(patch.d_localU));
    checkCudaError(cudaFree(patch.d_localUNew));

    checkCudaError(cudaFreeHost(patch.localU));

    MPI_Finalize();

    return 0;
}
