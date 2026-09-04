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
    // Note: both communicated layers share one stream - NCCL operations on a single
    //       communicator issued from concurrent streams may deadlock.
    cudaStream_t haloStream;
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
    // Note: NCCL requires a distinct device per rank of a communicator
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = rank % numDevicesPerNode;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "Rank " << rank << " using device " << deviceId << std::endl;

    // initialize NCCL
    // Note: NCCL provides no launcher and no rendezvous of its own - the unique ID
    //       has to be distributed out of band, here with MPI
    ncclUniqueId ncclId;
    if (0 == rank)
        checkNcclError(ncclGetUniqueId(&ncclId));
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Note: the communicator binds to the current device, so this has to follow cudaSetDevice
    ncclComm_t ncclComm;
    checkNcclError(ncclCommInitRank(&ncclComm, numRanks, ncclId, rank));

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
    checkCudaError(cudaStreamCreate(&patch.haloStream));
    checkCudaError(cudaStreamCreate(&patch.bulkStream));

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
        stencil2D<<<ceilingDivide(patch.localNumCellsX - 2, 256), 256, 0, patch.haloStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, 1, 2, patch.localNumCellsX);
        stencil2D<<<ceilingDivide(patch.localNumCellsX - 2, 256), 256, 0, patch.haloStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, patch.localNumCellsY - 2, patch.localNumCellsY - 1, patch.localNumCellsX);

        // start bulk compute
        stencil2D<<<patch.gridSize, patch.blockSize, 0, patch.bulkStream>>>(
            patch.d_localU, patch.d_localUNew, 1, patch.localNumCellsX - 1, 2, patch.localNumCellsY - 2, patch.localNumCellsX);

        // exchange halos
        // Note: no host synchronization is needed before sending - the send is enqueued
        //       behind the kernel that produces the layer, on the very same stream
        // Note: the group call allows the matching send/ receive pairs to be resolved
        //       together instead of deadlocking on each other
        checkNcclError(ncclGroupStart());
        if (rank > 0) {
            checkNcclError(ncclRecv(&patch.d_localUNew[0 * patch.localNumCellsX],
                patch.localNumCellsX, ncclDouble, rank - 1, ncclComm, patch.haloStream));
            checkNcclError(ncclSend(&patch.d_localUNew[1 * patch.localNumCellsX],
                patch.localNumCellsX, ncclDouble, rank - 1, ncclComm, patch.haloStream));
        }
        if (rank < numRanks - 1) {
            checkNcclError(ncclRecv(&patch.d_localUNew[(patch.localNumCellsY - 1) * patch.localNumCellsX],
                patch.localNumCellsX, ncclDouble, rank + 1, ncclComm, patch.haloStream));
            checkNcclError(ncclSend(&patch.d_localUNew[(patch.localNumCellsY - 2) * patch.localNumCellsX],
                patch.localNumCellsX, ncclDouble, rank + 1, ncclComm, patch.haloStream));
        }
        checkNcclError(ncclGroupEnd());

        // synchronize both streams
        // Note: these waits are only required because the buffer swap below happens on
        //       the host - they are not communication waits
        checkCudaError(cudaStreamSynchronize(patch.haloStream));
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
    // Note: NCCL collectives operate on device buffers - accumulating on the GPU keeps the
    //       reduction where the data already is, so neither the field nor a staging value
    //       has to cross the PCIe/ NVLink boundary before the collective
    double *d_temperature;
    checkCudaError(cudaMalloc((void **)&d_temperature, sizeof(double)));
    checkCudaError(cudaMemsetAsync(d_temperature, 0, sizeof(double), patch.bulkStream));

    const size_t numInnerCells = (patch.localNumCellsX - 2) * (patch.localNumCellsY - 2);
    const int accumulateGridSize = static_cast<int>(std::max<size_t>(1, std::min<size_t>(
        1024, ceilingDivide(numInnerCells, accumulateBlockSize))));

    accumulateTemperature2D<<<accumulateGridSize, accumulateBlockSize, 0, patch.bulkStream>>>(
        patch.d_localU, patch.localNumCellsX, patch.localNumCellsY, d_temperature);

    // reduce the total temperature across GPUs, in place
    // Note: no host synchronization is needed before the collective - it is enqueued behind
    //       the kernel producing its send buffer, on the very same stream
    // Note: NCCL reduces in place when the send and the receive buffer are the same
    //       pointer, so a single element is enough
    checkNcclError(ncclReduce(
        d_temperature,          // send buffer
        d_temperature,          // receive buffer, in place, only written on the root
        1,                      // count
        ncclDouble,             // datatype
        ncclSum,                // operation
        0,                      // root
        ncclComm,               // communicator
        patch.bulkStream));     // stream

    // Note: as every NCCL operation, the reduction is stream-ordered
    checkCudaError(cudaStreamSynchronize(patch.bulkStream));

    // a single 8-byte transfer replaces the full-field copy back
    if (0 == rank) {
        double temperature = 0.0;
        checkCudaError(cudaMemcpy(&temperature, d_temperature, sizeof(double), cudaMemcpyDeviceToHost));

        std::cout << "  Total temperature is " << temperature << std::endl;
    }

    checkCudaError(cudaFree(d_temperature));

    // clean up
    checkCudaError(cudaStreamDestroy(patch.haloStream));
    checkCudaError(cudaStreamDestroy(patch.bulkStream));

    checkCudaError(cudaFree(patch.d_localU));
    checkCudaError(cudaFree(patch.d_localUNew));

    checkCudaError(cudaFreeHost(patch.localU));

    checkNcclError(ncclCommDestroy(ncclComm));

    MPI_Finalize();

    return 0;
}
