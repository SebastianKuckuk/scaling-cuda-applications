#include "../stencil-2d-util.h"

#include "../cuda-util.h"

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>


__global__ void stencil2D(const double *__restrict__ u, double *__restrict__ uNew,
                          size_t globalInnerBeginX, size_t globalInnerEndX,
                          size_t globalInnerBeginY, size_t globalInnerEndY,
                          size_t globalNumCellsX, size_t patchHeight) {

    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tidy = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t i0 = globalInnerBeginX + tidx;
    const size_t i1 = globalInnerBeginY + tidy;

    const auto pe = nvshmem_my_pe();
    const auto numPEs = nvshmem_n_pes();

    if (i0 < globalInnerEndX && i1 < globalInnerEndY) {
        // temporary variables for stencil contributions
        double left, right, top, bottom;

        // left and right are always local
        left   = u[(i0 - 1) + i1 * globalNumCellsX];
        right  = u[(i0 + 1) + i1 * globalNumCellsX];

        // top and bottom may be remote
        if (i1 == globalInnerBeginY && pe > 0) {
            // fetch from neighboring PE below
            auto remotePE = pe - 1;
            auto remoteIndex = i0 + (patchHeight + 2 - 2) * globalNumCellsX;
            bottom = nvshmem_double_g(&u[remoteIndex], remotePE);
        } else {
            bottom = u[i0 + (i1 - 1) * globalNumCellsX];
        }

        if (i1 == globalInnerEndY - 1 && pe < numPEs - 1) {
            // fetch from neighboring PE above
            auto remotePE = pe + 1;
            auto remoteIndex = i0 + 1 * globalNumCellsX;
            top = nvshmem_double_g(&u[remoteIndex], remotePE);
        } else {
            top = u[i0 + (i1 + 1) * globalNumCellsX];
        }

        uNew[i0 + i1 * globalNumCellsX] = u[i0 + i1 * globalNumCellsX] + alpha * (left + right + top + bottom - 4 * u[i0 + i1 * globalNumCellsX]);
    }
}


struct Patch {
    size_t globalInnerBeginX;
    size_t globalInnerEndX;
    size_t globalInnerBeginY;
    size_t globalInnerEndY;

    dim3 blockSize;
    dim3 gridSize;

    size_t localNumCellsX;
    size_t localNumCellsY;
    size_t localSize;

    double* localU;
    double* localUNew;
};


int main(int argc, char *argv[]) {
    // initialize MPI
    MPI_Init(&argc, &argv);

    // initialize NVSHMEM
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int numPEs = nvshmem_n_pes();
    int pe = nvshmem_my_pe();

    std::cout << "PE " << pe << " of " << numPEs << std::endl;

    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // choose GPU
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = pe % numDevicesPerNode;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "PE " << pe << " using device " << deviceId << std::endl;

    // initialize patches
    int numPatches = numPEs; // one patch per PE
    checkCudaError(cudaSetDevice(deviceId));

    Patch patch;
    int patchIdx = pe;

    size_t patchHeight = ceilingDivide(globalNumCellsY - 2, numPatches);
    patch.globalInnerBeginX = 1;
    patch.globalInnerEndX = globalNumCellsX - 1;
    patch.globalInnerBeginY = patchIdx * patchHeight + 1;
    patch.globalInnerEndY = std::min((patchIdx + 1) * patchHeight + 1, globalNumCellsY - 1);

    patch.blockSize = dim3(16, 16); // semi-arbitrary choice
    patch.gridSize = dim3(ceilingDivide(patch.globalInnerEndX - patch.globalInnerBeginX, patch.blockSize.x),
                          ceilingDivide(patch.globalInnerEndY - patch.globalInnerBeginY, patch.blockSize.y));

    patch.localNumCellsX = patch.globalInnerEndX - patch.globalInnerBeginX + 2;   // two halo layers of size one each
    patch.localNumCellsY = patch.globalInnerEndY - patch.globalInnerBeginY + 2;
    // for NVSHMEM, we need to allocate symmetric memory
    patch.localSize = globalNumCellsX * (patchHeight + 2) * sizeof(double);

    // allocate CPU
    double *u;
    checkCudaError(cudaMallocHost(&u, patch.localSize));
    double *uNew;
    checkCudaError(cudaMallocHost(&uNew, patch.localSize));

    // allocate GPU per patch
    patch.localU = (double *)nvshmem_malloc(patch.localSize);
    patch.localUNew = (double *)nvshmem_malloc(patch.localSize);

    // init temperature fields including their boundaries
    initTemperaturePatch(u, uNew, globalNumCellsX, globalNumCellsY, patch.localNumCellsX, patch.localNumCellsY, patch.globalInnerBeginX, patch.globalInnerBeginY);

    // copy data to GPU
    checkCudaError(cudaMemcpy(patch.localU, u, patch.localSize, cudaMemcpyHostToDevice));

    // initialize uNew by copying from u
    checkCudaError(cudaMemcpy(patch.localUNew, patch.localU, patch.localSize, cudaMemcpyDeviceToDevice));

    // define print and work
    auto print = [&](size_t it) {
        if (0 == pe)
            std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        checkCudaError(cudaMemcpy(u, patch.localU, patch.localSize, cudaMemcpyDeviceToHost));

        // Note: brute-force full-synchronize approach for simplicity - optimize with point-to-point messages triggering next write
        for (int printPE = 0; printPE < numPEs; ++printPE) {
            nvshmem_barrier_all();
            if (pe == printPE)
                writeTemperaturePatchNpy("../output/temperature_" + idx + ".npy",
                    patch.localU,
                    globalNumCellsX, globalNumCellsY, patch.localNumCellsX, patch.localNumCellsY,
                    numPatches, pe);
        }
    };

    auto work = [&](size_t it) {
        stencil2D<<<patch.gridSize, patch.blockSize>>>(
            patch.localU, patch.localUNew,
            1, patch.localNumCellsX - 1,
            1, patch.localNumCellsY - 1,
            patch.localNumCellsX, patchHeight);

        checkCudaError(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        std::swap(patch.localU, patch.localUNew);

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    checkCudaError(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    nvtxRangePush("work");
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    checkCudaError(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    auto end = std::chrono::steady_clock::now();
    nvtxRangePop();

    // print stats and diagnostic result
    if (0 == pe)
        printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    checkCudaError(cudaMemcpy(u, patch.localU, patch.localSize, cudaMemcpyDeviceToHost));
    auto peTotalTemperature = accumulateTemperature(u, patch.localNumCellsX, patch.localNumCellsY);
    std::cout << "  Total temperature on pe " << pe << " is " << peTotalTemperature << std::endl;
    double totalTemperature = 0.0;
    MPI_Reduce(
        &peTotalTemperature,    // send buffer
        &totalTemperature,      // receive buffer
        1,                      // count
        MPI_DOUBLE,             // datatype
        MPI_SUM,                // operation
        0,                      // root
        MPI_COMM_WORLD);        // communicator
    if (0 == pe)
        std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    nvshmem_free(patch.localU);
    nvshmem_free(patch.localUNew);
    
    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(uNew));

    nvshmem_finalize();

    MPI_Finalize();

    return 0;
}
