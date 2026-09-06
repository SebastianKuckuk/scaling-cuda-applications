#include "../stencil-2d-util.h"

#include "../cuda-util.h"


__global__ void stencil2D(const double *__restrict__ u, double *__restrict__ uNew,
                          size_t globalInnerBeginX, size_t globalInnerEndX,
                          size_t globalInnerBeginY, size_t globalInnerEndY,
                          size_t globalNumCellsX) {

    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tidy = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t i0 = globalInnerBeginX + tidx;
    const size_t i1 = globalInnerBeginY + tidy;

    if (i0 < globalInnerEndX && i1 < globalInnerEndY) {
        uNew[i0 + i1 * globalNumCellsX] = u[i0 + i1 * globalNumCellsX]
            + alpha * (
                      u[(i0 - 1) +  i1      * globalNumCellsX]
                +     u[(i0 + 1) +  i1      * globalNumCellsX]
                +     u[ i0      + (i1 + 1) * globalNumCellsX]
                +     u[ i0      + (i1 - 1) * globalNumCellsX]
                - 4 * u[ i0      +  i1      * globalNumCellsX]);
    }
}


struct Patch {
    // patch boundaries in global coordinates excluding boundaries
    size_t globalInnerBeginX;
    size_t globalInnerEndX;
    size_t globalInnerBeginY;
    size_t globalInnerEndY;

    // execution configuration
    dim3 blockSize;
    dim3 gridSize;

    // patch stream
    cudaStream_t stream;
};


int main(int argc, char *argv[]) {
    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // initialize patches
    constexpr int numPatches = 8; // fixed number of patches

    Patch *patches = new Patch[numPatches];

    size_t patchHeight = ceilingDivide(globalNumCellsY - 2, numPatches);
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        auto &patch = patches[patchIdx];

        // no partitioning in the x-dimension
        patch.globalInnerBeginX = 1;
        patch.globalInnerEndX = globalNumCellsX - 1;

        patch.globalInnerBeginY = 1 + patchIdx * patchHeight;
        patch.globalInnerEndY   = std::min(     // the end is either
            1 + (patchIdx + 1) * patchHeight,   // the beginning of the next patch, or
            globalNumCellsY - 1);               // the end of the global domain

        // execution configuration
        auto numInnerCellsX = patch.globalInnerEndX - patch.globalInnerBeginX;
        auto numInnerCellsY = patch.globalInnerEndY - patch.globalInnerBeginY;

        patch.blockSize = dim3(16, 16);
        patch.gridSize  = dim3(
            ceilingDivide(numInnerCellsX, patch.blockSize.x),
            ceilingDivide(numInnerCellsY, patch.blockSize.y));

        // create stream
        checkCudaError(cudaStreamCreate(&patch.stream));            
    }

    // allocate
    double *u;
    checkCudaError(cudaMallocManaged(&u, globalNumCellsX * globalNumCellsY * sizeof(double)));
    double *uNew;
    checkCudaError(cudaMallocManaged(&uNew, globalNumCellsX * globalNumCellsY * sizeof(double)));

    // init temperature fields including their boundaries
    initTemperature(u, uNew, globalNumCellsX, globalNumCellsY);

    // prefetch to GPU
    int deviceId = 0;
    checkCudaError(cudaGetDevice(&deviceId));

    cudaMemLocation deviceID{cudaMemLocationTypeDevice, deviceId};
    cudaMemLocation hostID{cudaMemLocationTypeHost, 0};
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        const auto &patch = patches[patchIdx];

        auto startIdx = (patch.globalInnerBeginY - 1) * globalNumCellsX;
        auto endIdx = (patch.globalInnerEndY + 1) * globalNumCellsX;
        auto size = (endIdx - startIdx) * sizeof(double);

        checkCudaError(cudaMemPrefetchAsync(u + startIdx, size, deviceID, 0, patch.stream));
        checkCudaError(cudaMemPrefetchAsync(uNew + startIdx, size, deviceID, 0, patch.stream));
    }

    // define print and work
    auto print = [&](size_t it) {
        std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        // Note: this could be optimized - see the course 'Fundamentals of Accelerated Computing with Modern CUDA C++'
        checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), hostID, 0));
        writeTemperatureNpy("../output/temperature_" + idx + ".npy", u, globalNumCellsX, globalNumCellsY);
        checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), deviceID, 0));
    };

    auto work = [&](size_t it) {
        for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
            const auto &patch = patches[patchIdx];

            stencil2D<<<patch.gridSize, patch.blockSize, 0, patch.stream>>>(
                u, uNew,                                            // global arrays
                patch.globalInnerBeginX, patch.globalInnerEndX,     // global x interval coordinates for the current patch
                patch.globalInnerBeginY, patch.globalInnerEndY,     // global y interval coordinates for the current patch
                globalNumCellsX                                     // global stride required for index linearization
            );
        }

        checkCudaError(cudaDeviceSynchronize());

        std::swap(u, uNew);

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    checkCudaError(cudaDeviceSynchronize());
    nvtxRangePushA("work");
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    checkCudaError(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    nvtxRangePop();

    // print stats and diagnostic result
    printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), hostID, 0));
    auto totalTemperature = accumulateTemperature(u, globalNumCellsX, globalNumCellsY);
    std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        auto &patch = patches[patchIdx];
        checkCudaError(cudaStreamDestroy(patch.stream));
    }

    checkCudaError(cudaFree(u));
    checkCudaError(cudaFree(uNew));

    delete[] patches;

    return 0;
}
