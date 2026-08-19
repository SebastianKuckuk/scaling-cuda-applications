#include "../stencil-2d-util.h"

#include "../cuda-util.h"


TODO: convert this function to a kernel
// inline void stencil2D(const double *__restrict__ u, double *__restrict__ uNew, size_t globalNumCellsX, size_t globalNumCellsY) {
//     for (size_t i1 = 1; i1 < globalNumCellsY - 1; ++i1) {
//         for (size_t i0 = 1; i0 < globalNumCellsX - 1; ++i0) {
//             uNew[i0 + i1 * globalNumCellsX] = u[i0 + i1 * globalNumCellsX]
//                 + alpha * (
//                           u[(i0 - 1) +  i1      * globalNumCellsX]
//                     +     u[(i0 + 1) +  i1      * globalNumCellsX]
//                     +     u[ i0      + (i1 + 1) * globalNumCellsX]
//                     +     u[ i0      + (i1 - 1) * globalNumCellsX]
//                     - 4 * u[ i0      +  i1      * globalNumCellsX]
//                 );
//         }
//     }
// }

int main(int argc, char *argv[]) {
    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // allocation
    TODO: allocate u and uNew with cudaMallocManaged

    // init temperature fields including their boundaries
    initTemperature(u, uNew, globalNumCellsX, globalNumCellsY);

    // prefetch to GPU
    int deviceId = 0;
    checkCudaError(cudaGetDevice(&deviceId));

    // CUDA 13 removed the int-device overload of cudaMemPrefetchAsync - the target is a cudaMemLocation now
    cudaMemLocation gpuLocation{cudaMemLocationTypeDevice, deviceId};
    cudaMemLocation cpuLocation{cudaMemLocationTypeHost, 0};
    TODO: prefetch u and uNew to device

    // define execution configuration
    TODO: define gridSize and blockSize

    // print function
    auto print = [&](size_t it) {
        std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        // Note: this could be optimized - see the course 'Fundamentals of Accelerated Computing with Modern CUDA C++'
        checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), cpuLocation, 0));
        writeTemperatureNpy("../output/temperature_" + idx + ".npy", u, globalNumCellsX, globalNumCellsY);
        checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), gpuLocation, 0));
    };

    // work function
    auto work = [&](size_t it) {
        TODO: launch stencil2D
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

    TODO: prefetch u to CPU
    auto totalTemperature = accumulateTemperature(u, globalNumCellsX, globalNumCellsY);
    std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    checkCudaError(cudaFree(u));
    checkCudaError(cudaFree(uNew));

    return 0;
}
