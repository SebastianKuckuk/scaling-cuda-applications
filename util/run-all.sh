#!/bin/bash

# Detect how many GPUs are available
max_gpus=$(nvidia-smi -L | wc -l)

echo "Detected $max_gpus GPUs"

# Helper to generate CUDA_VISIBLE_DEVICES string
make_devlist () {
    n="$1"
    out=""
    for ((i=0; i<n; i++)); do
        if [ -z "$out" ]; then out="$i"; else out="$out,$i"; fi
    done
    echo "$out"
}

single_only_bins="03-gpu-baseline 04-work-partitioning 05-streams"
non_mpi_bins="06-mgpu 07-halos 07-halos-2 08-p2p 09-overlap"
mpi_bins="11-mpi 12-overlap 13-nvshmem"

# Sweep GPU counts in powers of two
gpus=1
while [ "$gpus" -le "$max_gpus" ]; do
    echo
    echo "${gpus} GPU(s)"
    echo "=========="

    devs=$(make_devlist $gpus)

    # Only run these for a single GPU
    if [ "$gpus" -eq 1 ]; then
        for name in $single_only_bins; do
            printf "%-20s" "$name"
            CUDA_VISIBLE_DEVICES="$devs" ../build/"$name" $((32 * 1024)) 256 2 2 0 | grep "bandwidth"
        done
    fi

    # Single-process multi-GPU codes
    for name in $non_mpi_bins; do
        printf "%-20s" "$name"
        CUDA_VISIBLE_DEVICES="$devs" ../build/"$name" $((32 * 1024)) 256 2 2 0 | grep "bandwidth"
    done

    # MPI codes (always run)
    for name in $mpi_bins; do
        printf "%-20s" "$name"
        CUDA_VISIBLE_DEVICES="$devs" mpirun -np "$gpus" ../build/"$name" $((32 * 1024)) 256 2 2 0 | grep "bandwidth"
    done

    # Next power of two
    gpus=$(( gpus * 2 ))
done
