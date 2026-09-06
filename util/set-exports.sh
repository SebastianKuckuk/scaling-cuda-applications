export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

ml nvhpc cuda openmpi

export NVSHMEM_ROOT=$NVHPC_ROOT/Linux_x86_64/26.3/comm_libs/nvshmem
export NCCL_ROOT=$NVHPC_ROOT/Linux_x86_64/26.3/comm_libs/nccl

export LD_LIBRARY_PATH=$NVSHMEM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export LIBRARY_PATH=$NVSHMEM_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LIBRARY_PATH=$NCCL_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}


export NVSHMEM_REMOTE_TRANSPORT=NONE

export UCX_WARN_UNUSED_ENV_VARS=n

