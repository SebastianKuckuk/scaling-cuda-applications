export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export NVSHMEM_ROOT=$NVHPC_ROOT/Linux_x86_64/25.5/comm_libs/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_ROOT/lib

export NVSHMEM_REMOTE_TRANSPORT=NONE

export UCX_WARN_UNUSED_ENV_VARS=n
