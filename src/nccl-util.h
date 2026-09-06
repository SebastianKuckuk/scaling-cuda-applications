#pragma once

#include <iostream>

#include <nccl.h>


#define checkNcclError(...) \
    checkNcclErrorImpl(__FILE__, __LINE__, __VA_ARGS__)

inline void checkNcclErrorImpl(const std::string &file, int line, ncclResult_t code) {
    if (ncclSuccess != code) {
        std::cerr << "NCCL Error (" << file << " : " << line << ") --- " << ncclGetErrorString(code) << std::endl;
        exit(1);
    }
}
