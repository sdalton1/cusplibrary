#pragma once

#include <cuda.h>
#include <iostream>

void set_device(int device_id)
{
    cudaSetDevice(device_id);
}


void list_devices(void)
{
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
        std::cout << "There is no device supporting CUDA" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                std::cout << "There is no device supporting CUDA." << std::endl;
            else if (deviceCount == 1)
                std::cout << "There is 1 device supporting CUDA" << std:: endl;
            else
                std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
        }

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
        std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    }
    std::cout << std::endl;
}


template <typename T>
T l2_error(size_t N, const T * a, const T * b)
{
    T numerator   = 0;
    T denominator = 0;
    for(size_t i = 0; i < N; i++)
    {
        numerator   += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += (b[i] * b[i]);
    }

    return numerator/denominator;
}

template <typename T>
T l2_error(size_t N, const cusp::complex<T> * a, const cusp::complex<T> * b)
{
    T numerator   = 0;
    T denominator = 0;
    for(size_t i = 0; i < N; i++)
    {
        numerator   += (a[i].real() - b[i].real()) * (a[i].real() - b[i].real());
        denominator += (b[i].real() * b[i].real());
    }

    return numerator/denominator;
}



