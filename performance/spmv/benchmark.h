#pragma once

#include <string>
#include <iostream>
#include <stdio.h>

#include "bytes_per_spmv.h"

#include "../timer.h"

const char * BENCHMARK_OUTPUT_FILE_NAME = "benchmark_output.log";

template <typename HostMatrix, typename TestMatrix, typename TestKernel>
float check_spmv(HostMatrix& host_matrix, TestMatrix& test_matrix, TestKernel test_kernel)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array1d<ValueType,cusp::host_memory> host_x(N);
    cusp::array1d<ValueType,cusp::host_memory> host_y(M);

    for(IndexType i = 0; i < N; i++) host_x[i] = (int(i % 21) - 10);
    for(IndexType i = 0; i < M; i++) host_y[i] = 0;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(host_x);
    cusp::array1d<ValueType, MemorySpace> test_y(host_y);

    // compute SpMV on host and device
    cusp::multiply(host_matrix, host_x, host_y);
    test_kernel(test_matrix, test_x, test_y, ValueType(1), ValueType(0));

    // compare results
    cusp::array1d<ValueType, cusp::host_memory> test_y_copy(test_y);
    cusp::blas::axpy(host_y, test_y_copy, ValueType(-1));

    using namespace thrust::placeholders;
    double numerator   = abs(thrust::transform_reduce(test_y_copy.begin(), test_y_copy.end(), cusp::norm<ValueType>(), ValueType(0), thrust::plus<ValueType>()));
    double denominator = abs(cusp::blas::dotc(host_y, host_y));
    double error       = numerator / denominator;

    return error;
}


template <typename TestMatrix, typename TestKernel>
float time_spmv(TestMatrix& test_matrix, TestKernel test_spmv, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = test_matrix.num_rows;
    const IndexType N = test_matrix.num_cols;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(N);
    cusp::array1d<ValueType, MemorySpace> test_y(M);

    // warmup
    timer time_one_iteration;
    test_spmv(test_matrix, test_x, test_y, ValueType(1), ValueType(0));
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();

    // determine # of seconds dynamically
    size_t num_iterations;
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) );

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        test_spmv(test_matrix, test_x, test_y, ValueType(1), ValueType(0));
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;

    return sec_per_iteration;
}


template <typename HostMatrix, typename TestMatrixOnHost, typename TestMatrixOnDevice, typename TestKernel>
void test_spmv(std::string         kernel_name,
               HostMatrix&         host_matrix,
               TestMatrixOnHost&   test_matrix_on_host,
               TestMatrixOnDevice& test_matrix_on_device,
               TestKernel          test_spmv)
{
    float error = check_spmv(host_matrix, test_matrix_on_device, test_spmv);
    float time  = time_spmv(              test_matrix_on_device, test_spmv);
    float gbyte = bytes_per_spmv(test_matrix_on_host);

    float GFLOPs = (time == 0) ? 0 : (2 * host_matrix.num_entries / time) / 1e9;
    float GBYTEs = (time == 0) ? 0 : (gbyte / time)                       / 1e9;

    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, GFLOPs, GBYTEs, error);

    //record results to file
    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", kernel_name.c_str(), GFLOPs, GBYTEs, 1e3 * time);
    fclose(fid);
}


/////////////////////////////////////////////////////
// These methods test specific formats and kernels //
/////////////////////////////////////////////////////

template <typename HostMatrix>
void test_coo(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::coo_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("coo",     host_matrix, test_matrix_on_host, test_matrix_on_device,
              cusp::detail::device::spmv_coo    <DeviceMatrix,DeviceArray,DeviceArray,ValueType>);
}

template <typename HostMatrix>
void test_csr(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer csr_matrix to device
    typedef typename cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("csr_scalar",     host_matrix, test_matrix_on_host, test_matrix_on_device,
              cusp::detail::device::spmv_csr_scalar    <DeviceMatrix,DeviceArray,DeviceArray,ValueType>);
}

template <typename HostMatrix>
void test_dia(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::dia_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to DIA format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::dia_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("dia",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::detail::device::spmv_dia    <DeviceMatrix,DeviceArray,DeviceArray,ValueType>);
}

template <typename HostMatrix>
void test_ell(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to ELL format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::ell_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("ell",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::detail::device::spmv_ell    <DeviceMatrix,DeviceArray,DeviceArray,ValueType>);
}

template <typename HostMatrix>
void test_hyb(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("hyb",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::detail::device::spmv_hyb    <DeviceMatrix,DeviceArray,DeviceArray,ValueType>);
}

