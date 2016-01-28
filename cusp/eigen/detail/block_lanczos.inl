/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>

#include <cusp/blas/blas.h>
#include <cusp/blas/cublas/blas.h>

#include <cusp/io/matrix_market.h>
#include <cusp/iterator/random_iterator.h>

#include <cusp/lapack/lapack.h>

#include "cublas_v2.h"

#include <sstream>
#include <iostream>
#include <iomanip>

namespace cusp
{
namespace eigen
{
namespace detail
{

template <typename ValueType, unsigned int THREADS_PER_BLOCK>
__launch_bounds__(THREADS_PER_BLOCK,1)
__global__ void SimpleGEMMKernel(const ValueType alpha, const ValueType beta,
                                 const int m, const int n, const int k, const int lda,
                                 const ValueType* A, const ValueType* B, ValueType* C,
                                 bool transa, bool transb)
{
    enum
    {
        K = 4,
    };

    struct Storage
    {
        ValueType data[K*K];
    };

    __shared__ Storage storage;

    const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    ValueType vals[K];

    if(threadIdx.x < K*K)
        storage.data[threadIdx.x] = B[threadIdx.x];

    if(thread_id < m) {
#pragma unroll
        for(int i = 0; i < K; i++)
            vals[i] = A[i*lda + thread_id];
    }

    __syncthreads();

    if(thread_id < m)
    {
        if(beta == 0)
#pragma unroll
            for(int i = 0; i < K; i++)
                C[i*lda + thread_id] = alpha*(vals[0]*storage.data[i*K+0] + vals[1]*storage.data[i*K+1] + vals[2]*storage.data[i*K+2] + vals[3]*storage.data[i*K+3]);
        else
#pragma unroll
            for(int i = 0; i < K; i++)
                C[i*lda + thread_id] = beta*C[i*lda + thread_id] +
                                       alpha*(vals[0]*storage.data[0*K+i] + vals[1]*storage.data[1*K+i] + vals[2]*storage.data[2*K+i] + vals[3]*storage.data[3*K+i]);
    }
}

template<typename Array2d>
void SimpleGEMM(const Array2d& A, const Array2d& B, Array2d& C,
                cublasOperation_t transa = CUBLAS_OP_N,
                cublasOperation_t transb = CUBLAS_OP_N,
                typename Array2d::value_type alpha = 1,
                typename Array2d::value_type beta  = 0)
{
    using namespace cusp::system::cuda;

    typedef typename Array2d::value_type ValueType;

    const size_t THREADS_PER_BLOCK  = 256;
    const size_t NUM_BLOCKS = DIVIDE_INTO(A.num_rows, THREADS_PER_BLOCK);

    const int m = A.num_rows;
    const int n = B.num_cols;
    const int k = B.num_rows;
    const int lda = A.pitch;

    const ValueType * A_values = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_values = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_values = thrust::raw_pointer_cast(&C(0,0));

    SimpleGEMMKernel<ValueType, THREADS_PER_BLOCK><<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(alpha,beta,m,n,k,lda,
            A_values,B_values,C_values,transa==CUBLAS_OP_N,transb==CUBLAS_OP_N);
    /* gpuErrchk(cudaDeviceSynchronize()); */
}

template<typename Array2d1, typename Array2d2>
void modifiedGramSchmidt(Array2d1& Q, Array2d2& R, cusp::array2d_format, cusp::array2d_format)
{
    typedef typename Array2d1::value_type ValueType;

    for(size_t i = 0; i < Q.num_cols; i++)
    {
        R(i,i) = cusp::blas::nrm2(Q.column(i));

        if(R(i,i) < std::numeric_limits<ValueType>::epsilon())
        {
            std::cerr << "R(" << i << "," << i << ") = " << R(i,i) << std::endl;
            throw cusp::runtime_exception("Gram-Schmidt orthogonalization failed.");
        }

        cusp::blas::scal(Q.column(i), 1.0/R(i,i));

        for(size_t j = i+1; j < Q.num_cols; j++)
        {
            R(i,j) = cusp::blas::dot(Q.column(i), Q.column(j));
            cusp::blas::axpy(Q.column(i), Q.column(j), -R(i,j));
        }
    }
}

template<typename Array2d1, typename Array2d2>
void modifiedGramSchmidt(Array2d1& Q, Array2d2& v, cusp::array2d_format, cusp::array1d_format)
{
    typedef typename Array2d1::value_type ValueType;

    for(size_t i = 0; i < Q.num_cols; i++)
    {
        ValueType dot_product = cusp::blas::dot(Q.column(i), v);
        cusp::blas::axpy(v, Q.column(i), -dot_product);
    }
}

template<typename Array2d1, typename Array2d2>
void modifiedGramSchmidt(Array2d1& Q, Array2d2& R)
{
    modifiedGramSchmidt(Q,R,typename Array2d1::format(), typename Array2d2::format());
}

template<typename Array2d1, typename Array1d1, typename Array1d2>
void gemv(cublasHandle_t& cublas_handle,
          const Array2d1& A,
          const Array1d1& x,
          const Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    cublasOperation_t trans = CUBLAS_OP_N;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t result = cusp::blas::cublas::detail::gemv(cublas_handle,
                                                             trans, m, n, alpha,
                                                             A_p, lda, x_p, 1, beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void gemm(cublasHandle_t& cublas_handle,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C,
          cublasOperation_t transa = CUBLAS_OP_N,
          cublasOperation_t transb = CUBLAS_OP_N,
          typename Array2d1::value_type alpha = 1,
          typename Array2d1::value_type beta  = 0)
{
    typedef typename Array2d1::value_type ValueType;

    int m = transa == CUBLAS_OP_N ? A.num_rows : A.num_cols;
    int n = transb == CUBLAS_OP_N ? B.num_cols : B.num_rows;
    int k = transb == CUBLAS_OP_N ? B.num_rows : B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result = cusp::blas::cublas::detail::gemm(cublas_handle,
                                                             transa, transb,
                                                             m, n, k, alpha, A_p, lda,
                                                             B_p, ldb, beta, C_p, ldc);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemm failed!");
}

int syev(const int s, float * T_hp, float * eigvals_p)
{
    return LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', s, T_hp, s, eigvals_p);
}

int syev(const int s, double * T_hp, double * eigvals_p)
{
    return LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', s, T_hp, s, eigvals_p);
}

} // end namespace detail

class timer
{
    cudaEvent_t start;
    cudaEvent_t end;

public:
    timer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start,0);
    }

    ~timer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float milliseconds_elapsed()
    {
        float elapsed_time;
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        return elapsed_time;
    }
    float seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};

template<typename MatrixType,
         typename Array1d,
         typename Array2d>
void block_lanczos(const MatrixType& A,
                   Array1d& eigVals,
                   Array2d& eigVecs,
                   const size_t blocksize,
                   const size_t maxouter,
                   const size_t maxinner_)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef cusp::array2d<ValueType,MemorySpace,cusp::column_major> Array2dColumn;
    typedef typename Array2dColumn::view Array2dColumnView;

    const size_t N  = A.num_rows;
    const size_t maxinner = std::min(N, maxinner_);

    const size_t St = blocksize * blocksize;
    const size_t Sx = N * blocksize;
    const size_t s  = (maxinner+1) * blocksize;

    cublasHandle_t cublas_handle;

    if(cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("cublasCreate failed");

    // initialize starting vector to random values in [0,1)
    cusp::array2d<ValueType,MemorySpace,cusp::column_major> X0(N, blocksize);
    cusp::blas::copy(cusp::random_array<ValueType>(X0.num_entries), X0.values);

    timer initial_timer;
    Array2dColumn AX(N,blocksize);
    Array2dColumn X(N,s);
    Array2dColumn V(s,s);
    Array2dColumn Evects(N,s);
    cusp::array1d<ValueType,MemorySpace> TD(s*blocksize,0);
    cusp::array1d<ValueType,MemorySpace> TE(s*blocksize,0);

    cusp::array1d<ValueType,cusp::host_memory> eigvals(s);
    cusp::array1d<ValueType,cusp::host_memory> TD_h(s*blocksize);
    cusp::array1d<ValueType,cusp::host_memory> TE_h(s*blocksize);
    cusp::array2d<ValueType,cusp::host_memory> T_h(s,s,0);
    cusp::array2d<ValueType,MemorySpace> temp1(N,blocksize);
    cusp::array2d<ValueType,MemorySpace> temp2(N,blocksize);

    ValueType *T_hp  = thrust::raw_pointer_cast(&T_h(0,0));
    ValueType *eigvals_p = thrust::raw_pointer_cast(&eigvals[0]);

    // initialize starting vector to random values in [0,1)
    Array2dColumnView X_start(N, blocksize, N, cusp::make_array1d_view(X.values));
    thrust::copy(X0.values.begin(), X0.values.end(), X_start.values.begin());

    // normalize v0
    cusp::array1d<ValueType,MemorySpace> ones(N,ValueType(1));
    cusp::blas::scal(ones, ValueType(1) / cusp::blas::nrm2(ones));
    detail::modifiedGramSchmidt(X_start, ones);

    Array2dColumnView TD_init(blocksize, blocksize, blocksize, cusp::make_array1d_view(TD));
    detail::modifiedGramSchmidt(X_start, TD_init);
    std::cout << " Initialization time : " << initial_timer.milliseconds_elapsed() << " (ms)." << std::endl;

    float multiply_time = 0;
    float gemm1_time = 0;
    float gemm2_time = 0;
    float ortho_time = 0;

    for(int i = 0; i < maxouter; i++)
    {
        Array2dColumnView TD_start(blocksize, blocksize, blocksize, cusp::make_array1d_view(TD));
        Array2dColumnView X_start(N, blocksize, N, cusp::make_array1d_view(X.values));

        timer multiply_timer;
        /*cusp::transpose(X_start, temp1);*/
        cusp::multiply(A, X_start, AX);
        /*cusp::detail::device::spmv_csr_block(A, blocksize,*/
        /*                   thrust::raw_pointer_cast(&temp1(0,0)),*/
        /*                   thrust::raw_pointer_cast(&temp2(0,0)));*/
        /*cusp::transpose(temp2, AX);*/
        multiply_time += multiply_timer.milliseconds_elapsed();

        timer gemm_timer;
        detail::gemm(cublas_handle, X_start, AX, TD_start, CUBLAS_OP_T, CUBLAS_OP_N);
        gemm2_time += gemm_timer.milliseconds_elapsed();

        timer inner_loop_timer;
        for(int j = 0; j < maxinner; j++)
        {
            int xdrag = (j-1)*Sx;
            int xstart= j*Sx;
            int xstop = (j+1)*Sx;

            int tstart= j*St;
            int tstop = (j+1)*St;

            Array2dColumnView X_drag (N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xdrag,Sx)));
            Array2dColumnView X_start(N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xstart,Sx)));
            Array2dColumnView X_stop (N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xstop,Sx)));

            Array2dColumnView TD_start(blocksize, blocksize, blocksize, cusp::make_array1d_view(TD.subarray(tstart,St)));
            Array2dColumnView TD_stop (blocksize, blocksize, blocksize, cusp::make_array1d_view(TD.subarray(tstop,St)));

            Array2dColumnView TE_start(blocksize, blocksize, blocksize, cusp::make_array1d_view(TE.subarray(tstart,St)));
            Array2dColumnView TE_stop (blocksize, blocksize, blocksize, cusp::make_array1d_view(TE.subarray(tstop,St)));

            timer gemm1_timer;
            detail::SimpleGEMM(X_start, TD_start, X_stop);
            /*gemm(X_start, TD_start, X_stop);*/
            gemm1_time += gemm1_timer.milliseconds_elapsed();

            cusp::blas::axpby(AX.values, X_stop.values, X_stop.values, ValueType(1), ValueType(-1));

            if( j > 0 )
            {
                timer gemm_timer;
                detail::SimpleGEMM(X_drag, TE_start, X_stop, CUBLAS_OP_N, CUBLAS_OP_T, ValueType(-1), ValueType(1));
                /*gemm(X_drag, TE_start, X_stop, CUBLAS_OP_N, CUBLAS_OP_T, ValueType(-1), ValueType(1));*/
                gemm1_time += gemm_timer.milliseconds_elapsed();
            }

            timer ortho_timer;
            detail::modifiedGramSchmidt(X_stop, TE_stop);
            ortho_time += ortho_timer.milliseconds_elapsed();

            timer multiply_timer;
            /*cusp::transpose(X_stop, temp1);*/
            cusp::multiply(A, X_stop, AX);
            /*cusp::detail::device::spmv_csr_block_tex(A, blocksize,*/
            /*               thrust::raw_pointer_cast(&temp1(0,0)),*/
            /*               thrust::raw_pointer_cast(&temp2(0,0)));*/
            /*cusp::transpose(temp2, AX);*/
            multiply_time += multiply_timer.milliseconds_elapsed();

            timer gemm2_timer;
            detail::gemm(cublas_handle, X_stop, AX, TD_stop, CUBLAS_OP_T, CUBLAS_OP_N);
            /*std::cout << " X_stop.T*AX time : " << gemm2_timer.milliseconds_elapsed() << " (ms)." << std::endl;*/
            gemm2_time += gemm2_timer.milliseconds_elapsed();
        }
        std::cout << " Inner loop time : " << inner_loop_timer.milliseconds_elapsed() << " (ms)." << std::endl;

        timer eigen_init_timer;
        TD_h = TD;
        TE_h = TE;

        for(int i = 0; i <= maxinner; i++)
        {
            for(int j = 0; j < blocksize; j++)
            {
                for(int k = j; k < blocksize; k++)
                {
                    T_h(i*blocksize+j,i*blocksize+k) = TD_h[i*St + j*blocksize + k];
                    if(i > 0)
                        T_h((i-1)*blocksize+k,i*blocksize+j) = TE_h[i*St + k*blocksize + j];
                }
            }
        }
        std::cout << " Eigensolver (INIT) time : " << eigen_init_timer.milliseconds_elapsed() << " (ms)." << std::endl;

        timer syev_timer;
        detail::syev(s, T_hp, eigvals_p);
        int info = detail::syev(s, T_hp, eigvals_p);
        std::cout << " Eigensolver (SYEV) time : " << syev_timer.milliseconds_elapsed() << " (ms)." << std::endl;

        V = T_h;

        timer vector_timer;
        /*gemm(X, V, Evects);*/
        detail::gemv(cublas_handle, X, V.column(0), Evects.column(0));
        std::cout << " Eigensolver (VECTOR) time : " << vector_timer.milliseconds_elapsed() << " (ms)." << std::endl;
    }

    printf("Total SpMV time : %4.4f (ms)\n", multiply_time);
    printf("Total MGS  time : %4.4f (ms)\n", ortho_time);
    printf("Total GEMM(1) time : %4.4f (ms)\n", gemm1_time);
    printf("Total GEMM(2) time : %4.4f (ms)\n", gemm2_time);
}

} // end namespace eigen
} // end namespace cusp

