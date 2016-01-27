/*#define CUSP_USE_TEXTURE_MEMORY*/

#include <cusp/blas/blas.h>
#include <cusp/blas/cublas/blas.h>

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/connected_components.h>

#include <cusp/io/matrix_market.h>
#include <cusp/iterator/random_iterator.h>

#include <cusp/lapack/lapack.h>

#include "cublas_v2.h"
#include "../timer.h"

template<typename MatrixType, typename ArrayType>
void analyze(const MatrixType& G, const ArrayType& P)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    size_t N = G.num_rows;
    size_t M = G.num_entries;
    size_t num_parts = 2;

    MatrixType A(G);
    ArrayType  partition(P);

    cusp::array1d<IndexType,MemorySpace> row_indices(M);
    cusp::array1d<IndexType,MemorySpace> edge_cuts(M);
    cusp::offsets_to_indices(A.row_offsets, row_indices);
    thrust::transform(thrust::make_permutation_iterator(partition.begin(), row_indices.begin()),
                      thrust::make_permutation_iterator(partition.begin(), row_indices.begin()) + M,
                      thrust::make_permutation_iterator(partition.begin(), A.column_indices.begin()),
                      edge_cuts.begin(), thrust::not_equal_to<IndexType>());

    size_t num_edge_cuts = thrust::reduce(edge_cuts.begin(), edge_cuts.end()) / 2; // Divide by 2 to account for symmetry
    std::cout << " Edges cuts : " << num_edge_cuts << std::endl;

    thrust::replace_if(A.column_indices.begin(), A.column_indices.end(),
                       edge_cuts.begin(), thrust::identity<IndexType>(), -1);

    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> G_host(A);

    cusp::array1d<IndexType,cusp::host_memory> components(N);
    size_t num_comp = cusp::graph::connected_components(G_host, components);

    cusp::array1d<IndexType,cusp::host_memory> component_sizes(num_comp);
    thrust::sort(components.begin(), components.end());
    thrust::reduce_by_key(components.begin(), components.end(), thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(), component_sizes.begin());
    std::cout << "Partition yielded " << num_comp << " components" << std::endl;
    /*cusp::print(component_sizes);*/

    cusp::array1d<IndexType,MemorySpace> part_sizes(num_parts, 0);
    thrust::sort(partition.begin(), partition.end());
    thrust::reduce_by_key(partition.begin(), partition.end(), thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(), part_sizes.begin(), thrust::equal_to<IndexType>());

    ValueType S_p   = *thrust::max_element(part_sizes.begin(), part_sizes.end());
    ValueType S_opt = std::ceil(ValueType(N/num_parts));
    ValueType imbalance = ((S_p/S_opt) * 100.0) - 100.0;
    std::cout << "Largest partition size is " << S_p << ", while S_opt is " << S_opt
              << " representing " << imbalance << "% imbalance" << std::endl;
    for(size_t i = 0; i < num_parts; i++)
        std::cout << "Size of partition " << i << " is " << part_sizes[i] << std::endl;

}

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
    typedef typename Array2d::value_type ValueType;

    const size_t THREADS_PER_BLOCK  = 256;
    const size_t NUM_BLOCKS = 0; //cusp::detail::device::DIVIDE_INTO(A.num_rows, THREADS_PER_BLOCK);

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
        /*cusp::blas::axpby(Q.column(i), v, Q.column(i), 1.0, -dot_product);*/
        cusp::blas::axpy(v, Q.column(i), -dot_product);
    }
}

template<typename Array2d1, typename Array2d2>
void modifiedGramSchmidt(Array2d1& Q, Array2d2& R)
{
    modifiedGramSchmidt(Q,R,typename Array2d1::format(), typename Array2d2::format());
}

template<typename MatrixType, typename Array2d, typename Array1d>
void Partition(const MatrixType& A, const Array2d& X0, Array1d& partition, int blocksize, int maxouter, int maxinner)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef cusp::array2d<ValueType,MemorySpace,cusp::column_major> Array2dColumn;
    typedef typename Array2dColumn::view Array2dColumnView;

    const int N  = A.num_rows;
    const int St = blocksize * blocksize;
    const int Sx = N * blocksize;
    const int s  = (maxinner+1) * blocksize;
    maxinner = std::min(N, maxinner);

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }
    cusp::cublas::execution_policy cublas(handle);

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
    modifiedGramSchmidt(X_start, ones);

    Array2dColumnView TD_init(blocksize, blocksize, blocksize, cusp::make_array1d_view(TD));
    modifiedGramSchmidt(X_start, TD_init);
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
        /* gemm(cublas, X_start, AX, TD_start, CUBLAS_OP_T, CUBLAS_OP_N); */
        gemm2_time += gemm_timer.milliseconds_elapsed();

        timer inner_loop_timer;
        for(int j = 0; j < maxinner; j++)
        {
            int xdrag = (j-1)*Sx;
            int xstart= j*Sx;
            int xstop = (j+1)*Sx;
            int xshift= (j+2)*Sx;

            int tstart= j*St;
            int tstop = (j+1)*St;
            int tshift= (j+2)*St;

            Array2dColumnView X_drag (N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xdrag,xstart-1)));
            Array2dColumnView X_start(N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xstart,xstop-1)));
            Array2dColumnView X_stop (N, blocksize, N, cusp::make_array1d_view(X.values.subarray(xstop,xshift-1)));

            Array2dColumnView TD_start(blocksize, blocksize, blocksize, cusp::make_array1d_view(TD.subarray(tstart,tstop-1)));
            Array2dColumnView TD_stop (blocksize, blocksize, blocksize, cusp::make_array1d_view(TD.subarray(tstop,tshift-1)));

            Array2dColumnView TE_start(blocksize, blocksize, blocksize, cusp::make_array1d_view(TE.subarray(tstart,tstop-1)));
            Array2dColumnView TE_stop (blocksize, blocksize, blocksize, cusp::make_array1d_view(TE.subarray(tstop,tshift-1)));

            timer gemm1_timer;
            SimpleGEMM(X_start, TD_start, X_stop);
            /*gemm(X_start, TD_start, X_stop);*/
            gemm1_time += gemm1_timer.milliseconds_elapsed();

            cusp::blas::axpby(AX.values, X_stop.values, X_stop.values, ValueType(1), ValueType(-1));
            if( j > 0 )
            {
                timer gemm_timer;
                SimpleGEMM(X_drag, TE_start, X_stop, CUBLAS_OP_N, CUBLAS_OP_T, ValueType(-1), ValueType(1));
                /*gemm(X_drag, TE_start, X_stop, CUBLAS_OP_N, CUBLAS_OP_T, ValueType(-1), ValueType(1));*/
                gemm1_time += gemm_timer.milliseconds_elapsed();
            }

            timer ortho_timer;
            modifiedGramSchmidt(X_stop, TE_stop);
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
            /* cusp::blas::gemm(cublas, X_stop, AX, TD_stop, CUBLAS_OP_T, CUBLAS_OP_N); */
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
        int info = 0; //cusp::lapack::syev(s, T_hp, eigvals);
        std::cout << " Eigensolver (SYEV) time : " << syev_timer.milliseconds_elapsed() << " (ms)." << std::endl;
        if(info) throw cusp::runtime_exception("SYEV Failed!\n");

        V = T_h;

        timer vector_timer;
        /*gemm(X, V, Evects);*/
        /* cusp::blas::gemv(cublas, X, V.column(0), Evects.column(0)); */
        std::cout << " Eigensolver (VECTOR) time : " << vector_timer.milliseconds_elapsed() << " (ms)." << std::endl;
    }

    timer cut_timer;
    cusp::array1d<ValueType,MemorySpace> fiedler(Evects.column(0));
    thrust::sort(fiedler.begin(), fiedler.end());
    ValueType vmed = fiedler[N/2];
    thrust::transform(Evects.column(0).begin(), Evects.column(0).end(),
                      thrust::constant_iterator<ValueType>(vmed),
                      partition.begin(), thrust::greater_equal<ValueType>());
    std::cout << " Cut time : " << cut_timer.milliseconds_elapsed() << " (ms)." << std::endl;

    printf("Total SpMV time : %4.4f (ms)\n", multiply_time);
    printf("Total MGS  time : %4.4f (ms)\n", ortho_time);
    printf("Total GEMM(1) time : %4.4f (ms)\n", gemm1_time);
    printf("Total GEMM(2) time : %4.4f (ms)\n", gemm2_time);
}

int main(int argc, char*argv[])
{
    typedef int IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    size_t size = 10;
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    size_t N = A.num_rows;
    size_t M = A.num_entries;
    size_t blocksize = 4;
    size_t maxouter  = 1;
    size_t maxinner  = 10;
    std::cout << "with shape ("  << N << "," << N << ") and " << M << " entries" << "\n\n";

    cusp::array1d<IndexType, MemorySpace> partition(N);

    // initialize starting vector to random values in [0,1)
    cusp::array2d<ValueType,MemorySpace,cusp::column_major> X0(N,4);
    cusp::blas::copy(cusp::random_array<ValueType>(X0.num_entries), X0.values);

    timer t;
    Partition(A, X0, partition, blocksize, maxouter, maxinner);
    std::cout << " Bisect time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;

    analyze(A, partition);

    return EXIT_SUCCESS;
}

