/*#define CUSP_USE_TEXTURE_MEMORY*/

#include <cusp/blas/blas.h>

#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/eigen/block_lanczos.h>
#include <cusp/gallery/poisson.h>
#include <cusp/graph/connected_components.h>
#include <cusp/graph/pseudo_peripheral.h>
#include <cusp/io/matrix_market.h>
#include <cusp/lapack/lapack.h>

#include "../../timer.h"

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <limits>

struct partition_stats
{
    float initial_guess_time;
    float bisect_time;
    float cut_time;
    float imbalance;
    int   num_edge_cuts;
    int   min_edge_cuts;
    int   max_edge_cuts;
    int   avg_edge_cuts;
    int   num_comps;
    int   max_part_size;
};

struct partition_config
{
    size_t blocksize;
    size_t maxouter;
    size_t maxinner;
    bool verbose;
};

typedef std::map<std::string, std::string> ArgumentMap;
ArgumentMap args;

std::string process_args(int argc, char ** argv)
{
    std::string filename;

    for(int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);

        if (arg.substr(0,2) == "--")
        {
            std::string::size_type n = arg.find('=',2);

            if (n == std::string::npos)
                args[arg.substr(2)] = std::string();              // (key)
            else
                args[arg.substr(2, n - 2)] = arg.substr(n + 1);   // (key,value)
        }
        else
        {
            filename = arg;
        }
    }

    return filename;
}

template<typename MatrixType, typename Array2dColumn>
void generate_initial_guess(const MatrixType& A, Array2dColumn& Q)
{
    using namespace thrust::placeholders;
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    cusp::array1d<IndexType,MemorySpace> A_row_lengths(A.num_rows);
    thrust::transform(A.row_offsets.begin() + 1, A.row_offsets.end(),
                      A.row_offsets.begin(), A_row_lengths.begin(),
                      thrust::minus<IndexType>());

    cusp::array1d<IndexType,MemorySpace> level_x(A.num_rows);
    IndexType vertex = cusp::graph::pseudo_peripheral_vertex(A, level_x);
    IndexType last_vertex = thrust::find(level_x.begin(), level_x.end(), 0) - level_x.begin();

    cusp::array1d<IndexType,MemorySpace> level_y(A.num_rows);
    cusp::graph::breadth_first_search(A, last_vertex, level_y);

    cusp::blas::copy(level_x, Q.column(0));
    cusp::blas::copy(level_y, Q.column(1));

    for(size_t i = 2; i < Q.num_cols; i++)
    {
        vertex = rand() % A.num_rows;
        cusp::graph::breadth_first_search(A, vertex, level_x);
        cusp::blas::copy(level_x, Q.column(i));
    }
}

template<typename MatrixType, typename ArrayType>
void analyze(const MatrixType& G, const ArrayType& P, partition_stats& stats, bool verbose)
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
    if(verbose) std::cout << " Edges cuts : " << num_edge_cuts << std::endl;

    thrust::replace_if(A.column_indices.begin(), A.column_indices.end(),
                       edge_cuts.begin(), thrust::identity<IndexType>(), -1);

    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> G_host(A);

    cusp::array1d<IndexType,cusp::host_memory> components(N);
    size_t num_comps = cusp::graph::connected_components(G_host, components);
    stats.num_comps = num_comps;

    cusp::array1d<IndexType,cusp::host_memory> component_sizes(num_comps);
    thrust::sort(components.begin(), components.end());
    thrust::reduce_by_key(components.begin(), components.end(), thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(), component_sizes.begin());

    /* if(verbose) */
    /*     cusp::print(component_sizes); */

    cusp::array1d<IndexType,MemorySpace> part_sizes(num_parts, 0);
    thrust::sort(partition.begin(), partition.end());
    thrust::reduce_by_key(partition.begin(), partition.end(), thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(), part_sizes.begin(), thrust::equal_to<IndexType>());

    ValueType S_p   = *thrust::max_element(part_sizes.begin(), part_sizes.end());
    ValueType S_opt = std::ceil(ValueType(N/num_parts));
    ValueType imbalance = ((S_p/S_opt) * 100.0) - 100.0;

    if(verbose)
    {
        std::cout << "Largest partition size is " << S_p << ", while S_opt is " << S_opt
                  << " representing " << imbalance << "% imbalance" << std::endl;
        for(size_t i = 0; i < num_parts; i++)
            std::cout << "Size of partition " << i << " is " << part_sizes[i] << std::endl;
    }

    stats.imbalance = imbalance;
    stats.num_edge_cuts = num_edge_cuts;
    stats.max_part_size = *thrust::max_element(part_sizes.begin(), part_sizes.end());
}

template<typename MatrixType>
void run_tests(const MatrixType& A, const partition_config& config, partition_stats& stats)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    size_t N = A.num_rows;
    cusp::array1d<IndexType, MemorySpace> partition(N);

    // initialize starting vector to random values in [0,1)
    cusp::array2d<ValueType,MemorySpace> X0(N,config.blocksize);
    /* cusp::blas::copy(cusp::random_array<ValueType>(X0.num_entries), X0.values); */

    timer initial_guess_timer;
    generate_initial_guess(A, X0);
    // normalize X0
    cusp::constant_array<ValueType> ones(A.num_rows, ValueType(1) / std::sqrt(A.num_rows));
    cusp::eigen::detail::modifiedGramSchmidt(X0, ones);
    stats.initial_guess_time = initial_guess_timer.milliseconds_elapsed();
    if(config.verbose)
        std::cout << " Initial guess time : " << stats.initial_guess_time << " (ms)." << std::endl;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("cublasCreate failed");
    }
    cusp::cublas::execution_policy cublas(handle);

    cusp::array1d<ValueType, MemorySpace> eigVals(config.blocksize);
    cusp::array2d<ValueType, MemorySpace> eigVecs(N, config.blocksize);

    timer bisect_timer;
    cusp::eigen::block_lanczos(cublas, A, X0, eigVals, eigVecs,
                               config.blocksize, config.maxouter,
                               config.maxinner, config.verbose);
    stats.bisect_time = bisect_timer.milliseconds_elapsed();
    if(config.verbose)
        std::cout << " Bisect time : " << stats.bisect_time << " (ms)." << std::endl;

    timer cut_timer;
    cusp::array1d<ValueType,MemorySpace> fiedler(eigVecs.column(0));
    thrust::sort(fiedler.begin(), fiedler.end());
    ValueType vmed = fiedler[N/2];
    thrust::transform(eigVecs.column(0).begin(), eigVecs.column(0).end(),
                      thrust::constant_iterator<ValueType>(vmed),
                      partition.begin(), thrust::greater_equal<ValueType>());
    stats.cut_time = cut_timer.milliseconds_elapsed();

    analyze(A, partition, stats, config.verbose);
}

template<typename IndexType, typename ValueType, typename MemorySpace>
void laplacian_transform(cusp::csr_matrix<IndexType,ValueType,MemorySpace>& A)
{
    cusp::blas::fill(A.values, -1);

    cusp::array1d<IndexType,MemorySpace> A_row_lengths(A.num_rows);
    thrust::transform(A.row_offsets.begin() + 1, A.row_offsets.end(),
                      A.row_offsets.begin(), A_row_lengths.begin(),
                      thrust::minus<IndexType>());
    cusp::blas::axpy(cusp::constant_array<IndexType>(A.num_rows, 1),
                     A_row_lengths, -1);

    cusp::array1d<IndexType,MemorySpace> A_row_indices(A.num_entries);
    cusp::offsets_to_indices(A.row_offsets, A_row_indices);

    cusp::array1d<IndexType,MemorySpace> diagonal_indices(A.num_rows);
    thrust::scatter_if(thrust::counting_iterator<IndexType>(0),
                       thrust::counting_iterator<IndexType>(A.num_entries),
                       A_row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(A_row_indices.begin(), A.column_indices.begin())),
                           cusp::equal_pair_functor<IndexType>()),
                       diagonal_indices.begin());

    thrust::copy(A_row_lengths.begin(), A_row_lengths.end(),
                 thrust::make_permutation_iterator(A.values.begin(), diagonal_indices.begin()));
}

int main(int argc, char*argv[])
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    srand(time(NULL));

    std::string filename = process_args(argc, argv);

    if (filename.empty())
    {
        const size_t size = 25;
        cusp::gallery::poisson5pt(A, size, size);
    }
    else
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, filename);
        std::cout << "Read matrix (" << filename << ") ";
    }

    laplacian_transform(A);

    size_t N = A.num_rows;
    size_t M = A.num_entries;
    std::cout << "with shape ("  << N << "," << N << ") and " << M << " entries" << "\n";

    partition_stats base_stats;
    base_stats.initial_guess_time = 0;
    base_stats.bisect_time = 0;
    base_stats.cut_time = 0;
    base_stats.imbalance = 0;
    base_stats.min_edge_cuts = INT_MAX;
    base_stats.max_edge_cuts = 0;
    base_stats.avg_edge_cuts = 0;
    base_stats.num_edge_cuts = 0;
    base_stats.num_comps = 0;
    base_stats.max_part_size = 0;

    partition_config config;
    size_t num_iterations = args.count("numiter")   ? atoi(args["numiter"].c_str())   : 10;
    config.blocksize      = args.count("blocksize") ? atoi(args["blocksize"].c_str()) : 4;
    config.maxinner       = args.count("maxinner")  ? atoi(args["maxinner"].c_str())  : 10;
    config.maxouter       = args.count("maxouter")  ? atoi(args["maxouter"].c_str())  : 1;
    config.verbose        = args.count("v")         ? true  : false;

    std::vector<partition_stats> stats_array(num_iterations);
    for(size_t i = 0; i < num_iterations; i++)
    {
        run_tests(A, config, stats_array[i]);

        base_stats.initial_guess_time += stats_array[i].initial_guess_time;
        base_stats.bisect_time        += stats_array[i].bisect_time;
        base_stats.cut_time           += stats_array[i].cut_time;
        base_stats.imbalance          += stats_array[i].imbalance;
        base_stats.num_edge_cuts      += stats_array[i].num_edge_cuts;
        base_stats.min_edge_cuts      =  min(base_stats.min_edge_cuts, stats_array[i].num_edge_cuts);
        base_stats.max_edge_cuts      =  max(base_stats.max_edge_cuts, stats_array[i].num_edge_cuts);
        base_stats.num_comps          =  max(base_stats.num_comps, stats_array[i].num_comps);
        base_stats.max_part_size      =  max(base_stats.max_part_size, stats_array[i].max_part_size);
    }

    base_stats.initial_guess_time /= num_iterations;
    base_stats.bisect_time        /= num_iterations;
    base_stats.cut_time           /= num_iterations;
    base_stats.imbalance          /= num_iterations;
    base_stats.avg_edge_cuts       =  base_stats.num_edge_cuts / num_iterations;

    printf("blocksize=%lu\n",            config.blocksize);
    printf("maxinner=%lu\n",             config.maxinner);

    printf("initial_guess_time=%4.2f\n", base_stats.initial_guess_time);
    printf("bisect_time=%4.2f\n",        base_stats.bisect_time);
    printf("cut_time=%4.2f\n",           base_stats.cut_time);
    printf("total_time=%4.2f\n",         base_stats.initial_guess_time + base_stats.bisect_time);
    printf("min_edge_cuts=%d\n",         base_stats.min_edge_cuts);
    printf("max_edge_cuts=%d\n",         base_stats.max_edge_cuts);
    printf("avg_edge_cuts=%d\n",         base_stats.avg_edge_cuts);
    printf("num_comps=%d\n",             base_stats.num_comps);
    printf("max_part_size=%d\n",         base_stats.max_part_size);

    return EXIT_SUCCESS;
}

