/*#define CUSP_USE_TEXTURE_MEMORY*/

#include <cusp/blas/blas.h>

#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/connected_components.h>

#include <cusp/io/matrix_market.h>

#include <cusp/eigen/block_lanczos.h>

#include "../../timer.h"

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

int main(int argc, char*argv[])
{
    typedef int IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    size_t size = 4;
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);

        for(size_t i = 0; i < A.num_rows; i++)
            for(IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
                if(IndexType(i) == A.column_indices[j])
                    A.values[j] = A.row_offsets[i + 1] - A.row_offsets[i] - 1;
                else
                    A.values[j] = -1;
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

    cusp::array1d<ValueType, MemorySpace> eigVals(1);
    cusp::array2d<ValueType, MemorySpace> eigVecs(1,N);

    // initialize starting vector to random values in [0,1)
    cusp::array2d<ValueType,MemorySpace,cusp::column_major> X0(N, 4);
    cusp::blas::copy(cusp::random_array<ValueType>(X0.num_entries), X0.values);

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A_d(A);

    timer t;
    cusp::eigen::block_lanczos(A_d, eigVals, eigVecs, blocksize, maxouter, maxinner);
    std::cout << " Bisect time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;

    cusp::array1d<IndexType, MemorySpace> partition(N);
    analyze(A_d, partition);

    return EXIT_SUCCESS;
}

