#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/breadth_first_search.h>
#include <cusp/io/matrix_market.h>

#include <thrust/extrema.h>
#include "../../timer.h"

int main(int argc, char*argv[])
{
    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;

    size_t size = 1024;
    size_t source = 0;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc >= 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    if (argc == 3)
    {
        source = atoi(argv[2]);
        std::cout << "read source vertex (" << atoi(argv[2]) << ") ";
    }

    std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

    cusp::array1d<IndexType,MemorySpace> labels(A.num_rows);

    timer t;
    cusp::graph::breadth_first_search(A, source, labels);
    float bfs_time = t.milliseconds_elapsed();

    size_t num_levels = *thrust::max_element(labels.begin(), labels.end()) + 1;
    std::cout << "BFS time : " << bfs_time << " (ms), number of levels : "
              << num_levels << ", source : " << source << std::endl;

    return EXIT_SUCCESS;
}

