#include <cusp/grapple/grapple.h>

#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>

#include <iostream>

int main(int argc, char ** argv)
{
    typedef int    IndexType;
    typedef float  ValueType;

    cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> A;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        cusp::gallery::poisson5pt(A, 500, 500);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
    }

    std::cout << "Input matrix has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    grapple_system exec;

    cusp::array1d<IndexType,cusp::device_memory> A_row_indices(A.num_entries);
    cusp::offsets_to_indices(exec, A.row_offsets, A_row_indices);
    cusp::compute_optimal_entries_per_row(exec, A.row_offsets);

    return 0;
}

