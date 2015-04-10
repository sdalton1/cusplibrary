#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>

#include <cusp/gallery/poisson.h>

#include <iostream>

#include <cusp/grapple/grapple.h>

int main(int argc, char ** argv)
{
    typedef int                  IndexType;
    typedef float                ValueType;
    typedef cusp::device_memory  MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    cusp::gallery::poisson5pt(A, 500, 500);

    std::cout << "Input matrix has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    grapple_system exec;

    cusp::array1d<IndexType,MemorySpace> A_row_indices(A.num_entries);
    cusp::offsets_to_indices(exec, A.row_offsets, A_row_indices);

    cusp::array1d<IndexType,MemorySpace> x(A.num_cols, 1);
    cusp::array1d<IndexType,MemorySpace> y(A.num_rows, 0);
    cusp::multiply(thrust::omp::par, A, x, y);

    return 0;
}

