#include <cusp/grapple.h>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/krylov/cg.h>

#include <cusp/gallery/poisson.h>

#include <iostream>

int main(int argc, char ** argv)
{
    using namespace grapple;

    typedef int                  IndexType;
    typedef float                ValueType;
    typedef cusp::device_memory  MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    cusp::gallery::poisson5pt(A, 500, 500);

    std::cout << "Input matrix has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
    cusp::monitor<ValueType> monitor(b, 1, 1e-10);

    grapple_system exec;
    cusp::krylov::cg(exec, A, x, b, monitor);

    return 0;
}

