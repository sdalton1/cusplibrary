#include <cusp/multilevel.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>

#include <iostream>

#include "../timer.h"

int main(int argc, char ** argv)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;
    typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> MatrixType;

    // create an empty sparse matrix structure
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;

    IndexType N = 256;

    // create 2D Poisson problem
    cusp::gallery::poisson5pt(A, N, N);

    // solve without preconditioning
    std::cout << "\nSolving with no preconditioner" << std::endl;

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria (iteration_limit = 10000, relative_tolerance = 1e-10)
    cusp::monitor<ValueType> monitor(b, 10000, 1e-10);

    // setup generic multilevel solver
    cusp::multilevel<MatrixType> M(A);

    // solve
    timer cg_time;
    cusp::krylov::cg(M, x, b, monitor);
    std::cout << "solved system  in " << cg_time.milliseconds_elapsed() << " ms " << std::endl;

    // report status
    monitor.print();

    return EXIT_SUCCESS;
}

