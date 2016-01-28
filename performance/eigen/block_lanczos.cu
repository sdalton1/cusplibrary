#include <cusp/csr_matrix.h>
#include <cusp/eigen/block_lanczos.h>
#include <cusp/gallery/poisson.h>

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    // initialize matrix
    cusp::gallery::poisson5pt(A, 1024, 1024);
    // allocate storage and initialize eigenpairs
    cusp::array1d<float, cusp::device_memory> eigVals(5,0);
    cusp::array2d<float, cusp::device_memory> eigVecs(A.num_rows, 5);

    // Compute the largest eigenpair of A
    cusp::eigen::block_lanczos(A, eigVals, eigVecs, 4, 1, 10);
    std::cout << "Largest eigenvalue : " << eigVals[4] << std::endl;
    return 0;
}
