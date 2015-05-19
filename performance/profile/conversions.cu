#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/blas/blas.h>

#include <thrust/binary_search.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <stdio.h>

#include <cusp/grapple.h>

template <typename SourceType, typename DestinationType, typename InputType>
float time_conversion(const InputType& A)
{
    unsigned int N = 1;

    SourceType S;
    grapple::grapple_system exec;

    try
    {
        S = A;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    try
    {
        DestinationType D(S);
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    for(unsigned int i = 0; i < N; i++)
    {
        DestinationType D;
        cusp::convert(exec, S, D);
    }

    return 0;
}

template <typename SourceType, typename InputType>
void for_each_destination(const std::string& from, const InputType& A)
{
    typedef typename SourceType::index_type   I;
    typedef typename SourceType::value_type   V;
    typedef typename SourceType::memory_space M;

    typedef cusp::coo_matrix<I,V,M> COO;
    typedef cusp::csr_matrix<I,V,M> CSR;
    typedef cusp::dia_matrix<I,V,M> DIA;
    typedef cusp::ell_matrix<I,V,M> ELL;
    typedef cusp::hyb_matrix<I,V,M> HYB;

    std::cout << from << "->COO" << std::endl;
    time_conversion<SourceType, COO>(A);
    std::cout << from << "->CSR" << std::endl;
    time_conversion<SourceType, CSR>(A);
    std::cout << from << "->DIA" << std::endl;
    time_conversion<SourceType, DIA>(A);
    std::cout << from << "->ELL" << std::endl;
    time_conversion<SourceType, ELL>(A);
    std::cout << from << "->HYB" << std::endl;
    time_conversion<SourceType, HYB>(A);
}

template <typename MemorySpace, typename InputType>
void for_each_source(const InputType& A)
{
    typedef typename InputType::index_type I;
    typedef typename InputType::value_type V;

    typedef cusp::coo_matrix<I,V,MemorySpace> COO;
    typedef cusp::csr_matrix<I,V,MemorySpace> CSR;
    typedef cusp::dia_matrix<I,V,MemorySpace> DIA;
    typedef cusp::ell_matrix<I,V,MemorySpace> ELL;
    typedef cusp::hyb_matrix<I,V,MemorySpace> HYB;

    for_each_destination<COO>("COO",A); printf("\n");
    for_each_destination<CSR>("CSR",A); printf("\n");
    for_each_destination<DIA>("DIA",A); printf("\n");
    for_each_destination<ELL>("ELL",A); printf("\n");
    for_each_destination<HYB>("HYB",A); printf("\n");
}

int main(int argc, char ** argv)
{
    #ifdef __CUDACC__
    cudaSetDevice(0);
    #endif

    typedef int    IndexType;
    typedef float  ValueType;

    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;

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

    printf("Host Conversions (milliseconds per conversion)\n");
    for_each_source<cusp::host_memory>(A);

    printf("\n\n");

    printf("Device Conversions (milliseconds per conversion)\n");
    for_each_source<cusp::device_memory>(A);

    return 0;
}

