#include <cusp/convert.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/gallery/poisson.h>

#include <iostream>

#include "my_execution_policy.h"

int main(void)
{
    cusp::dia_matrix<int,float,cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 4, 4);


    my_policy exec;
    cusp::csr_matrix<int,float,cusp::device_memory> B;
    cusp::convert(exec, A, B);

    return 0;
}

