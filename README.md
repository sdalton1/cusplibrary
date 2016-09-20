<hr>
This branch demonstrates the utility of integrating execution-policies
into applications for the purpose of profiling and tuning.
<br><hr>
<h3>CUSP : A C++ Templated Sparse Matrix Library</h3>

Current release    : v0.5.1 (April 28, 2015)

| Linux | Windows | Coverage |
| ----- | ------- | -------- |
| [![Linux](https://travis-ci.org/sdalton1/cusplibrary.png)](https://travis-ci.org/sdalton1/cusplibrary) | [![Windows](https://ci.appveyor.com/api/projects/status/36pf1oqwkfq6xekn?svg=true)](https://ci.appveyor.com/project/StevenDalton/cusplibrary) | [![Coverage](https://coveralls.io/repos/sdalton1/cusplibrary/badge.svg?branch=master)](https://coveralls.io/r/sdalton1/cusplibrary?branch=master) |

View the project at [CUSP Website](http://cusplibrary.github.io) and the [cusp-users discussion forum](http://groups.google.com/group/cusp-users) for information and questions.

<br><hr>
<h3>A Profiling Example</h3>
#include <cusp/csr_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

#include <iostream>

#include "my_execution_policy.h"

int main(int argc, char** argv)
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;

    if (argc == 1)
    {
        std::cout << "Using default matrix (5-pt Laplacian stencil)" << std::endl;
        cusp::gallery::poisson5pt(A, 1000, 1000);
    }
    else
    {
        std::cout << "Reading matrix from file: " << argv[1] << std::endl;
        cusp::io::read_matrix_market_file(A, std::string(argv[1]));
    }

    size_t N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N,0);
    cusp::array1d<ValueType, MemorySpace> b(N,1);
    cusp::monitor<ValueType> monitor(b, 2, 1e-5);
    cusp::identity_operator<ValueType, MemorySpace> M(N, N);

    my_policy exec;
    cusp::krylov::cg(exec, A, x, b, monitor, M);

    return 0;
}
<br><hr>
Using default matrix (5-pt Laplacian stencil)
[ 0]cusp_cg                 : 12.8329  (ms), allocated : 0          bytes
[ 1]    cusp_multiply           : 1.78803  (ms), allocated : 0          bytes
[ 2]            cusp_multiply           : 1.77875  (ms), allocated : 0          bytes
[ 3]    cusp_blas_axpby         : 0.3448   (ms), allocated : 0          bytes
[ 4]    cusp_multiply           : 0.251712 (ms), allocated : 0          bytes
[ 5]            cusp_blas_copy          : 0.243776 (ms), allocated : 0          bytes
[ 6]    cusp_blas_copy          : 0.242304 (ms), allocated : 0          bytes
[ 7]    cusp_blas_dotc          : 0.436832 (ms), allocated : 0          bytes
[ 8]    cusp_blas_nrm2          : 0.39568  (ms), allocated : 0          bytes
[ 9]    cusp_multiply           : 1.78525  (ms), allocated : 0          bytes
[10]            cusp_multiply           : 1.77725  (ms), allocated : 0          bytes
[11]    cusp_blas_dotc          : 0.42     (ms), allocated : 0          bytes
[12]    cusp_blas_axpy          : 0.341088 (ms), allocated : 0          bytes
[13]    cusp_blas_axpy          : 0.360384 (ms), allocated : 0          bytes
[14]    cusp_multiply           : 0.25072  (ms), allocated : 0          bytes
[15]            cusp_blas_copy          : 0.242624 (ms), allocated : 0          bytes
[16]    cusp_blas_dotc          : 0.420992 (ms), allocated : 0          bytes
[17]    cusp_blas_axpby         : 0.352896 (ms), allocated : 0          bytes
[18]    cusp_blas_nrm2          : 0.39136  (ms), allocated : 0          bytes
[19]    cusp_multiply           : 1.77923  (ms), allocated : 0          bytes
[20]            cusp_multiply           : 1.77117  (ms), allocated : 0          bytes
[21]    cusp_blas_dotc          : 0.419264 (ms), allocated : 0          bytes
[22]    cusp_blas_axpy          : 0.342816 (ms), allocated : 0          bytes
[23]    cusp_blas_axpy          : 0.356352 (ms), allocated : 0          bytes
[24]    cusp_multiply           : 0.253696 (ms), allocated : 0          bytes
[25]            cusp_blas_copy          : 0.245472 (ms), allocated : 0          bytes
[26]    cusp_blas_dotc          : 0.421984 (ms), allocated : 0          bytes
[27]    cusp_blas_axpby         : 0.350016 (ms), allocated : 0          bytes
[28]    cusp_blas_nrm2          : 0.42064  (ms), allocated : 0          bytes


<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, cusp::device_memory> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b);

    return 0;
}
```

<br><hr>
<h3>Stable Releases</h3>

CUSP releases are labeled using version identifiers having three fields:

| Date | Version | Date | Version |
| ---- | ------- | ---- | ------- |
|            |                                                                              | 03/13/2015 | [CUSP v0.5.0](https://github.com/cusplibrary/cusplibrary/archive/v0.5.0.zip) |
|            |                                                                              | 08/30/2013 | [CUSP v0.4.0](https://github.com/cusplibrary/cusplibrary/archive/v0.4.0.zip) |
|            |                                                                              | 03/08/2012 | [CUSP v0.3.1](https://github.com/cusplibrary/cusplibrary/archive/v0.3.1.zip) |
|            |                                                                              | 02/04/2012 | [CUSP v0.3.0](https://github.com/cusplibrary/cusplibrary/archive/v0.3.0.zip) |
|            |                                                                              | 05/30/2011 | [CUSP v0.2.0](https://github.com/cusplibrary/cusplibrary/archive/v0.2.0.zip) |
| 04/28/2015 | [CUSP v0.5.1](https://github.com/cusplibrary/cusplibrary/archive/v0.5.1.zip) | 07/10/2010 | [CUSP v0.1.0](https://github.com/cusplibrary/cusplibrary/archive/v0.1.0.zip) |


<br><hr>
<h3>Contributors</h3>

CUSP is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).
[Nathan Bell](http:github.com/wnbell) was the original creator and
[Steven Dalton](http://github.com/sdalton1) is the current primary contributor.

<br><hr>
<h3>Citing</h3>

```shell
@MISC{Cusp,
  author = "Steven Dalton and Nathan Bell and Luke Olson and Michael Garland",
  title = "Cusp: Generic Parallel Algorithms for Sparse Matrix and Graph Computations",
  year = "2014",
  url = "http://cusplibrary.github.io/",
  note = "Version 0.5.0"
}
```

<br><hr>
<h3>Open Source License</h3>

CUSP is available under the Apache open-source license:

```
Copyright 2008-2014 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
