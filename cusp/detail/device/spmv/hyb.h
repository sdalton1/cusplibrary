/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename Matrix, typename Array1, typename Array2, typename ScalarType>
void spmv_hyb(const Matrix&   A,
              const Array1&   x,
                    Array2&   y,
              const ScalarType& alpha,
              const ScalarType& beta)
{
    spmv_ell(A.ell, x, y, alpha, beta);
    spmv_coo(A.coo, x, y, alpha, ScalarType(1));
}

template <typename Matrix, typename Array1, typename Array2, typename ScalarType>
void spmv_hyb_tex(const Matrix&   A,
                  const Array1&   x,
                        Array2&   y,
                  const ScalarType& alpha,
                  const ScalarType& beta)
{
    spmv_ell_tex(A.ell, x, y, alpha, beta);
    spmv_coo(A.coo, x, y, alpha, ScalarType(1));
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

