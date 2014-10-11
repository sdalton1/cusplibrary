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

#include <cusp/multiply.h>
#include <cusp/blas/blas.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>

namespace cusp
{
namespace precond
{

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void jacobi_smoother_policy<ValueType,MemorySpace>
::presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    // x <- omega * D^-1 * b
    thrust::transform(b.begin(), b.end(),
                      jacobi_smoothers[i].diagonal.begin(),
                      x.begin(),
                      cusp::relaxation::detail::jacobi_presmooth_functor<ValueType>(jacobi_smoothers[i].default_omega));
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void jacobi_smoother_policy<ValueType,MemorySpace>
::postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    jacobi_smoothers[i](A, b, x);
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void polynomial_smoother_policy<ValueType,MemorySpace>
::presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    // Ignore the initial x and use b as the residual
    ValueType scale_factor = polynomial_smoothers[i].default_coefficients[0];
    cusp::blas::axpby(b, x, x, scale_factor, ValueType(0));

    for( size_t i = 1; i < polynomial_smoothers[i].default_coefficients.size(); i++ )
    {
        scale_factor = polynomial_smoothers[i].default_coefficients[i];

        cusp::multiply(A, x, polynomial_smoothers[i].y);
        cusp::blas::axpby(polynomial_smoothers[i].y, b, x, ValueType(1.0), scale_factor);
    }
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void polynomial_smoother_policy<ValueType,MemorySpace>
::postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    polynomial_smoothers[i](A, b, x);
}

} // end namespace precond
} // end namespace cusp
