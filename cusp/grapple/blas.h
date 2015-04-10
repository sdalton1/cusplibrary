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

#include <cusp/blas/blas.h>

namespace cusp
{
namespace blas
{

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(grapple_system &exec,
          const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    exec.start(CUSP_BLAS_AXPY);
    cusp::blas::axpy(exec.policy(get_system(x.begin(), y.begin())), x, y, alpha);
    exec.stop();
}

template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(grapple_system &exec,
           const Array1& x,
           const Array2& y,
           Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    exec.start(CUSP_BLAS_AXPBY);
    cusp::blas::axpby(exec.policy(get_system(x.begin(), y.begin(), output.begin())), x, y, alpha, beta);
    exec.stop();
}

template <typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(grapple_system &exec,
              const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma)
{
    exec.start(CUSP_BLAS_AXPBYPCZ);
    cusp::blas::axpbypcz(exec.policy(get_system(x.begin(), y.begin(), z.begin(), output.begin())), x, y, z, alpha, beta, gamma);
    exec.stop();
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(grapple_system &exec,
         const Array1& x,
         const Array2& y,
         Array3& output)
{
    exec.start(CUSP_BLAS_XMY);
    cusp::blas::xmy(exec.policy(get_system(x.begin(), y.begin(), output.begin())), x, y, output);
    exec.stop();
}

template <typename Array1,
          typename Array2,
          typename Array3>
void copy(grapple_system &exec,
          const Array1& x,
          const Array2& y)
{
    exec.start(CUSP_BLAS_COPY);
    cusp::blas::copy(exec.policy(get_system(x.begin(), y.begin())), x, y);
    exec.stop();
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(grapple_system &exec,
    const Array1& x,
    const Array2& y)
{
    exec.start(CUSP_BLAS_DOT);
    typename Array1::value_type ret = cusp::blas::dot(exec.policy(get_system(x.begin(), y.begin())), x, y);
    exec.stop();

    return ret;
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dotc(grapple_system &exec,
     const Array1& x,
     const Array2& y)
{
    exec.start(CUSP_BLAS_DOTC);
    typename Array1::value_type ret = cusp::blas::dotc(exec.policy(get_system(x.begin(), y.begin())), x, y);
    exec.stop();

    return ret;
}

template <typename Array,
          typename ScalarType>
void fill(grapple_system &exec,
          Array& array,
          ScalarType alpha)
{
    exec.start(CUSP_BLAS_FILL);
    cusp::blas::fill(exec.policy(get_system(array.begin())), array, alpha);
    exec.stop();
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(grapple_system &exec,
     const Array& array)
{
    typedef typename cusp::detail::norm_type<typename Array::value_type>::type ValueType;

    exec.start(CUSP_BLAS_NRM1);
    ValueType ret = cusp::blas::nrm1(exec.policy(get_system(array.begin())), array);
    exec.stop();

    return ret;
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(grapple_system &exec,
     const Array& array)
{
    typedef typename cusp::detail::norm_type<typename Array::value_type>::type ValueType;

    exec.start(CUSP_BLAS_NRM2);
    ValueType ret = cusp::blas::nrm2(exec.policy(get_system(array.begin())), array);
    exec.stop();

    return ret;
}

template <typename Array>
typename Array::value_type
nrmmax(grapple_system &exec,
     const Array& array)
{
    typedef typename Array::value_type ValueType;

    exec.start(CUSP_BLAS_NRMMAX);
    ValueType ret = cusp::blas::nrmmax(exec.policy(get_system(array.begin())), array);
    exec.stop();

    return ret;
}

template <typename Array,
          typename ScalarType>
void scal(grapple_system &exec,
          Array& x,
          ScalarType alpha)
{
    exec.start(CUSP_BLAS_SCAL);
    cusp::blas::scal(exec.policy(get_system(x.begin())), x, alpha);
    exec.stop();
}

template <typename Array2d,
          typename Array1,
          typename Array2>
void gemv(grapple_system &exec,
          const Array2d& A,
          const Array1&  x,
                Array2&  y)
{
    exec.start(CUSP_BLAS_GEMV);
    cusp::blas::gemv(exec.policy(get_system(A.values.begin(), x.begin(), y.begin())), A, x, y);
    exec.stop();
}

template <typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(grapple_system &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    exec.start(CUSP_BLAS_GEMM);
    cusp::blas::gemm(exec.policy(get_system(A.values.begin(), B.values.begin(), C.values.begin())), A, B, C);
    exec.stop();
}

} // end blas namespace
} // end cusp nemespace

