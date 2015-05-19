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

namespace grapple
{

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(grapple_system &exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::axpy;

    exec.start(CUSP_BLAS_AXPY);
    axpy(exec.policy(), x, y, alpha);
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
                 Array3& z,
                 ScalarType1 alpha,
                 ScalarType2 beta)
{
    using cusp::blas::axpby;

    exec.start(CUSP_BLAS_AXPBY);
    axpby(exec.policy(), x, y, z, alpha, beta);
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
    using cusp::blas::axpbypcz;

    exec.start(CUSP_BLAS_AXPBYPCZ);
    axpbypcz(exec.policy(), x, y, z, output, alpha, beta, gamma);
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
    using cusp::blas::xmy;

    exec.start(CUSP_BLAS_XMY);
    xmy(exec.policy(), x, y, output);
    exec.stop();
}

template <typename Array1,
          typename Array2,
          typename Array3>
void copy(grapple_system &exec,
          const Array1& x,
          const Array2& y)
{
    using cusp::blas::thrustblas::copy;

    exec.start(CUSP_BLAS_COPY);
    copy(exec.policy(), x, y);
    exec.stop();
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(grapple_system &exec,
    const Array1& x,
    const Array2& y)
{
    using cusp::blas::thrustblas::dot;

    exec.start(CUSP_BLAS_DOT);
    typename Array1::value_type ret = dot(exec.policy(), x, y);
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
    using cusp::blas::thrustblas::dotc;

    exec.start(CUSP_BLAS_DOTC);
    typename Array1::value_type ret = dotc(exec.policy(), x, y);
    exec.stop();

    return ret;
}

template <typename Array,
          typename ScalarType>
void fill(grapple_system &exec,
          Array& x,
          ScalarType alpha)
{
    using cusp::blas::thrustblas::fill;

    exec.start(CUSP_BLAS_FILL);
    fill(exec.policy(), x, alpha);
    exec.stop();
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(grapple_system &exec,
     const Array& x)
{
    using cusp::blas::thrustblas::nrm1;

    typedef typename cusp::detail::norm_type<typename Array::value_type>::type ValueType;

    exec.start(CUSP_BLAS_NRM1);
    ValueType ret = nrm1(exec.policy(), x);
    exec.stop();

    return ret;
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(grapple_system &exec,
     const Array& x)
{
    using cusp::blas::thrustblas::nrm2;

    typedef typename cusp::detail::norm_type<typename Array::value_type>::type ValueType;

    exec.start(CUSP_BLAS_NRM2);
    ValueType ret = nrm2(exec.policy(), x);
    exec.stop();

    return ret;
}

template <typename Array>
typename Array::value_type
nrmmax(grapple_system &exec,
       const Array& x)
{
    using cusp::blas::thrustblas::nrmmax;

    typedef typename Array::value_type ValueType;

    exec.start(CUSP_BLAS_NRMMAX);
    ValueType ret = nrmmax(exec.policy(), x);
    exec.stop();

    return ret;
}

template <typename Array,
          typename ScalarType>
void scal(grapple_system &exec,
          Array& x,
          ScalarType alpha)
{
    using cusp::blas::thrustblas::scal;

    exec.start(CUSP_BLAS_SCAL);
    scal(exec.policy(), x, alpha);
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
    using cusp::blas::thrustblas::gemv;

    exec.start(CUSP_BLAS_GEMV);
    gemv(exec.policy(), A, x, y);
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
    using cusp::blas::thrustblas::gemm;

    exec.start(CUSP_BLAS_GEMM);
    gemm(exec.policy(), A, B, C);
    exec.stop();
}

} // end grapple namespace

