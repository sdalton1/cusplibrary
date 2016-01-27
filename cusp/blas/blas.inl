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

#ifndef __CUSP_BLAS_H__
#define __CUSP_BLAS_H__

#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/exception.h>
#include <cusp/verify.h>

#include <cusp/execution_policy.h>
#include <cusp/blas/blas_policy.h>
#include <cusp/blas/thrustblas/blas.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace blas
{

template <typename DerivedPolicy,
          typename ArrayType>
int amax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const ArrayType& x)
{
    using cusp::blas::thrustblas::amax;

    return amax(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename ArrayType>
int amax(const ArrayType& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::blas::amax(select_system(system), x);
}

template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x)
{
    using cusp::blas::thrustblas::asum;

    return asum(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(const ArrayType& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::blas::asum(select_system(system), x);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
          const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::axpy;

    cusp::assert_same_dimensions(x, y);

    axpy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y, alpha);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::axpy(select_system(system1,system2), x, y, alpha);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
                 ScalarType1 alpha,
                 ScalarType2 beta)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::axpby(select_system(system1,system2,system3), x, y, z, alpha, beta);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
                 ScalarType1 alpha,
                 ScalarType2 beta)
{
    using cusp::blas::thrustblas::axpby;

    cusp::assert_same_dimensions(x, y, z);

    axpby(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
          x, y, z, alpha, beta);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& output,
                    ScalarType1 alpha,
                    ScalarType2 beta,
                    ScalarType3 gamma)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;
    typedef typename ArrayType4::memory_space System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    cusp::blas::axpbypcz(select_system(system1,system2,system3,system4), x, y, z, output, alpha, beta, gamma);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& output,
                    ScalarType1 alpha,
                    ScalarType2 beta,
                    ScalarType3 gamma)
{
    using cusp::blas::thrustblas::axpbypcz;

    cusp::assert_same_dimensions(x, y, z, output);

    axpbypcz(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
             x, y, z, output, alpha, beta, gamma);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::xmy(select_system(system1,system2,system3), x, y, z);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z)
{
    using cusp::blas::thrustblas::xmy;

    cusp::assert_same_dimensions(x, y, z);

    xmy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y, z);
}

template <typename ArrayType1,
          typename ArrayType2>
void copy(const ArrayType1& x,
                ArrayType2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::copy(select_system(system1,system2), x, y);
}

template <typename ArrayType,
          typename RandomAccessIterator>
void copy(const ArrayType& x,
          cusp::array1d_view<RandomAccessIterator> y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System1;
    typedef typename thrust::iterator_system<RandomAccessIterator>::type System2;

    System1 system1;
    System2 system2;

    cusp::blas::copy(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
void copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const ArrayType1& x,
                ArrayType2& y)
{
    using cusp::blas::thrustblas::copy;

    cusp::assert_same_dimensions(x, y);

    copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(const ArrayType1& x,
    const ArrayType2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::blas::dot(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    const ArrayType1& x,
    const ArrayType2& y)
{
    using cusp::blas::thrustblas::dot;

    cusp::assert_same_dimensions(x, y);

    return dot(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

// TODO properly harmonize heterogenous types
template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(const ArrayType1& x,
     const ArrayType2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::blas::dotc(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType1& x,
     const ArrayType2& y)
{
    using cusp::blas::thrustblas::dotc;

    cusp::assert_same_dimensions(x, y);

    return dotc(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

template <typename RandomAccessIterator,
          typename ScalarType>
void fill(cusp::array1d_view<RandomAccessIterator> x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

    System system;

    cusp::blas::fill(select_system(system), x, alpha);
}

template <typename ArrayType,
          typename ScalarType>
void fill(ArrayType& x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    cusp::blas::fill(select_system(system), x, alpha);
}

template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void fill(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          ArrayType& x,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::fill;

    fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, alpha);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(const ArrayType& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::blas::nrm1(select_system(system), x);
}

template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x)
{
    using cusp::blas::thrustblas::nrm1;

    return nrm1(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(const ArrayType& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::blas::nrm2(select_system(system), x);
}

template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x)
{
    using cusp::blas::thrustblas::nrm2;

    return nrm2(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(const ArrayType& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::blas::nrmmax(select_system(system), x);
}

template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       const ArrayType& x)
{
    using cusp::blas::thrustblas::nrmmax;

    return nrmmax(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename ArrayType,
          typename ScalarType>
void scal(ArrayType& x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    cusp::blas::scal(select_system(system), x, alpha);
}

template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void scal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ArrayType& x,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::scal;

    scal(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, alpha);
}

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const Array2d& A,
          const Array1& x,
                Array2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1::memory_space  System2;
    typedef typename Array2::memory_space  System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::gemv(select_system(system1,system2,system3), A, x, y);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1,
          typename Array2>
void gemv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
          const Array1&  x,
                Array2&  y)
{
    using cusp::blas::thrustblas::gemv;

    gemv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, y);
}

template<typename Array1d1,
         typename Array1d2,
         typename Array2d1>
void ger(const Array1d1& x,
         const Array1d2& y,
               Array2d1& A)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1d1::memory_space System1;
    typedef typename Array1d2::memory_space System2;
    typedef typename Array2d1::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::ger(select_system(system1,system2,system3), x, y, A);
}

template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1>
void ger(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A)
{
    using cusp::blas::thrustblas::ger;

    ger(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y, A);
}

template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void symv(const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array1d1::memory_space System2;
    typedef typename Array1d2::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::symv(select_system(system1,system2,system3), A, x, y);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    using cusp::blas::thrustblas::symv;

    symv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, y);
}

template<typename Array1d,
         typename Array2d>
void syr(const Array1d& x,
               Array2d& A)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1d::memory_space System1;
    typedef typename Array2d::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::syr(select_system(system1,system2), x, A);
}

template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d>
void syr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d& x,
               Array2d& A)
{
    using cusp::blas::thrustblas::syr;

    syr(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, A);
}

template<typename Array2d,
         typename Array1d>
void trmv(const Array2d& A,
                Array1d& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::trmv(select_system(system1,system2), A, x);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    using cusp::blas::thrustblas::trmv;

    trmv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x);
}

template<typename Array2d,
         typename Array1d>
void trsv(const Array2d& A,
                Array1d& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::trsv(select_system(system1,system2), A, x);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    using cusp::blas::thrustblas::trsv;

    trsv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;
    typedef typename Array2d3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::gemm(select_system(system1,system2,system3), A, B, C);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using cusp::blas::thrustblas::gemm;

    gemm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;
    typedef typename Array2d3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::symm(select_system(system1,system2,system3), A, B, C);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void symm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using cusp::blas::thrustblas::symm;

    symm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C);
}

template<typename Array2d1,
         typename Array2d2>
void syrk(const Array2d1& A,
                Array2d2& B)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::syrk(select_system(system1,system2), A, B);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void syrk(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B)
{
    using cusp::blas::thrustblas::syrk;

    syrk(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;
    typedef typename Array2d3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::syr2k(select_system(system1,system2,system3), A, B, C);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void syr2k(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    using cusp::blas::thrustblas::syr2k;

    syr2k(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C);
}

template<typename Array2d1,
         typename Array2d2>
void trmm(const Array2d1& A,
                Array2d2& B)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::trmm(select_system(system1,system2), A, B);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trmm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B)
{
    using cusp::blas::thrustblas::trmm;

    trmm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B);
}

template<typename Array2d1,
         typename Array2d2>
void trsm(const Array2d1& A,
                Array2d2& B)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::trsm(select_system(system1,system2), A, B);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trsm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B)
{
    using cusp::blas::thrustblas::trsm;

    trsm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B);
}

} // end namespace blas
} // end namespace cusp

#endif // __CUSP_BLAS_H__

