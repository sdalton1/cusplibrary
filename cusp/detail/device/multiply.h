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

#include <cusp/format.h>
#include <cusp/coo_matrix.h>

// SpMV
#include <cusp/detail/device/spmv/coo.h>
#include <cusp/detail/device/spmv/csr_vector.h>
#include <cusp/detail/device/spmv/dia.h>
#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/hyb.h>

// SpMM
#include <cusp/detail/device/spmm/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{

///////////////////////////////////
// Sparse Matrix-Vector Multiply //
///////////////////////////////////
template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::coo_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::device::spmv_coo(A, B, C, 1, 0);
}

template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::device::spmv_csr_vector(A, B, C, 1, 0);
}

template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::dia_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::device::spmv_dia(A, B, C, 1, 0);
}

template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::ell_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::device::spmv_ell(A, B, C, 1, 0);
}

template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::hyb_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::device::spmv_hyb(A, B, C, 1, 0);
}

/////////////////////////////////
// Permutation Matrix Multiply //
/////////////////////////////////
template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::permutation_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    thrust::gather(A.permutation.begin(), A.permutation.end(), B.begin(), C.begin());
}

// Ensure 2D arrays are stored in column-major format
template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
              Vector2& C,
              cusp::sparse_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    cusp::detail::device::multiply(A, B, C,
                                   cusp::sparse_format(),
                                   cusp::array2d_format(),
                                   cusp::array2d_format(),
                                   typename Vector1::orientation(),
                                   typename Vector2::orientation());
}

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////
template <typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
              Matrix3& C,
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    cusp::detail::device::spmm_coo(A,B,C);
}

template <typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
              Matrix3& C,
              cusp::sparse_format,
              cusp::sparse_format,
              cusp::sparse_format)
{
    // other formats use COO * COO
    cusp::coo_matrix<typename Matrix1::index_type,typename Matrix1::value_type,cusp::device_memory> A_(A);
    cusp::coo_matrix<typename Matrix2::index_type,typename Matrix2::value_type,cusp::device_memory> B_(B);
    cusp::coo_matrix<typename Matrix3::index_type,typename Matrix3::value_type,cusp::device_memory> C_;

    cusp::detail::device::spmm_coo(A_,B_,C_);

    cusp::convert(C_, C);
}

/////////////////
// Entry Point //
/////////////////
template <typename Matrix,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(const Matrix&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C)
{
    cusp::detail::device::multiply(A, B, C,
                                   typename Matrix::format(),
                                   typename MatrixOrVector1::format(),
                                   typename MatrixOrVector2::format());
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

