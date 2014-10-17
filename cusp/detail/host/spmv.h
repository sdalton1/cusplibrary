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

/*! \file spmv.h
 *  \brief Host SpMV routines
 */

#pragma once

#include <cusp/functional.h>

//MW: add some OpenMP pragmas
namespace cusp
{
namespace detail
{
namespace host
{

//////////////
// COO SpMV //
//////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(size_t n = 0; n < A.num_entries; n++)
    {
        const IndexType& i   = A.row_indices[n];
        const IndexType& j   = A.column_indices[n];
        const ValueType& Aij = A.values[n];
        const ValueType& xj  = x[j];

        y[i] = reduce(y[i], combine(Aij, xj));
    }
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_coo(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_coo(A, x, y,
             cusp::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}


//////////////
// CSR SpMV //
//////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_csr(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
    {
        const IndexType& row_start = A.row_offsets[i];
        const IndexType& row_end   = A.row_offsets[i+1];

        ValueType accumulator = initialize(y[i]);

        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType& j   = A.column_indices[jj];
            const ValueType& Aij = A.values[jj];
            const ValueType& xj  = x[j];

            accumulator = reduce(accumulator, combine(Aij, xj));
        }

        y[i] = accumulator;
    }
}


template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_csr(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_csr(A, x, y,
             cusp::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}


//////////////
// DIA SpMV //
//////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_dia(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    const size_t num_diagonals = A.values.num_cols;

    for(size_t i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(size_t i = 0; i < num_diagonals; i++)
    {
        const IndexType k = A.diagonal_offsets[i];

        const size_t i_start = std::max<IndexType>(0, -k);
        const size_t j_start = std::max<IndexType>(0,  k);

        // number of elements to process in this diagonal
        const size_t N = std::min(A.num_rows - i_start, A.num_cols - j_start);

        for(size_t n = 0; n < N; n++)
        {
            const ValueType Aij = A.values(i_start + n, i);

            const ValueType  xj = x[j_start + n];
                  ValueType& yi = y[i_start + n];

            yi = reduce(yi, combine(Aij, xj));
        }
    }
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_dia(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_dia(A, x, y,
             cusp::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}

//////////////
// ELL SpMV //
//////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_ell(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    const size_t& num_entries_per_row = A.column_indices.num_cols;

    const IndexType invalid_index = Matrix::invalid_index;

    for(size_t i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(size_t n = 0; n < num_entries_per_row; n++)
    {
        for(size_t i = 0; i < A.num_rows; i++)
        {
            const IndexType& j   = A.column_indices(i, n);
            const ValueType& Aij = A.values(i,n);

            if (j != invalid_index)
            {
                const ValueType& xj = x[j];
                y[i] = reduce(y[i], combine(Aij, xj));
            }
        }
    }
}


template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_ell(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_ell(A, x, y,
             cusp::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

