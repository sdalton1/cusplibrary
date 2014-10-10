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

namespace cusp
{
namespace precond
{

template <typename ValueType>
template <typename MatrixType>
lu_solve_policy<ValueType>
::lu_solve_policy(const MatrixType& A_)
{
    CUSP_PROFILE_SCOPED();

    assert(A_.num_rows == A_.num_cols);

    A = A_;
    int n = A.num_rows;
    pivot.resize(n);

    // For each row and column, k = 0, ..., n-1,
    for (int k = 0; k < n; k++)
    {
        // find the pivot row
        pivot[k] = k;
        ValueType max = std::fabs(A(k,k));

        for (int j = k + 1; j < n; j++)
        {
            if (max < std::fabs(A(j,k)))
            {
                max = std::fabs(A(j,k));
                pivot[k] = j;
            }
        }

        // and if the pivot row differs from the current row, then
        // interchange the two rows.
        if (pivot[k] != k)
            for (int j = 0; j < n; j++)
                std::swap(A(k,j), A(pivot[k],j));

        // and if the matrix is singular, return error
        if (A(k,k) == 0.0)
            throw cusp::runtime_exception("matrix is singular");

        // otherwise find the lower triangular matrix elements for column k.
        for (int i = k + 1; i < n; i++)
            A(i,k) /= A(k,k);

        // update remaining matrix
        for (int i = k + 1; i < n; i++)
            for (int j = k + 1; j < n; j++)
                A(i,j) -= A(i,k) * A(k,j);
    }
}

template <typename ValueType>
template <typename VectorType1, typename VectorType2>
void lu_solve_policy<ValueType>
::coarse_solve(VectorType1& b, VectorType2& x)
{
    CUSP_PROFILE_SCOPED();

    const int n = A.num_rows;

    // copy rhs to x
    temp_x = b;

    // Solve the linear equation Lx = b for x, where L is a lower
    // triangular matrix with an implied 1 along the diagonal.
    for (int k = 0; k < n; k++)
    {
        if (pivot[k] != k)
            std::swap(temp_x[k],temp_x[pivot[k]]);

        for (int i = 0; i < k; i++)
            temp_x[k] -= A(k,i) * temp_x[i];
    }

    // Solve the linear equation Ux = y, where y is the solution
    // obtained above of Lx = b and U is an upper triangular matrix.
    for (int k = n - 1; k >= 0; k--)
    {
        for (int i = k + 1; i < n; i++)
            temp_x[k] -= A(k,i) * temp_x[i];

        if (A(k,k) == 0)
            throw cusp::runtime_exception("matrix is non-invertible");

        temp_x[k] /= A(k,k);
    }

    b = temp_x;
}

} // end namespace precond
} // end namespace cusp
