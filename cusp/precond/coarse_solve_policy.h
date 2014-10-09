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

namespace cusp
{
namespace precond
{

template<typename ValueType>
class lu_solve_policy
{
private:
    cusp::array2d<ValueType,cusp::host_memory> A;
    cusp::array1d<int,cusp::host_memory>       pivot;

public:

    lu_solve_policy() {}

    template <typename MatrixType>
    lu_solve_policy(const MatrixType& A_);

    template <typename VectorType1, typename VectorType2>
    void coarse_solve(const VectorType1& x, VectorType2& y) const;
};

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/coarse_solve_policy.inl>
