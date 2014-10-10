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

#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>

namespace cusp
{
namespace precond
{

template <typename ValueType, typename MemorySpace>
class jacobi_smoother_policy
{
  public:
    typedef cusp::relaxation::jacobi<ValueType,MemorySpace> SmootherType;

    std::vector<SmootherType> jacobi_smoothers;

    template<typename LevelType>
    void generate_smoother(const LevelType& level, const ValueType omega);

    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i);

    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i);
};

template <typename ValueType, typename MemorySpace>
class polynomial_smoother_policy
{
  public:
    typedef cusp::relaxation::polynomial<ValueType,MemorySpace> SmootherType;

    std::vector<SmootherType> polynomial_smoothers;

    template<typename LevelType>
    void generate_smoother(const LevelType& level);

    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i);

    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x, const size_t i);
};

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/smoother_policy.inl>
