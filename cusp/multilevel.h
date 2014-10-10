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

/*! \file multilevel.h
 *  \brief Multilevel hierarchy
 *
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/linear_operator.h>


namespace cusp
{
namespace detail
{
// Forward definition
template<typename MatrixType, typename SetupPolicy, typename CyclePolicy> struct multilevel_policy;
} // end namespace detail

/*! \addtogroup iterative_solvers Multilevel hiearchy
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p multilevel : multilevel hierarchy
 *
 *
 *  TODO
 */
template <typename MatrixType,
          typename SetupPolicy = thrust::use_default,
          typename CyclePolicy = thrust::use_default>
class multilevel :
  private detail::multilevel_policy<MatrixType,SetupPolicy,CyclePolicy>::setup_policy,
  private detail::multilevel_policy<MatrixType,SetupPolicy,CyclePolicy>::cycle_policy,
  public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
public:

    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef typename detail::multilevel_policy<MatrixType,SetupPolicy,CyclePolicy>::setup_policy setup_policy;
    typedef typename detail::multilevel_policy<MatrixType,SetupPolicy,CyclePolicy>::cycle_policy cycle_policy;

    using setup_policy::setup_initialize;
    using setup_policy::extend_hierarchy;
    using cycle_policy::cycle_initialize;
    using cycle_policy::cycle;

    struct level
    {
        MatrixType R;  // restriction operator
        MatrixType A;  // matrix
        MatrixType P;  // prolongation operator

        level() {}

        cusp::array1d<ValueType,MemorySpace> x;
        cusp::array1d<ValueType,MemorySpace> b;
        cusp::array1d<ValueType,MemorySpace> residual;

        template<typename LevelType>
        level(const LevelType& L) : R(L.R), A(L.A), P(L.P),
                                    x(L.x), b(L.b), residual(L.residual) {}
    };

    const MatrixType * A;
    const size_t max_levels;
    const size_t min_level_size;

    std::vector<level> levels;

    multilevel() : max_levels(0), min_level_size(0) {};

    multilevel(const MatrixType& A, const size_t max_levels = 10, const size_t min_level_size = 100);

    template<typename MatrixType2>
    multilevel(const multilevel<MatrixType2, SetupPolicy, CyclePolicy>& M);

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y);

    template <typename Array1, typename Array2>
    void solve(const Array1& b, Array2& x);

    template <typename Array1, typename Array2, typename Monitor>
    void solve(const Array1& b, Array2& x, Monitor& monitor);

    void print(void);

    double operator_complexity(void);

    double grid_complexity(void);
};
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/multilevel.inl>

