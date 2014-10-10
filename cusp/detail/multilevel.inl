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
#include <cusp/monitor.h>
#include <cusp/blas/blas.h>

#include <cusp/precond/smoothed_aggregation_policy.h>
#include <cusp/precond/cycle_policy.h>

namespace cusp
{
namespace detail
{

template<typename MatrixType, typename SetupPolicy, typename CyclePolicy>
struct multilevel_policy {

    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef typename cusp::precond::smoothed_aggregation_policy<IndexType,ValueType,MemorySpace> SmoothedAggregationPolicy;
    typedef typename cusp::precond::cycle_policy<ValueType,MemorySpace>                          VJacobiLUPolicy;

    typedef typename thrust::detail::eval_if<
      thrust::detail::is_same<SetupPolicy,thrust::use_default>::value,
      thrust::detail::identity_<SmoothedAggregationPolicy>,
      thrust::detail::identity_<SetupPolicy>
    >::type setup_policy;

    typedef typename thrust::detail::eval_if<
      thrust::detail::is_same<CyclePolicy,thrust::use_default>::value,
      thrust::detail::identity_<VJacobiLUPolicy>,
      thrust::detail::identity_<CyclePolicy>
    >::type cycle_policy;
};

} // end namespace detail

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
multilevel<MatrixType,SetupPolicy,CyclePolicy>
::multilevel(const MatrixType& A, const size_t max_levels, const size_t min_level_size)
  : A(&A), max_levels(max_levels), min_level_size(min_level_size)
{
    CUSP_PROFILE_SCOPED();

    // initialize the setup components to build the hierarchy
    setup_initialize(A);

    // reserve room for maximum number of levels
    levels.reserve(max_levels);

    // build heirarchy
    do
    {
        // create container to store next level
        levels.push_back(level());

        // construct level
        extend_hierarchy(levels.back().R, levels.back().A, levels.back().P);

        // allocate buffers for cycling on current level
        size_t N = levels.back().A.num_rows;
        levels.back().b.resize(N);
        levels.back().x.resize(N);
        levels.back().residual.resize(N);

    } while ((levels.back().A.num_rows > min_level_size) &&
             (levels.size() < max_levels));

    // construct additional solve phase components
    cycle_initialize(levels);
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
template <typename MatrixType2>
multilevel<MatrixType,SetupPolicy,CyclePolicy>
::multilevel(const multilevel<MatrixType2,SetupPolicy,CyclePolicy>& M)
  : A(M.A), max_levels(M.max_levels), min_level_size(M.min_level_size)
{
    for(size_t lvl = 0; lvl < M.levels.size(); lvl++)
        levels.push_back(M.levels[lvl]);
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SetupPolicy,CyclePolicy>
::operator()(const Array1& b, Array2& x)
{
    CUSP_PROFILE_SCOPED();

    // perform 1 V-cycle
    cycle(levels, b, x, 0);
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SetupPolicy,CyclePolicy>
::solve(const Array1& b, Array2& x)
{
    CUSP_PROFILE_SCOPED();

    cusp::monitor<ValueType> monitor(b);

    solve(b, x, monitor);
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
template <typename Array1, typename Array2, typename Monitor>
void multilevel<MatrixType,SetupPolicy,CyclePolicy>
::solve(const Array1& b, Array2& x, Monitor& monitor)
{
    CUSP_PROFILE_SCOPED();

    const size_t n = A->num_rows;

    // compute initial residual
    cusp::multiply(*A, x, levels[0].residual);
    cusp::blas::axpby(b, levels[0].residual, levels[0].residual, 1.0, -1.0);

    while(!monitor.finished(levels[0].residual))
    {
        // execute cycle
        cycle(levels, levels[0].residual, levels[0].x, 0);

        // x += M * r
        cusp::blas::axpy(levels[0].x, x, ValueType(1.0));

        // update residual
        cusp::multiply(*A, x, levels[0].residual);
        cusp::blas::axpby(b, levels[0].residual, levels[0].residual, 1.0, -1.0);
        ++monitor;
    }
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
void multilevel<MatrixType,SetupPolicy,CyclePolicy>
::print(void)
{
    size_t num_levels = levels.size();

    std::cout << "\tNumber of Levels:\t" << num_levels << std::endl;
    std::cout << "\tOperator Complexity:\t" << operator_complexity() << std::endl;
    std::cout << "\tGrid Complexity:\t" << grid_complexity() << std::endl;
    std::cout << "\tlevel\tunknowns\tnonzeros:\t" << std::endl;

    double nnz = 0;

    for(size_t index = 0; index < num_levels; index++)
        nnz += levels[index].A.num_entries;

    for(size_t index = 0; index < num_levels; index++)
    {
        double percent = levels[index].A.num_entries / nnz;
        std::cout << "\t" << index << "\t" << levels[index].A.num_cols << "\t\t" \
                  << levels[index].A.num_entries << " \t[" << 100*percent << "%]" \
                  << std::endl;
    }
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
double multilevel<MatrixType,SetupPolicy,CyclePolicy>
::operator_complexity(void)
{
    size_t nnz = 0;

    for(size_t index = 0; index < levels.size(); index++)
        nnz += levels[index].A.num_entries;

    return (double) nnz / (double) levels[0].A.num_entries;
}

template <typename MatrixType, typename SetupPolicy, typename CyclePolicy>
double multilevel<MatrixType,SetupPolicy,CyclePolicy>
::grid_complexity(void)
{
    size_t unknowns = 0;
    for(size_t index = 0; index < levels.size(); index++)
        unknowns += levels[index].A.num_rows;

    return (double) unknowns / (double) levels[0].A.num_rows;
}

} // end namespace cusp

