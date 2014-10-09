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
#include <cusp/precond/smoother_policy.h>
#include <cusp/precond/solve_policy.h>
#include <cusp/precond/coarse_solve_policy.h>

namespace cusp
{
namespace detail
{

template<typename MatrixType, typename SetupPolicy, typename SolvePolicy>
struct multilevel_policy {

    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef typename cusp::precond::aggregation::smoothed_aggregation_policy<IndexType,ValueType,MemorySpace> SmoothedAggregationPolicy;
    typedef typename cusp::precond::jacobi_smoother_policy<ValueType,MemorySpace>       JacobiSmootherPolicy;
    typedef typename cusp::precond::lu_solve_policy<ValueType>                          LuSolvePolicy;
    typedef typename cusp::precond::v_cycle_policy<JacobiSmootherPolicy,LuSolvePolicy>  VJacobiLUPolicy;

    typedef typename thrust::detail::eval_if<
      thrust::detail::is_same<SetupPolicy,thrust::use_default>::value,
      thrust::detail::identity_<SmoothedAggregationPolicy>,
      thrust::detail::identity_<SetupPolicy>
    >::type setup_policy;

    typedef typename thrust::detail::eval_if<
      thrust::detail::is_same<SolvePolicy,thrust::use_default>::value,
      thrust::detail::identity_<VJacobiLUPolicy>,
      thrust::detail::identity_<SolvePolicy>
    >::type solve_policy;
};

} // end namespace detail

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
multilevel<MatrixType,SetupPolicy,SolvePolicy>
::multilevel(const MatrixType& A, const size_t max_levels, const size_t min_level_size)
  : A(&A), max_levels(max_levels), min_level_size(min_level_size)
{
    CUSP_PROFILE_SCOPED();

    // reserve room for maximum number of levels
    levels.reserve(max_levels);

    // add the first level
    levels.push_back(level());

    // build heirarchy
    while ((levels.back().A.num_rows > min_level_size) &&
           (levels.size() < max_levels))
    {
        extend_hierarchy();
    }

    // construct additional solve phase components
    initialize_solve();
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
template <typename MatrixType2>
multilevel<MatrixType,SetupPolicy,SolvePolicy>
::multilevel(const multilevel<MatrixType2,SetupPolicy,SolvePolicy>& M)
{
    for(size_t lvl = 0; lvl < M.levels.size(); lvl++)
        levels.push_back(M.levels[lvl]);
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SetupPolicy,SolvePolicy>
::operator()(const Array1& b, Array2& x)
{
    CUSP_PROFILE_SCOPED();

    // perform 1 V-cycle
    cycle(b, x, 0);
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SetupPolicy,SolvePolicy>
::solve(const Array1& b, Array2& x)
{
    CUSP_PROFILE_SCOPED();

    cusp::monitor<ValueType> monitor(b);

    solve(b, x, monitor);
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
template <typename Array1, typename Array2, typename Monitor>
void multilevel<MatrixType,SetupPolicy,SolvePolicy>
::solve(const Array1& b, Array2& x, Monitor& monitor)
{
    CUSP_PROFILE_SCOPED();

    const size_t n = levels[0].A.num_rows;

    // use simple iteration
    cusp::array1d<ValueType,MemorySpace> update(n);
    cusp::array1d<ValueType,MemorySpace> residual(n);

    // compute initial residual
    cusp::multiply(levels[0].A, x, residual);
    cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));

    while(!monitor.finished(residual))
    {
        cycle(residual, update, 0);

        // x += M * r
        cusp::blas::axpy(update, x, ValueType(1.0));

        // update residual
        cusp::multiply(levels[0].A, x, residual);
        cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));
        ++monitor;
    }
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
void multilevel<MatrixType,SetupPolicy,SolvePolicy>
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

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
double multilevel<MatrixType,SetupPolicy,SolvePolicy>
::operator_complexity(void)
{
    size_t nnz = 0;

    for(size_t index = 0; index < levels.size(); index++)
        nnz += levels[index].A.num_entries;

    return (double) nnz / (double) levels[0].A.num_entries;
}

template <typename MatrixType, typename SetupPolicy, typename SolvePolicy>
double multilevel<MatrixType,SetupPolicy,SolvePolicy>
::grid_complexity(void)
{
    size_t unknowns = 0;
    for(size_t index = 0; index < levels.size(); index++)
        unknowns += levels[index].A.num_rows;

    return (double) unknowns / (double) levels[0].A.num_rows;
}

} // end namespace cusp

