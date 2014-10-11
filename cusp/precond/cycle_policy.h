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

#include <cusp/precond/smoother_policy.h>
#include <cusp/precond/solve_policy.h>

namespace cusp
{
namespace precond
{

class v_cycle_policy;

template<typename ValueType, typename MemorySpace, typename CycleTypePolicy>
struct select_cycle_type_policy {

    typedef typename cusp::precond::v_cycle_policy VCyclePolicy;

    typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<CycleTypePolicy,thrust::use_default>::value,
           thrust::detail::identity_<VCyclePolicy>,
           thrust::detail::identity_<CycleTypePolicy>
           >::type type;
};

template<typename ValueType, typename MemorySpace, typename SmootherPolicy>
struct select_smoother_policy {

    typedef typename cusp::precond::jacobi_smoother_policy<ValueType,MemorySpace> JacobiSmootherPolicy;

    typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<SmootherPolicy,thrust::use_default>::value,
           thrust::detail::identity_<JacobiSmootherPolicy>,
           thrust::detail::identity_<SmootherPolicy>
           >::type type;
};

template<typename ValueType, typename SolverPolicy>
struct select_solve_policy {

    typedef typename cusp::precond::lu_solve_policy<ValueType> LuSolvePolicy;

    typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<SolverPolicy,thrust::use_default>::value,
           thrust::detail::identity_<LuSolvePolicy>,
           thrust::detail::identity_<SolverPolicy>
           >::type type;
};

template <typename ValueType,
         typename MemorySpace,
         typename CycleTypePolicy = thrust::use_default,
         typename SmootherPolicy  = thrust::use_default,
         typename SolvePolicy     = thrust::use_default>
class cycle_policy :
    private select_cycle_type_policy<ValueType,MemorySpace,CycleTypePolicy>::type,
    private select_smoother_policy<ValueType,MemorySpace,SmootherPolicy>::type,
    private select_solve_policy<ValueType,SolvePolicy>::type
{
protected:

    typedef typename select_cycle_type_policy<ValueType,MemorySpace,CycleTypePolicy>::type   cycle_type_policy;
    typedef typename select_smoother_policy<ValueType,MemorySpace,SmootherPolicy>::type      smoother_policy;
    typedef typename select_solve_policy<ValueType,SolvePolicy>::type                        solve_policy;

    typedef typename smoother_policy::SmootherType                                           SmootherType;

    using cycle_type_policy::cycle;
    using solve_policy::coarse_initialize;
    using solve_policy::coarse_solve;

    struct cycle_level
    {
        cycle_level() {}

        cusp::array1d<ValueType,MemorySpace> x;
        cusp::array1d<ValueType,MemorySpace> b;
        cusp::array1d<ValueType,MemorySpace> residual;

        SmootherType                         smoother;

        template<typename LevelType>
        cycle_level(const LevelType& L) : x(L.x), b(L.b), residual(L.residual) {}
    };

    std::vector<cycle_level> cycle_levels;

    template<typename SetupLevels>
    void cycle_initialize(const SetupLevels& setup_levels)
    {
        for(int i = 0; i < setup_levels.size(); i++)
        {
            cycle_levels.push_back(cycle_level());
            cycle_levels.back().smoother = SmootherType(setup_levels[i].A);

            int N = setup_levels[i].A.num_rows;
            cycle_levels.back().x.resize(N);
            cycle_levels.back().b.resize(N);
            cycle_levels.back().residual.resize(N);
        }

        coarse_initialize(setup_levels.back().A);
    }
};

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/cycle_policy.inl>
