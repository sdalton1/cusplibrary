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

template <typename SmootherPolicy, typename SolverPolicy>
class v_cycle_policy
  : private SmootherPolicy, private SolverPolicy
{
    protected:
    typedef SmootherPolicy smoother_policy;
    typedef SolverPolicy   solver_policy;

    using solver_policy::coarse_solve;
    using smoother_policy::presmooth;
    using smoother_policy::postsmooth;

    template<typename Levels, typename Array1, typename Array2>
    void cycle(Levels& levels, const Array1& b, Array2& x, const size_t i);

    void initialize_solve(void) {}
};

template <typename SmootherPolicy, typename SolverPolicy>
class w_cycle_policy
  : private SmootherPolicy, private SolverPolicy
{
    protected:
    typedef SmootherPolicy smoother_policy;
    typedef SolverPolicy   solver_policy;

    using solver_policy::coarse_solve;
    using smoother_policy::presmooth;
    using smoother_policy::postsmooth;

    template<typename Levels, typename Array1, typename Array2>
    void cycle(Levels& levels, const Array1& b, Array2& x, const size_t i);

    void initialize_solve(void) {}
};

template <typename SmootherPolicy, typename SolverPolicy>
class f_cycle_policy
  : private SmootherPolicy, private SolverPolicy
{
    protected:
    typedef SmootherPolicy smoother_policy;
    typedef SolverPolicy   solver_policy;

    using solver_policy::coarse_solve;
    using smoother_policy::presmooth;
    using smoother_policy::postsmooth;

    v_cycle_policy<SmootherPolicy,SolverPolicy> v;

    template<typename Levels, typename Array1, typename Array2>
    void cycle(Levels& levels, const Array1& b, Array2& x, const size_t i);

    void initialize_solve(void) {}
};

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/cycle_policy.inl>
