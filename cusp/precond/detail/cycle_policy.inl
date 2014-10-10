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

template <typename SmootherPolicy, typename SolverPolicy>
template<typename Levels, typename Array1, typename Array2>
void v_cycle_policy<SmootherPolicy, SolverPolicy>
::cycle(Levels& levels, const Array1& b, Array2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    if (i + 1 == levels.size())
    {
        // coarse grid solve
        coarse_solve(levels[i].b, levels[i].x);
    }
    else
    {
        // presmooth
        presmooth(levels[i].A, levels[i].b, levels[i].x, i);

        // compute residual <- b - A*x
        cusp::multiply(levels[i].A, levels[i].x, levels[i].residual);
        cusp::blas::axpby(levels[i].b, levels[i].residual, levels[i].residual, 1.0, -1.0);

        // restrict to coarse grid
        cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

        // compute coarse grid solution
        cycle(levels, levels[i + 1].residual, levels[i + 1].x, i + 1);

        // apply coarse grid correction
        cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
        cusp::blas::axpy(levels[i].residual, levels[i].x, 1.0);

        // postsmooth
        postsmooth(levels[i].A, levels[i].b, levels[i].x, i);
    }
}

template <typename SmootherPolicy, typename SolverPolicy>
template<typename Levels, typename Array1, typename Array2>
void w_cycle_policy<SmootherPolicy, SolverPolicy>
::cycle(Levels& levels, const Array1& b, Array2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    if (i + 1 == levels.size())
    {
        // coarse grid solve
        coarse_solve(levels[i].b, levels[i].x);
    }
    else
    {
        // presmooth
        presmooth(levels[i].A, levels[i].b, levels[i].x);

        // compute residual <- b - A*x
        cusp::multiply(levels[i].A, levels[i].x, levels[i].residual);
        cusp::blas::axpby(levels[i].b, levels[i].residual, levels[i].residual, 1.0, -1.0);

        // restrict to coarse grid
        cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

        // compute coarse grid solution
        cycle(levels, levels[i + 1].residual, levels[i + 1].x, i + 1);
        cycle(levels, levels[i + 1].residual, levels[i + 1].x, i + 1);

        // apply coarse grid correction
        cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
        cusp::blas::axpy(levels[i].residual, levels[i].x, 1.0);

        // postsmooth
        postsmooth(levels[i].A, levels[i].b, levels[i].x);
    }
}

template <typename SmootherPolicy, typename SolverPolicy>
template<typename Levels, typename Array1, typename Array2>
void f_cycle_policy<SmootherPolicy, SolverPolicy>
::cycle(Levels& levels, const Array1& b, Array2& x, const size_t i)
{
    CUSP_PROFILE_SCOPED();

    if (i + 1 == levels.size())
    {
        // coarse grid solve
        coarse_solve(levels[i].b, levels[i].x);
    }
    else
    {
        // presmooth
        presmooth(levels[i].A, b, x);

        // compute residual <- b - A*x
        cusp::multiply(levels[i].A, levels[i].x, levels[i].residual);
        cusp::blas::axpby(levels[i].b, levels[i].residual, levels[i].residual, 1.0, -1.0);

        // restrict to coarse grid
        cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

        // compute coarse grid solution
        cycle  (levels, levels[i + 1].residual, levels[i + 1].x, i + 1);
        v.cycle(levels, levels[i + 1].residual, levels[i + 1].x, i + 1);

        // apply coarse grid correction
        cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
        cusp::blas::axpy(levels[i].residual, levels[i].x, 1.0);

        // postsmooth
        postsmooth(levels[i].A, b, x);
    }
}

} // end namespace precond
} // end namespace cusp
