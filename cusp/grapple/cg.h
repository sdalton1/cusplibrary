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

#include <cusp/krylov/cg.h>

namespace cusp
{
namespace krylov
{

template <class LinearOperator, class Vector>
void cg(grapple_system &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b)
{
    exec.start(CUSP_KRYLOV_CG);
    cusp::krylov::cg(exec.policy(), A, x, b);
    exec.stop();
}

template <class LinearOperator, class Vector, class Monitor>
void cg(grapple_system &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    exec.start(CUSP_KRYLOV_CG);
    cusp::krylov::cg(exec.policy(), A, x, b, monitor);
    exec.stop();
}

template <class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(grapple_system &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
    exec.start(CUSP_KRYLOV_CG);
    cusp::krylov::cg(exec.policy(), A, x, b, monitor, M);
    exec.stop();
}

}
}
