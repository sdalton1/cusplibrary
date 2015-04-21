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

#include <cusp/elementwise.h>

namespace cusp
{

template <typename MatrixType1, typename MatrixType2, typename MatrixType3, typename BinaryFunction>
void elementwise(grapple_system& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                 MatrixType3& C,
                 BinaryFunction op)
{
    exec.start(CUSP_ELEMENTWISE);
    cusp::elementwise(exec.policy(), A, B, C, op);
    exec.stop();
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void add(grapple_system& exec,
         const MatrixType1& A,
         const MatrixType2& B,
               MatrixType3& C)
{
    exec.start(CUSP_ADD);
    cusp::add(exec.policy(), A, B, C);
    exec.stop();
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void subtract(grapple_system& exec,
              const MatrixType1& A,
              const MatrixType2& B,
                    MatrixType3& C)
{
    exec.start(CUSP_SUBTRACT);
    cusp::subtract(exec.policy(), A, B, C);
    exec.stop();
}

} // end namespace cusp

