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

#include <cusp/multiply.h>

namespace grapple
{

template <typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(grapple_system &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C)
{
    exec.start(CUSP_MULTIPLY);
    cusp::multiply(exec.policy(), A, B, C);
    exec.stop();
}

template <typename LinearOperator,
         typename Vector1,
         typename Vector2,
         typename Vector3,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spmv(grapple_system &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                            Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    exec.start(CUSP_GENERALIZED_SPMV);
    cusp::generalized_spmv(exec.policy(), A, x, y, z, combine, reduce);
    exec.stop();
}

} // end namespace cusp

