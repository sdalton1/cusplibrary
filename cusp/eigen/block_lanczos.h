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

/*! \file block lanczos.h
 *  \brief Block Lanczos method
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/eigen/lanczos.h>

namespace cusp
{
namespace eigen
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup eigensolvers EigenSolvers
 *  \ingroup iterative_solvers
 *  \{
 */

template<typename MatrixType,
         typename Array1d,
         typename Array2d>
void block_lanczos(const MatrixType& A,
                   Array1d& eigVals,
                   Array2d& eigVecs,
                   const size_t blocksize,
                   const size_t maxouter,
                   const size_t maxinner);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/block_lanczos.inl>
