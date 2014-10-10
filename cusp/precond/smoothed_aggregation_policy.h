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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/transpose.h>

#include <cusp/detail/spectral_radius.h>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/aggregation/aggregate.h>
#include <cusp/precond/aggregation/smooth.h>
#include <cusp/precond/aggregation/strength.h>
#include <cusp/precond/aggregation/tentative.h>

#include <vector> // TODO replace with host_vector

namespace cusp
{
namespace precond
{
namespace aggregation
{
template <typename IndexType, typename ValueType, typename MemorySpace> struct amg_container;

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 */
template <typename IndexType, typename ValueType, typename MemorySpace>
class smoothed_aggregation_policy
{
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type SetupMatrixType;

public:

    struct sa_level
    {
        cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
        cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates

        ValueType rho_DinvA;

        sa_level() : rho_DinvA(0) {}

        template<typename SaLevelType>
        sa_level(const SaLevelType& sa_level)
            : aggregates(sa_level.aggregates), B(sa_level.B), rho_DinvA(sa_level.rho_DinvA) {}
    };

    ValueType omega;
    SetupMatrixType A_; // matrix
    std::vector<sa_level> sa_levels;

    template<typename MatrixType2>
    void extend_hierarchy(MatrixType2& lvl_R, MatrixType2& lvl_A, MatrixType2& lvl_P)
    {
        CUSP_PROFILE_SCOPED();

        cusp::array1d<IndexType,MemorySpace> aggregates;
        {
            // compute stength of connection matrix
            ValueType theta = 0;
            SetupMatrixType C;
            cusp::precond::aggregation::symmetric_strength_of_connection(A_, C, theta);

            // compute aggregates
            aggregates.resize(C.num_rows);
            cusp::blas::fill(aggregates,IndexType(0));
            cusp::precond::aggregation::standard_aggregation(C, aggregates);
        }

        SetupMatrixType P;
        cusp::array1d<ValueType,MemorySpace>  B_coarse;
        {
            // compute tenative prolongator and coarse nullspace vector
            SetupMatrixType 				T;
            cusp::precond::aggregation::fit_candidates(aggregates, sa_levels.back().B, T, B_coarse);

            // compute prolongation operator
            ValueType rho_DinvA = 0;
            cusp::precond::aggregation::smooth_prolongator(A_, T, P, omega, rho_DinvA);
        }

        // compute restriction operator (transpose of prolongator)
        SetupMatrixType R;
        cusp::transpose(P,R);

        // construct Galerkin product R*A*P
        SetupMatrixType RAP;
        SetupMatrixType AP;
        cusp::multiply(A_, P, AP);
        cusp::multiply(R, AP, RAP);

        lvl_R = R;
        lvl_P = P;
        lvl_A = RAP;

        A_.swap(RAP);

        sa_levels.back().aggregates.swap(aggregates);
        sa_levels.push_back(sa_level());
        sa_levels.back().B.swap(B_coarse);
    }
};
/*! \}
 */

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::host_memory>
{
    // use CSR on host
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> setup_type;
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> solve_type;
};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::device_memory>
{
    // use COO on device
    typedef typename cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> setup_type;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> solve_type;
};

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp
