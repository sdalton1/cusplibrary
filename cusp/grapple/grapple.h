/*
*  Copyright 2008-2013 NVIDIA Corporation
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

#include <grapple/grapple.h>

// define complex function markers
enum
{
    CUSP_BLAS_AXPY,
    CUSP_BLAS_AXPBY,
    CUSP_BLAS_AXPBYPCZ,
    CUSP_BLAS_XMY,
    CUSP_BLAS_COPY,
    CUSP_BLAS_DOT,
    CUSP_BLAS_DOTC,
    CUSP_BLAS_FILL,
    CUSP_BLAS_NRM1,
    CUSP_BLAS_NRM2,
    CUSP_BLAS_NRMMAX,
    CUSP_BLAS_SCAL,
    CUSP_BLAS_GEMV,
    CUSP_BLAS_GEMM,

    CUSP_CONVERT,

    CUSP_COPY,

    CUSP_ELEMENTWISE,
    CUSP_ADD,
    CUSP_SUBTRACT,

    CUSP_OFFSETS_TO_INDICES,
    CUSP_INDICES_TO_OFFSETS,
    CUSP_COMPUTE_MAX_ENTRIES_PER_ROW,
    CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW,
    CUSP_EXTRACT_DIAGONAL,
    CUSP_COUNT_DIAGONALS,

    CUSP_MULTIPLY,
    CUSP_GENERALIZED_SPMV,

    CUSP_COUNTING_SORT,
    CUSP_COUNTING_SORT_BY_KEY,
    CUSP_SORT_BY_ROW,
    CUSP_SORT_BY_ROW_AND_COLUMN,

    CUSP_TRANSPOSE,
};

// grapple intercept file

// insert example function markers and names
// into global grapple map
struct cusp_grapple_init
{
    cusp_grapple_init(void)
    {
        grapple_thrust_map.insert(CUSP_BLAS_AXPY, "blas::axpy");
        grapple_thrust_map.insert(CUSP_BLAS_AXPBY, "blas::axpby");
        grapple_thrust_map.insert(CUSP_BLAS_AXPBYPCZ, "blas::axpbypcz");
        grapple_thrust_map.insert(CUSP_BLAS_XMY, "blas::xmy");
        grapple_thrust_map.insert(CUSP_BLAS_COPY, "blas::copy");
        grapple_thrust_map.insert(CUSP_BLAS_DOT, "blas::dot");
        grapple_thrust_map.insert(CUSP_BLAS_DOTC, "blas::dotc");
        grapple_thrust_map.insert(CUSP_BLAS_FILL, "blas::fill");
        grapple_thrust_map.insert(CUSP_BLAS_NRM1, "blas::nrm1");
        grapple_thrust_map.insert(CUSP_BLAS_NRM2, "blas::nrm2");
        grapple_thrust_map.insert(CUSP_BLAS_NRMMAX, "blas::nrmmax");
        grapple_thrust_map.insert(CUSP_BLAS_SCAL, "blas::scal");
        grapple_thrust_map.insert(CUSP_BLAS_GEMV, "blas::gemv");
        grapple_thrust_map.insert(CUSP_BLAS_GEMM, "blas::gemm");

        grapple_thrust_map.insert(CUSP_CONVERT, "convert");

        grapple_thrust_map.insert(CUSP_COPY, "copy");

        grapple_thrust_map.insert(CUSP_ELEMENTWISE, "elementwise");
        grapple_thrust_map.insert(CUSP_ADD, "add");
        grapple_thrust_map.insert(CUSP_SUBTRACT, "subtract");

        grapple_thrust_map.insert(CUSP_OFFSETS_TO_INDICES, "offsets_to_indices");
        grapple_thrust_map.insert(CUSP_INDICES_TO_OFFSETS, "indices_to_offsets");
        grapple_thrust_map.insert(CUSP_COMPUTE_MAX_ENTRIES_PER_ROW, "compute_max_entries_per_row");
        grapple_thrust_map.insert(CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW, "compute_optimal_entries_per_row");
        grapple_thrust_map.insert(CUSP_EXTRACT_DIAGONAL, "extract_diagonal");
        grapple_thrust_map.insert(CUSP_COUNT_DIAGONALS, "count_diagonals");

        grapple_thrust_map.insert(CUSP_MULTIPLY, "multiply");
        grapple_thrust_map.insert(CUSP_GENERALIZED_SPMV, "generalized_spmv");

        grapple_thrust_map.insert(CUSP_COUNTING_SORT, "counting_sort");
        grapple_thrust_map.insert(CUSP_COUNTING_SORT_BY_KEY, "counting_sort_by_key");
        grapple_thrust_map.insert(CUSP_SORT_BY_ROW, "sort_by_row");
        grapple_thrust_map.insert(CUSP_SORT_BY_ROW_AND_COLUMN, "sort_by_row_and_column");

        grapple_thrust_map.insert(CUSP_TRANSPOSE, "transpose");
    }
};
static cusp_grapple_init initialize_cusp_grapple;

#include "blas.h"
#include "convert.h"
#include "copy.h"
#include "elementwise.h"
#include "format_utils.h"
#include "multiply.h"
#include "sort.h"
#include "temporary_array.h"
#include "transpose.h"

