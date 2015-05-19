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

#include <cusp/format_utils.h>

namespace grapple
{

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(grapple_system &exec,
                        const OffsetArray& offsets,
                        IndexArray& indices)
{
    using cusp::system::detail::generic::offsets_to_indices;

    exec.start(CUSP_OFFSETS_TO_INDICES);
    offsets_to_indices(exec.policy(), offsets, indices);
    exec.stop();
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(grapple_system &exec,
                        const IndexArray& indices,
                        OffsetArray& offsets)
{
    using cusp::system::detail::generic::indices_to_offsets;

    exec.start(CUSP_INDICES_TO_OFFSETS);
    indices_to_offsets(exec.policy(), indices, offsets);
    exec.stop();
}

template <typename ArrayType>
size_t compute_max_entries_per_row(grapple_system &exec,
                                   const ArrayType& row_offsets)
{
    using cusp::system::detail::generic::compute_max_entries_per_row;

    exec.start(CUSP_COMPUTE_MAX_ENTRIES_PER_ROW);
    size_t ret = compute_max_entries_per_row(exec.policy(), row_offsets);
    exec.stop();

    return ret;
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(grapple_system &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
    using cusp::system::detail::generic::compute_optimal_entries_per_row;

    exec.start(CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW);
    size_t ret = compute_optimal_entries_per_row(exec.policy(), row_offsets, relative_speed, breakeven_threshold);
    exec.stop();

    return ret;
}

template <typename MatrixType, typename ArrayType>
void extract_diagonal(grapple_system &exec,
                      const MatrixType& A,
                      ArrayType& output)
{
    using cusp::system::detail::generic::extract_diagonal;

    output.resize(thrust::min(A.num_rows, A.num_cols));

    exec.start(CUSP_EXTRACT_DIAGONAL);
    extract_diagonal(exec.policy(), A, output, typename MatrixType::format());
    exec.stop();
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(grapple_system &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    using cusp::system::detail::generic::count_diagonals;

    exec.start(CUSP_COUNT_DIAGONALS);
    size_t ret = count_diagonals(exec.policy(), num_rows, num_cols, row_indices, column_indices);
    exec.stop();

    return ret;
}

} //end cusp namespace
