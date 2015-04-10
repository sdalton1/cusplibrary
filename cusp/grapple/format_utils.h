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

namespace cusp
{

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(grapple_system &exec,
                        const OffsetArray& offsets,
                        IndexArray& indices)
{
    exec.start(CUSP_OFFSETS_TO_INDICES);
    cusp::offsets_to_indices(exec.policy(get_system(offsets.begin(),indices.begin())), offsets, indices);
    exec.stop();
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(grapple_system &exec,
                        const IndexArray& indices,
                        OffsetArray& offsets)
{
    exec.start(CUSP_INDICES_TO_OFFSETS);
    cusp::indices_to_offsets(exec.policy(get_system(indices.begin(),offsets.begin())), indices, offsets);
    exec.stop();
}

template <typename ArrayType>
size_t compute_max_entries_per_row(grapple_system &exec,
                                   const ArrayType& row_offsets)
{
    exec.start(CUSP_COMPUTE_MAX_ENTRIES_PER_ROW);
    size_t ret = cusp::compute_max_entries_per_row(exec.policy(get_system(row_offsets.begin())), row_offsets);
    exec.stop();

    return ret;
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(grapple_system &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
    exec.start(CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW);
    size_t ret = cusp::compute_optimal_entries_per_row(exec.policy(get_system(row_offsets.begin())), row_offsets, relative_speed, breakeven_threshold);
    exec.stop();

    return ret;
}

template <typename MatrixType, typename ArrayType>
void extract_diagonal(grapple_system &exec,
                      const MatrixType& A,
                      ArrayType& output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    exec.start(CUSP_EXTRACT_DIAGONAL);
    cusp::extract_diagonal(exec.policy(select_system(system1,system2)), A, output);
    exec.stop();
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(grapple_system &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    exec.start(CUSP_COUNT_DIAGONALS);
    size_t ret = cusp::count_diagonals(exec.policy(get_system(row_indices.begin(), column_indices.begin())),
                                       num_rows, num_cols, row_indices, column_indices);
    exec.stop();

    return ret;
}

} //end cusp namespace
