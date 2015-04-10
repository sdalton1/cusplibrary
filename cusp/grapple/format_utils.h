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
    cusp::offsets_to_indices(exec.policy(), offsets, indices);
    exec.stop();
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(grapple_system &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
    exec.start(CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW);
    size_t ret = cusp::compute_optimal_entries_per_row(exec.policy(), row_offsets, relative_speed, breakeven_threshold);
    exec.stop();

    return ret;
}

template <typename ArrayType>
size_t compute_max_entries_per_row(grapple_system &exec,
                                   const ArrayType& row_offsets)
{
    exec.start(CUSP_COMPUTE_MAX_ENTRIES_PER_ROW);
    size_t ret = cusp::compute_max_entries_per_row(exec.policy(), row_offsets);
    exec.stop();

    return ret;
}

} //end cusp namespace
