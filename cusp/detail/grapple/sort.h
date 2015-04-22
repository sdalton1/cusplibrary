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

#include <cusp/sort.h>

namespace grapple
{

template <typename ArrayType>
void counting_sort(grapple_system &exec,
                   ArrayType& keys,
                   typename ArrayType::value_type min, typename ArrayType::value_type max)
{
    exec.start(CUSP_COUNTING_SORT);
    cusp::counting_sort(exec.policy(), keys, min, max);
    exec.stop();
}

template <typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(grapple_system &exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
    exec.start(CUSP_COUNTING_SORT_BY_KEY);
    cusp::counting_sort_by_key(exec.policy(), keys, vals, min, max);
    exec.stop();
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(grapple_system &exec,
                 ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                 const int min_row = -1, const int max_row = -1)
{
    exec.start(CUSP_SORT_BY_ROW);
    cusp::sort_by_row(exec.policy(), row_indices, column_indices, values, min_row, max_row);
    exec.stop();
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(grapple_system &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            const int min_row = -1, const int max_row = -1,
                            const int min_col = -1, const int max_col = -1)
{
    exec.start(CUSP_SORT_BY_ROW_AND_COLUMN);
    cusp::sort_by_row_and_column(exec.policy(), row_indices, column_indices, values, min_row, max_row, min_col, max_col);
    exec.stop();
}

} // end namespace cusp

