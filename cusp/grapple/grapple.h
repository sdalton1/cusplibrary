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
    CUSP_OFFSETS_TO_INDICES,
    CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW,
    CUSP_COMPUTE_MAX_ENTRIES_PER_ROW,
    CUSP_COPY,
};

// grapple intercept file

// insert example function markers and names
// into global grapple map
struct cusp_grapple_init
{
    cusp_grapple_init(void)
    {
        grapple_thrust_map.insert(CUSP_OFFSETS_TO_INDICES, "offsets_to_indices");
        grapple_thrust_map.insert(CUSP_COMPUTE_OPTIMAL_ENTRIES_PER_ROW, "compute_optimal_entries_per_row");
        grapple_thrust_map.insert(CUSP_COMPUTE_MAX_ENTRIES_PER_ROW, "compute_max_entries_per_row");
        grapple_thrust_map.insert(CUSP_COPY, "copy");
    }
};
static cusp_grapple_init initialize_cusp_grapple;

#include "format_utils.h"
