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

#include <cusp/copy.h>

namespace cusp
{

template <typename SourceType, typename DestinationType>
void copy(grapple_system &exec,
          const SourceType& src, DestinationType& dst)
{
    exec.start(CUSP_COPY);
    cusp::copy(exec.policy(), src, dst);
    exec.stop();
}

}

