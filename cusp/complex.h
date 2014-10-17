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

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <cusp/detail/config.h>

#if THRUST_VERSION >= 100800
#include <thrust/complex.h>
#else
#include <cusp/detail/thrust/complex.h>
#endif

namespace cusp
{

template <typename T>
struct norm_type {
    typedef T type;
};

template <typename T>
struct norm_type< thrust::complex<T> > {
    typedef T type;
};

} // end namespace cusp
