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

#include <cusp/detail/config.h>

// we need to carefully undefine and then redefined these macros to ensure that multiple
// versions of moderngpu can coexist in the same program
// push_macro & pop_macro were introduced to gcc in version 4.3

// if the macros are already defined, save them and undefine them

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef MGPU_NS_PREFIX
#    pragma push_macro("MGPU_NS_PREFIX")
#    undef MGPU_NS_PREFIX
#    define MGPU_NS_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef MGPU_NS_POSTFIX
#    pragma push_macro("MGPU_NS_POSTFIX")
#    undef MGPU_NS_POSTFIX
#    define MGPU_NS_POSTFIX_NEEDS_RESTORE
#  endif
#  ifdef MGPU_CDP
#    pragma push_macro("MGPU_CDP")
#    undef MGPU_CDP
#    define MGPU_CDP_NEEDS_RESTORE
#  endif
#  ifdef cub
#    pragma push_macro("mgpu")
#    undef mgpu
#    define MGPU_NEEDS_RESTORE
#  endif
#endif // __GNUC__

// define the macros while we #include our version of cub
#define MGPU_NS_PREFIX namespace cusp { namespace system { namespace cuda { namespace detail {
#define MGPU_NS_POSTFIX               }                  }                }                  }

#if __BULK_HAS_CUDART__
#define MGPU_CDP 1
#endif

// rename "cub" so it doesn't collide with another installation elsewhere
#define mgpu mgpu_

#include <cusp/system/cuda/detail/mgpu/util_namespace.cuh>
#include <cusp/system/cuda/detail/mgpu/mgpu.cuh>

// undef the top-level namespace name
#undef mgpu

// undef the macros
#undef MGPU_NS_PREFIX
#undef MGPU_NS_POSTFIX

#ifdef MGPU_CDP
#  undef MGPU_CDP
#endif

// redefine the macros if they were defined previously

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef MGPU_NS_PREFIX_NEEDS_RESTORE
#    pragma pop_macro("MGPU_NS_PREFIX")
#    undef MGPU_NS_PREFIX_NEEDS_RESTORE
#  endif
#  ifdef MGPU_NS_POSTFIX_NEEDS_RESTORE
#    pragma pop_macro("MGPU_NS_POSTFIX")
#    undef MGPU_NS_POSTFIX_NEEDS_RESTORE
#  endif
#  ifdef MGPU_CDP_NEEDS_RESTORE
#    pragma pop_macro("MGPU_CDP")
#    undef MGPU_CDP_NEEDS_RESTORE
#  endif
#  ifdef MGPU_NEEDS_RESTORE
#    pragma pop_macro("mgpu")
#    undef MGPU_NEEDS_RESTORE
#  endif
#endif // __GNUC__

