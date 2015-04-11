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

/*! \file temporary_array.h
 *  \brief Container-like class temporary storage inside algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>

namespace cusp
{
namespace detail
{

template<typename T, typename Space, typename System>
  class temporary_array
    : public thrust::detail::temporary_array<T,System>
{
  private:

    typedef thrust::detail::temporary_array<T,System> super_t;

  public:

    typedef typename super_t::size_type size_type;

    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system) : super_t(system) {};

    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system, size_type n) : super_t(system, n) {};

    // provide a kill-switch to explicitly avoid initialization
    __host__ __device__
    temporary_array(int uninit, thrust::execution_policy<System> &system, size_type n) : super_t(uninit, system, n) {};

    template<typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    size_type n) : super_t(system, first, n) {}

    template<typename InputIterator, typename InputSystem>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    size_type n) : super_t(system, input_system, first, n) {}

    template<typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    InputIterator last) : super_t(system, first, last) {}

    template<typename InputSystem, typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    InputIterator last) : super_t(system, input_system, first, last) {}

}; // end temporary_array


} // end detail
} // end cusp

