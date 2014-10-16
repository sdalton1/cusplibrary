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

/*! \file array1d.h
 *  \brief One-dimensional array
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/exception.h>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/detail/vector_base.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
// forward definitions
template <typename RandomAccessIterator> class array1d_view;

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/*! \p array1d : One-dimensional array container
 *
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 *
 * \TODO example
 */
template <typename T, typename MemorySpace>
class array1d : public thrust::detail::vector_base<T, typename cusp::default_memory_allocator<T, MemorySpace>::type>
{
private:
    typedef typename cusp::default_memory_allocator<T, MemorySpace>::type Alloc;
    typedef typename thrust::detail::vector_base<T,Alloc> Parent;

public:
    typedef MemorySpace memory_space;
    typedef cusp::array1d_format format;

    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::array1d<T, MemorySpace2> type;
    };

    /*! equivalent container type
     */
    typedef typename cusp::array1d<T,MemorySpace> container;

    /*! equivalent view type
     */
    typedef typename cusp::array1d_view<typename Parent::iterator> view;

    /*! equivalent const_view type
     */
    typedef typename cusp::array1d_view<typename Parent::const_iterator> const_view;

    typedef typename Parent::size_type  size_type;
    typedef typename Parent::value_type value_type;

    array1d(void) : Parent() {}

    explicit array1d(size_type n)
        : Parent()
    {
        if(n > 0)
        {
            Parent::m_storage.allocate(n);
            Parent::m_size = n;
        }
    }

    array1d(size_type n, const value_type &value)
        : Parent(n, value) {}

    template<typename Array>
    array1d(const Array& a, typename thrust::detail::enable_if<!thrust::detail::is_convertible<Array,size_type>::value>::type * = 0)
        : Parent(a.begin(), a.end()) {}

    template<typename InputIterator>
    array1d(InputIterator first, InputIterator last)
        : Parent(first, last) {}

    template<typename Array>
    array1d &operator=(const Array& a)
    {
        Parent::assign(a.begin(), a.end());
        return *this;
    }

    view subarray(size_type start_index, size_type num_entries)
    {
        return view(Parent::begin() + start_index, Parent::begin() + start_index + num_entries);
    }

    T* raw_data(void)
    {
        return thrust::raw_pointer_cast(&Parent::m_storage[0]);
    }

    const T* raw_data(void) const
    {
        return thrust::raw_pointer_cast(&Parent::m_storage[0]);
    }

    // TODO specialize resize()
}; // class array1d
/*! \}
 */

/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/*! \p array1d_view : One-dimensional array view
 *
 * \tparam RandomAccessIterator Underlying iterator type
 *
 * \TODO example
 */
template<typename Iterator>
class array1d_view : public thrust::iterator_adaptor<array1d_view<Iterator>, Iterator>
{
  public :

    typedef cusp::array1d_format format;
    typedef Iterator iterator;

    typedef thrust::iterator_adaptor<array1d_view<iterator>, iterator>  super_t;
    typedef typename cusp::array1d_view<iterator>                       view;

    typedef typename super_t::value_type                                value_type;
    typedef typename super_t::pointer                                   pointer;
    typedef typename super_t::reference                                 reference;
    typedef typename super_t::difference_type                           size_type;
    typedef typename super_t::difference_type                           difference_type;
    typedef typename thrust::iterator_system<iterator>::type            memory_space;
    typedef const    value_type*                                        const_pointer;
    typedef const_pointer                                               const_iterator;

    array1d_view(void)
        : m_size(0), m_capacity(0) {}

    template <typename Array>
    array1d_view(Array& a)
        : super_t(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}

    template <typename InputIterator>
    array1d_view(InputIterator begin, InputIterator end)
        : super_t(begin), m_size(end-begin), m_capacity(end-begin) {}

    friend class thrust::iterator_core_access;

    reference front(void) const
    {
        return *begin();
    }

    reference back(void) const
    {
        return *(begin() + (size() - 1));
    }

    reference operator[](difference_type n) const
    {
        return *(begin() + n);
    }

    iterator begin(void) const
    {
        return this->base();
    }

    iterator end(void) const
    {
        return begin() + m_size;
    }

    size_type size(void) const
    {
        return m_size;
    }

    size_type capacity(void) const
    {
        return m_capacity;
    }

    pointer data(void)
    {
        return &front();
    }

    value_type* raw_data(void)
    {
        return thrust::raw_pointer_cast(&front());
    }

    const value_type* raw_data(void) const
    {
        return thrust::raw_pointer_cast(&front());
    }

    void resize(size_type new_size)
    {
        if (new_size <= m_capacity)
            m_size = new_size;
        else
            // XXX is not_implemented_exception the right choice?
            throw cusp::not_implemented_exception("array1d_view cannot resize() larger than capacity()");
    }

    view subarray(size_type start_index, size_type num_entries)
    {
        return view(begin() + start_index, begin() + start_index + num_entries);
    }

protected:
    size_type m_size;
    size_type m_capacity;

  private :
};

/*! \p counting_array : One-dimensional counting array view
 *
 * \tparam ValueType iterator type
 *
 * \TODO example
 */
template <typename ValueType>
class counting_array : public cusp::array1d_view< thrust::counting_iterator<ValueType> >
{
    typedef thrust::counting_iterator<ValueType> iterator;
    typedef cusp::array1d_view<iterator> Parent;

public:

    counting_array(ValueType size) : Parent(iterator(0), iterator(size)) {}
    counting_array(ValueType start, ValueType finish) : Parent(iterator(start), iterator(finish)) {}
};

/*! \p constant_array : One-dimensional constant array view
 *
 * \tparam ValueType iterator type
 *
 * \TODO example
 */
template <typename ValueType>
class constant_array : public cusp::array1d_view< thrust::constant_iterator<ValueType> >
{
    typedef thrust::constant_iterator<ValueType> iterator;
    typedef cusp::array1d_view<iterator> Parent;

public:

    constant_array(ValueType value, size_t size) : Parent(iterator(value), iterator(value) + size) {}
};

/* Convenience functions */

template <typename Iterator>
array1d_view<Iterator> make_array1d_view(Iterator first, Iterator last)
{
    return array1d_view<Iterator>(first, last);
}

template <typename Iterator>
array1d_view<Iterator> make_array1d_view(const array1d_view<Iterator>& a)
{
    return make_array1d_view(a.begin(), a.end());
}

template <typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::view make_array1d_view(array1d<T,MemorySpace>& a)
{
    return make_array1d_view(a.begin(), a.end());
}

template <typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::const_view make_array1d_view(const array1d<T,MemorySpace>& a)
{
    return make_array1d_view(a.begin(), a.end());
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/array1d.inl>

