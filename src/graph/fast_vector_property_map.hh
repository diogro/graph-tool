// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

// Copyright (C) Vladimir Prus 2003.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/graph/vector_property_map.html for
// documentation.
//

//
// This is a modification of boost's vector property map which optionally
// disables bound checking for better performance.
//

#ifndef FAST_VECTOR_PROPERTY_MAP_HH
#define FAST_VECTOR_PROPERTY_MAP_HH

#include <boost/property_map/property_map.hpp>
#include <memory>
#include <vector>

#include "value_convert.hh"

namespace boost {

template <typename T, typename IndexMap>
class unchecked_vector_property_map;

template <typename T, typename IndexMap = identity_property_map>
class checked_vector_property_map
{
public:
    typedef typename property_traits<IndexMap>::key_type key_type;
    typedef T value_type;
    typedef typename std::iterator_traits<
        typename std::vector<T>::iterator >::reference reference;
    typedef boost::lvalue_property_map_tag category;

    template <typename Type, typename Index>
    friend class unchecked_vector_property_map;

    typedef unchecked_vector_property_map<T, IndexMap> unchecked_t;
    typedef IndexMap index_map_t;
    typedef checked_vector_property_map<T,IndexMap> self_t;

    checked_vector_property_map(const IndexMap& idx = IndexMap())
        : _store(std::make_shared<std::vector<T>>()), _index(idx) {}

    checked_vector_property_map(const IndexMap& idx, size_t initial_size)
        : _store(std::make_shared<std::vector<T>>(initial_size)), _index(idx) {}

    checked_vector_property_map(size_t initial_size)
        : _store(std::make_shared<std::vector<T>>(initial_size)), _index(IndexMap()) {}

    typename std::vector<T>::iterator storage_begin()
    {
        return _store->begin();
    }

    typename std::vector<T>::iterator storage_end()
    {
        return _store->end();
    }

    typename std::vector<T>::const_iterator storage_begin() const
    {
        return _store->begin();
    }

    typename std::vector<T>::const_iterator storage_end() const
    {
        return _store->end();
    }

    void reserve(size_t size) const
    {
        if (size > _store->size())
            _store->resize(size);
    }

    void resize(size_t size) const
    {
        _store->resize(size);
    }

    void shrink_to_fit() const
    {
        _store->shrink_to_fit();
    }

    std::vector<T>& get_storage() const { return *_store; }

    void swap(checked_vector_property_map& other)
    {
        _store->swap(*other._store);
    }

    unchecked_t& get_unchecked(size_t size = 0)
    {
        reserve(size);
        return reinterpret_cast<unchecked_t&>(*this);
    }

    const unchecked_t& get_unchecked(size_t size = 0) const
    {
        reserve(size);
        return reinterpret_cast<const unchecked_t&>(*this);
    }

    // deep copy
    template <class T2 = T>
    checked_vector_property_map<T2, IndexMap> copy() const
    {
        checked_vector_property_map<T2, IndexMap> pmap(_index);
        if constexpr (std::is_same_v<T, T2>)
            *(pmap._store) = *_store;
        else
            *(pmap._store) = graph_tool::convert<std::vector<T2>>(*_store);
        return pmap;
    }

    [[gnu::always_inline]]
    reference operator[](const key_type& v) const
    {
        auto i = get(_index, v);
        auto& store = *_store;
        if (static_cast<size_t>(i) >= store.size())
            store.resize(i + 1);
        return store[i];
    }

protected:
    template <class T2, class Idx>
    friend class checked_vector_property_map;

    std::shared_ptr<std::vector<T>> _store;
    IndexMap _index;
};

template <typename T, typename IndexMap>
[[gnu::always_inline]] inline
typename checked_vector_property_map<T, IndexMap>::reference
get(checked_vector_property_map<T, IndexMap>& m,
    const typename checked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap>
[[gnu::always_inline]] inline
const typename checked_vector_property_map<T, IndexMap>::reference
get(const checked_vector_property_map<T, IndexMap>& m,
    const typename checked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap, typename Val>
[[gnu::always_inline]] inline
void put(checked_vector_property_map<T, IndexMap>& m,
         const typename checked_vector_property_map<T, IndexMap>::key_type& k, const Val& v)
{
    m[k] = v;
}

template <typename T, typename IndexMap>
checked_vector_property_map<T, IndexMap>
make_checked_vector_property_map(IndexMap index)
{
    return checked_vector_property_map<T, IndexMap>(index);
}

template <typename T, typename IndexMap = identity_property_map>
class unchecked_vector_property_map
    : private checked_vector_property_map<T, IndexMap>
{
public:
    typedef typename property_traits<IndexMap>::key_type  key_type;
    typedef T value_type;
    typedef typename std::iterator_traits<
        typename std::vector<T>::iterator >::reference reference;
    typedef boost::lvalue_property_map_tag category;

    typedef checked_vector_property_map<T, IndexMap> checked_t;

    template <class... Ts>
    unchecked_vector_property_map(Ts&&... as)
        : checked_t(std::forward<Ts>(as)...)
    {}

    typename std::vector<T>::iterator storage_begin()
    {
        return checked_t::_store->begin();
    }

    typename std::vector<T>::iterator storage_end()
    {
        return checked_t::_store->end();
    }

    typename std::vector<T>::const_iterator storage_begin() const
    {
        return checked_t::_store->begin();
    }

    typename std::vector<T>::const_iterator storage_end() const
    {
        return checked_t::_store->end();
    }

    void reserve(size_t size) const { checked_t::reserve(size); }
    void resize(size_t size) const { checked_t::resize(size); }
    void shrink_to_fit() const { checked_t::shrink_to_fit(); }

    [[gnu::always_inline]] [[gnu::flatten]]
    reference operator[](const key_type& v) const
    {
        return (*checked_t::_store)[get(checked_t::_index, v)];
    }

    std::vector<T>& get_storage() const { return checked_t::get_storage(); }

    void swap(unchecked_vector_property_map& other)
    {
        get_storage().swap(other.get_storage());
    }

    checked_t& get_checked()
    {
        return reinterpret_cast<checked_t&>(*this);
    }

    const checked_t& get_checked_t() const
    {
        return reinterpret_cast<const checked_t&>(*this);
    }

    // deep copy
    template <class T2 = T>
    unchecked_vector_property_map<T2, IndexMap> copy() const
    {
        return checked_t::template copy<T2>().get_unchecked();
    }
};

template <typename T, typename IndexMap>
[[gnu::always_inline]] inline
typename unchecked_vector_property_map<T, IndexMap>::reference
get(unchecked_vector_property_map<T, IndexMap>& m,
    const typename unchecked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap>
[[gnu::always_inline]] inline
const typename unchecked_vector_property_map<T, IndexMap>::reference
get(const unchecked_vector_property_map<T, IndexMap>& m,
    const typename unchecked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap, typename Val>
[[gnu::always_inline]] inline
void put(unchecked_vector_property_map<T, IndexMap>& m,
         const typename unchecked_vector_property_map<T, IndexMap>::key_type& k,
         const Val& v)
{
    m[k] = v;
}

template <typename T, typename IndexMap>
unchecked_vector_property_map<T, IndexMap>
make_unchecked_vector_property_map(IndexMap index)
{
    return unchecked_vector_property_map<T, IndexMap>(index);
}


template <class Type, class Index>
unchecked_vector_property_map<Type, Index>&
get_unchecked(checked_vector_property_map<Type, Index>& prop)
{
    return prop.get_unchecked();
}

template <class Prop>
Prop&&
get_unchecked(Prop&& prop)
{
    return std::forward<Prop>(prop);
}

template <class Type, class Index>
checked_vector_property_map<Type, Index>&
get_checked(unchecked_vector_property_map<Type, Index>& prop)
{
    return prop.get_checked();
}

template <class Prop>
Prop&&
get_checked(Prop&& prop)
{
    return std::forward<Prop>(prop);
}

}

#endif
