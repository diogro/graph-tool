// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2026 Tiago de Paula Peixoto <tiago@skewed.de>
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

    template <class... As>
    checked_vector_property_map(const IndexMap& idx = IndexMap(), As&&... as)
        : _store(std::make_shared<std::vector<T>>(std::forward<As>(as)...)),
          _index(idx) {}

    template <class... As>
    checked_vector_property_map(size_t size, As&&... as)
        : _store(std::make_shared<std::vector<T>>(size, std::forward<As>(as)...)),
          _index(IndexMap()) {}

    checked_vector_property_map(const checked_vector_property_map&) = default;
    checked_vector_property_map& operator=(const checked_vector_property_map&) = default;

    template <class... As>
    void reserve(size_t size, As&&... as) const
    {
        if (size > _store->size())
            _store->resize(size, std::forward<As>(as)...);
    }

    template <class... As>
    void soft_reserve(size_t size, As&&... as) const
    {
        if (size > _store->size())
            resize(std::max(size, 2 * _store->size()), std::forward<As>(as)...);
    }

    template <class... As>
    void resize(size_t size, As&&... as) const
    {
        _store->resize(size, std::forward<As>(as)...);
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

    reference operator[](const key_type& v) const
    {
        auto i = get(_index, v);
        soft_reserve(i + 1);
        return (*_store)[i];
    }

protected:
    template <class T2, class Idx>
    friend class checked_vector_property_map;

    std::shared_ptr<std::vector<T>> _store;
    IndexMap _index;
};

template <typename T, typename IndexMap>
typename checked_vector_property_map<T, IndexMap>::reference
get(checked_vector_property_map<T, IndexMap>& m,
    const typename checked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap>
const typename checked_vector_property_map<T, IndexMap>::reference
get(const checked_vector_property_map<T, IndexMap>& m,
    const typename checked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap, typename Val>
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

    template <class... As>
    unchecked_vector_property_map(const IndexMap& idx = IndexMap(), As&&... as)
        : checked_t(idx, std::forward<As>(as)...) {}

    template <class... As>
    unchecked_vector_property_map(size_t size, As&&... as)
        : checked_t(size, std::forward<As>(as)...) {}

    unchecked_vector_property_map(const unchecked_vector_property_map&) = default;
    unchecked_vector_property_map& operator=(const unchecked_vector_property_map&) = default;

    unchecked_vector_property_map& operator=(const checked_vector_property_map<T, IndexMap>& o)
    {
        *this = o.get_unchecked();
        return *this;
    }

    template <class... As>
    void reserve(As&&... as) const { checked_t::reserve(std::forward<As>(as)...); }

    template <class... As>
    void soft_reserve(As&&... as) const { checked_t::soft_reserve(std::forward<As>(as)...); }

    template <class... As>
    void resize(As&&... as) const { checked_t::resize(std::forward<As>(as)...); }
    template <class... As>

    void shrink_to_fit() const { checked_t::shrink_to_fit(); }

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

    const checked_t& get_checked_c() const
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
typename unchecked_vector_property_map<T, IndexMap>::reference
get(unchecked_vector_property_map<T, IndexMap>& m,
    const typename unchecked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap>
const typename unchecked_vector_property_map<T, IndexMap>::reference
get(const unchecked_vector_property_map<T, IndexMap>& m,
    const typename unchecked_vector_property_map<T, IndexMap>::key_type& k)
{
    return m[k];
}

template <typename T, typename IndexMap, typename Val>
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

template<typename T>
concept CheckedPMap =
    requires(T a)
    {
        a.get_unchecked();
    };

template<typename T>
concept UncheckedPMap =
    requires(T a)
    {
        a.get_checked();
    };

template <class T>
struct uncheck
{
    typedef T type;

    decltype(auto) operator()(auto&& a)
    {
        return std::forward<T>(a);
    }
};

template <CheckedPMap T>
struct uncheck<T>
{
    typedef typename T::unchecked_t type;

    decltype(auto) operator()(auto&& a)
    {
        return a.get_unchecked();
    }
};

template <class T>
struct check
{
    typedef T type;

    decltype(auto) operator()(auto&& a)
    {
        return std::forward<T>(a);
    }
};

template <UncheckedPMap T>
struct check<T>
{
    typedef typename T::checked_t type;

    decltype(auto) operator()(auto&& a)
    {
        return a.get_checked();
    }
};

template <class PMap>
decltype(auto) get_unchecked(PMap&& prop)
{
    return uncheck<std::decay_t<PMap>>()(std::forward<PMap>(prop));
}

template <class PMap>
decltype(auto) get_checked(PMap&& prop)
{
    return check<std::decay_t<PMap>>()(std::forward<PMap>(prop));
}

}

#endif
