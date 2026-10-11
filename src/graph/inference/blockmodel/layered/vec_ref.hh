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

#ifndef VEC_REF_HH
#define VEC_REF_HH

#include <type_traits>
#include <vector>

namespace graph_tool
{

template <class Vec>
class VecRef
{
public:
    VecRef() = delete;
    VecRef(Vec& v) : _v(&v) {}

    using value_type = Vec::value_type;
    using iterator = Vec::iterator;
    using const_iterator = Vec::const_iterator;

    auto&& operator[](auto&& k)
    {
        return (*_v)[k];
    }

    auto&& operator[](auto&& k) const
    {
        return (*_v)[k];
    }

    auto begin()
    {
        return _v->begin();
    }

    auto begin() const
    {
        return _v->begin();
    }

    auto cbegin() const
    {
        return _v->cbegin();
    }

    auto end()
    {
        return _v->end();
    }

    auto end() const
    {
        return _v->end();
    }

    auto cend() const
    {
        return _v->cend();
    }

    template <class... Ts>
    auto emplace(Ts&&... as)
    {
        return _v->emplace(std::forward<Ts>(as)...);
    }

    template <class... Ts>
    auto insert(Ts&&... as)
    {
        return _v->insert(std::forward<Ts>(as)...);
    }

    template <class... Ts>
    auto erase(Ts&&... as)
    {
        return _v->erase(std::forward<Ts>(as)...);
    }

    void clear()
    {
        _v->clear();
    }

    auto size() const
    {
        return _v->size();
    }

    auto empty() const
    {
        return _v->empty();
    }

private:
    Vec* _v = nullptr;
};

template <class Vec>
auto vec_ref(Vec& v)
{
    return VecRef<Vec>(v);
}



}// namespace graph_tool

#endif // VEC_REF_HH
