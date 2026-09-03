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

#ifndef GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_TYPES_HH
#define GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_TYPES_HH

#include "../layered/types.hh"
#include "../weighted/types.hh"

namespace graph_tool
{

template <LayerCounts X>
auto get_count(const X& x)
{
    LayerCount<typename X::cval_t::wval_t, false, x._layered> w;
    for (auto&& [l, c] : x._h)
    {
        auto y = c.visit_count([](auto& z) { return z; });
        if (y == 0)
            continue;
        w._h.emplace(l, y);
    }
    return w;
}

template <LayerCounts X>
auto get_weight(const X& x, size_t i)
{
    LayerCount<typename X::cval_t::wval_t, false, x._layered> w;
    for (auto&& [l, c] : x._h)
    {
        auto y = c.visit_weight(i, [](auto& z) { return z; });
        if (y == 0)
            continue;
        w._h.emplace(l, y);
    }
    return w;
}

template <LayerCounts X>
auto get_dev(const X& x, size_t i)
{
    LayerCount<typename X::cval_t::wval_t, false, x._layered> w;
    for (auto&& [l, c] : x._h)
    {
        auto y = c.visit_dev(i, [](auto& z) { return z; });
        if (y == 0)
            continue;
        w._h.emplace(l, y);
    }
    return w;
}

template <LayerCounts X, class T>
requires (std::is_scalar_v<T>)
void set_weight(X&& x, size_t i, T y)
{
    for (auto&& [l, c] : x._h)
        c.visit_weight(i, [&](auto& z) { z = y; });
    if (y == 0)
        x.clean();
}

template <LayerCounts X, LayerCounts Y>
void set_weight(X&& x, size_t i, const Y& y)
{
    for (auto&& [l, c] : y._h)
    {
        assert(c != 0);
        if (c == 0)
            continue;
        typename std::decay_t<X>::cval_t w;
        set_weight(w, i, c);
        lc_insert_or_visit(x._h, std::make_pair(l, w),
                           [&](auto&& z) { set_weight(z.second, i, c); },
                           dummy_mutex());
    }

    for (auto&& [l, c] : x._h)
    {
        if (y.get(l) == 0)
            set_weight(c, i, 0);
    }

    x.clean();
}

template <LayerCounts X, class Y>
requires std::is_scalar_v<Y>
void set_dev(X&& x, size_t i, Y y)
{
    for (auto&& [l, c] : x._h)
        c.visit_dev(i, [&](auto& z) { z = y; });
    if (y == 0)
        x.clean();
}

template <LayerCounts X, LayerCounts Y>
void set_dev(X&& x, size_t i, const Y& y)
{
    for (auto&& [l, c] : y._h)
    {
        assert(c != 0);
        if (c == 0)
            continue;
        typename std::decay_t<X>::cval_t w;
        set_dev(w, i, c);
        lc_insert_or_visit(x._h, std::make_pair(l, w),
                           [&](auto&& z) { set_dev(z.second, i, c); },
                           dummy_mutex());
    }

    for (auto&& [l, c] : x._h)
    {
        if (y.get(l) == 0)
            set_dev(c, i, 0);
    }

    x.clean();
}


} // graph_tool namespace

#endif // GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_TYPES_HH
