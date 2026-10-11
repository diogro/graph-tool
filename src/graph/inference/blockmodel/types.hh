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

#pragma once

#include "config.h"

#include "graph_util.hh"

namespace graph_tool
{

template <class T1, class T2>
struct thresh_diff_imp
{
    auto operator()(T1 before, T2 after, int t = 0)
    {
        return (quiet_less_equal(before, t) && quiet_greater(after, t)) ? 1 : 0;
    }
};

template <class T1, class T2>
auto thresh_diff(T1&& before, T2&& after, int t = 0)
{
    return thresh_diff_imp<T1, T2>()(std::forward<T1>(before),
                                     std::forward<T2>(after), t);
}

} // namespace graph_tool
