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

#include "types.hh"

namespace graph_tool
{

template <class Val, class Graph>
double vterm(const Val& mrp, const Val& mrm, const ovweight_t& wr, bool deg_corr,
             const Graph& g)
{
    return vterm(int64_t(mrp), int64_t(mrm), int64_t(wr.size()), deg_corr, g);
}

template <class Graph, class Val, class Eval>
double eterm_dense(Val r, Val s, Eval&& ers, const ovweight_t& wr_r,
                   const ovweight_t& wr_s, bool multigraph, const Graph& g)
{
    return eterm_dense(r, s, ers, wr_r.size(), wr_s.size(), multigraph, g);
}

template <class Graph, class EW>
double get_parallel_entropy(const half_edge_adaptor<Graph>&, const EW&)
{
    return 0;
}

template <class Graph, class Delta>
double get_deg_entropy(const std::tuple<odeg_t, odeg_t>& degs, const Graph& g,
                       const Delta& delta)
{
    return get_deg_entropy(std::make_tuple(get<0>(degs)._k, get<1>(degs)._k), g, delta);
}

template <class T>
    requires(std::is_same_v<T, odeg_t>)
auto lgamma_fast(T&& x)
{
    return lgamma_fast(int64_t(x));
}

template <class Deg>
   requires(std::is_same_v<Deg, odeg_t>)
double deg_entropy(const Deg& d)
{
    return graph_tool::deg_entropy(d._k);
}

} // namespace graph_tool
