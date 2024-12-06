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

#ifndef GRAPH_BLOCKMODEL_UNCERTAIN_MARGINAL_HH
#define GRAPH_BLOCKMODEL_UNCERTAIN_MARGINAL_HH

#include "config.h"

#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"

#include <mutex>
#include <shared_mutex>

namespace graph_tool
{
using namespace boost;
using namespace std;

class dummy_property
{};

template <class Key>
constexpr double get(dummy_property&, Key&&) { return 0.; }

template <class Key, class Val>
constexpr void put(dummy_property&, Key&&, Val&&) {}


template <class Graph, class UGraph, class Eprop, class Xprop, class XSprop, class Cprop>
void collect_marginal(Graph& g, UGraph& u, Eprop ecount, Xprop x, XSprop xsum,
                      XSprop x2sum, [[maybe_unused]] Cprop xs,
                      [[maybe_unused]] Cprop xcount)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;

    std::vector<gt_hash_map<vertex_t, edge_t>> emap(num_vertices(g));
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             auto& em = emap[v];
             for (auto e : out_edges_range(v, g))
             {
                 auto w = target(e, g);
                 if (!graph_tool::is_directed(g) && v > w)
                     continue;
                 em[w] = e;
             }
         });

    std::shared_mutex mutex;

    parallel_vertex_loop
        (u,
         [&](auto v)
         {
             auto& em = emap[v];
             bool self = false;
             for (auto e : out_edges_range(v, u))
             {
                 auto w = target(e, u);
                 if (!graph_tool::is_directed(u) && (v > w || (v == w && self)))
                     continue;
                 if (v == w)
                     self = !self;
                 edge_t ge;
                 auto iter = em.find(w);
                 if (iter == em.end())
                 {
                     std::unique_lock lock(mutex);
                     ge = add_edge(v, w, g).first;
                     em[w] = ge;
                     put(ecount, ge, 0);
                     put(xsum, ge, 0);
                     put(x2sum, ge, 0);
                 }
                 else
                 {
                     ge = iter->second;
                 }

                 std::shared_lock lock(mutex);
                 put(ecount, ge, get(ecount, ge) + 1);
                 put(xsum, ge, get(xsum, ge) + get(x, e));
                 put(x2sum, ge, get(x2sum, ge) + get(x, e) * get(x, e));
                 if constexpr (!std::is_same_v<Cprop, dummy_property>)
                 {
                     auto xe = get(x, e);
                     auto& xs_e = xs[ge];
                     auto& xc_e = xcount[ge];
                     auto iter = std::lower_bound(xs_e.begin(), xs_e.end(), xe);
                     if (iter == xs_e.end() || *iter != xe)
                     {
                         iter = xs_e.insert(iter, xe);
                         xc_e.insert(xc_e.begin() + (iter - xs_e.begin()), 0);
                     }
                     xc_e[iter - xs_e.begin()]++;
                 }
             }
         });

    if constexpr (!std::is_same_v<Cprop, dummy_property>)
    {
        parallel_vertex_loop
            (g,
             [&](auto v)
             {
                 auto e = edge(v, v, u);
                 if (e.second)
                     return;
                 auto ge = edge(v, v, g);
                 if (!ge.second)
                 {
                     std::unique_lock lock(mutex);
                     ge = add_edge(v, v, g);
                 }
                 std::shared_lock lock(mutex);
                 auto& xs_e = xs[ge.first];
                 auto& xc_e = xcount[ge.first];
                 if (xs_e.empty() || xs_e.front() != 0)
                 {
                     xs_e.insert(xs_e.begin(), 0);
                     xc_e.insert(xc_e.begin(), 0);
                 }
                 xc_e.front()++;
             });
    }
}


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UNCERTAIN_MARGINAL_HH
