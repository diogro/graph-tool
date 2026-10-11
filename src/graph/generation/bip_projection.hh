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

#ifndef BIP_PROJECTION_HH
#define BIP_PROJECTION_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

#include "idx_map.hh"
#include "hash_map_wrap.hh"
#include "demangle.hh"

#include <mutex>
#include <shared_mutex>

#include <boost/python/object.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class PGraph, class Graph, class PMap, class VertexMap,
          class EdgeMap, class EdgeVMap>
void bipartite_projection(PGraph& ug, Graph& g, PMap pmap, VertexMap vmap,
                          EdgeMap emap, EdgeVMap evmap)
{
    for (auto v : vertices_range(g))
    {
        if (get(pmap, v))
            vmap[v] = -1;
        else
            vmap[v] = add_vertex(ug);
    }

    for (auto v : vertices_range(g))
    {
        if (get(pmap, v))
            continue;
        for (auto e : out_edges_range(v, g))
        {
            auto u = target(e, g);
            if (get(pmap, u))
            {
                for (auto w : out_neighbors_range(u, g))
                {
                    if (get(pmap, w) || w == v)
                        continue;
                    if constexpr (!is_directed_v<PGraph>)
                    {
                        if (vmap[v] > vmap[w])
                            continue;
                    }
                    auto ge = add_edge(size_t(vmap[v]), size_t(vmap[w]), ug).first;
                    evmap[ge] = u;
                }
            }
            else
            {
                // non-bipartite edge
                auto [ge, exists] = edge(size_t(vmap[v]), size_t(vmap[u]), ug);
                if (!exists)
                    ge = add_edge(size_t(vmap[v]), size_t(vmap[u]), ug).first;
                emap[ge] = e;
                evmap[ge] = -1;
            }
        }
    }
}

} // graph_tool namespace

#endif // BIP_PROJECTION_HH
