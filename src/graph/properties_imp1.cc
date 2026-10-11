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

#include "graph.hh"
#include "python_interface.hh"
#include "properties.hh"
#include "graph_filtering.hh"
#include "selectors.hh"
#include "graph_util.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;
using namespace graph_tool;

template <bool src>
struct do_edge_endpoint
{
    template <class Graph, class EdgeIndexMap, class VertexPropertyMap>
    void operator()(Graph& g, EdgeIndexMap, VertexPropertyMap prop,
                    std::any aeprop, size_t edge_index_range) const
    {
        typedef typename property_traits<VertexPropertyMap>::value_type vval_t;
        typedef std::conditional_t<std::is_same_v<vval_t, size_t>, int64_t, vval_t>
            val_t;
        typedef property_map_t<val_t, EdgeIndexMap> eprop_t;
        eprop_t eprop = std::any_cast<eprop_t>(aeprop);
        eprop.reserve(edge_index_range);

        bool is_object = std::is_same_v<val_t,python::object>;

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh() && !is_object)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 for (const auto& e : out_edges_range(v, g))
                 {
                     auto s = v;
                     auto t = target(e, g);
                     if (!is_directed(g) && s > t)
                         continue;
                     if (src)
                         eprop[e] = prop[s];
                     else
                         eprop[e] = prop[t];
                 }
             });
    }
};

void edge_endpoint(GraphInterface& gi, std::any prop,
                   std::any eprop, std::string endpoint)
{
    size_t edge_index_range = gi.get_edge_index_range();
    if (endpoint == "source")
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_edge_endpoint<true>()(
                     std::forward<decltype(graph)>(graph), gi.get_edge_index(),
                     std::forward<decltype(a2)>(a2), eprop, edge_index_range);
             },
             vertex_properties)(prop);
    else
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_edge_endpoint<false>()(
                     std::forward<decltype(graph)>(graph), gi.get_edge_index(),
                     std::forward<decltype(a2)>(a2), eprop, edge_index_range);
             },
             vertex_properties)(prop);
}
