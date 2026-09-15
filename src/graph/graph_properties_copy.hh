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

#ifndef GRAPH_PROPERTIES_COPY_HH
#define GRAPH_PROPERTIES_COPY_HH

#include "graph_properties.hh"
#include "graph_util.hh"
#include "hash_map_wrap.hh"

namespace graph_tool
{

//
// Property map copying
// ====================

template <class GraphSrc, class GraphTgt, class PropertySrc, class PropertyTgt>
void copy_external_edge_property_dispatch(const GraphSrc& src, const GraphTgt& tgt,
                                          PropertySrc src_map, PropertyTgt tgt_map)
{
    bool is_python =
        (get_underlying_value_type(tgt_map) != typeid(boost::python::object) ||
         get_underlying_value_type(src_map) != typeid(boost::python::object));

    GILRelease gil(!is_python);

    typedef typename boost::graph_traits<GraphTgt>::edge_descriptor edge_t;
    std::vector<gt_hash_map<size_t, std::deque<edge_t>>> tgt_edges(num_vertices(tgt));
    parallel_vertex_loop
        (tgt,
         [&](auto u)
         {
             auto& es = tgt_edges[u];
             for (auto e : out_edges_range(u, tgt))
             {
                 auto v = target(e, tgt);
                 if (!is_directed(tgt) && u > v)
                     continue;
                 es[v].push_back(e);
             }
         });

    bool parallel =
        (num_vertices(src) > get_openmp_min_thresh() && !is_python);

    std::exception_ptr eptr{};
    #pragma omp parallel if (parallel)
    eptr = parallel_vertex_loop_no_spawn<true>
        (src,
         [&](auto u)
         {
             if (u >= tgt_edges.size())
                 return;

             auto& es = tgt_edges[u];
             for (auto e : out_edges_range(u, src))
             {
                 auto v = target(e, src);
                 if (!is_directed(src) && u > v)
                     continue;
                 auto iter = es.find(v);
                 if (iter == es.end() || iter->second.empty())
                     continue;
                 put(tgt_map, iter->second.front(), get(src_map, e));
                 iter->second.pop_front();
             }
         });
    if (eptr)
        std::rethrow_exception(eptr);
}


} // namespace graph_tool

#endif // GRAPH_PROPERTIES_COPY_HH
