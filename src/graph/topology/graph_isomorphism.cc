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

#include "graph.hh"
#include "graph_filtering.hh"

#include <boost/graph/isomorphism.hpp>

using namespace graph_tool;
using namespace boost;

struct check_iso
{

    template <class Graph1, class Graph2, class IsoMap, class InvMap,
              class VertexIndexMap>
    void operator()(Graph1& g1, Graph2& g2, InvMap cinv_map1, InvMap cinv_map2,
                    int64_t max_inv, IsoMap map, VertexIndexMap index1,
                    VertexIndexMap index2, bool& result) const
    {
        auto inv_map1 = cinv_map1.get_unchecked(num_vertices(g1));
        auto inv_map2 = cinv_map2.get_unchecked(num_vertices(g2));

        vinv_t<decltype(inv_map1)> vinv1(inv_map1, max_inv);
        vinv_t<decltype(inv_map2)> vinv2(inv_map2, max_inv);

        result = isomorphism(g1, g2,
                             isomorphism_map(map.get_unchecked(num_vertices(g1))).
                             vertex_invariant1(vinv1).
                             vertex_invariant2(vinv2).
                             vertex_index1_map(index1).
                             vertex_index2_map(index2));
    }

    template <class Prop>
    struct vinv_t
    {
        vinv_t(Prop& prop, int64_t max)
            : _prop(prop), _max(max) {}
        Prop& _prop;
        int64_t _max;

        template <class Vertex>
        int64_t operator()(Vertex v) const
        {
            return _prop[v];
        };

        int64_t max() const { return _max; }

        typedef int64_t result_type;
        typedef size_t argument_type;
    };
};

bool check_isomorphism(GraphInterface& gi1, GraphInterface& gi2,
                       std::any ainv_map1, std::any ainv_map2,
                       int64_t max_inv, std::any aiso_map)
{
    bool result;

    typedef vprop_map_t<int32_t> iso_map_t;
    auto iso_map = std::any_cast<iso_map_t>(aiso_map);

    typedef vprop_map_t<int64_t> inv_map_t;
    auto inv_map1 = std::any_cast<inv_map_t>(ainv_map1);
    auto inv_map2 = std::any_cast<inv_map_t>(ainv_map2);

    if (gi1.get_directed() != gi2.get_directed())
        return false;
    if (gi1.get_directed())
    {
        gt_dispatch<>()
            ([&](auto&& graph, auto&& a2)
             {
                 vprop_map_t<size_t> imap(get(vertex_index, graph));

                 check_iso()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), inv_map1, inv_map2,
                      max_inv, imap, gi1.get_vertex_index(),
                      gi2.get_vertex_index(), result);

                 if (result)
                 {
                     for (auto v : vertices_range(graph))
                         iso_map[v] = imap[v];
                 }
             },
             always_directed,
             always_directed)(gi1.get_graph_view(), gi2.get_graph_view());
    }
    else
    {
        gt_dispatch<>()
            ([&](auto&& graph, auto&& a2)
             {
                 vprop_map_t<size_t> imap(get(vertex_index, graph));

                 check_iso()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), inv_map1, inv_map2,
                      max_inv, imap, gi1.get_vertex_index(),
                      gi2.get_vertex_index(), result);

                 if (result)
                 {
                     for (auto v : vertices_range(graph))
                         iso_map[v] = imap[v];
                 }
             },
             never_directed,
             never_directed)(gi1.get_graph_view(), gi2.get_graph_view());
    }

    return result;
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
    def("check_isomorphism", &check_isomorphism);
 });
