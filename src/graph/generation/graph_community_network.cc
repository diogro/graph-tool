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
#include "graph_python_interface.hh"
#include "graph_filtering.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"

#include <boost/mpl/push_back.hpp>
#include <boost/python.hpp>

#include "graph_community_network.hh"

using namespace std;
using namespace boost;

using namespace graph_tool;

typedef UnityPropertyMap<int,GraphInterface::edge_t> no_eweight_map_t;
typedef UnityPropertyMap<int,GraphInterface::vertex_t> no_vweight_map_t;
typedef eprop_map_t<int32_t> ::unchecked_t ecount_map_t;
typedef vprop_map_t<int32_t> ::unchecked_t vcount_map_t;

struct get_community_network_vertices_dispatch
{
    template <class Graph, class CommunityGraph, class CommunityMap,
              class VertexWeightMap>
    void operator()(const Graph& g, CommunityGraph& cg,
                    CommunityMap s_map, std::any acs_map,
                    VertexWeightMap vweight, std::any vcount) const
    {
        typename CommunityMap::checked_t cs_map = std::any_cast<typename CommunityMap::checked_t>(acs_map);

        typedef typename boost::mpl::if_<std::is_same<no_vweight_map_t, VertexWeightMap>,
                                         vcount_map_t, VertexWeightMap>::type vweight_t;
        typename vweight_t::checked_t vertex_count = std::any_cast<typename vweight_t::checked_t>(vcount);

        get_community_network_vertices()(g, cg, s_map, cs_map, vweight, vertex_count);
    }

};

struct get_community_network_edges_dispatch
{
    get_community_network_edges_dispatch(bool self_loops): _self_loops(self_loops) {}
    bool _self_loops;

    template <class Graph, class CommunityGraph, class CommunityMap,
              class EdgeWeightMap, class EdgeIndex>
    void operator()(const Graph& g, CommunityGraph& cg, EdgeIndex cedge_index,
                    CommunityMap s_map, std::any acs_map,
                    EdgeWeightMap eweight, std::any ecount) const
    {
        typename CommunityMap::checked_t cs_map = std::any_cast<typename CommunityMap::checked_t>(acs_map);

        typedef typename boost::mpl::if_<std::is_same<no_eweight_map_t, EdgeWeightMap>,
                                         ecount_map_t, EdgeWeightMap>::type eweight_t;

        typename eweight_t::checked_t edge_count = std::any_cast<typename eweight_t::checked_t>(ecount);
        get_community_network_edges()(g, cg, cedge_index, s_map,
                                      cs_map, eweight, edge_count,
                                      _self_loops);
    }
};

void community_network_edges(GraphInterface& gi, GraphInterface& cgi,
                             std::any community_property,
                             std::any condensed_community_property,
                             std::any edge_count, std::any eweight,
                             bool self_loops, bool parallel_edges);

void community_network(GraphInterface& gi, GraphInterface& cgi,
                       std::any community_property,
                       std::any condensed_community_property,
                       std::any vertex_count, std::any edge_count,
                       std::any vweight, std::any eweight, bool self_loops,
                       bool parallel_edges)
{
    auto vweight_properties = hana::append(writable_vertex_scalar_properties,
                                           hana::type<no_vweight_map_t>());

    if (!vweight.has_value())
        vweight = no_vweight_map_t();

    run_action<>(false)
        (gi,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_community_network_vertices_dispatch()
                 (std::forward<decltype(graph)>(graph), cgi.get_graph(),
                  std::forward<decltype(a2)>(a2), condensed_community_property,
                  std::forward<decltype(a3)>(a3), vertex_count);
         },
         writable_vertex_properties,
         vweight_properties)(community_property, vweight);

    community_network_edges(gi, cgi, community_property,
                            condensed_community_property, edge_count,
                            eweight, self_loops, parallel_edges);
}

using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("community_network", &community_network);
 });
