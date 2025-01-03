// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_bip_projection.hh"

using namespace graph_tool;
using namespace boost;

boost::python::object
bipartite_projection_dispatch(GraphInterface& ugi, GraphInterface& gi,
                              std::any apmap, std::any avmap, std::any aevmap)
{
    typedef eprop_map_t<GraphInterface::edge_t> emap_t;
    emap_t emap(gi.get_edge_index());
    emap.reserve(gi.get_edge_index_range());

    bool is_object = belongs(hana::tuple_t<vprop_map_t<python::object>>, apmap);

    gt_dispatch<false>(false)
        ([&](auto& ug, auto& g, auto& pmap, auto& vmap, auto& evmap)
         {
             GILRelease gil(!is_object);
             bipartite_projection(ug, g, pmap, vmap, emap, evmap);
         },
         never_filtered_never_reversed,
         all_graph_views,
         hana::tuple_t<vprop_map_t<uint8_t>, vdprop_map_t<uint8_t>>,
         hana::tuple_t<vprop_map_t<int64_t>>,
         hana::tuple_t<eprop_map_t<int64_t>>)
        (ugi.get_graph_view(),
         gi.get_graph_view(),
         vprop_map_as_dynamic(apmap, vprop_map_t<uint8_t>()),
         avmap, aevmap);

    return boost::python::object(std::any(emap));
}

void projection_copy_eprop(GraphInterface& ugi, std::any aevmap,
                           std::any uprop, std::any prop)
{
    bool is_object = belongs(hana::tuple_t<eprop_map_t<python::object>>, uprop);
    gt_dispatch<>(false)
        ([&](auto& ug, auto& evmap, auto& uprop, auto& prop)
         {
             GILRelease gil(!is_object);
             for (auto e : edges_range(ug))
             {
                 auto v = evmap[e];
                 if (v == -1)
                     continue;
                 uprop[e] = prop[v];
             }
         },
         never_filtered_never_reversed,
         hana::tuple_t<eprop_map_t<int64_t>>,
         writable_edge_properties,
         hana::tuple_t<vprop_same_t>)
        (ugi.get_graph_view(), aevmap, uprop, prop);
}

void projection_copy_reprop(GraphInterface& ugi, std::any aemap,
                            std::any uprop, std::any prop)
{
    typedef eprop_map_t<GraphInterface::edge_t> emap_t;

    bool is_object = belongs(hana::tuple_t<eprop_map_t<python::object>>, uprop);
    gt_dispatch<>(false)
        ([&](auto& ug, auto& emap, auto& uprop, auto& prop)
         {
             GILRelease gil(!is_object);
             for (auto e : edges_range(ug))
             {
                 auto& eg = emap.get_checked()[e];
                 if (eg == GraphInterface::edge_t())
                     continue;
                 uprop[e] = prop[eg];
             }
         },
         never_filtered_never_reversed,
         hana::tuple_t<emap_t>,
         writable_edge_properties,
         hana::tuple_t<eprop_same_t>)
        (ugi.get_graph_view(), aemap, uprop, prop);
}

void projection_copy_vprop(GraphInterface& gi, std::any avmap,
                           std::any uprop, std::any prop)
{
    bool is_object = belongs(hana::tuple_t<vprop_map_t<python::object>>, uprop);
    gt_dispatch<>(false)
        ([&](auto& g, auto& vmap, auto& uprop, auto& prop)
         {
             GILRelease gil(!is_object);
             for (auto v : vertices_range(g))
             {
                 auto u = vmap[v];
                 if (u < 0)
                     continue;
                 uprop[u] = prop[v];
             }
         },
         never_filtered_never_reversed,
         hana::tuple_t<vprop_map_t<int64_t>>,
         writable_vertex_properties,
         hana::tuple_t<vprop_same_t>)
        (gi.get_graph_view(), avmap, uprop, prop);
}

#include <boost/python.hpp>
using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("bipartite_projection", &bipartite_projection_dispatch);
     def("projection_copy_eprop", &projection_copy_eprop);
     def("projection_copy_reprop", &projection_copy_reprop);
     def("projection_copy_vprop", &projection_copy_vprop);
 });
