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

#include "graph_merge.hh"

using namespace graph_tool;
using namespace boost;

typedef vprop_map_t<int64_t> vprop_t;

void edge_property_merge(GraphInterface& ugi, GraphInterface& gi,
                         std::any avmap, std::any aemap,
                         std::any uprop, std::any aprop, merge_t merge,
                         bool simple)
{
    typedef eprop_map_t<GraphInterface::edge_t> emap_t;
    emap_t emap = std::any_cast<emap_t>(aemap);

    auto owedge_sum_properties =
        hana::append(hana::concat(writable_edge_scalar_properties,
                                  edge_scalar_vector_properties),
                     hana::type<eprop_map_t<python::object>>());

    auto oedge_sum_properties =
        hana::append(hana::concat(edge_scalar_properties,
                                  edge_scalar_vector_properties),
                     hana::type<eprop_map_t<python::object>>());

    auto oedge_scalar_vector_properties =
        hana::append(edge_scalar_vector_properties,
                     hana::type<eprop_map_t<python::object>>());

    auto osedge_scalar_vector_properties =
        hana::concat(oedge_scalar_vector_properties,
                     hana::tuple_t<eprop_map_t<std::string>,
                                   eprop_map_t<vector<std::string>>>);

    check_belongs(vertex_properties, avmap);
    check_pmap_is_convertible(hana::tuple_t<vprop_map_t<int64_t>>, avmap);
    if (!belongs(hana::tuple_t<vertex_index_map_t>, avmap))
        avmap = vprop_map_as_dynamic(avmap, vprop_map_t<int64_t>());
    auto vmap_ts = hana::tuple_t<vertex_index_map_t, vprop_map_t<int64_t>,
                                 vdprop_map_t<int64_t>>;

    switch (merge)
    {
    case merge_t::set:
        check_belongs(writable_edge_properties, uprop);
        check_belongs(edge_properties, aprop);
        gt_dispatch<>(false)
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::set>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             writable_edge_properties,
             hana::concat(dynamic_prop_type, same_arg_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             uprop,
             eprop_map_as_dynamic(aprop, uprop));
        break;
    case merge_t::sum: [[fallthrough]];
    case merge_t::diff:
        check_belongs(owedge_sum_properties, uprop);
        check_belongs(oedge_sum_properties, aprop);
        gt_dispatch<>(false)
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 switch (merge)
                 {
                 case merge_t::sum:
                     graph_tool::property_merge<merge_t::sum>()
                         (g1, g2, vmap, emap, uprop, prop, simple);
                     break;
                 case merge_t::diff:
                     graph_tool::property_merge<merge_t::diff>()
                         (g1, g2, vmap, emap, uprop, prop, simple);
                     break;
                 default:
                     break;
                 }
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             owedge_sum_properties,
             hana::concat(dynamic_prop_type, same_arg_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             uprop,
             eprop_map_as_dynamic(aprop, uprop));
        break;
    case merge_t::idx_inc:
        check_belongs(edge_scalar_vector_properties, uprop);
        check_belongs(hana::concat(edge_scalar_properties,
                                   edge_scalar_vector_properties), aprop);
        gt_dispatch<>(false)
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::idx_inc>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             edge_scalar_vector_properties,
             hana::tuple_t<eprop_map_t<int32_t>, edprop_map_t<int32_t>,
                           eprop_map_t<vector<double>>, edprop_map_t<vector<double>>>)
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             uprop,
             belongs(edge_scalar_properties, aprop) ?
                 eprop_map_as_dynamic(aprop, eprop_map_t<int32_t>()) :
                 eprop_map_as_dynamic(aprop, eprop_map_t<vector<double>>()));
        break;
    case merge_t::append:
        check_belongs(oedge_scalar_vector_properties, uprop);
        check_belongs(hana::append(edge_scalar_properties,
                                   hana::type<eprop_map_t<python::object>>()),
                      aprop);
        gt_dispatch<>(false)
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::append>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             oedge_scalar_vector_properties,
             hana::concat(velem_dprop_type, velem_eprop_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             uprop,
             eprop_map_as_dvelem(aprop, uprop));
        break;
    case merge_t::concat:
        check_belongs(osedge_scalar_vector_properties, uprop);
        check_belongs(osedge_scalar_vector_properties, aprop);
        gt_dispatch<>(false)
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::concat>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             osedge_scalar_vector_properties,
             hana::concat(dynamic_prop_type, eprop_same_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             uprop,
             eprop_map_as_dynamic(aprop, uprop));
        break;
    default:
        break;
    };
}

#include <boost/python.hpp>
using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("edge_property_merge", &edge_property_merge);
 });
