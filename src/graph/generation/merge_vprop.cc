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
#include "graph_filtering.hh"

#include "merge.hh"

using namespace graph_tool;
using namespace boost;

typedef vprop_map_t<int64_t> vprop_t;

void vertex_property_merge(GraphInterface& ugi, GraphInterface& gi,
                           std::any avmap, std::any aemap,
                           std::any auprop, std::any aprop, merge_t merge,
                           bool simple)
{
    check_belongs(vertex_properties, avmap);
    check_belongs(vertex_properties, auprop);
    check_belongs(vertex_properties, aprop);

    typedef eprop_map_t<GraphInterface::edge_t> emap_t;
    emap_t emap = std::any_cast<emap_t>(aemap);

    auto owvertex_sum_properties =
        hana::append(hana::concat(writable_scalar_vertex_properties,
                                  scalar_vector_vertex_properties),
                     hana::type<vprop_map_t<python::object>>());
    auto overtex_sum_properties =
        hana::append(hana::concat(scalar_vertex_properties,
                                  scalar_vector_vertex_properties),
                     hana::type<vprop_map_t<python::object>>());

    auto oscalar_vector_vertex_properties =
        hana::append(scalar_vector_vertex_properties,
                     hana::type<vprop_map_t<python::object>>());

    auto osscalar_vector_vertex_properties =
        hana::concat(oscalar_vector_vertex_properties,
                     hana::tuple_t<vprop_map_t<std::string>,
                                   vprop_map_t<vector<std::string>>>);

    check_pmap_is_convertible(hana::tuple_t<vprop_map_t<int64_t>>, avmap);
    if (!belongs(hana::tuple_t<vertex_index_map_t>, avmap))
        avmap = vprop_map_as_dynamic(avmap, vprop_map_t<int64_t>());
    auto vmap_ts = hana::tuple_t<vertex_index_map_t, vprop_map_t<int64_t>,
                                 vdprop_map_t<int64_t>>;

    switch (merge)
    {
    case merge_t::set:
        check_belongs(writable_vertex_properties, auprop);
        check_belongs(vertex_properties, aprop);
        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::set>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             writable_vertex_properties,
             hana::concat(dynamic_prop_type, same_arg_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             auprop,
             vprop_map_as_dynamic(aprop, auprop));
        break;
    case merge_t::sum: [[fallthrough]];
    case merge_t::diff:
        check_belongs(owvertex_sum_properties, auprop);
        check_belongs(overtex_sum_properties, aprop);
        gt_dispatch<{.gil_release=false}>()
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
             owvertex_sum_properties,
             hana::concat(dynamic_prop_type, same_arg_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             auprop,
             vprop_map_as_dynamic(aprop, auprop));
        break;
    case merge_t::idx_inc:
        check_belongs(scalar_vector_vertex_properties, auprop);
        check_belongs(hana::concat(scalar_vertex_properties,
                                   scalar_vector_vertex_properties), aprop);
        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::idx_inc>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             scalar_vector_vertex_properties,
             hana::tuple_t<vprop_map_t<int32_t>, vdprop_map_t<int32_t>,
                           vprop_map_t<vector<double>>, vdprop_map_t<vector<double>>>)
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             vprop_map_as_dynamic(avmap, vprop_map_t<int64_t>()),
             auprop,
             belongs(scalar_vertex_properties, aprop) ?
                 vprop_map_as_dynamic(aprop, vprop_map_t<int32_t>()) :
                 vprop_map_as_dynamic(aprop, vprop_map_t<vector<double>>()));
        break;
    case merge_t::append:
        check_belongs(oscalar_vector_vertex_properties, auprop);
        check_belongs(hana::append(scalar_vertex_properties,
                                   hana::type<vprop_map_t<python::object>>()),
                      aprop);
        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::append>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             oscalar_vector_vertex_properties,
             hana::concat(velem_dprop_type, velem_vprop_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             auprop,
             vprop_map_as_dvelem(aprop, auprop));
        break;
    case merge_t::concat:
        check_belongs(osscalar_vector_vertex_properties, auprop);
        check_belongs(osscalar_vector_vertex_properties, auprop);
        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g1, auto& g2, auto& vmap, auto& uprop, auto& prop)
             {
                 graph_tool::property_merge<merge_t::concat>()
                     (g1, g2, vmap, emap, uprop, prop, simple);
             },
             always_directed_never_reversed,
             always_directed_never_reversed,
             vmap_ts,
             osscalar_vector_vertex_properties,
             hana::concat(dynamic_prop_type, same_arg_type))
            (ugi.get_graph_view(),
             gi.get_graph_view(),
             avmap,
             auprop,
             vprop_map_as_dynamic(aprop, auprop));
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
     def("vertex_property_merge", &vertex_property_merge);
 });
