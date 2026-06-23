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

#include "graph_filtering.hh"

#include <boost/python.hpp>
#include <boost/graph/astar_search.hpp>

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_util.hh"

#include "graph_astar.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

inline constexpr auto graph_view_ptrs =
    hana::transform(all_graph_views,
                    [](auto t)
                    {
                        return hana::type<std::shared_ptr<typename decltype(+t)::type>>();
                    });

void a_star_search_implicit(GraphInterface& gi, size_t source,
                            std::any adist, std::any apred,
                            std::any acost, std::any aweight,
                            python::object vis, python::object cmp,
                            python::object cmb, python::object zero,
                            python::object inf, python::object h)
{
    run_action<{.tr=graph_tool::type<decltype(graph_view_ptrs)>,
                .dargs={ .uncheck=false, .gil_release=false}}>()
        (gi,
         [&](auto&& gp, auto&& pred, auto&& dist, auto&& cost)
         {
             checked_vector_property_map<default_color_type,
                                         decltype(get(vertex_index, *gp))>
                    color(get(vertex_index, *gp));
             typedef typename property_traits<std::decay_t<decltype(dist)>>::value_type
                 dtype_t;
             typedef std::remove_reference_t<decltype(*gp)> g_t;
             typedef typename graph_traits<g_t>::edge_descriptor edge_t;
             DynamicPropertyMapWrap<dtype_t, edge_t> weight(aweight,
                                                            edge_properties);
             astar_search_no_init(*gp, source,
                                  AStarH<std::decay_t<decltype(gp)>,dtype_t>(gp, h),
                                  AStarVisitorWrapper<std::decay_t<decltype(gp)>>(gp, vis),
                                  pred, cost, dist, weight,
                                  color, get(vertex_index, *gp), AStarCmp(cmp),
                                  AStarCmb(cmb), python::extract<dtype_t>(inf),
                                  python::extract<dtype_t>(zero));
         },
         writable_vertex_scalar_properties, writable_vertex_properties, same_arg_type)
         (apred, adist, acost);
}

#define __MOD__ search
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     using namespace boost::python;
     def("astar_search_implicit", &a_star_search_implicit);
 });
