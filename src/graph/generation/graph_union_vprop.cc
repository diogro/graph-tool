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

#include "graph_union.hh"

using namespace graph_tool;
using namespace boost;

typedef vprop_map_t<int64_t> vprop_t;

typedef eprop_map_t<GraphInterface::edge_t> eprop_t;

void vertex_property_union(GraphInterface& ugi, GraphInterface& gi,
                           std::any p_vprop, std::any p_eprop,
                           std::any uprop, std::any prop)
{
    vprop_t vprop = std::any_cast<vprop_t>(p_vprop);
    eprop_t eprop = std::any_cast<eprop_t>(p_eprop);

    gt_dispatch<>()
        (
            [&](auto&& graph, auto&& a2, auto&& a3)
            {
                return graph_tool::property_union()
                    (std::forward<decltype(graph)>(graph),
                     std::forward<decltype(a2)>(a2), vprop, eprop,
                     std::forward<decltype(a3)>(a3), prop);
            },
            always_directed, always_directed, writable_vertex_properties)(
            ugi.get_graph_view(), gi.get_graph_view(), uprop);
}

#include <boost/python.hpp>
using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("vertex_property_union", &vertex_property_union);
 });
