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
#include "graph.hh"
#include "graph_properties.hh"

#include "graph_arf.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void arf_layout(GraphInterface& g, std::any pos, std::any weight, double d,
                double a, double dt, size_t max_iter, double epsilon,
                size_t dim)
{
    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    auto edge_props_t = hana::append(edge_scalar_properties,
                                     hana::type<weight_map_t>());

    if(!weight.has_value())
        weight = weight_map_t();

    run_action<{.tr=graph_tool::type<decltype(graph_tool::never_directed)>, .dargs={.gil_release=false}}>()
        (g,
         [&](auto& g, auto pos, auto weight)
         {
             return get_arf_layout()(g, pos, weight, a, d, dt, epsilon,
                                     max_iter, dim);
         },
         vertex_floating_vector_properties, edge_props_t)(pos, weight);
}

#include <boost/python.hpp>

#define __MOD__ layout
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     boost::python::def("arf_layout", &arf_layout);
 });
