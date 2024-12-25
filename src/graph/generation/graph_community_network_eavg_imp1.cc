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

#include <boost/mpl/insert_range.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/python.hpp>

#include "graph_community_network.hh"

using namespace std;
using namespace boost;

using namespace graph_tool;

typedef UnityPropertyMap<int,GraphInterface::edge_t> no_eweight_map_t;
typedef eprop_map_t<int32_t> ecount_map_t;

void sum_eprops(GraphInterface& gi, GraphInterface& cgi,
                std::any community_property,
                std::any condensed_community_property, std::any ceprop,
                std::any eprop, bool self_loops, bool parallel_edges)
{
    auto eprops_t = hana::append(hana::concat(writable_edge_scalar_properties,
                                              edge_scalar_vector_properties),
                                 hana::type<eprop_map_t<boost::python::object>>());

    run_action<>(false)
        (gi,
         [&](auto& g, auto s_map, auto cs_map, auto eprop, auto ceprop)
         {
             get_edge_community_property_sum()(g, cgi.get_graph(), s_map, cs_map.get_checked(),
                                               eprop, ceprop.get_checked(),
                                               self_loops, parallel_edges);
         },
         writable_vertex_properties, same_arg_type, eprops_t,
         same_arg_type)
        (community_property, condensed_community_property,
         eprop, ceprop);
}
