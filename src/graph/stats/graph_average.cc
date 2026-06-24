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
#include "graph_properties.hh"

#include "graph_average.hh"

#include <boost/python.hpp>

#include <boost/mpl/joint_view.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

// this will return the vertex average of degrees or scalar properties
python::object
get_vertex_average(GraphInterface& gi, GraphInterface::deg_t deg)
{
    python::object a, dev;
    size_t count;

    auto vertex_numeric_properties_c =
        hana::concat(vertex_scalar_properties,
                     vertex_scalar_vector_properties);

    auto python_properties =
        hana::tuple_t<vprop_map_t<python::object>>;

    auto vertex_numeric_properties =
        hana::concat(vertex_numeric_properties_c,
                     python_properties);

    auto numeric_selectors =
        hana::concat(degree_selectors,
                     hana::transform(vertex_numeric_properties,
                                     [](auto t){ return hana::type<scalarS<typename decltype(+t)::type>>(); }));

    run_action<{.dargs={.gil_release=false}}>()(gi, get_average<VertexAverageTraverse>(a,dev,count),
                        numeric_selectors)(degree_selector(deg));
    return python::make_tuple(a, dev, count);
}

// this will return the edge average of scalar properties
python::object
get_edge_average(GraphInterface& gi, std::any prop)
{
    auto edge_numeric_properties_c =
        hana::concat(edge_scalar_properties,
                     edge_scalar_vector_properties);

    auto python_properties =
        hana::tuple_t<eprop_map_t<python::object>>;

    auto edge_numeric_properties =
        hana::concat(edge_numeric_properties_c,
                     python_properties);

    python::object a, dev;
    size_t count;
    run_action<{.tr=graph_tool::type<decltype(graph_tool::always_directed)>, .dargs={.gil_release=false}}>()
        (gi, get_average<EdgeAverageTraverse>(a, dev, count),
         edge_numeric_properties)(prop);
    return python::make_tuple(a, dev, count);
}

using namespace boost::python;

#define __MOD__ stats
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("get_vertex_average", &get_vertex_average);
     def("get_edge_average", &get_edge_average);
 });
