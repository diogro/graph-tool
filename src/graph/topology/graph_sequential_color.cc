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

#include "graph_filtering.hh"
#include "graph.hh"
#include "graph_properties.hh"

#include <boost/graph/sequential_vertex_coloring.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

struct get_coloring
{
    template <class Graph, class OrderMap, class ColorMap>
    void operator()(Graph& g, OrderMap order, ColorMap color, size_t& nc) const
    {
        nc = sequential_vertex_coloring(g, order, color);
    }
};

size_t sequential_coloring(GraphInterface& gi, std::any order,
                           std::any color)
{
    auto int_properties = hana::tuple_t<vprop_map_t<int32_t>,
                                        vprop_map_t<int64_t>>;
    size_t nc = 0;
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_coloring()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3), nc);
         },
         integer_vertex_properties, int_properties)(order, color);
    return nc;
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("sequential_coloring", &sequential_coloring);
 });
