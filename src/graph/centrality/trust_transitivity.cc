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
#include "selectors.hh"
#include "trust_transitivity.hh"

#include <boost/python.hpp>


using namespace std;
using namespace boost;
using namespace graph_tool;

void trust_transitivity(GraphInterface& g, int64_t source, int64_t target,
                    std::any c, std::any t)
{
    if (!belongs(floating_edge_properties, c))
        throw ValueException("edge property must be of floating point value type");
    if (!belongs(floating_vector_vertex_properties, t))
        throw ValueException("vertex property must be of floating point valued vector type");

    run_action<>()
        (g,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_trust_transitivity()
                 (std::forward<decltype(graph)>(graph), g.get_vertex_index(),
                  source, target, std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3));
         },
         floating_edge_properties, floating_vector_vertex_properties)(c, t);
}

#define __MOD__ centrality
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     using namespace boost::python;
     def("get_trust_transitivity", &trust_transitivity);
 });
