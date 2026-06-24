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
#include "graph_selectors.hh"
#include "graph_katz.hh"

#include <boost/python.hpp>


using namespace std;
using namespace boost;
using namespace graph_tool;

void katz(GraphInterface& g, std::any w, std::any c, std::any beta,
          long double alpha, double epsilon, size_t max_iter)
{
    if (w.has_value() && !belongs(writable_scalar_edge_properties, w))
        throw ValueException("edge property must be writable");
    if (!belongs(floating_vertex_properties, c))
        throw ValueException("centrality vertex property must be of floating point"
                             " value type");
    if (beta.has_value() && !belongs(floating_vertex_properties, beta))
        throw ValueException("personalization vertex property must be of floating point"
                             " value type");

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    auto weight_props = hana::append(writable_scalar_edge_properties,
                                    hana::type<weight_map_t>());

    if (!w.has_value())
        w = weight_map_t();

    typedef UnityPropertyMap<int,GraphInterface::vertex_t> beta_map_t;
    auto beta_props = hana::append(floating_vertex_properties,
                                     hana::type<beta_map_t>());

    if (!beta.has_value())
        beta = beta_map_t();

    run_action<>()
        (g,
         [&](auto&& graph, auto&& a2, auto&& a3, auto&& a4)
         {
             return get_katz()
                 (std::forward<decltype(graph)>(graph), g.get_vertex_index(),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3),
                  std::forward<decltype(a4)>(a4), alpha, epsilon, max_iter);
         },
         weight_props, floating_vertex_properties,
         beta_props)(w, c, beta);
}

#define __MOD__ centrality
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     using namespace boost::python;
     def("get_katz", &katz);
 });
