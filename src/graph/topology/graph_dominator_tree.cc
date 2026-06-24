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

#include <boost/graph/dominator_tree.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

struct get_dominator_tree
{
    template <class Graph, class PredMap>
    void operator()(const Graph& g, size_t entry, PredMap pred_map) const
    {
        lengauer_tarjan_dominator_tree(g, vertex(entry,g), pred_map);
    }
};

void dominator_tree(GraphInterface& gi, size_t entry, std::any pred_map)
{
    auto pred_properties = hana::tuple_t<vprop_map_t<int32_t>>;

    run_action<{.tr=graph_tool::type<decltype(graph_tool::always_directed)>}>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return get_dominator_tree()
                 (std::forward<decltype(graph)>(graph), entry,
                  std::forward<decltype(a2)>(a2));
         },
         pred_properties)(pred_map);
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("dominator_tree", &dominator_tree);
 });
