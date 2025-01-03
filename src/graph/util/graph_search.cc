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

#include "graph_search.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

inline constexpr auto graph_view_ptrs =
    hana::transform(all_graph_views,
                    [](auto t)
                    {
                        return hana::type<std::shared_ptr<typename decltype(+t)::type>>();
                    });


// find vertices which match a certain (inclusive) property range
python::list
find_vertex_range(GraphInterface& gi, GraphInterface::deg_t adeg,
                  python::tuple range)
{
    python::list ret;
    run_action<decltype(graph_view_ptrs)>(false)
        (gi,
         [&](auto&& gp, auto&& deg)
         {
             return find_vertices()(*gp, gp, deg, range, ret);
         },
         all_selectors)(degree_selector(adeg));
    return ret;
}

// find edges which match a certain (inclusive) property range
python::list
find_edge_range(GraphInterface& gi, std::any eprop, python::tuple range)
{
    python::list ret;

    GraphInterface::edge_index_map_t eindex =
        std::any_cast<GraphInterface::edge_index_map_t>(gi.get_edge_index());
    run_action<decltype(graph_view_ptrs)>(false)
        (gi,
         [&](auto&& gp, auto&& prop)
         {
             return find_edges()(*gp, gp, eindex, prop, range, ret);
         },
         edge_properties)(eprop);
    return ret;
}


using namespace boost::python;

void export_search()
{
    def("find_vertex_range", &find_vertex_range);
    def("find_edge_range", &find_edge_range);
}
