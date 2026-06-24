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

#include "graph_geometric.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

void geometric(GraphInterface& gi, python::object opoints, double r,
               python::object orange, bool periodic, std::any pos)
{
    python::object shape = opoints.attr("shape");
    size_t size = python::extract<size_t>(shape[0]);
    vector<vector<double> > points(size);
    vector<pair<double, double> > range(python::len(orange));

    auto prop_types = hana::tuple_t<vprop_map_t<vector<double>>>;

    size = python::extract<size_t>(shape[1]);
    for(size_t i = 0; i < points.size(); ++i)
    {
        points[i].resize(size);
        for (size_t j = 0; j < points[i].size(); ++j)
            points[i][j] = python::extract<double>(opoints[i][j]);
    }

    for(size_t i = 0; i < range.size(); ++i)
    {
        range[i].first = python::extract<double>(orange[i][0]);
        range[i].second = python::extract<double>(orange[i][1]);
    }

    auto gviews =
        get_graph_views(hana::tuple_t<std::false_type>,
                        hana::tuple_t<std::false_type>,
                        hana::tuple_t<std::false_type>);

    run_action<{.tr=graph_tool::type<decltype(gviews)>}>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return get_geometric()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2), points, range, r, periodic);
         },
         prop_types)(pos);
}

using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("geometric", &geometric);
 });
