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

// based on code written by Alexandre Hannud Abdo <abdo@member.fsf.org>

#include "graph_filtering.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_extended_clustering.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

void extended_clustering(GraphInterface& g, boost::python::list aprops)
{
    std::any aprop = boost::python::extract<std::any>(aprops[0])();

    run_action<>(false)
        (g,
         [&](auto&& graph, auto&& prop)
         {
             typedef std::remove_reference_t<decltype(prop)> prop_t;
             std::vector<prop_t> props({prop});

             for (int i = 1; i < boost::python::len(aprops); ++i)
                 props.push_back(std::any_cast<typename prop_t::checked_t>(boost::python::extract<std::any>(aprops[i])()).get_unchecked());

             GILRelease gil_release;
             return get_extended_clustering()
                 (graph, get(vertex_index_t(), graph), props);
         },
         writable_vertex_scalar_properties)(aprop);
}

#define __MOD__ clustering
#include "module_registry.hh"
REGISTER_MOD
([]
 {
    def("extended_clustering", &extended_clustering);
 });
