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

#include "merge.hh"

using namespace graph_tool;
using namespace boost;

boost::python::tuple graph_merge_dispatch_imp(GraphInterface& ugi,
                                              GraphInterface& gi,
                                              std::any avmap, std::any aw1,
                                              std::any aw2, bool multiset,
                                              bool diff, bool sym_diff,
                                              bool intersect, bool symmetric,
                                              bool simple);

boost::python::tuple graph_merge_dispatch(GraphInterface& ugi, GraphInterface& gi,
                                          std::any avmap, std::any aw1,
                                          std::any aw2, bool multiset,
                                          bool diff, bool sym_diff, bool intersect,
                                          bool symmetric, bool simple)
{
    if (!ugi.get_directed())
        return graph_merge_dispatch_imp(ugi, gi, avmap, aw1, aw2, multiset,
                                        diff, sym_diff, intersect, symmetric,
                                        simple);

    // the same choice of view as GraphInterface::get_graph_view()
    bool reversed = ugi.get_reversed();
    bool filtered = (ugi.is_vertex_filter_active() ||
                     ugi.is_edge_filter_active());
    auto f = (reversed ?
              (filtered ? graph_merge_dispatch_dir<true, true> :
                          graph_merge_dispatch_dir<true, false>) :
              (filtered ? graph_merge_dispatch_dir<false, true> :
                          graph_merge_dispatch_dir<false, false>));
    return f(ugi, gi, avmap, aw1, aw2, multiset, diff, sym_diff, intersect,
             symmetric, simple);
}

#include <boost/python.hpp>
using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     enum_<merge_t>("merge_t")
        .value("set", merge_t::set)
        .value("sum", merge_t::sum)
        .value("diff", merge_t::diff)
        .value("idx_inc", merge_t::idx_inc)
        .value("append", merge_t::append)
        .value("concat", merge_t::concat);

     def("graph_merge", &graph_merge_dispatch);
 });
