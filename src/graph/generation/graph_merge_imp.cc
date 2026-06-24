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

#include "graph_merge.hh"

using namespace graph_tool;
using namespace boost;

boost::python::tuple graph_merge_dispatch_imp(GraphInterface& ugi, GraphInterface& gi,
                                              std::any avmap, std::any aw1,
                                              std::any aw2, bool multiset,
                                              bool diff, bool sym_diff, bool intersect,
                                              bool symmetric, bool simple)
{
    auto writable_weight_props =
        hana::append(writable_scalar_edge_properties,
                     hana::type<null_weight_map_t>());

    if (!aw1.has_value())
        aw1 = null_weight_map_t();

    if (!aw2.has_value())
        aw2 = null_weight_map_t();

    typedef eprop_map_t<GraphInterface::edge_t> emap_t;
    emap_t emap(gi.get_edge_index());
    emap.reserve(gi.get_edge_index_range());

    gt_dispatch<{.gil_release=false}>()
        ([&](auto& ug, auto& g, auto& vmap, auto& w1, auto& w2)
         {
             graph_merge(ug, g, vmap, emap, w1, w2,
                         multiset, diff, sym_diff,
                         intersect, symmetric,
                         simple, ugi.get_keep_ehash());
         },
         never_directed,
         never_directed,
         integer_vertex_properties,
         writable_weight_props,
         same_arg_type)
        (ugi.get_graph_view(), gi.get_graph_view(),
         avmap, aw1, aw2);

    return boost::python::make_tuple(avmap, std::any(emap));
}
