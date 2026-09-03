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
#include "graph_properties.hh"

#include "graph_properties_copy.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void copy_external_edge_property(const GraphInterface& src_g,
                                 const GraphInterface& tgt_g,
                                 std::any prop_src,
                                 std::any prop_tgt)
{
    gt_dispatch<{.gil_release=false}>()
        ([&](auto& src, auto& tgt, auto tgt_map, auto src_map)
         {
             copy_external_edge_property_dispatch(src, tgt, src_map, tgt_map);
         },
         all_graph_views,
         all_graph_views,
         writable_edge_properties,
         hana::concat(eprop_same_type, dynamic_prop_type))
         (src_g.get_graph_view(), tgt_g.get_graph_view(), prop_tgt,
          eprop_map_as_dynamic(prop_src, prop_tgt));
}
