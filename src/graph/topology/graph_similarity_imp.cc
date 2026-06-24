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


#include "graph_python_interface.hh"

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_properties.hh"
#include "graph_selectors.hh"

#include "graph_similarity.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

python::object similarity_fast(GraphInterface& gi1, GraphInterface& gi2,
                               std::any weight1, std::any weight2,
                               std::any label1, std::any label2,
                               double norm, bool asym)
{
    typedef UnityPropertyMap<size_t,GraphInterface::edge_t> ecmap_t;
    auto weight_props_t = hana::append(scalar_edge_properties,
                                       hana::type<ecmap_t>());

    if (!weight1.has_value())
        weight1 = ecmap_t();
    if (!weight2.has_value())
        weight2 = ecmap_t();
    python::object s;
    gt_dispatch<{.gil_release=false}>()
        ([&](const auto& g1, const auto& g2, auto ew1, auto ew2, auto l1, auto l2)
         {
             GILRelease gil_release;
             auto ret = get_similarity_fast(g1, g2, ew1, get_unchecked(ew2),
                                            l1, get_unchecked(l2), norm, asym);
             gil_release.restore();
             s = python::object(ret);
         },
         all_graph_views,
         all_graph_views,
         weight_props_t,
         same_arg_type,
         hana::tuple_t<vprop_map_t<int64_t>, vdprop_map_t<int64_t>>,
         same_arg_type)
        (gi1.get_graph_view(), gi2.get_graph_view(), weight1, weight2,
         vprop_map_as_dynamic(label1, vprop_map_t<int64_t>()),
         vprop_map_as_dynamic(label2, vprop_map_t<int64_t>()));
    return s;
}
