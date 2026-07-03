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
#include "graph_properties.hh"
#include "idx_map.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class Graph, class Weight>
double get_reciprocity(const Graph& g, Weight w, bool self_loops)
{
    typedef typename property_traits<Weight>::value_type wval_t;
    typedef promote_value_t<wval_t> val_t;
    val_t L = 0, Lbd = 0;

    idx_map<size_t, val_t> out_sum;

    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        reduction(+: L, Lbd) firstprivate(out_sum)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             for (auto e : out_edges_range(v, g))
             {
                 if (!self_loops && target(e, g) == v)
                     continue;
                 out_sum[target(e, g)] += w[e];
             }

             for (auto&& [u, w_u] : out_sum)
             {
                 val_t wr = 0;
                 for (auto e : out_edges_range(u, g))
                 {
                     if (target(e, g) == v)
                         wr += w[e];
                 }
                 Lbd += std::min(wr, w_u);
                 L += w_u;
             }

             out_sum.clear();
         });

    if constexpr (std::is_floating_point_v<val_t>)
        return Lbd / L;
    else
        return Lbd / double(L);
};

double reciprocity(GraphInterface& gi, std::any aw, bool self_loops)
{
    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    auto weight_props_t = hana::append(scalar_edge_properties,
                                       hana::type<weight_map_t>());

    if (!aw.has_value())
        aw = weight_map_t();

    double ret = 0;
    run_action<>()(gi,
                   [&](auto& g, auto w)
                   {
                       ret = get_reciprocity(g, w, self_loops);
                   }, weight_props_t)(aw);
    return ret;
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("reciprocity", &reciprocity);
 });
