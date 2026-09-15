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

#include "graph_tool.hh"
#include "../support/util.hh"

#include "uncertain.hh"

using namespace boost;
using namespace graph_tool;

double marginal_count_entropy(GraphInterface& gi, std::any aexc, std::any aeh)
{
    typedef eprop_map_t<double> ehmap_t;
    auto eh = std::any_cast<ehmap_t>(aeh);

    double S_tot = 0;
    gt_dispatch<>()
        ([&](auto& g, auto exc)
         {
             parallel_edge_loop
                 (g,
                  [&](auto& e)
                  {
                      auto& S = eh[e];
                      S = 0;
                      size_t N = 0;
                      for (auto n : exc[e])
                      {
                          S -= xlogx_fast(int64_t(n));
                          N += n;
                      }
                      if (N == 0)
                          return;
                      S /= N;
                      S += safelog_fast(N);

                      #pragma omp atomic
                      S_tot += S;
                  });
         },
        all_graph_views, scalar_vector_edge_properties)
        (gi.get_graph_view(), aexc);
    return S_tot;
}

void marginal_graph_sample(GraphInterface& gi, std::any ap,
                           std::any ax, rng_t& rng_)
{
    gt_dispatch<>()
        ([&](auto& g, auto p, auto x)
         {
             parallel_edge_loop_rng
                 (g,
                  [&](auto& e, auto, auto& rng)
                  {
                      std::bernoulli_distribution sample(get(p, e));
                      put(x, e, sample(rng));
                  }, rng_);
         },
         all_graph_views,
         hana::tuple_t<eprop_map_t<double>, edprop_map_t<double>>,
         hana::tuple_t<eprop_map_t<int32_t>, edprop_map_t<int32_t>>)
        (gi.get_graph_view(),
         eprop_map_as_dynamic(ap, eprop_map_t<double>()),
         eprop_map_as_dynamic(ax, eprop_map_t<int32_t>()));
}

double marginal_graph_lprob(GraphInterface& gi, std::any ap,
                            std::any ax)
{
    double L = 0;
    gt_dispatch<>()
        ([&](auto& g, auto p, auto x)
         {
             for (auto e : edges_range(g))
             {
                 if (x[e] == 1)
                     L += std::log(p[e]);
                 else
                     L += std::log1p(-p[e]);
             }
         },
         all_graph_views, scalar_edge_properties,
         scalar_edge_properties)
        (gi.get_graph_view(), ap, ax);
    return L;
}

#include <boost/python.hpp>

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("marginal_count_entropy", &marginal_count_entropy);
    def("marginal_graph_sample", &marginal_graph_sample);
    def("marginal_graph_lprob", &marginal_graph_lprob);
}, MODULE_PRIO);
