// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
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
#include "uncertain_marginal.hh"
#include "parallel_rng.hh"

using namespace boost;
using namespace graph_tool;

void marginal_multigraph_sample(GraphInterface& gi, std::any axs, std::any axc,
                                std::any ax, rng_t& rng_)
{
    gt_dispatch<>()
        ([&](auto& g, auto xs, auto xc, auto x)
         {
             parallel_rng<rng_t> prng(rng_);
             parallel_edge_loop
                 (g,
                  [&](auto& e)
                  {
                      typedef typename property_traits<decltype(xs)>::value_type::value_type val_t;
                      Sampler<val_t> sample(get(xs, e), get(xc, e));

                      auto& rng = prng.get(rng_);
                      put(x, e, sample.sample(rng));
                  });
         },
         all_graph_views,
         hana::tuple_t<eprop_map_t<std::vector<int32_t>>, edprop_map_t<std::vector<int32_t>>>,
         hana::tuple_t<eprop_map_t<std::vector<double>>, edprop_map_t<std::vector<double>>>,
         hana::tuple_t<eprop_map_t<int32_t>, edprop_map_t<int32_t>>)
        (gi.get_graph_view(),
         eprop_map_as_dynamic(axs, eprop_map_t<std::vector<int32_t>>()),
         eprop_map_as_dynamic(axc, eprop_map_t<std::vector<double>>()),
         eprop_map_as_dynamic(ax, eprop_map_t<int32_t>()));
}

double marginal_multigraph_lprob(GraphInterface& gi, std::any axs, std::any axc,
                                 std::any ax)
{
    double L = 0;
    gt_dispatch<>()
        ([&](auto& g, auto xs, auto xc, auto x)
         {
             #pragma omp parallel reduction(+:L)
             parallel_edge_loop_no_spawn
                 (g,
                  [&](auto e)
                  {
                      size_t Z = 0;
                      size_t p = 0;
                      const auto& xse = get(xs, e);
                      const auto& xce = get(xc, e);
                      for (size_t i = 0; i < xse.size(); ++i)
                      {
                          size_t m = xse[i];
                          if (m == size_t(get(x, e)))
                              p = xce[i];
                          Z += xce[i];
                      }
                      if (p == 0)
                      {
                          L += -numeric_limits<double>::infinity();
                          return;
                      }
                      L += std::log(p) - std::log(Z);
                  });
         },
         all_graph_views,
         hana::tuple_t<eprop_map_t<std::vector<int32_t>>, edprop_map_t<std::vector<int32_t>>>,
         hana::tuple_t<eprop_map_t<std::vector<int32_t>>, edprop_map_t<std::vector<int32_t>>>,
         hana::tuple_t<eprop_map_t<int32_t>, edprop_map_t<int32_t>>)
        (gi.get_graph_view(),
         eprop_map_as_dynamic(axs, eprop_map_t<std::vector<int32_t>>()),
         eprop_map_as_dynamic(axc, eprop_map_t<std::vector<int32_t>>()),
         eprop_map_as_dynamic(ax, eprop_map_t<int32_t>()));
    return L;
}

#include <boost/python.hpp>

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("marginal_multigraph_sample", &marginal_multigraph_sample);
    def("marginal_multigraph_lprob", &marginal_multigraph_lprob);
});
