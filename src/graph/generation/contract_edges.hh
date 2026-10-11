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

#ifndef CONTRACT_EDGES_HH
#define CONTRACT_EDGES_HH

#include "graph.hh"
#include "graph_util.hh"
#include "random.hh"
#include "samplers.hh"
#include "idx_map.hh"

#include "parallel_util.hh"
#include <boost/scope/scope_exit.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph, class EWeight>
void contract_parallel_edges(Graph& g, EWeight eweight)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    gt_hash_map<size_t, edge_t> emap;
    gt_hash_set<size_t> loops;
    auto eidx = get(edge_index_t(), g);
    std::vector<edge_t> remove;

    base_graph(g).set_concurrent(true);
    auto lguard = boost::scope::make_scope_exit([&](){ base_graph(g).set_concurrent(false); });

    std::vector<std::shared_mutex> vmutex(num_vertices(g));

    #pragma omp parallel firstprivate(emap, remove, loops)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
            emap.clear();
            remove.clear();
            loops.clear();

            do_slock([&]()
                     {
                         for (auto e : out_edges_range(v, g))
                         {
                             auto u = target(e, g);

                             if (!is_directed(g) && u < v)
                                continue;

                             auto iter = emap.find(u);
                             if (iter == emap.end())
                             {
                                 emap[u] = e;
                                 if (u == v)
                                     loops.insert(eidx[e]);
                             }
                             else
                             {
                                 if (loops.find(eidx[e]) != loops.end()) // self-loops
                                     continue;
                                 if constexpr (std::is_convertible_v<typename boost::property_traits<EWeight>::category,
                                               boost::writable_property_map_tag>)
                                 {
                                     auto w = eweight[iter->second];
                                     put(eweight, iter->second, eweight[e] + w);
                                 }
                                 remove.push_back(e);
                                 if (u == v)
                                     loops.insert(eidx[e]);
                             }
                         }
                     }, vmutex[v]);

            for (auto& e : remove)
            {
                auto u = target(e, g);
                do_ulock_pair
                    ([&]()
                     {
                         remove_edge(e, g);
                     },
                     v, u, vmutex[v], vmutex[u]);
            }
        });
}

template <class Graph, class EWeight>
void expand_parallel_edges(Graph& g, EWeight eweight)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    std::vector<edge_t> edges;
    idx_set<size_t> loops(num_vertices(g));
    auto eidx = get(edge_index_t(), g);

    base_graph(g).set_concurrent(true);
    auto lguard = boost::scope::make_scope_exit([&](){ base_graph(g).set_concurrent(false); });

    std::vector<std::shared_mutex> vmutex(num_vertices(g));

    #pragma omp parallel firstprivate(edges, loops)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             edges.clear();
             if (!is_directed(g))
                 loops.clear();

             do_slock
                 ([&]()
                  {
                      for (auto e : out_edges_range(v, g))
                      {
                          auto u = target(e, g);

                          if (!is_directed(g))
                          {
                              if (v > u)
                                  continue;
                              if (v == u && loops.find(eidx[e]) != loops.end()) // self-loops
                                  continue;
                          }

                          edges.push_back(e);

                          if (!is_directed(g) && u == v)
                              loops.insert(eidx[e]);
                      }
                  }, vmutex[v]);

             for (auto& e : edges)
             {
                 auto u = target(e, g);
                 size_t w = eweight[e];
                 do_slock_pair
                     ([&]()
                      {
                          if (w == 0)
                          {
                              remove_edge(e, g);
                          }
                          else
                          {
                              for (size_t m = 0; m < w - 1; ++m)
                                  add_edge(v, u, g);
                          }
                      }, v, u, vmutex[v], vmutex[u]);
             }
         });
}

} // graph_tool namespace

#endif // CONTRACT_EDGES_HH
