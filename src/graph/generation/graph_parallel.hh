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

#ifndef GRAPH_PARALLEL_HH
#define GRAPH_PARALLEL_HH

#include "hash_map_wrap.hh"
#include "graph_util.hh"
#include "parallel_util.hh"
#include <boost/scope/scope_exit.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

// label parallel edges in the order they are found, starting from 1
template <class Graph, class ParallelMap>
void label_parallel_edges(const Graph& g, ParallelMap parallel, bool mark_only)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    typename property_map<Graph, edge_index_t>::type eidx = get(edge_index, g);

    gt_hash_map<vertex_t, edge_t> vset;
    gt_hash_map<size_t, bool> self_loops;

    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        firstprivate(vset) firstprivate(self_loops)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             for (auto e : out_edges_range(v, g))
             {
                 vertex_t u = target(e, g);

                 // do not visit edges twice in undirected graphs
                 if (!is_directed(g) && u < v)
                     continue;

                 if (u == v)
                 {
                     if (self_loops[eidx[e]])
                         continue;
                     self_loops[eidx[e]] = true;
                 }

                 auto iter = vset.find(u);
                 if (iter == vset.end())
                 {
                     vset[u] = e;
                 }
                 else
                 {
                     if (mark_only)
                     {
                         parallel[e] = true;
                     }
                     else
                     {
                         parallel[e] = parallel[iter->second] + 1;
                         iter->second = e;
                     }
                 }
             }
             vset.clear();
             self_loops.clear();
         });
}

// label self loops edges in the order they are found, starting from 1
template <class Graph, class SelfMap>
void label_self_loops(const Graph& g, SelfMap self, bool mark_only)
{
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
             size_t n = 1;
             for (auto e : out_edges_range(v, g))
             {
                 if (target(e, g) == v)
                     put(self, e, mark_only ? 1 : n++);
                 else
                     put(self, e, 0);
             }
         });
};

// remove edges with label larger than 0
template <class Graph, class LabelMap>
void remove_labeled_edges(Graph& g, LabelMap label)
{
    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    vector<edge_t> r_edges;

    base_graph(g).set_concurrent(true);
    auto lguard = boost::scope::make_scope_exit([&](){ base_graph(g).set_concurrent(false); });

    std::vector<std::shared_mutex> vmutex(num_vertices(g));

    #pragma omp parallel firstprivate(r_edges)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             do_slock
                 ([&]()
                  {
                      for (auto e : out_edges_range(v, g))
                      {
                          if constexpr (!is_directed_v<Graph>)
                          {
                              auto u = target(e, g);
                              if (u < v)
                                  continue;
                          }

                          if (label[e] > 0)
                          {
                              r_edges.push_back(e);
                              label[e] = 0;
                          }
                      }

                  }, vmutex[v]);

             for (auto& e : r_edges)
             {
                auto u = target(e, g);
                do_ulock_pair
                    ([&]()
                     {
                         remove_edge(e, g);
                     },
                     v, u, vmutex[v], vmutex[u]);
             }

             r_edges.clear();
         });
}

} // graph_tool namespace

#endif //GRAPH_PARALLEL_HH
