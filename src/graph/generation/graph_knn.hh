// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2023 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_KNN_HH
#define GRAPH_KNN_HH

#include <tuple>
#include <random>
#include <list>
#include <shared_mutex>
#include <boost/functional/hash.hpp>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "parallel_rng.hh"
#include "idx_map.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

#include "graph_contract_edges.hh"
#include "shared_heap.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <bool directed, bool parallel, class Dist>
class DistCache
{
public:
    DistCache(Dist&& d, size_t N)
        :_d(d), _cache(N), _mutex(parallel ? N : 0)

    {}

    double operator()(size_t u, size_t v)
    {
        if (!directed && (u > v))
            std::swap(u, v);

        auto& cache = _cache[v];

        double d;
        if constexpr (parallel)
        {
            auto& mutex = _mutex[v];
            {
                std::shared_lock lock(mutex);
                auto iter = cache.find(u);
                if (iter != cache.end())
                    return iter->second;
            }

            d = _d(u, v);

            #pragma omp atomic
            _miss_count++;

            {
                std::unique_lock lock(mutex);
                cache[u] = d;
            }
        }
        else
        {
            auto iter = cache.find(u);
            if (iter != cache.end())
                return iter->second;
            d = _d(u, v);
            _miss_count++;
            cache[u] = d;
        }

        return d;
    }

    size_t _miss_count = 0;

private:
    Dist& _d;
    std::vector<gt_hash_map<size_t, double>> _cache;
    std::vector<std::shared_mutex> _mutex;
};

template <bool directed, bool parallel, class Dist>
DistCache<directed, parallel, Dist> make_dist_cache(Dist&& d, size_t N)
{
    return DistCache<directed, parallel, Dist>(d, N);
}

template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
size_t gen_knn(Graph& g, Dist&& d, size_t k, double r, size_t max_rk,
               double epsilon, Weight eweight, Hint& hint, bool verbose, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double>>>
        B(num_vertices(g));

    std::vector<size_t> vs, vs_;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    vs_ = vs;

    idx_set<size_t> seen;

    size_t n_tot = 0;

    if (verbose)
        cout << "random init" << endl;

    #pragma omp parallel if (parallel) firstprivate(vs_) firstprivate(seen) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& rng = prng.get(rng_);
             seen.clear();
             auto& Bv = B[v];
             for (auto u : random_permutation_range(vs_, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(u, v);
                 ++n_tot;
                 Bv.emplace_back(u, l);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 seen.insert(u);
                 if (Bv.size() == k)
                     break;
             }
             for (auto u : all_neighbors_range(v, hint))
             {
                 if (seen.find(u) != seen.end() || u == v)
                     continue;
                 double l = d(u, v);
                 ++n_tot;
                 if (l < get<1>(Bv.front()))
                 {
                     std::pop_heap(Bv.begin(), Bv.end(), cmp);
                     Bv.back() = {u, l};
                     std::push_heap(Bv.begin(), Bv.end(), cmp);
                 }
             }
         });

    auto build_vertex = [&](auto v)
        {
            for (auto& [u, l] : B[v])
            {
                auto e = add_edge(u, v, g).first;
                eweight[e] = l;
            }
        };

    std::vector<std::vector<size_t>> out_neighbors(num_vertices(g));
    std::vector<std::vector<size_t>> hint_neighbors(num_vertices(hint));

    std::bernoulli_distribution rsample(r);

    size_t iter = 1;
    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        for (size_t v : vs)
            clear_vertex(v, g);
        for (auto v : vs)
            build_vertex(v);

        #pragma omp parallel if (parallel)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);
                 auto& us = out_neighbors[v];
                 auto& hus = hint_neighbors[v];
                 us.clear();
                 for (auto u : out_neighbors_range(v, g))
                     us.push_back(u);
                 for (auto u : all_neighbors_range(v, hint))
                     hus.push_back(u);
                 if (max_rk < us.size())
                 {
                     size_t i = 0;
                     for ([[maybe_unused]] auto u : random_permutation_range(us, rng))
                     {
                         if (++i == max_rk)
                             break;
                     }
                     us.erase(us.begin() + max_rk, us.end());
                 }

                 if (max_rk < hus.size())
                 {
                     size_t i = 0;
                     for ([[maybe_unused]] auto u : random_permutation_range(hus, rng))
                     {
                         if (++i == max_rk)
                             break;
                     }
                     hus.erase(hus.begin() + max_rk, hus.end());
                 }
             });

        if (verbose)
            cout << "update neighbors" << endl;

        size_t c = 0;
        size_t n = 0;

        #pragma omp parallel if (parallel) firstprivate(seen) reduction(+:c,n,n_tot)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);

                 auto& Bv = B[v];

                 seen.clear();
                 for (auto& [v, l] : Bv)
                     seen.insert(v);

                 auto update =
                     [&](size_t u, size_t w)
                     {
                         if (u == w || w == v)
                             return;

                         auto it = seen.find(w);
                         if (it != seen.end())
                             return;

                         if (!rsample(rng))
                             return;

                         double l = d(w, v);
                         ++n_tot;

                         if (l < get<1>(Bv.front()))
                         {
                             std::pop_heap(Bv.begin(), Bv.end(), cmp);
                             Bv.back() = {w, l};
                             std::push_heap(Bv.begin(), Bv.end(), cmp);
                             c++;
                         }

                         seen.insert(w);
                         n++;
                     };

                 for (auto u : all_neighbors_range(v, hint))
                 {
                     update(v, u);
                     for (auto w : hint_neighbors[u])
                         update(u, w);
                 }

                 for (auto u : in_neighbors_range(v, g))
                 {
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }

                 for (auto u : out_neighbors[v])
                 {
                     update(v, u);
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }
             });

        delta = c / double(vs.size() * k);

        if (verbose)
            cout << iter++ << " " << delta << " " << c << " " << n
                 << " " << n_tot << endl;
    }

    for (size_t v : vs)
        clear_vertex(v, g);
    for (auto v : vs)
        build_vertex(v);
    return n_tot;
}


template <bool parallel, class Graph, class Dist, class Weight, class RNG>
size_t gen_knn_local(Graph& g, Dist&& d, size_t k, double r, double epsilon,
                   Weight eweight, bool verbose, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double, bool>>>
        B(num_vertices(g));
    std::vector<gt_hash_set<size_t>> Bset(num_vertices(g));
    vector<gt_hash_set<size_t>> all_visited(num_vertices(g));

    std::vector<size_t> vs, vs_;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    vs_ = vs;

    if (verbose)
        cout << "random init" << endl;

    size_t n_tot = 0;

    #pragma omp parallel if (parallel) firstprivate(vs_) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& rng = prng.get(rng_);
             auto& visited = all_visited[v];
             for (auto u : random_permutation_range(vs_, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(u, v);
                 ++n_tot;
                 auto& Bv = B[v];
                 Bv.emplace_back(u, l, true);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 Bset[v].insert(u);
                 visited.insert(u);
                 if (Bv.size() == k)
                     break;
             }
         });

    std::vector<std::shared_mutex> mutex(num_vertices(g));

    std::vector<std::vector<size_t>> vnew(num_vertices(g));
    std::vector<std::vector<size_t>> rvnew(num_vertices(g));
    std::vector<std::vector<size_t>> vold(num_vertices(g));
    std::vector<std::vector<size_t>> rvold(num_vertices(g));

    idx_set<size_t, false, false> visited(num_vertices(g));
    std::bernoulli_distribution rsample(std::min(r, 1.));

    size_t iter = 0;
    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        parallel_loop(vs,
                      [&](auto, auto v)
                      {
                          auto& rng = prng.get(rng_);

                          vnew[v].clear();
                          vold[v].clear();
                          rvnew[v].clear();
                          rvold[v].clear();
                          for (auto& [u, l, m] : B[v])
                          {
                              if (m)
                              {
                                  if (rsample(rng))
                                  {
                                      vnew[v].push_back(u);
                                      m = false;
                                  }
                              }
                              else
                              {
                                  vold[v].push_back(u);
                              }
                          }
                      }, vs.size());

        for (auto v : vs)
        {
            for (auto u : vnew[v])
                rvnew[u].push_back(v);
            for (auto u : vold[v])
                rvold[u].push_back(v);
        }

        if (verbose)
            cout << "update neighbors (local)" << endl;

        size_t rK = std::max(size_t(ceil(r * k)), size_t(1));
        size_t c = 0;
        size_t n = 0;

        #pragma omp parallel if (parallel) firstprivate(visited) reduction(+:c,n,n_tot)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);

                 visited.clear();
                 for (auto u : vnew[v])
                     visited.insert(u);
                 size_t i = 0;
                 for (auto u : random_permutation_range(rvnew[v], rng))
                 {
                     if (visited.find(u) != visited.end())
                         continue;
                     visited.insert(u);
                     vnew[v].push_back(u);
                     if (++i == rK)
                         break;
                 }

                 visited.clear();
                 for (auto u : vold[v])
                     visited.insert(u);
                 i = 0;
                 for (auto u : random_permutation_range(rvold[v], rng))
                 {
                     if (visited.find(u) != visited.end())
                         continue;
                     visited.insert(u);
                     vold[v].push_back(u);
                     if (++i == rK)
                         break;
                 }

                 auto update_heap =
                     [&](auto s, auto l, auto& Bt, auto& Bst)
                     {
                         if (l < get<1>(Bt.front()))
                         {
                             if (Bst.find(s) != Bst.end())
                                 return;
                             Bst.erase(get<0>(Bt.front()));
                             Bst.insert(s);
                             std::pop_heap(Bt.begin(), Bt.end(), cmp);
                             Bt.back() = {s, l, true};
                             std::push_heap(Bt.begin(), Bt.end(), cmp);
                             c++;
                         }
                     };

                 auto update_pair =
                     [&](auto s, auto t)
                     {
                         auto& Bt = B[t];
                         auto& Bst = Bset[t];
                         auto& vis = all_visited[t];
                         n++;

                         auto do_ulocked =
                             [&](auto&& f)
                             {
                                 if constexpr (parallel)
                                 {
                                     std::unique_lock lock(mutex[t]);
                                     f();
                                 }
                                 else
                                 {
                                     f();
                                 }
                             };

                         auto do_slocked =
                             [&](auto&& f)
                             {
                                 if constexpr (parallel)
                                 {
                                     std::shared_lock lock(mutex[t]);
                                     return f();
                                 }
                                 else
                                 {
                                     return f();
                                 }
                             };

                         if (do_slocked([&]()
                                        {
                                            if (vis.find(s) != vis.end())
                                                return true;
                                            return false;
                                        }))
                             return;

                         double l = d(s, t);
                         ++n_tot;

                         do_ulocked([&]()
                                    {
                                        update_heap(s, l, Bt, Bst);
                                        vis.insert(s);
                                    });
                     };

                 for (auto s : vnew[v])
                 {
                     for (auto t : vnew[v])
                     {
                        if (t >= s)
                            continue;
                         update_pair(s, t);
                         update_pair(t, s);
                     }
                 }

                 for (auto s : vnew[v])
                 {
                     for (auto t : vold[v])
                     {
                         if (t == s)
                             continue;
                         update_pair(s, t);
                         update_pair(t, s);
                     }
                 }
             });

        delta = c / double(vs.size() * k);

        if (verbose)
            cout << iter++ << " " << delta << " " << c << " " << n << " "
                 << n_tot << endl;
    }

    for (size_t v : vs)
        clear_vertex(v, g);
    for (auto v : vs)
    {
        for (auto& [u, l, m] : B[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = l;
        }
    }

    return n_tot;
}

template <bool parallel, class Graph, class Dist, class Weight>
size_t gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight)
{
    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    std::vector<std::vector<std::tuple<size_t, double>>> us(num_vertices(g));
    size_t n_tot = 0;

    #pragma omp parallel if (parallel) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& ns = us[v];
             for (size_t u : vertices_range(g))
             {
                 if (u == v)
                     continue;
                 ns.emplace_back(u, d(u, v));
                 ++n_tot;
             }
             if (ns.size() <= k)
                 return;
             nth_element(ns.begin(),
                         ns.begin() + k,
                         ns.end(),
                         [](auto& x, auto& y)
                         {
                             return get<1>(x) < get<1>(y);
                         });
             ns.resize(k);
             ns.shrink_to_fit();
         });

    for (auto v : vs)
    {
        for (auto& [u, w] : us[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = w;
        }
    }
    return n_tot;
}

template <bool parallel, class Graph, class Dist, class Weight>
size_t gen_k_nearest_exact(Graph& g, Dist&& d, size_t k, bool directed,
                           Weight eweight)
{
    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, k,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);

    size_t n_tot = 0;

    #pragma omp parallel if (parallel) firstprivate(heap) reduction(+:n_tot)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             for (auto u : vs)
             {
                 if (u == v)
                     continue;
                 if (!directed && u > v)
                     continue;
                 auto l = d(u, v);
                 heap.push({{u, v}, l});
                 ++n_tot;
             }
         });

    heap.merge();

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }

    return n_tot;
}

template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
size_t gen_k_nearest(Graph& g, Dist&& d, size_t m, size_t max_rk,
                     double epsilon, Weight eweight, Hint& hint,
                     bool directed, bool verbose, RNG& rng_)
{
    auto get_st =
        [&](size_t s, size_t t)
        {
            if (!directed && s > t)
                return std::make_tuple(t, s);
            return make_tuple(s, t);
        };

    size_t N = num_vertices(g);

    auto sample_st =
        [&](auto& rng_)
        {
            std::uniform_int_distribution<size_t> sample_s(0, N-1), sample_t(0, N-2);
            size_t s = sample_s(rng_);
            size_t t = sample_t(rng_);
            if (s == t)
                t = N - 1;
            return get_st(s, t);
        };

    std::vector<std::tuple<size_t,size_t,double>> edges(m), new_edges(m);
    gt_hash_set<std::tuple<size_t,size_t>> eset, new_eset;

    if (verbose)
        cout << "setup" << endl;

    for (size_t i = 0; i < m; ++i)
    {
        size_t s, t;
        do
        {
            std::tie(s, t) = sample_st(rng_);
        }
        while (eset.find({s,t}) != eset.end());
        eset.insert({s, t});
        edges[i] = {s, t, 0};
    }

    size_t n_tot = 0;

    #pragma omp parallel for if (parallel) reduction(+:n_tot)
    for (size_t i = 0; i < m; ++i)
    {
        auto& [s, t, l] = edges[i];
        l = d(s, t);
        n_tot++;
    }

    auto build_graph =
        [&]()
        {
            for (auto v : vertices_range(g))
                clear_vertex(v, g);
            for (auto& [s, t, l] : edges)
            {
                auto e = add_edge(s, t, g).first;
                eweight[e] = l;
            }
        };


    size_t iter = 0;
    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        build_graph();

        if (max_rk > 0)
        {
            for (auto v : vertices_range(g))
            {
                std::vector<std::tuple<size_t, double>> ein, eout;
                for (auto e : out_edges_range(v, g))
                    eout.emplace_back(target(e, g), eweight[e]);
                for (auto e : in_edges_range(v, g))
                    ein.emplace_back(source(e, g), eweight[e]);
                clear_vertex(v, g);

                size_t i = 0;
                for (auto& [u, l] : random_permutation_range(eout, rng_))
                {
                    auto e = add_edge(u, v, g).first;
                    eweight[e] = l;
                    if (++i == max_rk)
                        break;
                }

                i = 0;
                for (auto& [u, l] : random_permutation_range(ein, rng_))
                {
                    auto e = add_edge(v, u, g).first;
                    eweight[e] = l;
                    if (++i == max_rk)
                        break;
                }
            }
        }

        if (verbose)
            cout << "find new edges" << endl;

        size_t c = 0;

        parallel_rng<rng_t> prng(rng_);

        #pragma omp parallel for if (parallel) reduction(+:c, n_tot)
        for (size_t i = 0; i < m; ++i)
        {
            auto& [s, t, le] = edges[i];
            auto best = edges[i];

            auto check_edge =
                [&](auto s_, auto t_)
                {
                    if (s_ == t_)
                        return;

                    std::tie(s_, t_) = get_st(s_, t_);

                    if (eset.find({s_, t_}) != eset.end())
                        return;

                    auto l = d(s_, t_);

                    if (l < get<2>(best))
                        best = {s_, t_, l};

                    n_tot++;
                };

            auto check_incident =
                [&](auto u, auto v, auto& g)
                {
                    for (auto w : all_neighbors_range(u, g))
                    {
                        check_edge(w, v);
                        if (directed)
                            check_edge(v, w);

                        // for (auto w2 : all_neighbors_range(w, g))
                        // {
                        //     check_edge(w2, v);
                        //     if (directed)
                        //         check_edge(v, w2);

                        //     for (auto w3 : all_neighbors_range(w, g))
                        //     {
                        //         check_edge(w2, w3);
                        //         if (directed)
                        //             check_edge(w3, w2);
                        //     }
                        // }

                        // for (auto w2 : all_neighbors_range(u, g))
                        // {
                        //     check_edge(w2, w);
                        //     if (directed)
                        //         check_edge(w, w2);
                        // }
                    }
                };

            check_incident(s, t, g);
            check_incident(t, s, g);

            check_incident(s, s, hint);
            check_incident(t, t, hint);

            if (directed)
                check_edge(t, s);


            auto [s_, t_, l_] = best;
            #pragma omp critical (eset)
            {
                assert(best == edges[i] || eset.find({s_, t_}) == eset.end());
                auto iter = new_eset.find({s_, t_});
                if (iter != new_eset.end())
                {
                    best = edges[i];
                    new_eset.insert({s, t});
                }
                else
                {
                    new_eset.insert({s_, t_});
                }
            }

            if (best != edges[i])
                c++;
            new_edges[i] = best;
        }

        edges.swap(new_edges);
        eset.swap(new_eset);

        delta = c / double(m);

        if (verbose)
            cout << iter++ << " " << delta << " " << c << " " << m
                 << " " << n_tot << endl;
    }

    build_graph();

    return n_tot;
}

template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(){}
    MaskFilter(DescriptorProperty& filtered_property)
        : _filtered_property(&filtered_property) {}

    template <class Descriptor>
    inline bool operator() (Descriptor&& d) const
    {
        return get(*_filtered_property, d);
    }

    DescriptorProperty& get_filter() { return *_filtered_property; }
    constexpr bool is_inverted() { return false; }

private:
    DescriptorProperty* _filtered_property;
};


template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
size_t gen_k_nearest_knn(Graph& g, Dist&& d, size_t m, double r, size_t max_rk,
                         double epsilon, Weight eweight, Hint& hint, bool local,
                         bool directed, bool verbose, RNG& rng)
{
    size_t N = num_vertices(g);
    std::vector<bool> select(N, true);
    typename eprop_map_t<bool>::type::unchecked_t eselect(get(edge_index_t(), g));

    auto u = make_filt_graph(g, MaskFilter<decltype(eselect)>(eselect),
                             [&](auto v) { return select[v]; });

    auto uhint = make_filt_graph(hint, keep_all(),
                                 [&](auto v) { return select[v]; });

    size_t n_tot = 0;
    size_t iter = 0;
    while (N > 1)
    {
        if (verbose)
            cout << "m = " << m <<  " nearest iteration: " << iter++ << endl;

        if (N * N <= 4 * m)
        {
            if (verbose)
                cout << "Running exact m nearest with N = " << N << endl;
            n_tot += gen_k_nearest_exact<parallel>(u, d, m, directed, eweight);
            break;
        }

        size_t nk = ceil((4. * m)/N);

        eselect.reserve(N * nk);

        if (verbose)
            cout << "Running KNN with N = " << N << " and k = " << nk << endl;
        if (local)
            n_tot += gen_knn_local<parallel>(u, d, nk, r, epsilon, eweight, verbose, rng);
        else
            n_tot += gen_knn<parallel>(u, d, nk, r, max_rk, epsilon, eweight, uhint, verbose, rng);

        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        std::vector<std::tuple<edge_t, double>> medges;

        // 2m shortest directed pairs
        auto heap = make_shared_heap(medges, 2 * m,
                                     [](auto& x, auto& y)
                                     {
                                         return get<1>(x) < get<1>(y);
                                     });
        if (verbose)
            cout << "Keeping 2m = " << 2 * m << " of "
                 << nk * N << " closest directed pairs..." << endl;

        #pragma omp parallel if (parallel) firstprivate(heap)
        parallel_edge_loop_no_spawn(u,
                                    [&](auto& e)
                                    { heap.push({e, eweight[e]}); });
        heap.merge();

        if (verbose)
            cout << "heap size: " << medges.size()
                 << ", top: " << get<1>(medges.front()) << endl;

        if (verbose)
            cout << "Selecting nodes..." << endl;

        typename eprop_map_t<bool>::type ekeep(get(edge_index_t(), g));
        for (auto& el : medges)
            ekeep[get<0>(el)] = true;

        N = 0;
        std::vector<bool> nselect(select);
        #pragma omp parallel if (parallel) reduction(+:N)
        parallel_vertex_loop_no_spawn
            (u,
             [&](auto v)
             {
                 nselect[v] = true;
                 for (auto e : in_edges_range(v, u))
                 {
                     if (!ekeep[e])
                     {
                         nselect[v] = false;
                         break;
                     }
                 }
                 if (nselect[v])
                     N++;
             });

        select = nselect;
        if (N > 1)
        {
            #pragma omp parallel if (parallel)
            parallel_edge_loop_no_spawn(u, [&](auto& e){ eselect[e] = false; });
        }
    };

    if (verbose)
        cout << "Removing parallel edges..." << endl;

    if (!directed)
    {
        undirected_adaptor g_u(g);
        contract_parallel_edges(g_u, dummy_property_map());
    }
    else
    {
        contract_parallel_edges(g, dummy_property_map());
    }

    if (verbose)
        cout << "Selecting best m = " << m << " out of "
             << num_edges(g) << " edges..." << endl;

    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, m,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    #pragma omp parallel if (parallel) firstprivate(heap)
    parallel_edge_loop_no_spawn
        (g,
         [&](auto& e)
         {
             size_t u = source(e, g);
             size_t v = target(e, g);
             if (!directed && u > v)
                 std::swap(u, v);
             auto l = eweight[e];
             heap.push({{u, v}, l});
         });
    heap.merge();

    if (verbose)
        cout << "E = " << pairs.size()
             << ", top: " << get<1>(pairs.front()) << endl;

    if (verbose)
        cout << "Building graph..." << endl;

    for (auto v : vertices_range(g))
        clear_vertex(v, g);

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }

    return n_tot;
}

} // graph_tool namespace

#endif // GRAPH_KNN_HH
