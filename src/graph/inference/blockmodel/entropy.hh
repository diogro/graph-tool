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

#pragma once

#include "../support/util.hh"

namespace graph_tool
{

// ====================
// Entropy calculation
// ====================

// Sparse entropy terms
// ====================

// microcanonical deg-corr entropy

template <bool directed, class Val, class Mval>
    requires std::is_scalar_v<Mval>
[[gnu::always_inline]] [[gnu::pure]]
inline double eterm_d(Val r, Val s, Mval mrs)
{
    assert(mrs >= 0);

    double val = lgamma_fast(mrs + 1);

    if constexpr (directed)
    {
        return -val;
    }
    else if (r != s)
    {
        return -val;
    }
    else
    {
#ifndef __clang__
        constexpr
#endif
        double log_2 = std::log(2);
        return -val - mrs * log_2;
    }
}

template <class Graph, class Val, class Mval>
    requires std::is_scalar_v<Mval>
[[gnu::always_inline]] [[gnu::pure]]
inline double eterm(Val r, Val s, Mval mrs, const Graph&)
{
    return eterm_d<is_directed_v<Graph>>(r, s, mrs);
}

template <bool directed, class Val>
    requires std::is_scalar_v<Val>
[[gnu::always_inline]] [[gnu::pure]]
inline double vterm_d(Val mrp, Val mrm, Val wr, bool deg_corr)
{
    if (deg_corr)
    {
        if constexpr (directed)
            return lgamma_fast(mrp + 1) + lgamma_fast(mrm + 1);
        else
            return lgamma_fast(mrp + 1);
    }
    else
    {
        if constexpr (directed)
            return (mrp + mrm) * safelog_fast(wr);
        else
            return mrp * safelog_fast(wr);
    }
}

template <class Graph, class Val>
    requires std::is_scalar_v<Val>
[[gnu::always_inline]] [[gnu::pure]]
inline double vterm(Val mrp, Val mrm, Val wr, bool deg_corr,
                    const Graph&)
{
    return vterm_d<is_directed_v<Graph>>(mrp, mrm, wr, deg_corr);
}

template <class W>
    requires std::is_scalar_v<W>
double deg_entropy(W k)
{
    return -lgamma_fast(k + 1);
}

template <class Graph, class W, class Delta>
    requires std::is_scalar_v<W>
double get_deg_entropy(const std::tuple<W, W>& degs, const Graph&,
                       const Delta& delta)
{
    auto [kin, kout] = degs;
    if constexpr (delta.size() > 0)
    {
        if constexpr (is_directed_v<Graph>)
            kin += delta[0];
        kout += delta[1];
    }
    if constexpr (is_directed_v<Graph>)
        return deg_entropy(kin) + deg_entropy(kout);
    else
        return deg_entropy(kout);
}

template <class W, class Graph>
double get_deg_entropy(const std::tuple<W, W>& degs, const Graph& g)
{
    if constexpr (std::is_scalar_v<W>)
        return get_deg_entropy(degs, g, std::array<W, 0>{});
    else
        return get_deg_entropy(degs, g, std::array<W, 2>{});
}

template <class V, class Val, class Graph>
    requires std::is_scalar_v<Val>
double parallel_term(V u, V v, Val m, const Graph&)
{
    double S = 0;
    if (m > 1)
    {
        if (u == v && !is_directed_v<Graph>)
        {
            assert(m % 2 == 0);
            S += lgamma_fast(m/2 + 1) + m * log(2) / 2;
        }
        else
        {
            S += lgamma_fast(m + 1);
        }
    }
    return S;
}

template <class Graph, class EW>
double get_parallel_entropy(const Graph& g, const EW& eweight)
{
    typedef typename property_traits<EW>::value_type weight_t;

    double S = 0;

    #pragma omp parallel reduction(+:S)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             gt_hash_map<decltype(v), weight_t> us;
             for (auto e : out_edges_range(v, g))
             {
                 auto u = target(e, g);
                 if (u < v && !is_directed_v<Graph>)
                     continue;
                 us[u] += eweight[e];
             }

             for (auto& uc : us)
             {
                 auto& u = uc.first;
                 auto m = uc.second;
                 S += parallel_term(u, v, m, g);
             }
         });
    return S;
}

template <class Graph, class BGraph>
double sparse_entropy(const Graph& g, auto&& eweight, auto&& degs,
                      const BGraph& bg, auto&& mrs, auto&& mrp, auto&& mrm,
                      auto&& wr, bool deg_corr, const auto& ea)
{
    double S = 0;

    #pragma omp parallel reduction(+:S)
    parallel_edge_loop_no_spawn
        (bg,
         [&](auto me)
         {
             S += eterm(source(me, bg), target(me, bg), mrs[me], bg);
         });

    #pragma omp parallel reduction(+:S)
    parallel_vertex_loop_no_spawn
        (bg,
         [&](auto r)
         {
             S += vterm(mrp[r], mrm[r], wr[r], deg_corr, bg);
         });

    if (ea.constants)
    {
        if (deg_corr && ea.deg_entropy)
        {
            #pragma omp parallel reduction(+:S)
            parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     S += get_deg_entropy(degs[v], g);
                 });
        }

        if (ea.multigraph && ea.parallel_entropy)
            S += get_parallel_entropy(g, eweight);
    }

    return S;
}

// Dense entropy
// =============

// "edge" term of the entropy
template <bool directed, class Val, class Eval, class Wval>
    requires std::is_scalar_v<Eval>
[[gnu::always_inline]] [[gnu::pure]]
inline double eterm_dense_d(Val r, Val s, Eval ers, Wval wr_r,
                            Wval wr_s, bool multigraph)
{
    assert(wr_r + wr_s >= 0);

    uint64_t nrns;
    if (directed || r != s) [[likely]]
    {
        nrns = wr_r * wr_s;
    }
    else
    {
        if (multigraph)
            nrns = (wr_r * (wr_r + 1)) / 2;
        else
            nrns = (wr_r * (wr_r - 1)) / 2;
    }

    if (multigraph)
        return lbinom_fast(nrns + ers - 1, ers);
    else
        return lbinom_fast(nrns, ers);
}


template <class Graph, class Val, class Eval, class Wval>
    requires std::is_scalar_v<Eval>
[[gnu::always_inline]] [[gnu::pure]]
inline double eterm_dense(Val r, Val s, Eval ers, Wval wr_r,
                          Wval wr_s, bool multigraph, const Graph&)
{
    return eterm_dense_d<is_directed_v<Graph>>(r, s, ers, wr_r, wr_s, multigraph);
}

template <class Graph>
double dense_entropy(Graph& bg, auto&& mrs, auto&& wr, bool deg_corr,
                     const auto& ea)

{
    if (deg_corr)
        throw GraphException("Dense entropy for degree corrected model not implemented!");

    double S = 0;
    #pragma omp parallel reduction(+:S)
    parallel_edge_loop_no_spawn
        (bg,
         [&](auto e)
         {
             auto r = source(e, bg);
             auto s = target(e, bg);
             S += eterm_dense(r, s, mrs[e], wr[r], wr[s],
                              ea.multigraph, bg);
         });
    return S;
}

// Edges description length
template <class Graph>
double get_edges_dl(int64_t B, int64_t E, Graph& g)
{
    int64_t BB = (is_directed(g)) ? B * B : (B * (B + 1)) / 2;
    return lbinom_fast(BB + E - 1, E);
}

// Edges description length, when the edges can only connect BB pairs of groups
// (e.g. groups of different classes, for multipartite graphs)
inline double get_edges_dl_pairs(int64_t BB, int64_t E)
{
    if (E == 0)
        return 0;
    if (BB == 0)
        return std::numeric_limits<double>::infinity();
    return lbinom_fast(BB + E - 1, E);
}

} // namespace graph_tool
