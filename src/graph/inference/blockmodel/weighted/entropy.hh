// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_WEIGHTED_BLOCKMODEL_ENTROPY_HH
#define GRAPH_WEIGHTED_BLOCKMODEL_ENTROPY_HH

#include "types.hh"

namespace graph_tool
{

// Weighted entropy terms
// ======================

// exponential
template <class DT>
double positive_w_log_P(DT N, double x, double alpha, double beta,
                        double epsilon)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
    {
        if (x < epsilon || N == 1)
            return 0.;
        else
            return safe_lgamma(N) - (N - 1) * std::log(x);
    }
    return (safe_lgamma(N + alpha) -
            safe_lgamma(alpha) + alpha * std::log(beta) -
            (alpha + N) * std::log(beta + x));
}

// normal
template <class DT>
double signed_w_log_P(DT N, double x, double x2, double m0, double k0, double v0,
                      double nu0, double epsilon)
{
    if (N == 0)
        return 0.;
    if (std::isnan(m0) && std::isnan(k0))
    {
        auto smu1 = x * (x / N);

        if (N < 2 || smu1 >= x2 || (x2 - smu1) < std::pow(epsilon, 2))
            return 0.;
        else
            return (safe_lgamma((N - 1) / 2.) + std::log(N) / 2.
                    - ((int(N) - 3) / 2.) * std::log(x2 - smu1)
                    - ((N - 1) / 2.) * std::log(M_PI));
    }
    auto v = x2 - x * (x / N);
    auto k_n = k0 + N;
    auto nu_n = nu0 + N;
    auto v_n = (v0 * nu0 + v + ((N * k0)/(k0 + N)) * pow(m0 - x/N, 2)) / nu_n;
    return (safe_lgamma(nu_n / 2.) -
            safe_lgamma(nu0 / 2.) + (std::log(k0) - std::log(k_n)) / 2. +
            (nu0 / 2.) * std::log(nu0 * v0) - (nu_n / 2.) * std::log(nu_n * v_n) - (N / 2.) *
            std::log(M_PI));
}

// discrete: geometric
template <class DT>
double geometric_w_log_P(DT N, double x, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return -lbinom((N - 1) + x, x);
    return lbeta(N + alpha, x + beta) - lbeta(alpha, beta);
}

// discrete: binomial
template <class DT>
double binomial_w_log_P(DT N, double x, int64_t n, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return -lbinom(N * n, x);
    return lbeta(x + alpha, N * n - x + beta) - lbeta(alpha, beta);
}

// discrete: Poisson
template <class DT>
double poisson_w_log_P(DT N, double x, double alpha, double beta)
{
    if (N == 0)
        return 0.;
    if (std::isnan(alpha) && std::isnan(beta))
        return safe_lgamma(x + 1) - x * std::log(N);
    return (safe_lgamma(x + alpha) - (x + alpha) * std::log(N + beta) -
            safe_lgamma(alpha) + alpha * std::log(beta));
}

// Entropy terms

template <class Val, WeightCounts Mval, class Graph>
double eterm(Val r, Val s, const Mval& mrs, const Graph& g)
{
    return eterm(r, s, int64_t(mrs), g);
}

template <WeightCounts Val, class Wval, class Graph>
double vterm(const Val& mrp, const Val& mrm, Wval wr, bool deg_corr,
             const Graph& g)
{
    return vterm(int64_t(mrp), int64_t(mrm), wr, deg_corr, g);
}

template <class V, WeightCounts Val, class Graph>
double parallel_term(V u, V v, const Val& m, const Graph& g)
{
    return parallel_term(u, v, int64_t(m), g);
}

template <class Graph, class Val, WeightCounts Eval>
double eterm_dense(Val r, Val s, const Eval& ers, uint64_t wr_r,
                   uint64_t wr_s, bool multigraph, const Graph& g)
{
    return eterm_dense(r, s, int64_t(ers), wr_r, wr_s, multigraph, g);
}

template <class Graph, WeightCounts W, class Delta>
double get_deg_entropy(const std::tuple<W, W>& degs, const Graph& g,
                       const Delta& delta)
{
    return get_deg_entropy(std::make_tuple(int64_t(std::get<0>(degs)),
                                           int64_t(std::get<1>(degs))),
                           g, std::array<int64_t, 2>{int64_t(delta[0]),
                                                     int64_t(delta[1])});
}

} // namespace graph_tool

#endif // GRAPH_WEIGHTED_BLOCKMODEL_ENTROPY_HH
