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

#ifndef GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_ENTROPY_HH
#define GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_ENTROPY_HH

#include "../layered/types.hh"
#include "../layered/entropy.hh"
#include "../weighted/types.hh"
#include "../weighted/entropy.hh"

namespace graph_tool
{

template <class F, class... Xs>
double layer_dispatch(F&& f, Xs&&... xs)
{
    double ret = 0;
    iter_counts([&](auto, auto&& as)
                {
                    ret += std::apply(f, as);
                }, xs._h...);
    return ret;
}

template <LayerCounts DT, LayerCounts X>
double positive_w_log_P(DT&& N, X&& x, double alpha, double beta,
                        double epsilon)
{
    return layer_dispatch([&](auto Nl, auto xl)
                          { return positive_w_log_P(Nl, xl, alpha, beta, epsilon); },
                          N, x);
}

template <LayerCounts DT, LayerCounts X>
double signed_w_log_P(DT N, X x, X x2, double m0, double k0, double v0,
                      double nu0, double epsilon)
{
    return layer_dispatch([&](auto Nl, auto xl, auto x2l)
                          { return signed_w_log_P(Nl, xl, x2l, m0, k0, v0, nu0, epsilon); },
                          N, x, x2);
}


template <LayerCounts DT, LayerCounts X>
double geometric_w_log_P(DT&& N, X&& x, double alpha, double beta)
{
    return layer_dispatch([&](auto Nl, auto xl)
                          { return geometric_w_log_P(Nl, xl, alpha, beta); },
                          N, x);
}

template <LayerCounts DT, LayerCounts X>
double binomial_w_log_P(DT&& N, X&& x, size_t n, double alpha, double beta)
{
    return layer_dispatch([&](auto Nl, auto xl)
                          { return binomial_w_log_P(Nl, xl, n, alpha, beta); },
                          N, x);
}

template <LayerCounts DT, LayerCounts X>
double poisson_w_log_P(DT&& N, X&& x,  double alpha, double beta)
{
    return layer_dispatch([&](auto Nl, auto xl)
                          { return poisson_w_log_P(Nl, xl, alpha, beta); },
                          N, x);
}

template <LayerCounts X>
double safelog(X&& x)
{
    return layer_dispatch([&](auto xl){ return safelog(xl); }, x);
}

template <LayerCounts X>
double safelog_fast(X&& x)
{
    return layer_dispatch([&](auto xl){ return safelog_fast(xl); }, x);
}

template <class T, LayerCounts X>
double lbinom(T&& n, X&& x)
{
    return layer_dispatch([&](auto xl){ return lbinom(n, xl); }, x);
}


} // namespace graph_tool

#endif // GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_ENTROPY_HH
