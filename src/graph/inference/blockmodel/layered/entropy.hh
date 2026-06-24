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

#ifndef GRAPH_LAYERED_BLOCKMODEL_ENTROPY_HH
#define GRAPH_LAYERED_BLOCKMODEL_ENTROPY_HH

#include "types.hh"

namespace graph_tool
{

// Specialization of entropy terms

template <class T, T... S, class F>
constexpr void
for_sequence(std::integer_sequence<T, S...>, F&& f)
{
    (void(f(std::integral_constant<T, S>{})), ...);
}

template <class F, class... Ts>
void iter_counts(F&& f, Ts&&... vs)
{
    auto iter = std::make_tuple(std::begin(vs)...);
    auto end = std::make_tuple(std::end(vs)...);

    typedef std::decay_t<decltype(std::get<0>(*get<0>(iter)))> l_t;
    //typedef std::decay_t<decltype(std::get<1>(*get<0>(iter)))> c_t;

    l_t l = std::numeric_limits<l_t>::max();
    for_sequence(std::make_index_sequence<sizeof...(Ts)>{},
                 [&](auto i)
                 {
                     if (std::get<i>(iter) != std::get<i>(end))
                         l = std::min(l, std::get<0>(*std::get<i>(iter)));
                 });

    std::tuple<std::decay_t<decltype(std::begin(vs)->second)>...> c;

    while (iter != end)
    {
        l_t l_n = std::numeric_limits<l_t>::max();
        for_sequence(std::make_index_sequence<sizeof...(Ts)>{},
                     [&](auto i)
                     {
                         if (std::get<i>(iter) == std::get<i>(end))
                         {
                             get<i>(c) = 0;
                             return;
                         }

                         const auto& [l_i, c_i] = *std::get<i>(iter);
                         if (l_i == l)
                         {
                             get<i>(c) = c_i;
                             ++std::get<i>(iter);
                             if (std::get<i>(iter) != std::get<i>(end))
                                 l_n = std::min(l_n, std::get<i>(iter)->first);
                         }
                         else
                         {
                             assert(l_i > l);
                             get<i>(c) = 0;
                             l_n = std::min(l_n, l_i);
                         }
                     });
        f(l, c);
        l = l_n;
    }
}

template <class F, class T, size_t N1, size_t N2>
void iter_counts_arrays(F&& f, const std::array<T, N1>& v1,
                        const std::array<T, N2>& v2)
{
    typedef std::decay_t<decltype(get<0>(*v1[0]->_h.begin()))> val_t;
    std::apply
        ([&](auto&&... vs)
         {
             iter_counts([&](auto l, auto&& c)
                         {
                             std::array<val_t, N1> v1s;
                             std::array<val_t, N2> v2s;

                             constexpr auto get_array = [](auto&& ... x){ return std::array{std::forward<decltype(x)>(x) ... }; };
                             auto a = std::apply(get_array, c);

                             for (size_t i = 0; std::cmp_less(i, N1); ++i)
                                 v1s[i] = a[i];
                             for (size_t i = 0; std::cmp_less(i, N2); ++i)
                                 v2s[i] = a[i + N1];
                             f(l, v1s, v2s);
                         }, vs->_h ...);
         },
         std::tuple_cat(std::tuple_cat(v1), std::tuple_cat(v2)));
}

template <class Val, LayerCounts Mval, class Graph>
double eterm(Val r, Val s, const Mval& mrs, const Graph& g)
{
    if constexpr (mrs._layered)
    {
        return mrs.layer_dispatch([&](auto x)
                                  {
                                      return eterm(r, s, x, g);
                                  });
    }
    else
    {
        double S = eterm(r, s, int64_t(mrs), g);
        S += lgamma_fast(int64_t(mrs) + 1);
        S += mrs.layer_dispatch
            ([&](auto x)
             {
                 return -lgamma_fast(x + 1);
             });
        return S;
    }
}

template <LayerCounts Val, LayerCounts VVal, class Graph>
double vterm(const Val& mrp, const Val& mrm, const VVal& wr, bool deg_corr,
             const Graph& g)
{
    if constexpr (mrp._layered)
    {
        double ret = 0;
        iter_counts([&](auto, const auto& c)
                    {
                        ret += vterm(get<0>(c), get<1>(c), get<2>(c), deg_corr, g);
                    }, mrp._h, mrm._h, wr._h);
        return ret;
    }
    else
    {
        return vterm(int64_t(mrp), int64_t(mrm), int64_t(wr), deg_corr, g);
    }
}

template <class V, LayerCounts Val, class Graph>
double parallel_term(V u, V v, const Val& m, const Graph& g)
{
    return m.layer_dispatch
        ([&](auto x)
         {
             return parallel_term(u, v, x, g);
         });
}

template <class Graph, class Val, LayerCounts Eval, LayerCounts Wval>
double eterm_dense(Val r, Val s, Eval&& ers, Wval&& wr_r,
                   Wval&& wr_s, bool multigraph, const Graph& g)
{
    if constexpr (ers._layered)
    {
        double ret = 0;
        iter_counts([&](auto, const auto& c)
                    {
                        ret += eterm_dense(r, s, get<0>(c), get<1>(c), get<2>(c), multigraph, g);
                    }, ers._h, wr_r._h, wr_s._h);
        return ret;
    }
    else
    {
        double S = eterm_dense(r, s, int64_t(ers), wr_r, wr_s, multigraph, g);
        S += lgamma_fast(int64_t(ers) + 1);
        S += ers.layer_dispatch
            ([&](auto x)
             {
                 return -lgamma_fast(x + 1);
             });
        return S;
    }
}

template <class Graph, class Val, LayerCounts Eval, class Wval>
   requires std::is_scalar_v<Wval>
double eterm_dense(Val r, Val s, Eval&& ers, Wval wr_r,
                   Wval wr_s, bool multigraph, const Graph& g)
{
    return ers.layer_dispatch
        ([&](auto x)
         {
             return eterm_dense(r, s, x, wr_r, wr_s, multigraph, g);
         });
}

template <LayerCounts Deg>
double deg_entropy(const Deg& degs)
{
    if constexpr (degs._layered)
    {
        double ret = 0;
        for (auto&& [l, c] : degs._h)
            ret += deg_entropy(c);
        return ret;
    }
    else
    {
        return deg_entropy(int64_t(degs));
    }
}

template <class Graph, class Degs, class Delta>
    requires LayerCounts<std::tuple_element_t<0, Degs>>
double get_deg_entropy(const Degs& degs, const Graph& g, const Delta& delta)
{
    if constexpr (get<0>(degs)._layered)
    {
        double ret = 0;
        iter_counts([&](auto, const auto& c)
                    {
                        ret += get_deg_entropy(std::make_tuple(get<0>(c), get<1>(c)), g,
                                               std::array<int64_t, 2>{get<2>(c), get<3>(c)});
                    },
                    std::get<0>(degs)._h, std::get<1>(degs)._h,
                    delta[0]._h, delta[1]._h);
        return ret;
    }
    else
    {
        return get_deg_entropy(std::make_tuple(int64_t(std::get<0>(degs)),
                                               int64_t(std::get<1>(degs))),
                               g, std::array<int64_t, 2>{int64_t(delta[0]),
                                                         int64_t(delta[1])});
    }
}

template <LayerCounts C, class Graph>
double get_edges_dl(size_t B, const C& E, Graph& g)
{
    double S = 0;
    size_t Et = 0;
    size_t L = 0;
    E.template cdispatch<true>([&](auto&& h)
                               {
                                   for (auto&& [l, c] : h)
                                   {
                                       S += get_edges_dl(B, c, g);
                                       Et += c;
                                       L++;
                                   }
                               });
    S += lbinom_fast(Et - 1, L - 1);
    return S;
}

} //namespace graph_tool

#endif // GRAPH_LAYERED_BLOCKMODEL_ENTROPY_HH
