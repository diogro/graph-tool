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

#ifndef GRAPH_DYNAMICS_BASE_IMP_HH
#define GRAPH_DYNAMICS_BASE_IMP_HH

#include "dynamics_base.hh"

namespace graph_tool
{

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_node_prob(size_t v)
{
    double L = 0;
    auto k = keep_k ? _k[v] : 0;
    auto theta = _spec.transform_theta(_theta[v], k);
    iter_time<true, true, false>
        (std::array<size_t,0>({}), v,
         [&](auto, auto, auto&&, auto m, int w, auto... s_v)
         {
             if constexpr (!keep_k)
                 L += _spec.log_P(theta, m, s_v...) * w;
             else
                 L += _spec.log_P(theta, m, k, s_v...) * w;
         });
    return L;
}

template double DState::base_t::get_node_prob(size_t);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_active_prob(size_t v)
{
    double L = 0;
    for (size_t n = 0; n < _active.size(); ++n)
    {
        auto& active = _active[n][v];
        size_t T = active.size();
        size_t M = 0;
        for (auto w : active)
            M += w;
        L -= lbinom_fast(T, M) + safelog_fast(T + 1);
    }
    return L;
}

template double DState::base_t::get_active_prob(size_t);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edge_dS_compressed(size_t u, size_t v, double x, double nx)
{
    return get_edge_dS_dispatch_indirect<true>(std::array<size_t,1>{u}, v,
                                               std::array<double,1>{x},
                                               std::array<double,1>{nx});
};

template double DState::base_t::get_edge_dS_compressed(size_t u, size_t v,
                                                       double x, double nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edge_dS_uncompressed(size_t u, size_t v, double x, double nx)
{
    return get_edge_dS_dispatch_direct<false,false>(std::array<size_t,1>{u}, v,
                                                    std::array<double,1>{x},
                                                    std::array<double,1>{nx});
};

template double DState::base_t::get_edge_dS_uncompressed(size_t u, size_t v,
                                                         double x, double nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edges_dS_compressed(const std::vector<size_t>& us, size_t v,
                        const std::vector<double>& x, const std::vector<double>& nx)
{
    return get_edge_dS_dispatch_indirect<true>(us, v, x, nx);
};

template double DState::base_t::get_edges_dS_compressed(const std::vector<size_t>& us,
                                                        size_t v,
                                                        const std::vector<double>& x,
                                                        const std::vector<double>& nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edges_dS_uncompressed(const std::vector<size_t>& us, size_t v,
                          const std::vector<double>& x, const std::vector<double>& nx)
{
    return get_edge_dS_dispatch_direct<false, true>(us, v, x, nx);
};

template double DState::base_t::get_edges_dS_uncompressed(const std::vector<size_t>& us,
                                                          size_t v,
                                                          const std::vector<double>& x,
                                                          const std::vector<double>& nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edges_dS_compressed(const std::array<size_t,2>& us, size_t v,
                        const std::array<double,2>& x, const std::array<double,2>& nx)
{
    return get_edge_dS_dispatch_indirect<true>(us, v, x, nx);
};

template double DState::base_t::get_edges_dS_compressed(const std::array<size_t,2>& us,
                                                        size_t v,
                                                        const std::array<double,2>& x,
                                                        const std::array<double,2>& nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_edges_dS_uncompressed(const std::array<size_t,2>& us, size_t v,
                          const std::array<double,2>& x,
                          const std::array<double,2>& nx)
{
    return get_edge_dS_dispatch_direct<false, false>(us, v, x, nx);
};

template double DState::base_t::get_edges_dS_uncompressed(const std::array<size_t,2>& us,
                                                          size_t v,
                                                          const std::array<double,2>& x,
                                                          const std::array<double,2>& nx);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_node_dS_compressed(size_t v, double t, double nt)
{
    return get_node_dS_dispatch<true>(v, t, nt);
}

template double DState::base_t::get_node_dS_compressed(size_t v, double t, double nt);

template <class Spec, bool discrete, bool keep_k, bool tshift>
[[gnu::hot]]
double NSumStateBase<Spec, discrete, keep_k, tshift>::
get_node_dS_uncompressed(size_t v, double t, double nt)
{
    return get_node_dS_dispatch<false>(v, t, nt);
}

template double DState::base_t::get_node_dS_uncompressed(size_t v, double t, double nt);


}// graph_tool namespace

#endif //GRAPH_DYNAMICS_BASE_IMP_HH
