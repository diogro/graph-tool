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

#ifndef DYNAMICS_TEST_HH
#define DYNAMICS_TEST_HH

#include <algorithm>
#include <iostream>
#include <shared_mutex>

#include "../../../../hash_map_wrap.hh"
#include "../../../../idx_map.hh"

#include "openmp.hh"

#include "../dynamics.hh"

namespace graph_tool
{

template <class PMap>
PMap extract_pmap(python::object o)
{
    if (!PyObject_HasAttrString(o.ptr(), "_get_any"))
        throw ValueException("invalid property map");
    auto oa = o.attr("_get_any")();
    boost::any& a = python::extract<boost::any&>(oa)();
    return boost::any_cast<PMap>(a);
}

template <class Graph, class State>
class TestStateBase
    : public DStateBase
{
public:
    typedef typename eprop_map_t<double>::type emap_t;
    typedef typename vprop_map_t<double>::type vmap_t;

    TestStateBase(Graph& g, State& state, python::dict params)
        : _g(g),
          _state(state),
          _p(extract_pmap<emap_t>(params["p"])),
          _mu(extract_pmap<emap_t>(params["mu"])),
          _sigma(extract_pmap<emap_t>(params["sigma"])),
          _p_default(python::extract<double>(params["p_default"])),
          _mu_default(python::extract<double>(params["mu_default"])),
          _sigma_default(python::extract<double>(params["sigma_default"])),
          _mu_v(extract_pmap<vmap_t>(params["mu_v"])),
          _sigma_v(extract_pmap<vmap_t>(params["sigma_v"])),
          _self_loops(state._self_loops)
    {
        _theta.resize(num_vertices(state._u));
        for (auto v : vertices_range(state._u))
            _theta[v] = state._theta[v];

        for (auto e : edges_range(_g))
        {
            auto u = source(e, state._u);
            auto v = target(e, state._u);
            _edges[get_edge(u, v)] = e;
        }
    }

    Graph& _g;
    State& _state;
    emap_t _p;
    emap_t _mu;
    emap_t _sigma;
    double _p_default;
    double _mu_default;
    double _sigma_default;
    vmap_t _mu_v;
    vmap_t _sigma_v;
    bool _self_loops;
    std::vector<double> _theta;

    typedef typename graph_traits<Graph>::edge_descriptor edge_t;

    gt_hash_map<std::tuple<size_t,size_t>, edge_t> _edges;

    std::tuple<size_t, size_t> get_edge(size_t u, size_t v)
    {
        if (!graph_tool::is_directed(_g) && u > v)
            return {v, u};
        return {u, v};
    }

    void update_edge(size_t, size_t, double, double)
    {
    }

    void update_edges(const std::vector<size_t>&, size_t,
                      const std::vector<double>&,
                      const std::vector<double>&)
    {
    }

    double norm_lpmf(double a, double mu, double sigma)
    {
        //return norm_lpdf(a, mu, sigma) + log(_state._xdelta);
        mu = floor(mu / _state._xdelta) * _state._xdelta;
        if (mu == 0)
            return qlaplace_lprob(a, 1/sigma, _state._xdelta, true);
        double l = qlaplace_lprob(a - mu, 1/sigma, _state._xdelta, false);
        if ( _state._xdelta > 0)
            l -= log1p(-exp(qlaplace_lprob(-mu, 1/sigma, _state._xdelta, false)));
        return l;
    }

    double edge_lprob(double x, double p, double mu, double sigma)
    {
        if (x == 0)
            return log1p(-p);
        return log(p) + norm_lpmf(x, mu, sigma);
    }

    std::tuple<double, double, double> get_eparams(size_t u, size_t v)
    {
        double p = _p_default;
        double mu = _mu_default;
        double sigma = _sigma_default;
        auto iter = _edges.find(get_edge(u, v));
        if (iter != _edges.end())
        {
            auto e = iter->second;
            p = _p[e];
            mu = _mu[e];
            sigma = _sigma[e];
        }
        return {p, mu, sigma};
    }

    double get_edge_dS(size_t u, size_t v, double x, double nx)
    {
        if (!graph_tool::is_directed(_g) && u > v)
            return 0;
        auto [p, mu, sigma] = get_eparams(u, v);
        return -edge_lprob(nx, p, mu, sigma) + edge_lprob(x, p, mu, sigma);
    }

    template <class US, class XS>
    double get_edges_dS_dispatch(US&& us, size_t v, XS&& x, XS&& nx)
    {
        double dS = 0;
        for (size_t i = 0; i < us.size(); ++i)
            dS += get_edge_dS(us[i], v, x[i], nx[i]);
        return dS;
    }

    double get_edges_dS(const std::vector<size_t>& us, size_t v,
                        const std::vector<double>& x,
                        const std::vector<double>& nx)
    {
        return get_edges_dS_dispatch(us, v, x, nx);
    }

    double get_edges_dS(const std::array<size_t,2>& us, size_t v,
                        const std::array<double,2>& x,
                        const std::array<double,2>& nx)
    {
        return get_edges_dS_dispatch(us, v, x, nx);
    }

    double get_node_prob(size_t u)
    {
        double L = 0;
        for (auto e : out_edges_range(u, _g))
        {
            auto v = target(e, _g);
            auto p = _p[e];
            auto mu = _mu[e];
            auto sigma = _sigma[e];
            auto [m, x] = _state.edge_state(u, v);
            L += edge_lprob(x, p, mu, sigma);
        }

        size_t k = 0;
        for (auto e : out_edges_range(u, _state._u))
        {
            auto v = target(e, _g);
            auto iter = _edges.find(get_edge(u, v));
            if (iter != _edges.end())
                continue;
            auto x = _state._x[e];
            L += edge_lprob(x, _p_default, _mu_default, _sigma_default);
            k++;
        }

        size_t N = num_vertices(_g);
        if (!_self_loops)
            N--;

        L += (N - k) * edge_lprob(0, _p_default, _mu_default, _sigma_default);

        if (!graph_tool::is_directed(_g))
            L /= 2;

        L += norm_lpmf(_state._theta[u], _mu_v[u], _sigma_v[u]);

        return L;
    }

    double get_active_prob(size_t)
    {
        return 0;
    }

    double get_node_dS(size_t v, double t, double nt)
    {
        return -norm_lpmf(nt, _mu_v[v], _sigma_v[v]) +
            norm_lpmf(t, _mu_v[v], _sigma_v[v]);
    }

    double node_TE(size_t, size_t)
    {
        return 0;
    }

    double node_MI(size_t, size_t)
    {
        return 0;
    }

    double node_cov(size_t, size_t, bool, bool)
    {
        return 0;
    }
};

}// graph_tool namespace

#endif //DYNAMICS_TEST_HH
