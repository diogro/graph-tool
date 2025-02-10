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

#ifndef DYNAMICS_HH
#define DYNAMICS_HH

#include "config.h"

#include <vector>
#include <map>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "idx_map.hh"

#include "../../support/graph_state.hh"
#include "../../support/fibonacci_search.hh"
#include "../uncertain_util.hh"
#include "../../../generation/graph_knn.hh"

#include "../../../parallel_util.hh"

#include "dynamics_util.hh"

#include "bisection_sampler.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct dentropy_args_t:
        public uentropy_args_t
{
    dentropy_args_t(const uentropy_args_t& ea)
        : uentropy_args_t(ea) {}

    double alpha = 1;
    bool xdist = true;
    bool tdist = true;
    bool xdist_uniform = false;
    bool tdist_uniform = false;
    double xl1 = 0;
    double tl1 = 0;
    bool normal = false;
    double mu = 0;
    double sigma = 1;
    bool active = true;
};

typedef eprop_map_t<double> xmap_t;
typedef vprop_map_t<double> tmap_t;
typedef vprop_map_t<double> smap_t;

#define DYNAMICS_STATE_params                                                  \
    ((g, &, decltype(never_filtered_never_reversed), 1))                       \
    ((x,, xmap_t, 0))                                                          \
    ((params,, python::dict, 0))                                               \
    ((theta,, tmap_t, 0))                                                      \
    ((xdelta,, double, 0))                                                     \
    ((xdelta_min,, double, 0))                                                 \
    ((tdelta,, double, 0))                                                     \
    ((tdelta_min,, double, 0))                                                 \
    ((disable_xdist,, bool, 0))                                                \
    ((disable_tdist,, bool, 0))                                                \
    ((self_loops,, bool, 0))                                                   \
    ((max_m,, int, 0))

class DStateBase
{
public:

    virtual double get_edge_dS(size_t u, size_t v, double x, double nx) = 0;
    virtual double get_edges_dS(const std::vector<size_t>& us, size_t v,
                                const std::vector<double>& x,
                                const std::vector<double>& nx) = 0;
    virtual double get_edges_dS(const std::array<size_t,2>& us, size_t v,
                                const std::array<double,2>& x,
                                const std::array<double,2>& nx) = 0;

    virtual double get_node_dS(size_t v, double t, double nt) = 0;
    virtual double get_node_prob(size_t v) = 0;
    virtual double get_active_prob(size_t v) = 0;

    virtual void update_edge(size_t u, size_t v, double x, double nx) = 0;
    virtual void update_edges(const std::vector<size_t>& us, size_t v,
                              const std::vector<double>& x,
                              const std::vector<double>& nx) = 0;

    virtual double node_TE(size_t u, size_t v) = 0;
    virtual double node_MI(size_t u, size_t v) = 0;
    virtual double node_cov(size_t u, size_t v, bool, bool) = 0;

    virtual void set_params(boost::python::dict) {};
};

template <class BlockState>
struct Dynamics
{
    GEN_STATE_BASE(DynamicsStateBase, DYNAMICS_STATE_params)

    template <class... Ts>
    class DynamicsState
        : public DynamicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(DynamicsStateBase<Ts...>,
                         DYNAMICS_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, DYNAMICS_STATE_params)

        typedef typename property_traits<x_t>::value_type xval_t;
        typedef typename property_traits<theta_t>::value_type tval_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        DynamicsState(BlockState& block_state, ATs&&... args)
            : DynamicsStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state),
              _dummy_hint(num_vertices(_u)),
              _xc(_x.get_checked()),
              _e_mutex(num_vertices(_u)),
              _v_mutex(num_vertices(_u))
        {
            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                auto u = source(e, _u);
                auto v = target(e, _u);
                get_u_edge<true>(u, v) = e;
                if (_self_loops || u != v)
                {
                    hist_add(_x[e], _xhist, _xvals);
                    if (!_disable_xdist)
                    _M++;
                }
                _E += _eweight[e];
            }

            if (!_disable_tdist)
            {
                for (auto v : vertices_range(_u))
                    hist_add(_theta[v], _thist, _tvals);
            }
        }

        ~DynamicsState()
        {
            for (auto& es : _u_edges)
                for (auto& [v, e] : es)
                    delete e;
        }

        DynamicsState(const DynamicsState&) = delete;

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typedef typename BlockState::g_t u_t;
        u_t& _u = _block_state._g;
        adj_list<> _dummy_hint;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t*>> _u_edges;

        size_t _E = 0;
        size_t _M = 0;

        DStateBase* _dstate;

        void set_dstate(DStateBase& dstate)
        {
            _dstate = &dstate;
        }

        typename x_t::checked_t _xc;

        std::vector<double> _xvals;
        std::vector<double> _tvals;

        gt_hash_map<double, size_t> _xhist;
        gt_hash_map<double, size_t> _thist;

        std::vector<std::shared_mutex> _e_mutex;
        std::vector<std::mutex> _v_mutex;

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];

            GraphInterface::edge_t* e;

            if constexpr (insert)
            {
                std::unique_lock lock(_e_mutex[u]);
                auto& ep = qe[v];
                if (ep == nullptr)
                    ep = new GraphInterface::edge_t();
                e = ep;
            }
            else
            {
                std::shared_lock lock(_e_mutex[u]);
                auto iter = qe.find(v);
                if (iter != qe.end())
                    e = iter->second;
                else
                    e = &_null_edge;
            }

            return *e;
        }

        template <class Graph, class Elist>
        void _erase_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];

            std::unique_lock lock(_e_mutex[u]);
            auto iter = qe.find(v);
            delete iter->second;
            qe.erase(iter);
        }

        template <bool insert=false>
        auto& get_u_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _u, _u_edges);
        }

        void erase_u_edge(size_t u, size_t v)
        {
            return _erase_edge(u, v, _u, _u_edges);
        }

        std::tuple<size_t, xval_t> edge_state(size_t u, size_t v)
        {
            auto&& e = get_u_edge(u, v);
            if (e == _null_edge)
                return {0, 0};
            return {_eweight[e], _xc[e]};
        }

        double get_node_prob(size_t u)
        {
            return _dstate->get_node_prob(u);
        }

        double get_active_prob(size_t u)
        {
            return _dstate->get_active_prob(u);
        }

        template <class Hist>
        size_t get_count(Hist& h, double r)
        {
            auto iter = h.find(r);
            if (iter == h.end())
                return 0;
            return iter->second;
        }

        template <class Hist, class Vals>
        void hist_add(double r, Hist& h, Vals& vals, size_t delta=1)
        {
            auto& c = h[r];
            if (c == 0)
                vals.insert(std::upper_bound(vals.begin(), vals.end(), r), r);
            c += delta;
        }

        template <class Hist, class Vals>
        void hist_remove(double r, Hist& h, Vals& vals, size_t delta=1)
        {
            auto& c = h[r];
            c -= delta;
            if (c == 0)
            {
                h.erase(r);
                vals.erase(std::lower_bound(vals.begin(), vals.end(), r));
            }
        }

        template <class Hist, class Vals>
        double hist_entropy(size_t N, Hist& h, Vals& vals, bool uniform,
                            double l1, double delta, bool nonzero)
        {
            auto W = h.size();
            double S = 0;

            if (N == 0)
                return S;

            if (W > 0)
            {
                S -= qlaplace_lprob(vals.front(), l1, delta, nonzero);
                S -= qlaplace_lprob(vals.back(), l1, delta, nonzero);
                if (W >= 2)
                    S += lbinom_careful<double,double>
                        ((vals.back() - vals.front())/delta - 1 - int(nonzero),
                         W - 2);
            }

            if (uniform)
            {
                S += N * safelog_fast(W);
            }
            else
            {
                S += safelog_fast(N);
                S += lbinom_fast(N - 1, W - 1);
                S += lgamma_fast(N + 1);
                for (auto& [r, n] : h)
                    S -= lgamma_fast(n + 1);
            }

            return S;
        }

        template <class Hist, class Vals>
        double hist_move_dS(double r, double s, size_t N, Hist& h, Vals& vals,
                            bool uniform, double l1, double delta,
                            bool nonzero, size_t dn=1)
        {
            if (r == s)
                return 0;

            double Sa = 0;
            double Sb = 0;

            size_t nr = get_count(h, r);
            size_t ns = get_count(h, s);
            size_t W = h.size();
            int dW = 0;

            assert(nr > 0);
            assert(nr >= dn);

            double xmin = (W > 0) ? vals.front() : 0;
            double xmax = (W > 0) ? vals.back() : 0;
            double nxmin = xmin;
            double nxmax = xmax;

            if (nr == dn)
            {
                if (nxmin == r)
                    nxmin = (W > 1) ? vals[1] : numeric_limits<double>::infinity();
                if (nxmax == r)
                    nxmax = (W > 1) ? vals[vals.size() - 2] : -numeric_limits<double>::infinity();
                dW--;
            }

            if (ns == 0)
            {
                if (s < nxmin)
                    nxmin = s;
                if (s > nxmax)
                    nxmax = s;
                dW++;
            }

            auto get_S =
                [&](size_t nr, size_t ns, size_t W, double xmin, double xmax)
                {
                    double S = 0;

                    if (W > 0)
                    {
                        S -= qlaplace_lprob(xmin, l1, delta, nonzero);
                        S -= qlaplace_lprob(xmax, l1, delta, nonzero);
                        if (W >= 2)
                            S += lbinom_careful<double, double>
                                ((xmax - xmin)/delta - 1 - int(nonzero), W - 2);
                    }

                    if (uniform)
                    {
                        S += N * safelog_fast(W);
                    }
                    else
                    {
                        S += safelog_fast(N);
                        S += lbinom_fast(N - 1, W - 1);
                        S -= lgamma_fast(nr + 1) + lgamma_fast(ns + 1);
                    }
                    return S;
                };

            Sb += get_S(nr, ns, W, xmin, xmax);
            Sa += get_S(nr - dn, ns + dn, W + dW, nxmin, nxmax);

            assert(!isinf(Sa-Sb));

            return Sa - Sb;
        }

        template <bool Add, class Hist, class Vals>
        double hist_modify_dS(double r, size_t N, Hist& h, Vals& vals,
                              bool uniform, double l1, double delta, bool nonzero,
                              size_t dn=1)
        {
            double Sa = 0;
            double Sb = 0;

            size_t nr = get_count(h, r);
            size_t W = h.size();
            int dW = 0;
            int dN = 0;

            double xmin = (W > 0) ? vals.front() : 0;
            double xmax = (W > 0) ? vals.back() : 0;
            double nxmin = xmin;
            double nxmax = xmax;

            if constexpr (Add)
            {
                if (nr == 0)
                {
                    if (r < nxmin)
                        nxmin = r;
                    if (r > nxmax)
                        nxmax = r;
                    dW++;
                }
                dN += dn;
            }
            else
            {
                if (nr == dn)
                {
                    if (nxmin == r)
                        nxmin = (W > 1) ? vals[1] : 0;
                    if (nxmax == r)
                        nxmax = (W > 1) ? vals[vals.size() - 2] : 0;
                    dW--;
                }
                dN -= dn;
            }

            auto get_S =
                [&](size_t N, size_t nr, size_t W, double xmin, double xmax)
                {
                    double S = 0;
                    if (N == 0)
                        return S;

                    if (W > 0)
                    {
                        S -= qlaplace_lprob(xmin, l1, delta, nonzero);
                        S -= qlaplace_lprob(xmax, l1, delta, nonzero);
                        if (W >= 2)
                            S += lbinom_careful<double, double>
                                ((xmax - xmin)/delta - 1 - int(nonzero), W - 2);
                    }

                    if (uniform)
                    {
                        S += N * safelog_fast(W);
                    }
                    else
                    {
                        S += safelog_fast(N);
                        S += lbinom_fast(N - 1, W - 1);
                        S += lgamma_fast(N + 1);
                        S -= lgamma_fast(nr + 1);
                    }
                    return S;
                };

            Sb += get_S(N,      nr,      W,      xmin,  xmax);
            Sa += get_S(N + dN, nr + dN, W + dW, nxmin, nxmax);

            assert(!isinf(Sa-Sb));
            return Sa - Sb;
        }

        double quantize_x(double x)
        {
            return quantize(x, _xdelta);
        }

        double quantize_t(double x)
        {
            return quantize(x, _tdelta);
        }

        double entropy(const dentropy_args_t& ea)
        {
            double S = 0;

            if (ea.latent_edges)
            {
                #pragma omp parallel reduction(+:S)
                parallel_vertex_loop_no_spawn
                    (_u,
                     [&](auto v)
                     {
                         S -= _dstate->get_node_prob(v);
                     });
                S *= ea.alpha;
            }

            if (ea.active)
            {
                #pragma omp parallel reduction(+:S)
                parallel_vertex_loop_no_spawn
                    (_u,
                     [&](auto v)
                     {
                         S -= _dstate->get_active_prob(v);
                     });
            }

            if (ea.density)
                S += -(_E * log(ea.aE)) + lgamma_fast(_E + 1) - ea.aE;

            #pragma omp parallel reduction(+:S)
            parallel_edge_loop_no_spawn
                (_u,
                 [&](auto e)
                 {
                     if (source(e, _u) == target(e, _u) && !_self_loops)
                         return;
                     S += edge_x_S(_x[e], ea);
                 });

            size_t N = num_vertices(_u);
            size_t T = is_directed(_u) ?
                N * (_self_loops ? N : N-1) :
                (N * (_self_loops ? N + 1 : N - 1)) / 2;
            S += (T - _E) * edge_x_S(0, ea);

            if (!_disable_xdist && ea.xdist)
                S += hist_entropy(_M, _xhist, _xvals, ea.xdist_uniform,
                                  ea.xl1, _xdelta, true);

            if (!_disable_tdist && ea.tdist)
                S += hist_entropy(num_vertices(_u), _thist, _tvals,
                                  ea.tdist_uniform, ea.tl1, _tdelta, false);

            #pragma omp parallel reduction(+:S)
            parallel_vertex_loop_no_spawn
                (_u,
                 [&](auto v)
                 {
                    S += node_x_S(_theta[v], ea);
                 });

// #ifndef NDEBUG
//             for (auto v : vertices_range(_u))
//                 _dstate->check_m(*this, v);
// #endif
            return S;
        }

        double dstate_edge_dS(size_t u, size_t v, double x, double nx,
                              const dentropy_args_t& ea,
                              bool reciprocal = true)
        {
            if (!ea.latent_edges)
                return 0;
            double dS = 0;
            dS += _dstate->get_edge_dS(u, v, x, nx);
            if (u != v && !is_directed(_u) && !isinf(dS) && reciprocal)
                dS += _dstate->get_edge_dS(v, u, x, nx);
            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS * ea.alpha;
        }

        template <class VS, class VX>
        double dstate_edges_dS(const VS& us, size_t v, const VX& x, const VX& nx,
                               const dentropy_args_t& ea)
        {
            if (!ea.latent_edges)
                return 0;
            double dS = _dstate->get_edges_dS(us, v, x, nx);
            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS * ea.alpha;
        }

        double dstate_edges_dS_a(const std::array<size_t,2>& us, size_t v,
                                 const std::array<double,2>& x,
                                 const std::array<double,2>& nx,
                                 const dentropy_args_t& ea)
        {
            return dstate_edges_dS<std::array<size_t,2>,
                                   std::array<double,2>>(us, v, x, nx, ea);
        }

        double dstate_edges_dS(const std::vector<size_t>& us, size_t v,
                               const std::vector<double>& x,
                               const std::vector<double>& nx,
                               const dentropy_args_t& ea)
        {
            return dstate_edges_dS<std::vector<size_t>,
                                   std::vector<double>>(us, v, x, nx, ea);
        }

        double dstate_node_dS(size_t v, double t, double nt,
                              const dentropy_args_t& ea)
        {
            if (!ea.latent_edges)
                return 0;
            double dS = _dstate->get_node_dS(v, t, nt);
            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS * ea.alpha;
        }

        std::shared_mutex _sbm_mutex;
        std::shared_mutex _x_mutex;
        std::shared_mutex _t_mutex;

        double edge_x_S(xval_t x, const dentropy_args_t& ea)
        {
            if ((ea.sbm && x == 0) || ea.xdist || !ea.latent_edges)
                return 0.;
            double S = 0;
            if (ea.normal)
            {
                S -= norm_lpdf(x, ea.mu, ea.sigma);
            }
            else if (ea.xl1 > 0)
            {
                if (_xdelta == 0)
                    S -= laplace_lpdf(x, ea.xl1);
                else
                    S -= qlaplace_lprob(x, ea.xl1, _xdelta, ea.sbm);
            }
            return S;
        }

        double node_x_S(xval_t x, const dentropy_args_t& ea)
        {
            if (ea.tdist)
                return 0.;
            double S = 0;
            if (ea.tl1 > 0)
            {
                if (_tdelta == 0)
                    S -= laplace_lpdf(x, ea.tl1);
                else
                    S -= qlaplace_lprob(x, ea.tl1, _tdelta, false);
            }
            return S;
        }

        double remove_edge_dS(size_t u, size_t v, int dm,
                              const dentropy_args_t& ea, bool dstate = true,
                              bool lock = true)
        {
            if (dm == 0)
                return 0;

            auto& e = get_u_edge(u, v);
            auto x = _xc[e];

            assert(x != 0);

            double dS = 0;

            if (ea.sbm)
            {
                do_slock([&]()
                         {
                             dS += _block_state.modify_edge_dS(u, v, e, -dm, ea);
                         }, _sbm_mutex, lock);
            }

            if (ea.density)
            {
                dS += log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 - dm) - lgamma_fast(_E + 1);
            }

            if (_eweight[e] == dm && (_self_loops || u != v))
            {
                if (ea.latent_edges)
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, x, 0, ea);

                    dS += edge_x_S(0, ea) - edge_x_S(x, ea);

                    if (ea.xdist && !_disable_xdist)
                    {
                        do_slock([&]()
                                 {
                                     dS += hist_modify_dS<false>(x, _M, _xhist,
                                                                 _xvals,
                                                                 ea.xdist_uniform,
                                                                 ea.xl1, _xdelta,
                                                                 ea.sbm);
                                 }, _x_mutex, lock);
                    }
                }
            }

            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        double add_edge_dS(size_t u, size_t v, int dm, double x,
                           const dentropy_args_t& ea, bool dstate = true,
                           bool lock = true)
        {
            if (dm == 0)
                return 0;

            assert(x != 0);

            auto& e = get_u_edge(u, v);

            auto m = (e == _null_edge) ? 0 : _eweight[e];

            if (m + dm > _max_m)
                return numeric_limits<double>::infinity();

            double dS = 0;

            if (ea.sbm)
            {
                do_slock([&]()
                         {
                             dS += _block_state.modify_edge_dS(u, v, e, dm, ea);
                         }, _sbm_mutex, lock);
            }

            if (ea.density)
            {
                dS -= log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 + dm) - lgamma_fast(_E + 1);
            }

            assert(!std::isinf(dS) && !std::isnan(dS));

            if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
            {
                if (ea.latent_edges)
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, 0, x, ea);
                    assert(!std::isinf(dS) && !std::isnan(dS));

                    dS += edge_x_S(x, ea) - edge_x_S(0, ea);

                    assert(!std::isinf(dS) && !std::isnan(dS));
                    if (ea.xdist && !_disable_xdist)
                    {
                        do_slock([&]()
                                 {
                                     dS += hist_modify_dS<true>(x, _M, _xhist,
                                                                _xvals,
                                                                ea.xdist_uniform,
                                                                ea.xl1, _xdelta,
                                                                ea.sbm);
                                     assert(!isinf(dS));
                                 }, _x_mutex, lock);
                    }
                    assert(!std::isinf(dS) && !std::isnan(dS));
                }
            }

            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        double update_edge_dS(size_t u, size_t v, double nx, const dentropy_args_t& ea,
                              bool dstate = true, bool lock = true)
        {
            assert(nx != 0);

            double dS = 0;
            if (ea.latent_edges)
            {
                auto& e = get_u_edge(u, v);
                auto x = _x[e];

                if (x == nx)
                    return 0;

                if (_self_loops || u != v)
                {
                    if (dstate)
                        dS += dstate_edge_dS(u, v, x, nx, ea);

                    assert(!std::isinf(dS) && !std::isnan(dS));

                    dS += (edge_x_S(nx, ea) - edge_x_S(x, ea));

                    assert(!std::isinf(dS) && !std::isnan(dS));

                    if (ea.xdist && !_disable_xdist)
                    {
                        do_slock([&]()
                                 {
                                     dS += hist_move_dS(x, nx, _M, _xhist,
                                                        _xvals, ea.xdist_uniform,
                                                        ea.xl1, _xdelta, ea.sbm);
                                     assert(!std::isinf(dS) && !std::isnan(dS));
                                 }, _x_mutex, lock);
                    }
                }
            }

            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        template <class F>
        double update_edges_dS(F&& get_es, double x, double nx,
                               const dentropy_args_t& ea)
        {
            gt_hash_map<size_t, std::vector<size_t>> edges;
            std::vector<std::tuple<size_t, size_t, size_t>> eds;

            get_es([&](auto u, auto v, auto m)
                   {
                       edges[v].push_back(u);
                       if (!is_directed(_u))
                           edges[u].push_back(v);
                       eds.emplace_back(u, v, m);
                   });

            double dx = nx - x;
            if (dx == 0 || eds.empty())
                return 0.;

            std::vector<std::tuple<size_t, std::vector<size_t>*>> temp;
            for (auto& [v, us] : edges)
                temp.emplace_back(v, &us);

            double dS = 0;
            std::vector<double> xs, nxs;
            #pragma omp parallel for schedule(runtime) reduction(+:dS) \
                firstprivate(xs, nxs)
            for (size_t i = 0; i < temp.size(); ++i)
            {
                auto& [v, us] = temp[i];
                xs.resize(us->size());
                nxs.resize(us->size());
                std::fill(xs.begin(), xs.end(), x);
                std::fill(nxs.begin(), nxs.end(), nx);
                dS += dstate_edges_dS(*us, v, xs, nxs, ea);
            }

            if (x != 0 && nx != 0)
            {
                if (ea.xdist && !_disable_xdist)
                    dS += hist_move_dS(x, nx, _E, _xhist, _xvals,
                                       ea.xdist_uniform, ea.xl1, _xdelta, ea.sbm,
                                       eds.size());
                dS += (edge_x_S(nx, ea) - edge_x_S(x, ea)) * eds.size();
            }
            else if (x == 0)
            {
                assert(nx != 0);
                for (auto& [u, v, m] : eds)
                {
                    dS += add_edge_dS(u, v, m, nx, ea, false);
                    add_edge(u, v, m, nx, [](){}, false);
                }
                for (auto& [u, v, m] : eds)
                    remove_edge(u, v, m, [](){}, false);
            }
            else
            {
                std::vector<int> ms;
                for (auto& [u, v, m] : eds)
                {
                    dS += remove_edge_dS(u, v, m, ea, false);
                    remove_edge(u, v, m, [](){}, false);
                }
                for (auto& [u, v, m] : eds)
                    add_edge(u, v, m, x, [](){}, false);
            }

            return dS;
        }

        template <class Unlock = std::function<void(void)>>
        void remove_edge(size_t u, size_t v, int dm, Unlock&& unlock = [](){},
                         bool dstate=true, bool lock=true)
        {
            //serial part
            if (dm == 0)
            {
                unlock();
                return;
            }

            auto& e = get_u_edge(u, v);
            auto m = _eweight[e];
            auto x = _x[e];

            assert(e != _null_edge);

            do_ulock
                ([&]()
                 {
                     _block_state.template modify_edge<false>(u, v, e, dm);
                     if (e == _null_edge)
                         erase_u_edge(u, v);
                 }, _sbm_mutex, lock);

            #pragma omp atomic
            _E -= dm;

            if ((m == dm) && (_self_loops || u != v))
            {
                if (!_disable_xdist)
                {
                    do_ulock
                        ([&]()
                         {
                             hist_remove(x, _xhist, _xvals);
                         },
                         _x_mutex, lock);
                }

                #pragma omp atomic
                _M--;

                unlock();

                // parallel part
                if (dstate)
                {
                    _dstate->update_edge(u, v, x, 0);
                    if (u != v && !is_directed(_u))
                        _dstate->update_edge(v, u, x, 0);
                }
            }
            else
            {
                unlock();
            }

        }

        template <class Unlock = std::function<void(void)>>
        void add_edge(size_t u, size_t v, int dm, double nx,
                      Unlock&& unlock = [](){}, bool dstate = true,
                      bool lock = true)
        {
            // serial part
            if (dm == 0)
            {
                unlock();
                return;
            }

            assert (nx != 0 || (!_self_loops && u == v));

            auto& e = get_u_edge<true>(u, v);

            do_ulock
                ([&]()
                 {
                     _block_state.template modify_edge<true>(u, v, e, dm);
                 }, _sbm_mutex, lock);

            #pragma omp atomic
            _E += dm;

            if (_eweight[e] == dm)
            {
                _xc[e] = nx;
                if (_self_loops || u != v)
                {
                    if (!_disable_xdist)
                    {
                        do_ulock
                            ([&]()
                             {
                                 hist_add(nx, _xhist, _xvals);
                             }, _x_mutex, lock);
                    }

                    #pragma omp atomic
                    _M++;

                    unlock();

                    //parallel part
                    if (dstate)
                    {
                        _dstate->update_edge(u, v, 0, nx);
                        if (u != v && !is_directed(_u))
                            _dstate->update_edge(v, u, 0, nx);
                    }
                }
                else
                {
                    unlock();
                }
            }
            else
            {
                unlock();
            }
        }

        template <class Unlock = std::function<void(void)>>
        void update_edge(size_t u, size_t v, double nx, Unlock&& unlock = [](){},
                         bool dstate = true, bool lock = true)
        {
            if (_self_loops || u != v)
            {
                // serial part
                auto& e = get_u_edge(u, v);
                auto x = _x[e];
                if (x == nx)
                {
                    unlock();
                    return;
                }

                if (!_disable_xdist)
                {
                    do_ulock
                        ([&]()
                         {
                             hist_remove(x, _xhist, _xvals);
                             hist_add(nx, _xhist, _xvals);
                         }, _x_mutex, lock);
                }

                assert(nx != 0);
                _x[e] = nx;

                unlock();

                // parallel part
                if (dstate)
                {
                    _dstate->update_edge(u, v, x, nx);
                    if (u != v && !is_directed(_u))
                        _dstate->update_edge(v, u, x, nx);
                }
            }
            else
            {
                unlock();
            }
        }

        template <class F>
        void update_edges(F&& get_es, double x, double nx)
        {
            if (nx == x)
                return;

            gt_hash_map<size_t, std::vector<size_t>> edges;
            std::vector<std::tuple<size_t, size_t, size_t>> eds;

            get_es([&](auto u, auto v, auto m)
                   {
                       edges[v].push_back(u);
                       if (!is_directed(_u))
                           edges[u].push_back(v);
                       eds.emplace_back(u, v, m);
                   });

            std::vector<std::tuple<size_t, std::vector<size_t>*>> temp;
            for (auto& [v, us] : edges)
                temp.emplace_back(v, &us);

            std::vector<double> xs, nxs;
            #pragma omp parallel for schedule(runtime) firstprivate(xs, nxs)
            for (size_t i = 0; i < temp.size(); ++i)
            {
                auto& [v, us] = temp[i];
                xs.resize(us->size());
                nxs.resize(us->size());
                std::fill(xs.begin(), xs.end(), x);
                std::fill(nxs.begin(), nxs.end(), nx);
                _dstate->update_edges(*us, v, xs, nxs);
            }

            if (x != 0 && nx != 0)
            {
                for (auto& [u, v, m] : eds)
                {
                    auto e = get_u_edge(u, v);
                    _x[e] = nx;
                }

                if (!_disable_xdist)
                {
                    hist_remove(x, _xhist, _xvals, eds.size());
                    hist_add(nx, _xhist, _xvals, eds.size());
                }
            }
            else if (x == 0)
            {
                assert(nx != 0);
                for (auto& [u, v, m] : eds)
                    add_edge(u, v, m, nx, [](){}, false);
            }
            else
            {
                for (auto& [u, v, m] : eds)
                    remove_edge(u, v, m, [](){}, false);
            }
        }

        double update_node_dS(size_t v, double nt, const dentropy_args_t& ea,
                              bool dstate = true, bool lock = true)
        {
            auto t = _theta[v];
            if (nt == t)
                return 0;

            double dS = 0;
            if (dstate)
                dS += dstate_node_dS(v, t, nt, ea);

            if (ea.tdist && !_disable_tdist)
            {
                do_slock([&]()
                         {
                             dS += hist_move_dS(t, nt, num_vertices(_u), _thist,
                                                _tvals, ea.tdist_uniform, ea.tl1,
                                                _tdelta, false);
                         }, _t_mutex, lock);
            }

            dS += node_x_S(nt, ea) - node_x_S(t, ea);

            return dS;
        }

        template <class VS>
        double update_nodes_dS(VS& vs_, double x, double nx,
                               const dentropy_args_t& ea)
        {
            if (nx == x)
                return 0;
            double dS = 0;
            std::vector<size_t> vs(vs_.begin(), vs_.end());
            #pragma omp parallel for schedule(runtime) reduction(+:dS)
            for (size_t i = 0; i < vs.size(); ++i)
                dS += dstate_node_dS(vs[i], x, nx, ea);
            if (ea.tdist && !_disable_tdist)
                dS += hist_move_dS(x, nx, num_vertices(_u), _thist, _tvals,
                                   ea.tdist_uniform, ea.tl1, _tdelta, false,
                                   vs.size());
            dS += (node_x_S(nx, ea) - node_x_S(x, ea)) * vs.size();
            return dS;
        }

        void update_node(size_t v, double nt, bool lock = true)
        {
            auto t = _theta[v];

            if (nt == t)
                return;

            _theta[v] = nt;

            if (!_disable_tdist)
            {
                do_ulock
                    ([&]()
                     {
                         hist_remove(t, _thist, _tvals);
                         hist_add(nt, _thist, _tvals);
                     }, _t_mutex, lock);
            }
        }

        template <class VS>
        void update_nodes(VS& vs, double x, double nx)
        {
            if (x == nx)
                return;

            for (auto v : vs)
                _theta[v] = nx;

            if (!_disable_tdist)
            {
                hist_remove(x, _thist, _tvals, vs.size());
                hist_add(nx, _thist, _tvals, vs.size());
            }
        }

        double node_TE(size_t u, size_t v)
        {
            return _dstate->node_TE(u, v);
        }

        double node_MI(size_t u, size_t v)
        {
            return _dstate->node_MI(u, v);
        }

        double node_cov(size_t u, size_t v, bool toffset, bool pearson)
        {
            return _dstate->node_cov(u, v, toffset, pearson);
        }

        template <bool keep_iter, class Graph, class WMap, class IMap, class RNG>
        std::tuple<size_t, size_t, size_t>
        get_candidate_edges(Graph& g, size_t k, double r, size_t max_rk,
                            double epsilon, bool c_stop, size_t max_iter, WMap w,
                            IMap ei, const dentropy_args_t& ea, bool exact,
                            bool knn, bool keep_all, bool gradient, double h,
                            const bisect_args_t& ba, bool allow_edges,
                            bool include_edges, bool use_hint, size_t nrandom,
                            bool verbose, RNG& rng_)
        {
            constexpr bool directed = is_directed_::apply<g_t>::type::value;

            size_t N = num_vertices(g);
            size_t M = (N * (N - 1)) / 2;
            bool complete = knn ? k >= N - 1 : k >= M;
            if (complete)
                exact = true;

            parallel_rng<rng_t> prng(rng_);

            double xa = std::numeric_limits<double>::quiet_NaN();
            double xb = std::numeric_limits<double>::quiet_NaN();

            if (!_xvals.empty())
            {
                auto iter = std::lower_bound(_xvals.begin(), _xvals.end(), 0);
                if (iter != _xvals.end())
                    xb = *iter;
                if (iter != _xvals.begin())
                    --iter;
                xa = *iter;
            }

            auto d_ =
                [&](size_t u, size_t v, bool edges)
                {
                    auto [m_, x_] = edge_state(u, v);
                    auto m = m_; // workaround clang bug
                    auto x = x_;
                    if (m > 0 && !edges)
                        return std::numeric_limits<double>::infinity();

                    if (complete)
                        return 0.;

                    auto f =
                        [&](auto nx)
                        {
                            if (m == 0)
                            {
                                if (nx == 0)
                                    return 0.;
                                return add_edge_dS(u, v, 1, nx, ea, true, false);
                            }
                            else
                            {
                                return update_edge_dS(u, v, nx, ea, true, false);
                            }
                        };

                    if (gradient)
                    {
                        if (_xvals.empty())
                        {
                            auto get_dS =
                                [&](auto x, auto nx)
                                {
                                    return (dstate_edge_dS(u, v, x, nx, ea) +
                                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                                };

                            // central difference works better for L1
                            return -abs(get_dS(x - h, x + h) / (2 * h));
                        }
                        else
                        {
                            double dS = f(xa);
                            if (!std::isnan(xb) && xa != xb)
                                dS = std::min(dS, f(xb));
                            return dS;
                        }
                    }

                    if (ba.min_bound != ba.max_bound)
                    {
                        BisectionSampler sampler(f, ba);
                        auto nx = sampler.bisect(x, _xdelta);
                        double dS = sampler.f(nx);

                        if (!_xvals.empty())
                        {
                            auto& rng = prng.get(rng_);
                            nx = sampler.bisect_fb(_xvals, rng);
                            dS = std::min(dS, sampler.f(nx));
                        }

                        if (m > 0)
                            dS = std::min(dS, remove_edge_dS(u, v, m, ea, true,
                                                             false));

                        return dS;
                    }
                    else
                    {
                        return add_edge_dS(u, v, 1, ba.min_bound, ea, true, false);
                    }
                };

            auto d__ =
                [&](size_t u, size_t v, bool edges)
                {
                    if constexpr (directed)
                    {
                        if (u == v)
                            return d_(u, v, edges);
                        else
                            return std::min(d_(u, v, edges),
                                            d_(v, u, edges));
                    }
                    else
                    {
                        return d_(u, v, edges);
                    }
                };

            auto d = make_dist_cache<false, true, keep_iter>
                ([&](auto u, auto v){ return d__(u, v, allow_edges); },
                 num_vertices(_u));

            size_t n_tot = 0;
            size_t n_iter = 1;

            if (k > 0)
            {
                auto& g_r = get_reversed_graph(g);
                if (!knn)
                {
                    if (exact)
                        n_tot = gen_k_nearest_exact<true>(g_r, d, k, false, w);
                    else if (use_hint)
                        std::tie(n_tot, n_iter) =
                            gen_k_nearest<true>(g_r, d, k, r, max_rk, epsilon, c_stop,
                                                max_iter, w, _u, false, verbose, rng_);
                    else
                        std::tie(n_tot, n_iter) =
                            gen_k_nearest<true>(g_r, d, k, r, max_rk, epsilon, c_stop,
                                                max_iter, w, _dummy_hint, false,
                                                verbose, rng_);
                }
                else
                {
                    if (exact)
                        n_tot = gen_knn_exact<true>(g_r, d, k, w);
                    else if (use_hint)
                        n_tot = gen_knn<true>(g_r, d, k, r, max_rk, epsilon, c_stop,
                                              max_iter, w, _u, verbose, rng_);
                    else
                        n_tot = gen_knn<true>(g_r, d, k, r, max_rk, epsilon, c_stop,
                                              max_iter, w, _dummy_hint, verbose, rng_);
                }
            }

            if (keep_all)
            {
                for (auto v : vertices_range(g))
                    clear_vertex(v, g);
                for (auto v : vertices_range(g))
                {
                    for (const auto& [u, li] : d._cache[v])
                    {
                        double l;
                        if constexpr (keep_iter)
                            l = get<0>(li);
                        else
                            l = li;
                        if (std::isinf(l))
                            continue;
                        auto e = boost::add_edge(u, v, g).first;
                        w[e] = l;
                    }
                }
            }

            if constexpr (keep_iter)
            {
                for (auto e : edges_range(g))
                {
                    auto u = source(e, g);
                    auto v = target(e, g);
                    if (u > v)
                        std::swap(u, v);
                    ei[e] = get<1>(d._cache[v][u]);
                }
            }

            if (include_edges)
            {
                for (auto e : edges_range(_u))
                {
                    auto u = source(e, _u);
                    auto v = target(e, _u);
                    if (edge(u, v, g).second || edge(v, u, g).second)
                        continue;
                    boost::add_edge(u, v, g).first;
                }
            }

            if (_self_loops)
            {
                for (auto v : vertices_range(_u))
                {
                    if (edge(v, v, g).second)
                        continue;
                    boost::add_edge(v, v, g);
                }
            }

            std::uniform_int_distribution<size_t> sample(0, num_vertices(_u)-1);
            for (size_t i = 0; i < nrandom; ++i)
            {
                size_t u, v;
                do
                {
                    u = sample(rng_);
                    v = sample(rng_);
                }
                while ((u == v || _self_loops) && edge(v, u, g).second);
                auto ne = boost::add_edge(u, v, g).first;
                w[ne] = d__(u, v, true);
            }

            if constexpr (directed)
            {
                std::vector<std::tuple<size_t, size_t, double>> nes;
                for (auto e : edges_range(g))
                    nes.emplace_back(target(e, g), source(e, g), w[e]);
                for (auto& [s, t, l] : nes)
                {
                    if (s == t)
                        continue;
                    auto e = boost::add_edge(s, t, g).first;
                    w[e] = l;
                }
            }

            return {n_tot, n_iter, d._miss_count};
        }

        std::tuple<double, BisectionSampler>
        bisect_x_init(size_t u, size_t v, const dentropy_args_t& ea,
                      const bisect_args_t& ba)
        {
            size_t m;
            double x;
            std::tie(m, x) = edge_state(u, v);

            auto f =
                [=](auto nx)
                {
                    return (dstate_edge_dS(u, v, x, nx, ea) +
                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                };

            BisectionSampler sampler(f, ba);
            return {x, sampler};
        }

        template <class... RNG>
        std::tuple<double, BisectionSampler>
        bisect_x_disp(size_t u, size_t v, const dentropy_args_t& ea,
                      const bisect_args_t& ba, bool fb,
                      const std::vector<double>& xvals, RNG&... rng)
        {
            auto [x, sampler] = bisect_x_init(u, v, ea, ba);
            double nx;
            if (u == v && !_self_loops)
                nx = 0;
            else if (ba.min_bound == ba.max_bound)
                nx = ba.min_bound;
            else
                nx = fb ? sampler.bisect_fb(xvals, rng...) : sampler.bisect(x, _xdelta);
            sampler.f(nx, true);
            return {nx, sampler};
        }

        template <class... RNG>
        std::tuple<double, BisectionSampler>
        bisect_x(size_t u, size_t v, const dentropy_args_t& ea,
                 const bisect_args_t& ba, bool fb, RNG&... rng)
        {
            return bisect_x_disp(u, v, ea, ba, fb, _xvals, rng...);
        }

        std::tuple<double, BisectionSampler>
        bisect_t_init(size_t v, const dentropy_args_t& ea, const bisect_args_t& ba)
        {
            auto x = _theta[v];
            auto f =
                [=](auto nx)
                {
                    return dstate_node_dS(v, x, nx, ea) +
                        (node_x_S(nx, ea) - node_x_S(x, ea));
                };

            BisectionSampler sampler(f, ba);
            return {x, sampler};
        }

        template <class... RNG>
        std::tuple<double, BisectionSampler>
        bisect_t_disp(size_t v, const dentropy_args_t& ea, const bisect_args_t& ba,
                      bool fb, const std::vector<double>& tvals, RNG&... rng)
        {
            auto [x, sampler] = bisect_t_init(v, ea, ba);
            double nx;
            if (ba.min_bound == ba.max_bound)
                nx = ba.min_bound;
            else
                nx = fb ? sampler.bisect_fb(tvals, rng...) : sampler.bisect(x, _tdelta);
            sampler.f(nx, true);
            return {nx, sampler};
        }

        template <class... RNG>
        std::tuple<double, BisectionSampler>
        bisect_t(size_t v, const dentropy_args_t& ea, const bisect_args_t& ba,
                 bool fb, RNG&... rng)
        {
            return bisect_t_disp(v, ea, ba, fb, _tvals, rng...);
        }

        std::tuple<double, BisectionSampler>
        bisect_xl1(const dentropy_args_t& ea, const bisect_args_t& ba)
        {
            double S0 = entropy(ea);

            auto f =
                [=](auto nx)
                {
                    dentropy_args_t ea_ = ea;
                    ea_.xl1 = nx;
                    return entropy(ea_) - S0;
                };

            BisectionSampler sampler(f, ba);
            double nx = sampler.bisect(ea.xl1, 0);
            return {nx, sampler};
        }

        std::tuple<double, BisectionSampler>
        bisect_tl1(const dentropy_args_t& ea, const bisect_args_t& ba)
        {
            double S0 = entropy(ea);

            auto f =
                [=](auto nx)
                {
                    dentropy_args_t ea_ = ea;
                    ea_.tl1 = nx;
                    return entropy(ea_) - S0;
                };

            BisectionSampler sampler(f, ba);
            double nx = sampler.bisect(ea.tl1, 0);
            return {nx, sampler};
        }

        double edge_diff(size_t u, size_t v, double h, const dentropy_args_t& ea)
        {
            auto get_dS =
                [&](auto x, auto nx)
                {
                    return (dstate_edge_dS(u, v, x, nx, ea) +
                            (edge_x_S(nx, ea) - edge_x_S(x, ea)));
                };

            auto x = get<1>(edge_state(u, v));
            return get_dS(x - h, x + h) / (2 * h);
        }

        double node_diff(size_t v, double h, const dentropy_args_t& ea)
        {
            auto get_dS =
                [&](auto x, auto nx)
                {
                    return (dstate_node_dS(v, x, nx, ea) +
                            (node_x_S(nx, ea) - node_x_S(x, ea)));
                };

            auto x = _theta[v];
            return get_dS(x - h, x + h) / (2 * h);
        }

        void requantize_all_x(double delta)
        {
            if (delta == _xdelta)
                return;

            std::vector<std::mutex> vmutex(num_vertices(_u));

            auto dispatch =
                [&](size_t u, size_t v, auto&& f)
                {
                    bool dir = is_directed(_u) || u == v;

                    if (dir)
                        vmutex[v].lock();
                    else
                        std::lock(vmutex[u], vmutex[v]);

                    f();

                    if (dir)
                    {
                        vmutex[v].unlock();
                    }
                    else
                    {
                        vmutex[u].unlock();
                        vmutex[v].unlock();
                    }
                };

            parallel_edge_loop
                (_u,
                 [&](auto e)
                 {
                     size_t u = source(e, _u);
                     size_t v = target(e, _u);
                     auto x = _x[e];
                     double nx = quantize(x, delta);
                     if (nx == 0)
                         nx = x > 0 ? delta : -delta;
                     dispatch(u, v, [&]() { update_edge(u, v, nx); });
                 });

            _xdelta = delta;
        }

        void requantize_all_theta(double delta)
        {
            if (delta == _tdelta)
                return;

            parallel_vertex_loop
                (_u,
                 [&](auto v)
                 {
                     auto x = _theta[v];
                     double nx = quantize(x, delta);
                     update_node(v, nx);
                 });

            _tdelta = delta;
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_x_disp(size_t u, size_t v, double beta, const dentropy_args_t& ea,
                      const bisect_args_t& ba, bool fb, std::vector<double>& xvals,
                      RNG& rng)
        {
            auto [nx_, sampler] = bisect_x_disp(u, v, ea, ba, fb, xvals, rng);
            return {sampler.sample(beta, _xdelta, rng), sampler};
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_x(size_t u, size_t v, double beta, const dentropy_args_t& ea,
                 const bisect_args_t& ba, bool fb, RNG& rng)
        {
            return sample_x_disp(u, v, beta, ea, ba, fb, _xvals, rng);
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_t_disp(size_t v, double beta, const dentropy_args_t& ea,
                      const bisect_args_t& ba, bool fb, std::vector<double>& tvals,
                      RNG& rng)
        {
            auto [nx_, sampler] = bisect_t_disp(v, ea, ba, fb, tvals, rng);
            return {sampler.sample(beta, _tdelta, rng), sampler};
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_t(size_t v, double beta, const dentropy_args_t& ea,
                 const bisect_args_t& ba, bool fb, RNG& rng)
        {
            return sample_t_disp(v, beta, ea, ba, fb, _tvals, rng);
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_xl1(double beta, const dentropy_args_t& ea,
                   const bisect_args_t& ba, RNG& rng)
        {
            auto [nx_, sampler] = bisect_xl1(ea, ba);
            return {sampler.sample(beta, 0, rng), sampler};
        }

        template <class RNG>
        std::tuple<double, BisectionSampler>
        sample_tl1(double beta, const dentropy_args_t& ea,
                   const bisect_args_t& ba, RNG& rng)
        {
            auto [nx_, sampler] = bisect_tl1(ea, ba);
            return {sampler.sample(beta, 0, rng), sampler};
        }

        template <class Vals, class Val>
        std::tuple<Val, Val>
        get_close_int(Vals& vals, Val x,
                      Val skip = numeric_limits<Val>::quiet_NaN(),
                      Val add = numeric_limits<Val>::quiet_NaN())
        {
            auto [a, b, c] = bracket_closest(vals, x, skip, add);
            if (a == b)
                a = -std::numeric_limits<double>::infinity();
            else
                a += (b - a) / 2;
            if (c == b)
                c = std::numeric_limits<double>::infinity();
            else
                c -= (c - b) / 2;
            return {a, c};
        }

        template <bool try_zero, class F, class U, class Hist, class Vals,
                  class RNG>
        double vals_sweep(F&& f_, U& update_items, Hist& hist, Vals& vals,
                          double delta, double beta, double r,
                          size_t min_size, const bisect_args_t& ba, RNG& rng)
        {
            std::bernoulli_distribution skip(1-r);

            double S = 0;
            for (size_t xi = 0; xi < vals.size(); ++xi)
            {
                if ((get_count(hist, vals[xi]) < min_size) || skip(rng))
                    continue;

                auto f = [&](auto x)
                         {
                             return f_(xi, vals[xi], x);
                         };

                BisectionSampler sampler(f, ba);
                double nx = sampler.bisect(vals[xi], delta);
                double dS = sampler.f(nx, false);

                if (try_zero && std::isinf(beta))
                {
                    double dS0 = sampler.f(0);
                    if (dS0 < dS)
                        std::tie(nx, dS) = std::make_tuple(0., dS0);
                }

                if (std::isinf(beta))
                {
                    if (dS >= 0)
                        continue;
                }
                else
                {
                    nx = sampler.sample(beta, delta, rng);

                    if (!try_zero && nx == 0)
                        continue;

                    if (get_count(hist, nx) > 0)
                        continue;

                    dS = sampler.f(nx, false);

                    double lf = 0;
                    double lb = 0;

                    lf = sampler.lprob(nx, beta, delta);
                    lb = sampler.lprob(vals[xi], beta, delta);

                    std::uniform_real_distribution<> u(0, 1);

                    double a = -beta * dS + lb - lf;

                    if (std::isinf(lb) || (a <= 0 && exp(a) <= u(rng)))
                        continue;
                }

                auto x = vals[xi];

                update_items(xi, x, nx);

                S += dS;
            }
            return S;
        }


        template <class RNG>
        std::tuple<double, size_t>
        xvals_sweep(double beta, double r, size_t min_size,
                    const dentropy_args_t& ea_, const bisect_args_t& ba,
                    RNG& rng)
        {
            dentropy_args_t ea = ea_;

            std::vector<std::vector<std::tuple<size_t, size_t, size_t>>>
                eds(_xhist.size());

            std::vector<double> vals;
            for (auto& [x, c] : _xhist)
                vals.push_back(x);

            std::shuffle(vals.begin(), vals.end(), rng);

            gt_hash_map<double, size_t> vmap;
            for (auto x : vals)
            {
                auto pos = vmap.size();
                vmap[x] = pos;
            }

            for (auto e : edges_range(_u))
            {
                if (!_self_loops && (source(e, _u) == target(e, _u)))
                    continue;
                auto x = _x[e];
                assert(vmap.find(x) != vmap.end());
                auto pos = vmap[x];
                eds[pos].emplace_back(source(e, _u), target(e, _u),
                                      _eweight[e]);
            }

            auto f =
                [&](size_t xi, auto x, auto nx)
                {
                    auto& es = eds[xi];
                    return update_edges_dS([&](auto&& f)
                                           {
                                               for (auto& [u, v, m] : es)
                                                   f(u, v, m);
                                           }, x, nx, ea);
                };

            size_t nmoves = 0;

            auto update_edges =
                [&](size_t xi, double x, double nx)
                {
                    auto& es = eds[xi];
                    this->update_edges([&](auto&& f)
                                       {
                                           for (auto& [u, v, m] : es)
                                               f(u, v, m);
                                       }, x, nx);
                    nmoves += es.size();
                };

            if (std::isinf(beta))
                return {vals_sweep<true>(f, update_edges, _xhist, vals, _xdelta,
                                         beta, r, min_size, ba, rng), nmoves};
            else
                return {vals_sweep<false>(f, update_edges, _xhist, vals, _xdelta,
                                          beta, r, min_size, ba, rng), nmoves};
        }

        template <class RNG>
        std::tuple<double, size_t>
        tvals_sweep(double beta, double r, size_t min_size,
                    const dentropy_args_t& ea, const bisect_args_t& ba, RNG& rng)
        {
            std::vector<double> vals;
            for (auto& [x, c] : _thist)
                vals.push_back(x);

            std::shuffle(vals.begin(), vals.end(), rng);

            gt_hash_map<double, size_t> vmap;
            for (auto x : vals)
            {
                auto pos = vmap.size();
                vmap[x] = pos;
            }

            gt_hash_map<size_t, std::vector<size_t>> vertices;
            for (auto v : vertices_range(_u))
            {
                auto x = _theta[v];
                vertices[vmap[x]].push_back(v);
            }

            auto f =
                [&](size_t xi, auto x, auto nx)
                {
                    return update_nodes_dS(vertices[xi], x, nx, ea);
                };

            size_t nmoves = 0;

            auto update_vertices =
                [&](size_t xi, double x, double nx)
                {
                    auto& vs = vertices[xi];
                    update_nodes(vs, x, nx);
                    nmoves += vs.size();
                };

            return {vals_sweep<false>(f, update_vertices, _thist, vals, _tdelta,
                                      beta, r, min_size, ba, rng), nmoves};
        }

        std::vector<double>& get_xvals()
        {
            return _xvals;
        }

        std::vector<double>& get_tvals()
        {
            return _tvals;
        }

        void set_params(boost::python::dict p)
        {
            _dstate->set_params(p);
        }
    };
};


} // graph_tool namespace

#endif //DYNAMICS_HH
