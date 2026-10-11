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

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "../../support/segment_sampler.hh"
#include "samplers.hh"
#include "openmp.hh"

#include "dynamics_elist_state.hh"

#include "../../loops/mcmc_loop.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((state, State&, 0))                                                       \
    ((m, double, 0))                                                           \
    ((elist, elist_t, 0))                                                      \
    ((ecandidates, elist_t, 0))                                                \
    ((beta, double, 0))                                                        \
    ((pold, double, 0))                                                        \
    ((pnew, double, 0))                                                        \
    ((pxu, double, 0))                                                         \
    ((pm, double, 0))                                                          \
    ((premove, double, 0))                                                     \
    ((d, size_t, 0))                                                           \
    ((pself, double, 0))                                                       \
    ((puniform, double, 0))                                                    \
    ((pedge, double, 0))                                                       \
    ((pnearby, double, 0))                                                     \
    ((pcandidates, double, 0))                                                 \
    ((binary, bool, 0))                                                        \
    ((entropy_args, dentropy_args_t, 0))                                       \
    ((bisect_args, bisect_args_t, 0))                                          \
    ((verbose, int, 0))                                                        \
    ((sequential, bool, 0))                                                    \
    ((deterministic, bool, 0))                                                 \
    ((parallel, bool, 0))                                                      \
    ((pseudo, bool, 0))                                                        \
    ((niter, size_t, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    template <class... Ts>
    class MCMCDynamicsState
        : public MCMCDynamicsStateBase<Ts...>,
          public MetropolisStateBase
    {
    public:
        GET_PARAMS_USING(MCMCDynamicsStateBase<Ts...>,
                         MCMC_DYNAMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_DYNAMICS_STATE_params(State))

        enum class xmove_t { x_new = 0, x_old, remove, m };

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
            : MCMCDynamicsStateBase<Ts...>(as...),
              _edges(std::isinf(_beta) ? _elist.shape()[0] : _m * num_vertices(_state._u)),
              _vmutex(num_vertices(_state._u)),
              _xvals_mutex(_state._x_mutex)
        {
            _elist_states.reserve(size_t(get_num_threads()));
            for (size_t i = 0; i < size_t(get_num_threads()); ++i)
                _elist_states.emplace_back(_ecandidates, _candidates, _d,
                                           _state._self_loops ? _pself : 0,
                                           _puniform, _pedge, _pnearby,
                                           _pcandidates, _state._u);

            size_t E = std::max(num_edges(_state._u), _elist.shape()[0]);
            _state._eweight.reserve(3 * E * std::max(_m, 1.));
            _state._x.reserve(3 * E * std::max(_m, 1.));
            _xcaches.resize(get_num_threads());

            if (std::isinf(_beta))
            {
                for (size_t i = 0; i < _elist.shape()[0]; ++i)
                    _edges[i] = {_elist[i][0], _elist[i][1]};
            }

// #ifndef NDEBUG
//             for (auto v : vertices_range(_state._u))
//                 _state._dstate.check_m(_state, v);
// #endif
            if (_state._disable_xdist)
                std::tie(_pold, _pnew) = std::make_tuple(0., _pold + _pnew);
            if (_binary)
                _pm = 0;
            _probs.resize(get_num_threads());
        }

        std::vector<xmove_t> _moves = {xmove_t::x_new, xmove_t::x_old,
                                       xmove_t::remove, xmove_t::m};
        std::vector<std::vector<double>> _probs;

        typedef typename State::xval_t xval_t;

        std::vector<std::tuple<size_t, size_t>> _edges;

        typedef std::tuple<int, xval_t> move_t;

        constexpr static move_t _null_move = {0, 0};

        std::bernoulli_distribution _xmove;

        std::vector<std::mutex> _vmutex;

        std::vector<std::tuple<xmove_t, size_t, xval_t, double, double,
                               BisectionSampler<>>> _xcaches;

        bool proposal_lock(std::tuple<size_t, size_t>& uv)
        {
            if (!_parallel)
                return true;

            auto& [u, v] = uv;

            if (is_directed(_state._u) || u == v)
            {
                if (std::isinf(_beta) || _pseudo)
                    return _vmutex[v].try_lock();
                _vmutex[v].lock();
                return true;
            }
            else
            {
                if (std::isinf(_beta) || _pseudo)
                    return std::try_lock(_vmutex[u], _vmutex[v]) == -1;
                else
                    std::lock(_vmutex[u], _vmutex[v]);
                return true;
            }
        }

        void proposal_unlock(std::tuple<size_t, size_t>& uv)
        {
            if (!_parallel)
                return;
            auto& [u, v] = uv;
            _vmutex[v].unlock();
            if (!is_directed(_state._u) && u != v)
                _vmutex[u].unlock();
        }

        std::tuple<size_t, xval_t> edge_state(size_t u, size_t v)
        {
            return _state.edge_state(u, v);
        }

        template <class RNG>
        bool stage_proposal(std::tuple<size_t, size_t>& uv, RNG& rng)
        {
            // captures not supported with clang w/ openmp
            auto& u = get<0>(uv);
            auto& v = get<1>(uv);

            auto& es = get_elist_state();

            auto& [move, nm, nx, dS, lf, sampler] = _xcaches[get_thread_num()];
            nx = numeric_limits<double>::quiet_NaN();
            dS = 0;
            lf = 0;

            double lf_e = 0;
            if (!std::isinf(_beta))
            {
                do_slock_d
                    ([&]()
                     {
                         std::tie(u, v) = es.sample_edge(rng);
                         lf_e = es.log_prob(u, v);
                     },  _parallel, _state._sbm_mutex);
            }

            if (!proposal_lock(uv))
                return false;

            auto [m, x] = edge_state(u, v);
            nm = m;

            double pnew = _pnew;
            double pold = _pold;
            double premove = _premove;
            double pm = _pm;

            if (m == 0)
                premove = pm = 0;

            auto& probs = _probs[get_thread_num()];

            if (!_pseudo)
            {
                if (_parallel && (_state._self_loops || u != v))
                    _xvals_mutex.lock_shared(); // xvals can change

                if (_state._xvals.empty())
                {
                    if (pnew == 0)
                        pnew = 1;
                    pold = 0;
                }
            }

            probs = {pnew, pold, premove, pm};

            Sampler<xmove_t> move_sampler(_moves, probs);

            move = move_sampler(rng);

            if (_parallel && !_pseudo && (_state._self_loops || u != v) && move != xmove_t::x_old)
                _xvals_mutex.unlock_shared(); // local info only from this point

            bool new_nx;
            if (move == xmove_t::x_new || move == xmove_t::x_old || !std::isinf(_beta))
                std::tie(nx, dS, sampler, new_nx) = sample_nx(u, v, move == xmove_t::x_old, rng);

            if (_pseudo && move == xmove_t::x_old && std::isnan(nx))
            {
                if (pnew == 0)
                    pnew = 1;
                pold = 0;
                std::tie(nx, dS, sampler, new_nx) = sample_nx(u, v, false, rng);
            }

            double lf_x_old = -numeric_limits<double>::infinity();
            if (!std::isinf(_beta) && !new_nx)
                lf_x_old = sample_old_x_lprob(nx, sampler);

            if (_parallel && !_pseudo && (_state._self_loops || u != v) && move == xmove_t::x_old)
                _xvals_mutex.unlock_shared(); // local info only from this point

            if (m == 0 || move == xmove_t::m)
                nm = sample_m(m, rng);
            else if (move == xmove_t::x_new && nx == 0)
                move = xmove_t::remove;

            if (nx == 0)
                nm = 0;

            double ptot = pnew + pold + premove + pm;

            switch (move)
            {
            case xmove_t::remove:
                nx = 0;
                nm = 0;
                if (u != v || _state._self_loops)
                {
                    dS = _state.dstate_edge_dS(u, v, x, 0, _entropy_args);
                    dS += (_state.edge_x_S(0, _entropy_args) -
                           _state.edge_x_S(x, _entropy_args));
                }
                if (!std::isinf(_beta))
                {
                    lf = log(premove) - log(ptot);
                    if (pnew > 0)
                    {
                        if (u != v || _state._self_loops)
                            lf = log_sum_exp(lf,
                                             log(pnew) - log(ptot) +
                                             sample_new_x_lprob(0, sampler));
                        else
                            lf = log_sum_exp(lf,
                                             log(pnew) - log(ptot));
                    }
                }
                break;
            case xmove_t::x_new: [[fallthrough]];
            case xmove_t::x_old:
                if (!std::isinf(_beta))
                {
                    if (u == v && !_state._self_loops)
                    {
                        lf = log(pnew + pold) - log(ptot);
                    }
                    else
                    {
                        lf = log(pold) - log(ptot) + lf_x_old;
                        if (pnew > 0)
                        {
                            lf = log_sum_exp(lf,
                                             log(pnew) - log(ptot) +
                                             sample_new_x_lprob(nx, sampler));
                        }
                    }
                    if (m == 0 && nm > 0)
                        lf += sample_m_lprob(nm, m);
                }
                break;
            case xmove_t::m:
                nx = x;
                dS = 0;
                if (!std::isinf(_beta))
                {
                    lf = log(pm) - log(ptot);
                    lf += sample_m_lprob(nm, m);
                }
                break;
            default:
                break;
            }

            lf += lf_e;

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));
            assert(nm == 0 || nx != 0 || (!_state._self_loops && u == v));

            return true;
        }

        template <class RNG>
        move_t move_proposal(std::tuple<size_t, size_t>& uv, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(uv, rng);

            auto& [u, v] = uv;
            auto [m, x] = edge_state(u, v);
            auto& [move, nm, nx, dS, lf, sampler] = _xcaches[get_thread_num()];

            assert(!std::isinf(dS));

            int dm = int(nm) - int(m);

            return {dm, nx};
        }

        void lock_move()
        {
            if (!_pseudo)
                _move_mutex.lock();
        }

        void unlock_move()
        {
            if (!_pseudo)
                _move_mutex.unlock();
        }

        void perform_move(std::tuple<size_t, size_t>& uv, move_t& move)
        {
            size_t u = get<0>(uv);
            size_t v = get<1>(uv);
            auto [m, x] = _state.edge_state(u, v);
            auto& [dm, nx] = move;

            auto unlock =
                [&]()
                {
                    if (_parallel)
                    {
                        proposal_unlock(uv);
                        unlock_move();
                    }
                };

            if (dm == 0 && nx == x)
            {
                unlock();
                return;
            }

            if (dm == 0)
            {
                _state.update_edge(u, v, nx, unlock, true, _parallel);
            }
            else if (dm > 0)
            {
                if (m == 0)
                {
                    _state.add_edge(u, v, dm, nx, unlock, true, _parallel);
                }
                else
                {
                    _state.add_edge(u, v, dm, nx, [](){}, true, _parallel);
                    _state.update_edge(u, v, nx, unlock, true, _parallel);
                }
            }
            else
            {
                if (m + dm == 0)
                {
                    _state.remove_edge(u, v, -dm, unlock, true, _parallel);
                }
                else
                {
                    _state.remove_edge(u, v, -dm, [](){}, true, _parallel);
                    _state.update_edge(u, v, nx, unlock, true, _parallel);
                }
            }

            if (_verbose)
                cout << "u: " << u << ", v: " << v << ", m: " << m
                     << ", m + dm: " << m + dm << ", nx: "
                     << nx << " (moved)" << endl;
        }

        std::tuple<double, double>
        virtual_move(std::tuple<size_t, size_t>& uv, move_t& mv)
        {
            auto& u = get<0>(uv);  // captures are not supported by clang w/ openmp
            auto& v = get<1>(uv);

            auto mx = _state.edge_state(u, v);
            auto& m = get<0>(mx);
            auto& x = get<1>(mx);

            auto& dm = get<0>(mv);
            auto& nx = get<1>(mv);

            if (dm == 0 && nx == x)
                return {0., 0.};

            auto& [move, nm, nx_, dS_, lf_, sampler] = _xcaches[get_thread_num()];
            double dS = dS_;
            double lf = lf_;

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            if (dm == 0)
            {
                dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            else if (dm > 0)
            {
                dS += _state.add_edge_dS(u, v, dm, nx, ea, false);
                if (m > 0)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            else
            {
                dS += _state.remove_edge_dS(u, v, -dm, ea, false);
                if (m + dm > 0)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }

            assert(!std::isinf(dS) && !std::isnan(dS));

            double a = 0;
            double lb = 0;

            if (!std::isinf(_beta))
            {
                double pnew = _pnew;
                double pold = _pold;
                double premove = _premove;
                double pm = _pm;

                if (m + dm == 0)
                    premove = pm = 0;

                bool is_new_nx = false;
                bool is_last_x = false;
                do_slock_d
                    ([&]()
                     {
                        if (_state._xvals.size() == 1 &&
                            _state.get_count(_state._xhist, x) == 1 && m + dm == 0)
                        {
                            if (pnew == 0)
                                pnew = 1;
                            pold = 0;
                        }
                        is_new_nx = (_state.get_count(_state._xhist, nx) == 0);
                        is_last_x = (_state.get_count(_state._xhist, x) == 1);
                     }, _parallel && _pseudo, _xvals_mutex);

                double ptot = pnew + pold + premove + pm;

                double nx_add = is_new_nx ? nx : numeric_limits<double>::quiet_NaN();

                auto& es = get_elist_state();

                if (m == 0)
                {
                    lb = log(premove) - log(ptot);
                    if (pnew > 0)
                    {
                        if (u != v || _state._self_loops)
                            lb = log_sum_exp(lb,
                                             log(pnew) - log(ptot) +
                                             sample_new_x_lprob(0, sampler));
                        else
                            lb = log_sum_exp(lb,
                                             log(pnew) - log(ptot));
                    }
                    do_slock_d
                        ([&]()
                         {
                             lb += es.log_prob(u, v, 1, u, v);
                         }, _parallel && _pseudo, _state._sbm_mutex);
                }
                else
                {
                    switch (move)
                    {
                    case xmove_t::remove:
                        if (u != v || _state._self_loops)
                        {
                            if (pnew > 0)
                            {
                                lb = log(pnew) - log(ptot) +
                                    sample_new_x_lprob(x, sampler);
                                if (!is_last_x)
                                    lb = log_sum_exp(lb,
                                                     log(pold) - log(ptot) +
                                                     sample_old_x_lprob(x, sampler));
                            }
                            else
                            {
                                if (!is_last_x)
                                    lb = log(pold) - log(ptot) +
                                        sample_old_x_lprob(x, sampler);
                                else
                                    lb = -numeric_limits<double>::infinity();
                            }
                        }
                        else
                        {
                            lb = log(pnew + pold) - log(ptot);
                        }
                        lb += sample_m_lprob(m, 0);
                        do_slock_d
                            ([&]()
                             {
                                 lb += es.log_prob(u, v, -1, u, v);
                             }, _parallel && _pseudo, _state._sbm_mutex);
                        break;
                    case xmove_t::x_new: [[fallthrough]];
                    case xmove_t::x_old:
                        if (pnew > 0)
                        {
                            lb = log(pnew) - log(ptot) +
                                sample_new_x_lprob(x, sampler);
                            if (!is_last_x)
                                lb = log_sum_exp(lb,
                                                 log(pold) - log(ptot) +
                                                 sample_old_x_lprob(x, sampler,
                                                                    nx_add));
                        }
                        else
                        {
                            if (!is_last_x)
                                lb = log(pold) - log(ptot) +
                                    sample_old_x_lprob(x, sampler, nx_add);
                            else
                                lb = -numeric_limits<double>::infinity();
                        }
                        do_slock_d
                            ([&]()
                             {
                                 lb += es.log_prob(u, v);
                             }, _parallel && _pseudo, _state._sbm_mutex);
                        break;
                    case xmove_t::m:
                        lf = log(pm) - log(ptot);
                        lb += sample_m_lprob(m, m + dm);
                        do_slock_d
                            ([&]()
                             {
                                 lb += es.log_prob(u, v);
                             }, _parallel && _pseudo, _state._sbm_mutex);
                        break;
                    }
                }

                a = lb - lf;
            }

            if (_verbose)
                cout << "u: " << u << ", v: " << v << ", m: " << m
                     << ", m + dm: " << m + dm << ", x: " << x << ", nx: "
                     << nx << ", dS: " << dS << ", lf: " << lf << ", lb: " << lb << ", a: "
                     << a << ", -dS + a: " << -dS + a << endl;

            return {dS, a};
        }

        template <class RNG>
        auto sample_nx(size_t u, size_t v, bool xold, RNG& rng)
        {
            bool fb;
            std::vector<double> xvals_local;
            std::vector<double>* xvals;
            if (_parallel && _pseudo)
            {
                do_slock
                ([&]()
                 {
                     fb = xold && std::isinf(_beta) && !_state._xvals.empty();
                     xvals_local = _state._xvals;
                 }, _xvals_mutex);
                xvals = &xvals_local;
            }
            else
            {
                xvals = &_state._xvals;
                fb = xold && std::isinf(_beta) && !_state._xvals.empty();
            }

            auto [nx, sampler] =
                xold ?
                ((_pnew == 0 && _pxu == 1) ?
                 _state.bisect_x_init(u, v, _entropy_args, _bisect_args) :
                 _state.bisect_x_disp(u, v, _entropy_args, _bisect_args, fb, *xvals, rng)) :
                _state.sample_x_disp(u, v, _beta, _entropy_args, _bisect_args, fb, *xvals, rng);

            if (!_state._self_loops && u == v)
                return std::make_tuple(0., 0., sampler, false);

            bool new_x = true;
            if (xold)
            {
                do_slock_d
                    ([&]()
                     {
                         if (_state._xvals.empty())
                         {
                             nx = numeric_limits<double>::quiet_NaN();
                         }
                         else
                         {
                             SetBisectionSampler set_sampler(_state._xvals, _pxu, sampler);
                             nx = set_sampler.sample(_beta, rng);
                             new_x = false;
                         }
                     },
                     _parallel && _pseudo,
                     _xvals_mutex);

                if (std::isnan(nx))
                    return std::make_tuple(nx, nx, sampler, false);
            }
            else
            {
                do_slock_d
                    ([&]()
                     {
                         new_x = (_state.get_count(_state._xhist, nx) == 0);
                     },
                     _parallel && _pseudo,
                     _xvals_mutex);
            }

            assert(nx >= _bisect_args.min_bound);
            assert(nx <= _bisect_args.max_bound);

            auto dS = sampler.f(nx, false);

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));

            return std::make_tuple(nx, dS, sampler, new_x);
        }

        template <class Sampler>
        double sample_new_x_lprob(xval_t nx, Sampler& sampler)
        {
            return sampler.lprob(nx, _beta, _state._xdelta);
        }

        template <class Sampler>
        double sample_old_x_lprob(xval_t nx, Sampler& sampler,
                                  double add = numeric_limits<double>::quiet_NaN())
        {
            return
                do_slock_d
                ([&]()
                 {
                     if (_state._xvals.empty())
                         return std::numeric_limits<double>::quiet_NaN();

                     SetBisectionSampler set_sampler(_state._xvals, _pxu, sampler);

                     return set_sampler.lprob(nx, _beta,
                                              numeric_limits<double>::quiet_NaN(), add);
                 },
                 _parallel && _pseudo,
                 _xvals_mutex);
        }

        template <class RNG>
        size_t sample_m(size_t m, RNG& rng)
        {
            if (_binary || m == 0)
                return 1;
            double a = m + .5;
            double p = 1/(a+1);
            std::geometric_distribution<size_t> random(p);
            return random(rng) + 1;
        }

        double sample_m_lprob(size_t nm, size_t m)
        {
            if (_binary || m == 0)
            {
                if (nm == 1)
                    return 0;
                else
                    return -std::numeric_limits<double>::infinity();
            }
            double a = m + .5;
            double p = 1/(a+1);
            return (nm - 1) * log1p(-p) + log(p);
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        bool is_parallel()
        {
            return _parallel;
        }

        auto& get_vlist()
        {
            return _edges;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }

        auto& get_elist_state()
        {
            return _elist_states[get_thread_num()];
        }

    private:
        std::vector<elist_state_t<typename State::u_t>> _elist_states;
        std::vector<std::vector<size_t>> _candidates;
        std::shared_mutex _move_mutex;
        std::shared_mutex& _xvals_mutex;
    };
};

} // graph_tool namespace

namespace std
{
template <class T1, class T2>
struct formatter<std::tuple<T1, T2>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const std::tuple<T1, T2>& v, format_context& ctx) const
    {
        string temp;
        format_to(back_inserter(temp), "({}, {})", get<0>(v), get<1>(v));
        return formatter<string_view>().format(temp, ctx);
    }
};
}
