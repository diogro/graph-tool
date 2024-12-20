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

#ifndef DYNAMICS_MCMC_HH
#define DYNAMICS_MCMC_HH

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "openmp.hh"
#include "../../loops/mcmc_loop.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((step,, double, 0))                                                       \
    ((pold,, double, 0))                                                       \
    ((pxu,, double, 0))                                                        \
    ((intra_sweeps,, size_t, 0))                                               \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((bisect_args,, bisect_args_t, 0))                                         \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))

template <class State>
struct MCMCXDelta
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

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
        : MCMCDynamicsStateBase<Ts...>(as...),
          _vmutex(num_vertices(_state._u))
        {
            for (auto e : edges_range(_state._u))
            {
                auto u = source(e, _state._u);
                auto v = target(e, _state._u);
                if (!_state._self_loops && u == v)
                    continue;
                _edges.emplace_back(u, v);
            }
        }

        std::array<size_t,1> _vlist = {0};

        typedef double move_t;

        constexpr static move_t _null_move = std::numeric_limits<move_t>::quiet_NaN();

        move_t node_state(size_t)
        {
            return _state._xdelta;
        }

        std::vector<std::tuple<size_t, size_t>> _edges;
        std::tuple<double, std::vector<std::tuple<size_t, double>>> _temp, _next;
        std::vector<std::mutex> _vmutex;

        template <class XState>
        void push_state(XState& state)
        {
            auto& [delta, temp] = state;
            delta = _state._xdelta;
            temp.clear();
            for (auto& [u, v] : _edges)
            {
                auto [m, x] = _state.edge_state(u, v);
                assert(std::isinf(_beta) || m > 0);
                temp.emplace_back(m, x);
            }
        }

        void lock_edge(size_t u, size_t v)
        {
            if (is_directed(_state._u) || u == v)
                _vmutex[v].lock();
            else
                std::lock(_vmutex[u], _vmutex[v]);
        }

        void unlock_edge(size_t u, size_t v)
        {
            _vmutex[v].unlock();
            if (!is_directed(_state._u) && u != v)
                _vmutex[u].unlock();
        }

        template <class XState>
        void apply_state(XState& state)
        {
            auto& delta = get<0>(state);
            auto& temp = get<1>(state);
            _state._xdelta = delta;
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 0; i < _edges.size(); ++i)
            {
                auto& [u, v] = _edges[i];
                auto& [nm, nx] = temp[i];
                assert(std::isinf(_beta) || nm > 0);
                lock_edge(u, v);
                auto [m, x] = _state.edge_state(u, v);

                if (m == 0)
                {
                    if (nm > 0)
                        _state.add_edge(u, v, nm, nx);
                }
                else
                {
                    if (nm == 0)
                        _state.remove_edge(u, v, m);
                    else
                        _state.update_edge(u, v, nx);
                }
                unlock_edge(u, v);
            }
        }

        template <bool forward, bool soft, class XState, class RNG>
        std::tuple<double, double> mcmc_sweep(size_t niter,
                                              const XState& xstate, RNG& rng)
        {
            double dS = 0;
            double lp = 0;

            std::shared_mutex move_mutex;

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            for (size_t i = 0; i < niter; ++i)
            {
                #pragma omp parallel for schedule(runtime) reduction(+:dS,lp)
                for (size_t j = 0; j < _edges.size(); ++j)
                {
                    //parallel
                    auto& [u, v] = _edges[j];
                    lock_edge(u, v);
                    auto [m, x] = _state.edge_state(u, v);
                    auto [nx, ddS, xcache] = sample_nx(u, v, rng);
                    if constexpr (!forward)
                        nx = get<1>(xstate[j]);
                    double lf = 0;
                    if (!std::isinf(_beta))
                        lf = sample_x_lprob(nx, xcache);

                    move_mutex.lock();

                    auto unlock = [&]() { move_mutex.unlock(); };

                    // serial
                    if (nx != 0)
                    {
                        if (x != 0)
                            ddS += _state.update_edge_dS(u, v, nx, ea, false);
                        else
                            ddS += _state.add_edge_dS(u, v, 1, nx, ea, false);
                    }
                    else
                    {
                        if (x != 0)
                            ddS += _state.remove_edge_dS(u, v, m, ea, false);
                    }

                    double lb = 0;
                    if (!std::isinf(_beta))
                    {
                        auto x_skip = _state.get_count(_state._xhist, x) == 1 ?
                            x : numeric_limits<double>::quiet_NaN();
                        auto x_add = _state.get_count(_state._xhist, nx) == 0 ?
                            nx : numeric_limits<double>::quiet_NaN();
                        lb = sample_x_lprob(x, xcache, x_skip, x_add);
                    }

                    bool accept = false;
                    double a;
                    if (std::isinf(_beta))
                    {
                        a = accept = ddS < 0;
                    }
                    else
                    {
                        if (nx == 0)
                        {
                            ddS = numeric_limits<double>::infinity();
                            a = -numeric_limits<double>::infinity();
                            accept = false;
                        }
                        else
                        {
                            if constexpr (soft)
                            {
                                a = -_beta * ddS + lb;
                                a -= log_sum_exp(lf, a);
                            }
                            else
                            {
                                a = -_beta * ddS + lb - lf;
                                a = std::min(a, 0.);
                            }
                            std::uniform_real_distribution<> u(0, 1);
                            accept = u(rng) < exp(a);
                        }
                    }

                    lp += lf;
                    if (accept)
                    {
                        if (m > 0)
                        {
                            if (nx != 0)
                                _state.update_edge(u, v, nx, unlock);
                            else
                                _state.remove_edge(u, v, m, unlock);
                        }
                        else if (nx != 0)
                            _state.add_edge(u, v, 1, nx, unlock);
                        else
                            unlock();

                        dS += ddS;
                        lp += a;
                    }
                    else
                    {
                        unlock();
                        lp += log1p(-exp(a));
                    }

                    if (!std::isinf(_beta) && std::isinf(lp))
                        cout << x << " " << nx << " " << a << " "
                             << lf << " " << lb << " " << accept << endl;

                    unlock_edge(u, v);
                }

                if constexpr (!soft)
                    dS += get<0>(_state.xvals_sweep(_beta, 1, 1,
                                                    _entropy_args,
                                                    _bisect_args, rng));
            }
            return {dS, lp};
        }

        template <class RNG>
        double stage(double ndelta, RNG& rng)
        {
            double dS = -_state.entropy(_entropy_args);
            _state.requantize_all_x(ndelta);
            _state._xdelta = ndelta;
            dS += _state.entropy(_entropy_args);
            auto [ddS, lp] = mcmc_sweep<true, false>(_intra_sweeps,
                                                     std::array<std::tuple<size_t,double>, 0>(),
                                                     rng);
            dS += ddS;
            return dS;
        }

        template <class RNG>
        std::tuple<double, double> rescale(double ndelta, RNG& rng)
        {
            double dS = stage(ndelta, rng);
            auto [ddS, lp] = mcmc_sweep<true, true>(1, std::array<std::tuple<size_t,double>, 0>(),
                                                    rng);
            dS += ddS;
            return {dS, lp};
        }

        template <class XState, class RNG>
        double rescale_prob(double ndelta, XState& xstate, RNG& rng)
        {
            stage(ndelta, rng);
            auto [ddS, lp] = mcmc_sweep<false, true>(1, get<1>(xstate), rng);
            return lp;
        }

        double _dS = 0;
        double _lf = 0;
        double _lb = 0;

        template <class RNG>
        move_t move_proposal(size_t, RNG& rng)
        {
            double a = std::max(log(_state._xdelta) - log(_step),
                           log(_state._xdelta_min));
            double b = log(_state._xdelta) + log(_step);
            std::uniform_real_distribution<> r(a, b);
            double ndelta = exp(r(rng));

            push_state(_temp);

            std::tie(_dS, _lf) = rescale(ndelta, rng);

            push_state(_next);

            _lb = rescale_prob(_state._xdelta, _temp, rng);

            _lf += -log(ndelta) - log(b - a);

            a = std::max(log(ndelta) - log(_step), log(_state._xdelta_min));
            b = log(ndelta) + log(_step);

            _lb += -log(_state._xdelta) - log(b - a);

            apply_state(_temp);

            return ndelta;
        }

        void perform_move(size_t, move_t)
        {
            apply_state(_next);
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, move_t ndelta)
        {
            if (_verbose)
                cout << _state._xdelta << " " << ndelta << " "
                     << _dS << " " << _lf << " " << _lb << endl;
            return {_dS, _lb - _lf};
        }

        template <class RNG>
        auto sample_nx(size_t u, size_t v, RNG& rng)
        {
            double pold = _pold;
            if (_state._xvals.empty())
                pold = 0;
            std::bernoulli_distribution rold(pold);
            bool xold = rold(rng);

            auto [nx, sampler] =
                _state.sample_x(u, v, _beta, _entropy_args, _bisect_args, false,
                                rng);

            if (!_state._self_loops && u == v)
                return std::make_tuple(0., 0., sampler);

            if (xold)
            {
                SetBisectionSampler set_sampler(_state._xvals, _pxu, sampler);
                nx = set_sampler.sample(_beta, rng);
            }

            auto dS = sampler.f(nx, false);

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));

            return std::make_tuple(nx, dS, sampler);
        }

        template <class Sampler>
        double sample_new_x_lprob(double nx, Sampler& sampler)
        {
            return sampler.lprob(nx, _beta, _state._xdelta);
        }

        template <class Sampler>
        double sample_old_x_lprob(double nx, Sampler& sampler,
                                  double skip = numeric_limits<double>::quiet_NaN(),
                                  double add = numeric_limits<double>::quiet_NaN())
        {
            SetBisectionSampler set_sampler(_state._xvals, _pxu, sampler);
            return set_sampler.lprob(nx, _beta, skip, add);
        }

        template <class Sampler>
        double sample_x_lprob(double nx, Sampler& sampler,
                              double skip = numeric_limits<double>::quiet_NaN(),
                              double add = numeric_limits<double>::quiet_NaN())
        {
            if (_state._xvals.size() - int(!std::isnan(skip)) + int(!std::isnan(add)) == 0)
                return sample_new_x_lprob(nx, sampler);
            return log_sum_exp(log(_pold) + sample_old_x_lprob(nx, sampler,
                                                               skip, add),
                               log1p(-_pold) + sample_new_x_lprob(nx, sampler));
        }

        template <class RNG>
        double init_iter(RNG& rng)
        {
            std::shuffle(_edges.begin(), _edges.end(), rng);
            return 0;
        }

        constexpr bool is_deterministic()
        {
            return true;
        }

        constexpr bool is_sequential()
        {
            return true;
        }

        auto& get_vlist()
        {
            return _vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }
    };
};

} // graph_tool namespace

#endif //DYNAMICS_MCMC_HH
