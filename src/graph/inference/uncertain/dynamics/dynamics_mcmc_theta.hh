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

#ifndef DYNAMICS_THETA_MCMC_HH
#define DYNAMICS_THETA_MCMC_HH

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "../../loops/mcmc_loop.hh"
#include "../../support/fibonacci_search.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "openmp.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((pold,, double, 0))                                                       \
    ((pnew,, double, 0))                                                       \
    ((ptu,, double, 0))                                                        \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((bisect_args,, bisect_args_t, 0))                                         \
    ((verbose,, int, 0))                                                       \
    ((deterministic,, bool, 0))                                                \
    ((sequential,, bool, 0))                                                   \
    ((parallel,, bool, 0))                                                     \
    ((niter,, size_t, 0))


template <class State>
struct MCMCTheta
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    enum class xmove_t { x_old = 0, x_new};

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
            _vlist(num_vertices(_state._u)),
            _vmutex(_vlist.size())
        {
            std::iota(_vlist.begin(), _vlist.end(), 0);
            _xcaches.resize(get_num_threads());

            if (_state._disable_tdist)
                std::tie(_pold, _pnew) = std::make_tuple(0., 1.);

            std::vector<xmove_t> moves
                = {xmove_t::x_old, xmove_t::x_new};
            std::vector<double> probs
                = {_pold, _pnew};
            _move_sampler = Sampler<xmove_t, mpl::false_>(moves, probs);
        }

        Sampler<xmove_t, mpl::false_> _move_sampler;

        typedef typename State::tval_t tval_t;
        typedef tval_t move_t;

        std::vector<size_t> _vlist;

        constexpr static tval_t _null_move = numeric_limits<tval_t>::max();

        std::vector<std::mutex> _vmutex;

        constexpr bool proposal_lock(size_t)
        {
            return true;
        }

        constexpr void proposal_unlock(size_t)
        {
        }

        std::vector<std::tuple<xmove_t, tval_t, double, double,
                               BisectionSampler>> _xcaches;

        std::array<double, 2> _ws;

        template <class RNG>
        bool stage_proposal(size_t v, RNG& rng)
        {
            if (!proposal_lock(v))
                return false;

            auto& [move, nx, dS, lf, sampler] = _xcaches[get_thread_num()];
            nx = numeric_limits<double>::quiet_NaN();
            dS = numeric_limits<double>::quiet_NaN();
            lf = -numeric_limits<double>::infinity();

            move = _move_sampler(rng);

            if (_parallel)
                _tvals_mutex.lock_shared(); // tvals can change

            std::tie(nx, dS, sampler) = sample_nx(v, move == xmove_t::x_old, rng);

            double lf_x_old = -numeric_limits<double>::infinity();
            if (!std::isinf(_beta) && _state.get_count(_state._xhist, nx) > 0)
                lf_x_old = sample_old_x_lprob(nx, sampler);

            if (_parallel)
                _tvals_mutex.unlock_shared(); // tvals can change

            double ptot = _pold + _pnew;

            if (!std::isinf(_beta))
            {
                if (_pold > 0)
                    lf = log(_pold) - log(ptot) + lf_x_old;
                if (_pnew > 0)
                    lf = log_sum_exp(lf,
                                     log(_pnew) - log(ptot) +
                                     sample_new_x_lprob(nx, sampler));
            }

            return true;
        }

        template <class RNG>
        move_t move_proposal(size_t v, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(v, rng);

            auto& [move, nx, dS, lf, sampler] = _xcaches[get_thread_num()];

            return nx;
        }

        void lock_move()
        {
            _move_mutex.lock();
        }

        void unlock_move()
        {
            _move_mutex.unlock();
        }

        void perform_move(size_t v, move_t move)
        {
            if (_parallel)
                _tvals_mutex.lock();

            _state.update_node(v, move);

            if (_parallel)
            {
                _tvals_mutex.unlock();
                unlock_move();
            }
        }

        std::tuple<double, double>
        virtual_move_dS(size_t v, move_t nx)
        {
            auto x = _state._theta[v];
            if (x == nx)
                return {0., 0.};

            auto& [move, nx_, dS_, lf_, sampler] = _xcaches[get_thread_num()];
            double dS = dS_;
            double lf = lf_;

            auto ea = _entropy_args;
            if (!ea.tdist)
                ea.tl1 = 0;

            dS += _state.update_node_dS(v, nx, ea, false);

            double lb = 0;
            double a = 0;
            if (!std::isinf(_beta))
            {
                auto nx_add = _state.get_count(_state._thist, nx) == 0 ?
                    nx : numeric_limits<double>::quiet_NaN();

                double ptot = _pnew + _pold;
                if (_pnew > 0)
                {
                    lb = log(_pnew) - log(ptot) +
                        sample_new_x_lprob(x, sampler);
                    if (_state.get_count(_state._thist, x) > 1)
                        lb = log_sum_exp(lb,
                                         log(_pold) - log(ptot) +
                                         sample_old_x_lprob(x, sampler,
                                                            nx_add));
                }
                else
                {
                    if (_state.get_count(_state._thist, x) > 1)
                        lb = log(_pold) - log(ptot) +
                            sample_old_x_lprob(x, sampler, nx_add);
                    else
                        lb = -numeric_limits<double>::infinity();
                }

                a = lb - lf;
            }

            if (_verbose)
                cout << v << ", x: " << x << ", nx: "
                     << nx << ", dS: " << dS << ", lf: " << lf << ", lb: "
                     << lb << ", a: " << a << ", -dS + a: " << -dS + a << endl;

            return {dS, a};

        }

        template <class RNG>
        auto sample_nx(size_t v, bool told, RNG& rng)
        {
            auto [nx, sampler] = told ?
                ((_pnew == 0 && _ptu == 1) ?
                 _state.bisect_t_init(v, _entropy_args, _bisect_args) :
                 _state.bisect_t(v, _entropy_args, _bisect_args,
                                 told && std::isinf(_beta), rng)) :
                _state.sample_t(v, _beta, _entropy_args, _bisect_args,
                                told && std::isinf(_beta), rng);

            if (told)
            {
                SetBisectionSampler set_sampler(_state._tvals, _ptu, sampler);
                nx = set_sampler.sample(_beta, rng);
            }

            auto dS = sampler.f(nx, false);

            assert(!std::isinf(nx) && !std::isnan(nx));
            assert(!std::isinf(dS) && !std::isnan(dS));

            return std::make_tuple(nx, dS, sampler);
        }

        template <class Sampler>
        double sample_new_x_lprob(tval_t nx, Sampler& sampler)
        {
            return sampler.lprob(nx, _beta, _state._tdelta);
        }

        template <class Sampler>
        double sample_old_x_lprob(tval_t nx, Sampler& sampler,
                                  double x = numeric_limits<double>::quiet_NaN())
        {
            SetBisectionSampler set_sampler(_state._tvals, _ptu, sampler);
            return set_sampler.lprob(nx, _beta,
                                     numeric_limits<double>::quiet_NaN(),
                                     x);
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

    private:
        std::shared_mutex _move_mutex;
        std::shared_mutex _tvals_mutex;
    };
};

} // graph_tool namespace

#endif //DYNAMICS_THETA_MCMC_HH
