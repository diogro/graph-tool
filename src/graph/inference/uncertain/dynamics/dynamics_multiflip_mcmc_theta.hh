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

#ifndef DYNAMICS_MULTIFLIP_MCMC_THETA_HH
#define DYNAMICS_MULTIFLIP_MCMC_THETA_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "openmp.hh"

#include "idx_map.hh"
#include "../../loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((c,, double, 0))                                                          \
    ((psplit,, double, 0))                                                     \
    ((pmerge,, double, 0))                                                     \
    ((pmergesplit,, double, 0))                                                \
    ((nproposal, &, vector<size_t>&, 0))                                       \
    ((nacceptance, &, vector<size_t>&, 0))                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((bisect_args,, bisect_args_t, 0))                                         \
    ((verbose,, int, 0))                                                       \
    ((force_move,, bool, 0))                                                   \
    ((niter,, double, 0))

template <class State>
struct MCMCTheta
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    template <class... Ts>
    class MCMCDynamicsStateImp
        : public MCMCDynamicsStateBase<Ts...>,
          public MergeSplitStateBase
    {
    public:
        GET_PARAMS_USING(MCMCDynamicsStateBase<Ts...>,
                         MCMC_DYNAMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_DYNAMICS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsStateImp(ATs&&... as)
           : MCMCDynamicsStateBase<Ts...>(as...)
        {
            _dS.resize(get_num_threads());
        }

        typedef typename State::tval_t tval_t;

        constexpr static tval_t _null_group = std::numeric_limits<tval_t>::infinity();

        constexpr static double _psingle = 0;
        constexpr static double _pmovelabel = 0;

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 1;
        constexpr static double _pscoalesce = 0; // TODO: no detailed balance avaiable

        constexpr static bool _parallel = true;
        constexpr static bool _relabel = true;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (auto v : vertices_range(_state._u))
                f(v);
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto r : _state._tvals)
                f(r);
        }

        tval_t get_group(size_t v)
        {
            return _state._theta[v];
        }


        template <bool sample_branch=true, class RNG, class VS = std::array<size_t,0>>
        tval_t sample_new_group(size_t v, RNG& rng, VS&& except = VS())
        {
            std::bernoulli_distribution coin(_c);
            std::uniform_real_distribution<>
                unif(std::max(_state._tvals.front() - 2 * _state._tdelta,
                              _bisect_args.min_bound),
                     std::min(_state._tvals.back() + 2 * _state._tdelta,
                              _bisect_args.max_bound));

            auto [x, sampler] =
                _state.sample_t(v, 1., _entropy_args,
                                _bisect_args, false, rng);
            if (coin(rng))
                x = _state.quantize_t(unif(rng));
            while (_state.get_count(_state._thist, x) > 0 ||
                   std::find(except.begin(), except.end(), x) != except.end())
            {
                if (coin(rng))
                    x = _state.quantize_t(unif(rng));
                else
                    x = sampler.sample(1, _state._tdelta, rng);
                assert(x >= _bisect_args.min_bound && x <= _bisect_args.max_bound);
            }

            return x;
        }

        template <class VS = std::array<double,0>>
        double sample_new_group_lprob(size_t v, double s, VS&& except = VS())
        {
            if (_bisect_args.min_bound == _bisect_args.max_bound)
                return 0;

            // if (std::find(except.begin(), except.end(), s) != except.end())
            //     return -numeric_limits<double>::infinity();

            double d = (std::min(_state._tvals.back() + 2 * _state._tdelta,
                                 _bisect_args.max_bound) -
                        std::max(_state._tvals.front() - 2 * _state._tdelta,
                                 _bisect_args.min_bound));

            auto [nx_, sampler] = _state.bisect_t(v,_entropy_args,
                                                  _bisect_args, false);

            SegmentSampler seg = sampler.get_seg_sampler(1);

            if (_state._tdelta == 0)
                return log_sum_exp(log1p(-_c) + seg.lprob(s),
                                   log(_c) - log(d));

            auto get_lprob =
                [&](double x)
                {
                    return seg.lprob_int(x, _state._xdelta);
                };

            double l = get_lprob(s);
            double Z = get_lprob(0);
            for (auto x : _state._tvals)
            {
                if (std::find(except.begin(), except.end(), x) != except.end())
                    continue;
                if (x == s)  // relevant for reverse move!
                    continue;
                Z = log_sum_exp(Z, get_lprob(x));
            }
            for (auto x : except)
            {
                if (x == _null_group)
                    continue;
                Z = log_sum_exp(Z, get_lprob(x));
            }
            l -= log1p(-exp(Z));
            assert(!std::isinf(l));
            return log_sum_exp(log1p(-_c) + l,
                               log(_c) + log(_state._tdelta)
                               - log(d - (_state._tvals.size() +
                                          except.size() + 1) * _state._tdelta));
        }

        std::vector<std::array<std::tuple<double, double>,2>> _dS;
        std::mutex _move_mutex;

        void virtual_move_lock(size_t idx, double r, double s)
        {
            virtual_move_lock(idx, r, std::array<double,1>{s});
        }

        bool _move_locked = false;
        template <size_t d>
        void virtual_move_lock(size_t v, double r, const std::array<double,d>& s)
        {
            auto& dS = _dS[get_thread_num()];
            for (size_t i = 0; i < 2; ++i)
                dS[i] = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
            for (size_t i = 0; i < d; ++i)
            {
                if (std::isinf(s[i]))
                    continue;
                dS[i] = {s[i],
                         _state.dstate_node_dS(v, r, s[i], _entropy_args) +
                         (_state.node_x_S(s[i], _entropy_args) -
                          _state.node_x_S(r, _entropy_args))};
            }
            _move_mutex.lock();
            _move_locked = true;
        }

        constexpr void virtual_move_unlock(size_t)
        {
            _move_locked = false;
            _move_mutex.unlock();
        }

        void move_node(size_t v, double r, bool /*cache*/)
        {
            if (!_move_locked)
            {
                _state.update_node(v, r);
            }
            else
            {
                _state.update_node(v, r);
                _move_mutex.unlock();
            }
        }

        constexpr void reserve_empty_groups(size_t)
        {
        }

        constexpr bool allow_move(size_t, size_t)
        {
            return true;
        }

        double virtual_move(size_t v, double, double s)
        {
            auto& dSs = _dS[get_thread_num()];
            double dS = (get<0>(dSs[0]) == s) ? get<1>(dSs[0]) : get<1>(dSs[1]);
            auto ea = _entropy_args;
            if (!ea.tdist)
                ea.tl1 = 0;
            dS += _state.update_node_dS(v, s, ea, false);
            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        template <class RNG>
        double sample_group(size_t v, bool, //allow_empty,
                            RNG& rng)
        {
            auto r = _state._theta[v];

            std::bernoulli_distribution coin(_c);
            if (coin(rng))
                return uniform_sample(_state._tvals, rng);

            double xa = std::numeric_limits<double>::quiet_NaN();
            double xb = std::numeric_limits<double>::quiet_NaN();

            auto iter = std::lower_bound(_state._tvals.begin(),
                                         _state._tvals.end(), r);

            assert(*iter == r);

            if (iter != _state._tvals.begin())
                xa = *(iter - 1);
            if (iter + 1 != _state._tvals.end())
                xb = *(iter + 1);

            if (!std::isnan(xa) && !std::isnan(xb))
            {
                std::bernoulli_distribution random(.5);
                return random(rng) ? xa : xb;
            }
            else if (!std::isnan(xa))
            {
                return xa;
            }
            else
            {
                return xb;
            }
        };

        double get_move_prob(size_t, double r, double s, bool, //allow_empty,
                             bool)
        {
            double lr = log(_c);
            lr += -log(_state._xvals.size());

            double xa = std::numeric_limits<double>::quiet_NaN();
            double xb = std::numeric_limits<double>::quiet_NaN();

            auto iter = std::lower_bound(_state._tvals.begin(),
                                         _state._tvals.end(), r);

            assert(*iter == r);

            if (iter != _state._tvals.begin())
                xa = *(iter - 1);
            if (iter + 1 != _state._tvals.end())
                xb = *(iter + 1);

            double l = -numeric_limits<double>::infinity();
            if (!std::isnan(xa) && !std::isnan(xb))
            {
                if (s == xa || s == xb)
                    l = -log(2);
            }
            else if (!std::isnan(xa))
            {
                if (s == xa)
                    l = 0;
            }
            else
            {
                if (s == xb)
                    l = 0;
            }

            auto lp = log_sum_exp(l + log1p(-_c), lr);
            assert(lp <= 0);
            return lp;
        }

        template <class VS>
        std::tuple<double, double> relabel_group(double r, VS& vs)
        {
            BisectionSampler
                sampler([&](auto nx)
                        {
                            return _state.update_nodes_dS(vs, r, nx, _entropy_args);
                        }, _bisect_args);
            double nx = sampler.bisect(r, _state._tdelta);
            double dS = sampler.f(nx, false);
            return {nx, dS};
        }
    };

    class gmap_t :
        public gt_hash_map<double, gt_hash_set<size_t>> {};

    template <class T>
    using iset = idx_set<T>;

    template <class T, class V>
    using imap = idx_map<T, V>;

    template <class T>
    using gset = gt_hash_set<T>;

    template <class... Ts>
    class MCMCDynamicsState:
        public MergeSplit<MCMCDynamicsStateImp<Ts...>,
                          size_t,
                          double,
                          iset,
                          imap,
                          gset,
                          gmap_t, false, true>
    {
    public:
        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
           : MergeSplit<MCMCDynamicsStateImp<Ts...>,
                        size_t,
                        double,
                        iset,
                        imap,
                        gset,
                        gmap_t, false, true>(as...)
        {}
    };
};

} // graph_tool namespace

#endif //DYNAMICS_MULTIFLIP_MCMC_THETA_HH
