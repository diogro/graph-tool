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

#ifndef DYNAMICS_MULTIFLIP_MCMC_HH
#define DYNAMICS_MULTIFLIP_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "dynamics.hh"
#include "samplers.hh"
#include "openmp.hh"

#include "idx_map.hh"
#include "../../loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((state, State&, 0))                                                       \
    ((beta, double, 0))                                                        \
    ((c, double, 0))                                                           \
    ((psplit, double, 0))                                                      \
    ((pmerge, double, 0))                                                      \
    ((pmergesplit, double, 0))                                                 \
    ((nproposal, vector<size_t>&, 0))                                          \
    ((nacceptance, vector<size_t>&, 0))                                        \
    ((gibbs_sweeps, size_t, 0))                                                \
    ((bisect_args, bisect_args_t, 0))                                          \
    ((entropy_args, dentropy_args_t, 0))                                       \
    ((verbose, int, 0))                                                        \
    ((force_move, bool, 0))                                                    \
    ((niter, double, 0))

template <class State>
struct MCMC
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
            for (auto e : edges_range(_state._u))
            {
                auto u = source(e, _state._u);
                auto v = target(e, _state._u);
                if (u == v && !_state._self_loops)
                    continue;
                _elist.emplace_back(u, v);
            }
            _dS.resize(get_num_threads());
        }

        typedef typename State::xval_t xval_t;

        constexpr static xval_t _null_group = std::numeric_limits<xval_t>::infinity();

        constexpr static double _psingle = 0;
        constexpr static double _pmovelabel = 0;

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 1;
        constexpr static double _pscoalesce = 0; // TODO: broken detailed!

        constexpr static bool _parallel = true;
        constexpr static bool _has_replace = true;

        std::vector<std::tuple<size_t,size_t>> _elist;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (size_t i = 0; i < _elist.size(); ++i)
                f(i);
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto r : _state._xvals)
                f(r);
        }

        std::tuple<size_t, size_t> get_edge(size_t idx)
        {
            return _elist[idx];
        }

        xval_t get_group(size_t idx)
        {
            auto [u, v] = get_edge(idx);
            auto [m, x] = _state.edge_state(u, v);
            return x;
        }

        template <class F>
        auto do_lock_dispatch(F&& f, size_t u, size_t v)
        {
            if (is_directed(_state._u) || u == v)
                _state._v_mutex[v].lock();
            else
                std::lock(_state._v_mutex[u], _state._v_mutex[v]);

            auto ret = f();

            _state._v_mutex[v].unlock();
            if (!is_directed(_state._u) && u != v)
                _state._v_mutex[u].unlock();

            return ret;
        }

        template <class F>
        auto do_lock(F&& f, size_t u, size_t v)
        {
            return do_lock_dispatch
                ([&]()
                 {
                     if constexpr (std::is_same_v<typename std::result_of<F&()>::type,void>)
                     {
                         f();
                         return 0;
                     }
                     else
                     {
                         return f();
                     }
                 }, u, v);
        }

        template <bool sample_branch=true, class RNG, class VS = std::array<size_t,0>>
        xval_t sample_new_group(size_t idx, RNG& rng, VS&& except = VS())
        {
            auto [u_, v_] = get_edge(idx);
            auto u = u_; // workaround clang
            auto v = v_;
            std::bernoulli_distribution coin(_c);
            std::uniform_real_distribution<>
                unif(std::max(_state._xvals.front() - 2 * _state._xdelta,
                              _bisect_args.min_bound),
                     std::min(_state._xvals.back() + 2 * _state._xdelta,
                              _bisect_args.max_bound));

            auto x =
                do_lock([&]()
                        {
                            auto [x, sampler] =
                                _state.sample_x(u, v, 1., _entropy_args,
                                                _bisect_args, false, rng);
                            if (coin(rng))
                                x = _state.quantize_x(unif(rng));
                            while (_state.get_count(_state._xhist, x) > 0 ||
                                   x == 0 ||
                                   std::find(except.begin(), except.end(), x) != except.end())
                            {
                                if (coin(rng))
                                    x = _state.quantize_x(unif(rng));
                                else
                                    x = sampler.sample(1, _state._xdelta, rng);
                                assert(x >= _bisect_args.min_bound && x <= _bisect_args.max_bound);
                            }
                            return x;
                        }, u, v);

            assert(x != 0);
            return x;
        }

        template <class VS = std::array<double,0>>
        double sample_new_group_lprob(size_t idx, double s, VS&& except = VS())
        {
            auto [u, v] = get_edge(idx);

            if (_bisect_args.min_bound == _bisect_args.max_bound)
                return 0;

            // if (std::find(except.begin(), except.end(), s) != except.end())
            //     return -numeric_limits<double>::infinity();

            double d = (std::min(_state._xvals.back() + 2 * _state._xdelta,
                                 _bisect_args.max_bound) -
                        std::max(_state._xvals.front() - 2 * _state._xdelta,
                                 _bisect_args.min_bound));

            auto [nx_, sampler] = _state.bisect_x(u, v,_entropy_args,
                                                  _bisect_args, false);

            SegmentSampler seg = sampler.get_seg_sampler(1);

            if (_state._xdelta == 0)
                return log_sum_exp(log1p(-_c) + seg.lprob(s),
                                   log(_c) - log(d));

            auto get_lprob =
                [&](double x)
                {
                    return seg.lprob_int(x, _state._xdelta);
                };

            double l = get_lprob(s);
            double Z = get_lprob(0);
            for (auto x : _state._xvals)
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
            double L = log_sum_exp(log1p(-_c) + l,
                                   log(_c) + log(_state._xdelta)
                                   - log(d - (_state._xvals.size() +
                                              except.size() + 1) * _state._xdelta));
            assert(!std::isinf(L));
            return L;
        }

        std::vector<std::array<std::tuple<double, double>,2>> _dS;
        std::mutex _move_mutex;

        void virtual_move_lock(size_t idx, double s)
        {
            virtual_move_lock(idx, std::array<double,1>{s});
        }

        bool _move_locked = false;
        template <size_t d>
        void virtual_move_lock(size_t idx, const std::array<double,d>& s)
        {
            auto [u, v] = get_edge(idx);
            auto [m, r] = _state.edge_state(u, v);
            if (is_directed(_state._u) || u == v)
                _state._v_mutex[v].lock();
            else
                std::lock(_state._v_mutex[u], _state._v_mutex[v]);
            auto& dS = _dS[get_thread_num()];
            for (size_t i = 0; i < 2; ++i)
                dS[i] = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
            for (size_t i = 0; i < d; ++i)
            {
                if (std::isinf(s[i]))
                    continue;
                dS[i] = {s[i],
                         (r == s[i]) ?
                         0 : (_state.dstate_edge_dS(u, v, r, s[i], _entropy_args) +
                              (_state.edge_x_S(s[i], _entropy_args) -
                               _state.edge_x_S(r, _entropy_args)))};
                assert(!isinf(get<1>(dS[i])) && !isnan(get<1>(dS[i])));
            }
            _move_mutex.lock();
            _move_locked = true;
        }

        constexpr void virtual_move_unlock(size_t idx)
        {
            auto [u, v] = get_edge(idx);
            _move_locked = false;
            _move_mutex.unlock();
            _state._v_mutex[v].unlock();
            if (!is_directed(_state._u) && u != v)
                _state._v_mutex[u].unlock();
        }

        double virtual_move(size_t idx, double s)
        {
            auto [u, v] = get_edge(idx);
            auto [m, r] = _state.edge_state(u, v);

            if (s == r)
                return 0;

            auto& dSs = _dS[get_thread_num()];
            double dS = (get<0>(dSs[0]) == s) ? get<1>(dSs[0]) : get<1>(dSs[1]);
            assert(!isinf(dS) && !isnan(dS));

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            if (m > 0)
            {
                if (s != 0)
                    dS += _state.update_edge_dS(u, v, s, ea, false);
                else
                    dS += _state.remove_edge_dS(u, v, m, ea, false);
            }
            else if (s != 0)
                dS += _state.add_edge_dS(u, v, 1, s, ea, false);

            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        void move_node(size_t idx, double s, bool /*cache*/)
        {
            auto [u_, v_] = get_edge(idx);
            auto u = u_; // workaround clang bug
            auto v = v_; // workaround clang bug
            auto [m_, r_] = _state.edge_state(u, v);
            auto r = r_; // workaround clang bug
            auto m = m_; // workaround clang bug

            auto move =
                [&](auto&& unlock)
                {
                    if (r == s)
                    {
                        unlock();
                        return;
                    }

                    if (m > 0)
                    {
                        if (s != 0)
                            _state.update_edge(u, v, s, unlock);
                        else
                            _state.remove_edge(u, v, m, unlock);
                    }
                    else
                    {
                        assert(s != 0);
                        _state.add_edge(u, v, 1, s, unlock);
                    }
                };

            if (!_move_locked)
            {
                do_lock([&](){ move([](){}); }, u, v);
            }
            else
            {
                move([&]()
                     {
                         _move_locked = false;
                         _move_mutex.unlock();
                     });
                _state._v_mutex[v].unlock();
                if (!is_directed(_state._u) && u != v)
                    _state._v_mutex[u].unlock();
            }
        }

        constexpr void reserve_empty_groups(size_t)
        {
        }

        template <class RNG>
        double sample_group(size_t idx, bool, //allow_empty,
                            RNG& rng)
        {
            std::bernoulli_distribution coin(_c);
            if (coin(rng))
                return uniform_sample(_state._xvals, rng);

            auto [u, v] = get_edge(idx);
            auto [m, r] = _state.edge_state(u, v);

            double xa = std::numeric_limits<double>::quiet_NaN();
            double xb = std::numeric_limits<double>::quiet_NaN();

            auto iter = std::lower_bound(_state._xvals.begin(),
                                         _state._xvals.end(), r);

            assert(*iter == r);

            if (iter != _state._xvals.begin())
                xa = *(iter - 1);
            if (iter + 1 != _state._xvals.end())
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

            auto iter = std::lower_bound(_state._xvals.begin(),
                                         _state._xvals.end(), r);

            assert(*iter == r);

            if (iter != _state._xvals.begin())
                xa = *(iter - 1);
            if (iter + 1 != _state._xvals.end())
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
        std::tuple<double, double> replace_group(double r, VS& vs)
        {
            if (r == 0)
                return {0., 0.};

            if (vs.empty())
                return {r, 0.};

            auto f = [&](auto nx)
                     {
                         return _state.update_edges_dS([&](auto&& f)
                                                       {
                                                           for (auto idx : vs)
                                                           {
                                                               auto [u, v] = get_edge(idx);
                                                               auto [m, x] = _state.edge_state(u, v);
                                                               f(u, v, m);
                                                           }
                                                       }, r, nx, _entropy_args);
                     };

            BisectionSampler<> sampler(f, _bisect_args);
            double nx = sampler.bisect(r, _state._xdelta);
            double dS = sampler.f(nx, false);

            if (abs(nx) < _state._xdelta) // avoid zero
                nx = nx < 0 ? -_state._xdelta : _state._xdelta;

            dS = sampler.f(nx);

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

#endif //DYNAMICS_MULTIFLIP_MCMC_HH
