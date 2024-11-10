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

#ifndef DYNAMICS_MCMC_SWAP_HH
#define DYNAMICS_MCMC_SWAP_HH

#include "config.h"

#include <vector>
#include <mutex>

#include "graph_tool.hh"
#include "../../support/graph_state.hh"
#include "../../loops/mcmc_loop.hh"
#include "dynamics.hh"
#include "segment_sampler.hh"
#include "../../../generation/sampler.hh"
#include "openmp.hh"

#include "dynamics_elist_state.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((ecandidates,, elist_t, 0))                                               \
    ((beta,, double, 0))                                                       \
    ((preplace,, double, 0))                                                   \
    ((pswap,, double, 0))                                                      \
    ((d,, size_t, 0))                                                          \
    ((pself,, double, 0))                                                      \
    ((puniform,, double, 0))                                                   \
    ((pedge,, double, 0))                                                      \
    ((pnearby,, double, 0))                                                    \
    ((pcandidates,, double, 0))                                                \
    ((entropy_args,, dentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((parallel,, bool, 0))                                                     \
    ((niter,, size_t, 0))


enum class move_t { replace = 0, swap, null };

ostream& operator<<(ostream& s, move_t& v);

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

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
            : MCMCDynamicsStateBase<Ts...>(as...),
              _vlist(num_edges(_state._u)),
              _vmutex(num_vertices(_state._u))
        {
            _elist_states_free.reserve(size_t(get_num_threads()));
            for (size_t i = 0; i < size_t(get_num_threads()); ++i)
                _elist_states_free.emplace_back(_ecandidates, _candidates, _d,
                                                _state._self_loops ? _pself : 0,
                                                _puniform, _pedge, _pnearby,
                                                _pcandidates, _state._u);

            _elist_states_edges.reserve(size_t(get_num_threads()));
            for (size_t i = 0; i < size_t(get_num_threads()); ++i)
                _elist_states_edges.emplace_back(_ecandidates, _candidates, 1,
                                                 _state._self_loops ? _pself : 0,
                                                 0, 1, 0, 0,
                                                _state._u);

            std::iota(_vlist.begin(), _vlist.end(), 0);
            _state._eweight.reserve(2 * num_edges(_state._u));
            _state._x.reserve(2 * num_edges(_state._u));
            _dS.resize(get_num_threads());
            _swaps.resize(get_num_threads());

            std::vector<move_t> moves = {move_t::replace, move_t::swap};
            std::vector<double> probs = {_preplace, _pswap};
            _move_sampler = Sampler<move_t, mpl::false_>(moves, probs);
        }

        Sampler<move_t, mpl::false_> _move_sampler;

        typedef typename State::xval_t xval_t;

        std::vector<size_t> _vlist;

        struct swap_t
        {
            size_t u;
            size_t v;
            size_t s;
            size_t t;  // can be null

            size_t m_uv;
            size_t m_sv;
            size_t m_st;
            size_t m_ut;

            double x_uv;
            double x_sv;
            double x_st;
            double x_ut;
        };

        std::vector<swap_t> _swaps;

        constexpr static move_t _null_move = move_t::null;

        std::vector<std::recursive_mutex> _vmutex;

        std::vector<std::tuple<move_t, double, double, double>> _dS;

        template <class RNG>
        bool stage_proposal(size_t pos, RNG& rng)
        {
            auto& [u, v, s, t, m_uv, m_st, m_sv, m_ut, x_uv, x_st, x_sv, x_ut]
                = _swaps[get_thread_num()];
            auto& [move, dS, lf, lb] = _dS[get_thread_num()];
            dS = 0;
            lf = 0;
            lb = 0;

            move = _move_sampler(rng);

            auto& fs = get_elist_state_free();
            auto& es = get_elist_state_edges();

            if (_parallel)
                _move_mutex.lock_shared();

            std::tie(v, u) = es.sample_edge(rng);
            s = fs.sample_edge(v, rng);

            if (move == move_t::swap)
                t = es.sample_edge(s, rng);

            if (!std::isinf(_beta))
            {
                if (move == move_t::replace)
                {
                    lf = es.log_prob(v, u);
                    lf += fs.template log_prob<true>(v, s);
                    double lfa;
                    lfa = es.log_prob(v, s);
                    lfa += fs.template log_prob<true>(v, u);
                    lf = log_sum_exp(lf, lfa);
                }
                else
                {
                    lf = es.log_prob(v, u);
                    lf += fs.template log_prob<true>(v, s);
                    lf += es.template log_prob<true>(s, t);

                    double lfa;
                    lfa = es.log_prob(v, s);
                    lfa += fs.template log_prob<true>(v, u);
                    lfa += es.template log_prob<true>(u, t);
                    lf = log_sum_exp(lf, lfa);

                    lfa = es.log_prob(t, u);
                    lfa += fs.template log_prob<true>(t, s);
                    lfa += es.template log_prob<true>(s, v);
                    lf = log_sum_exp(lf, lfa);

                    lfa = es.log_prob(t, s);
                    lfa += fs.template log_prob<true>(t, u);
                    lfa += es.template log_prob<true>(u, v);
                    lf = log_sum_exp(lf, lfa);
                }
            }

            if (_parallel)
                _move_mutex.unlock_shared();

            if (_parallel)
            {
                // if (std::set({u, v, s, t}).size() < 4)
                // {
                //     move = move_t::null;
                //     return true;
                // }

                if (std::isinf(_beta))
                {
                    auto ret = (move == move_t::replace) ?
                        std::try_lock(_vmutex[u], _vmutex[v], _vmutex[s]) :
                        std::try_lock(_vmutex[u], _vmutex[v], _vmutex[s],
                                      _vmutex[t]);
                    if (ret != -1)
                        return false;
                }
                else
                {
                    if (move == move_t::replace)
                        std::lock(_vmutex[u], _vmutex[v], _vmutex[s]);
                    else
                        std::lock(_vmutex[u], _vmutex[v], _vmutex[s], _vmutex[t]);
                }
            }

            std::tie(m_uv, x_uv) = _state.edge_state(u, v);
            std::tie(m_sv, x_sv) = _state.edge_state(s, v);

            if (move == move_t::swap)
            {
                std::tie(m_st, x_st) = _state.edge_state(s, t);
                std::tie(m_ut, x_ut) = _state.edge_state(u, t);
            }

            switch (move)
            {
            case move_t::replace:
                {
                    if ((!_state._self_loops && (u == v || s == v)) ||
                        x_uv == x_sv)
                    {
                        proposal_unlock(pos);
                        move = move_t::null;
                        break;
                    }

                    // swap s->v and u->v
                    dS += _state.dstate_edges_dS_a({u, s}, v,
                                                   {x_uv, x_sv},
                                                   {x_sv, x_uv},
                                                   _entropy_args);
                    if (!graph_tool::is_directed(_state._u))
                    {
                        dS += _state.dstate_edge_dS(v, u, x_uv, x_sv,
                                                    _entropy_args, false);
                        dS += _state.dstate_edge_dS(v, s, x_sv, x_uv,
                                                    _entropy_args, false);
                    }
                }
                break;
            case move_t::swap:
                {
                    if ((!_state._self_loops && (u == v || s == v ||
                                                 s == t || u == t)) ||
                        (x_uv == x_sv && x_st == x_sv) || (u == s) || (v == t))
                    {
                        proposal_unlock(pos);
                        move = move_t::null;
                        break;
                    }

                    if (x_uv != x_sv)
                        dS += _state.dstate_edges_dS_a({u, s}, v,
                                                       {x_uv, x_sv},
                                                       {x_sv, x_uv},
                                                     _entropy_args);
                    if (x_st != x_ut)
                        dS += _state.dstate_edges_dS_a({s, u}, t,
                                                       {x_st, x_ut},
                                                       {x_ut, x_st},
                                                       _entropy_args);
                    if (!graph_tool::is_directed(_state._u))
                    {
                        if (x_uv != x_sv || x_ut != x_st)
                            dS += _state.dstate_edges_dS_a({v, t}, u,
                                                           {x_uv, x_ut},
                                                           {x_sv, x_st},
                                                           _entropy_args);
                        if (x_st != x_ut || x_sv != x_uv)
                            dS += _state.dstate_edges_dS_a({t, v}, s,
                                                           {x_st, x_sv},
                                                           {x_ut, x_uv},
                                                           _entropy_args);
                    }
                }
                break;
            default:
                break;
            }

            return true;
        }

        void proposal_unlock(size_t)
        {
            if (!_parallel)
                return;

            auto& [u, v, s, t, m_uv, m_st, m_sv, m_ut, x_uv, x_st, x_sv, x_ut]
                = _swaps[get_thread_num()];
            auto& [move, dS, lf, lb] = _dS[get_thread_num()];

            switch (move)
            {
            case move_t::replace:
                {
                    std::array<size_t, 3> vs({u,v,s});
                    for (auto w : vs)
                        _vmutex[w].unlock();
                }
                break;
            case move_t::swap:
                {
                    std::array<size_t, 4> vs({u,v,s,t});
                    for (auto w : vs)
                        _vmutex[w].unlock();
                }
                break;
            default:
                break;
            }
        }

        template <class RNG>
        move_t move_proposal(size_t pos, RNG& rng)
        {
            if (!_parallel)
                stage_proposal(pos, rng);
            auto& [move, dS, lf, lb] = _dS[get_thread_num()];
            return move;
        }

        void lock_move()
        {
            _move_mutex.lock();
        }

        void unlock_move()
        {
            _move_mutex.unlock();
        }

        void update_edge(size_t u, size_t v, size_t m, size_t nm, double x,
                         double nx, bool dstate = true)
        {
            if (m == 0 && nm == 0)
                return;
            if (nm > m)
            {
                if (nm != m)
                    _state.add_edge(u, v, nm - m, nx, [](){}, dstate);
                if (m > 0 && nx != x)
                    _state.update_edge(u, v, nx, [](){}, dstate);
            }
            else
            {
                if (nm != m)
                    _state.remove_edge(u, v, m - nm, [](){}, dstate);
                if (nm > 0 && nx != x)
                    _state.update_edge(u, v, nx, [](){}, dstate);
            }
        }

        double update_edge_dS(size_t u, size_t v, size_t m, size_t nm, double x,
                              double nx, entropy_args_t& ea)
        {
            double dS = 0;
            if (m == 0 && nm == 0)
                return dS;
            if (nm > m)
            {
                if (nm != m)
                    dS += _state.add_edge_dS(u, v, nm - m, nx, ea, false);
                if (m > 0 && nx != x)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            else
            {
                if (nm != m)
                    dS += _state.remove_edge_dS(u, v, m - nm, ea, false);
                if (nm > 0 && nx != x)
                    dS += _state.update_edge_dS(u, v, nx, ea, false);
            }
            return dS;
        }

        void perform_move(size_t pos, move_t move)
        {
            auto& [u, v, s, t, m_uv, m_st, m_sv, m_ut, x_uv, x_st, x_sv, x_ut]
                = _swaps[get_thread_num()];
            auto& [move_, dS, lf, lb] = _dS[get_thread_num()];

            switch (move)
            {
            case move_t::replace:
                update_edge(u, v, m_uv, m_sv,
                            x_uv, x_sv);
                update_edge(s, v, m_sv, m_uv,
                            x_sv, x_uv);
                if (_verbose > 0)
                    cout << "replace\t" << v << ":\t " << u
                         << " ( " << x_uv << " )"
                         << " ->\t" << s
                         << " ( " << x_sv << " )" << ",\t"
                         << dS << " " << lf << " " << lb << endl;
                break;
            case move_t::swap:
                assert(u != s);
                update_edge(u, v, m_uv, m_sv,
                                  x_uv, x_sv);
                update_edge(s, v, m_sv, m_uv,
                                  x_sv, x_uv);
                assert(t != v);
                update_edge(s, t, m_st, m_ut,
                                  x_st, x_ut);
                update_edge(u, t, m_ut, m_st,
                                  x_ut, x_st);

                if (_verbose > 0)
                {
                    cout << "swap (\t" << u << ",\t" << v << ") -> (\t"
                         << s << ",\t" << t << "),\t"
                         << dS << " " << lf << " " << lb << endl;
                    cout << x_uv << " " << x_st << " "
                         << x_ut << " " << x_sv << endl;
                }
                break;
            default:
                break;
            }
            proposal_unlock(pos); //TODO: rename-> unlock_proposal/move_unlock
            unlock_move();
        }


        std::tuple<double, double>
        virtual_move_dS(size_t, move_t move)
        {
            auto& [u, v, s, t, m_uv, m_st, m_sv, m_ut, x_uv, x_st, x_sv, x_ut]
                = _swaps[get_thread_num()];
            auto& [move_, dS_, lf_, lb] = _dS[get_thread_num()];
            auto& es = get_elist_state_edges();
            auto& fs = get_elist_state_free();

            double dS = dS_;
            dS *= _entropy_args.alpha;
            double lf = lf_;
            lb = 0;

            assert(move == move_);

            auto ea = _entropy_args;
            if (!ea.xdist)
                ea.xl1 = 0;
            ea.normal = false;

            switch (move)
            {
            case move_t::replace:
                if (u != s)
                {
                    dS += update_edge_dS(u, v, m_uv, m_sv,
                                               x_uv, x_sv, ea);
                    update_edge(u, v, m_uv, m_sv,
                                      x_uv, x_sv, false);
                    dS += update_edge_dS(s, v, m_sv, m_uv,
                                               x_sv, x_uv, ea);

                    if (!std::isinf(_beta))
                    {
                        update_edge(s, v, m_sv, m_uv,
                                          x_sv, x_uv, false);

                        lb = es.log_prob(v, s);
                        lb += fs.template log_prob<true>(v, u);

                        double lba;
                        lba = es.log_prob(v, u);
                        lba += fs.template log_prob<true>(v, s);

                        lb = log_sum_exp(lb, lba);

                        update_edge(s, v, m_uv, m_sv,
                                          x_uv, x_sv, false);
                    }

                    update_edge(u, v, m_sv, m_uv,
                                      x_sv, x_uv, false);
                }
                // if (_verbose > 0)
                //     cout << "replace " << v << ": " << u << " -> " << s << ", "
                //          << dS << " " << lf << " " << lb << endl;
                break;
            case move_t::swap:
                assert(u != s);
                dS += update_edge_dS(u, v, m_uv, m_sv,
                                           x_uv, x_sv, ea);
                update_edge(u, v, m_uv, m_sv,
                                  x_uv, x_sv, false);
                dS += update_edge_dS(s, v, m_sv, m_uv,
                                           x_sv, x_uv, ea);
                update_edge(s, v, m_sv, m_uv,
                                  x_sv, x_uv, false);
                assert(t != v);
                dS += update_edge_dS(u, t, m_ut, m_st,
                                           x_ut, x_st, ea);
                update_edge(u, t, m_ut, m_st,
                                  x_ut, x_st, false);
                dS += update_edge_dS(s, t, m_st, m_ut,
                                           x_st, x_ut, ea);

                if (!std::isinf(_beta))
                {
                    update_edge(s, t, m_st, m_ut,
                                      x_st, x_ut, false);

                    lb = es.log_prob(v, u);
                    lb += fs.template log_prob<true>(v, s);
                    lb += es.template log_prob<true>(s, t);

                    double lba;
                    lba = es.log_prob(v, s);
                    lba += fs.template log_prob<true>(v, u);
                    lba += es.template log_prob<true>(u, t);
                    lb = log_sum_exp(lb, lba);

                    lba = es.log_prob(t, u);
                    lba += fs.template log_prob<true>(t, s);
                    lba += es.template log_prob<true>(s, v);
                    lb = log_sum_exp(lb, lba);

                    lba = es.log_prob(t, s);
                    lba += fs.template log_prob<true>(t, u);
                    lba += es.template log_prob<true>(u, v);
                    lb = log_sum_exp(lb, lba);

                    update_edge(s, t, m_ut, m_st,
                                      x_ut, x_st, false);
                }

                update_edge(u, t, m_st, m_ut,
                                  x_st, x_ut, false);
                update_edge(s, v, m_uv, m_sv,
                                  x_uv, x_sv, false);
                update_edge(u, v, m_sv, m_uv,
                                  x_sv, x_uv, false);
                // if (_verbose > 0)
                //     cout << "replace (" << u << ", " << v << ") -> ("
                //          << s << ", " << t << "), "
                //          << dS << " " << lf << " " << lb << endl;
                break;
            default:
                break;
            }

            return {dS, lb-lf};
        }

        constexpr bool is_deterministic()
        {
            return false;
        }

        constexpr bool is_sequential()
        {
            return false;
        }

        bool is_parallel()
        {
            return _parallel;
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

        template <class T>
        constexpr bool node_state(T&)
        {
            return false;
        }

        auto& get_elist_state_free()
        {
            return _elist_states_free[get_thread_num()];
        }

        auto& get_elist_state_edges()
        {
            return _elist_states_edges[get_thread_num()];
        }

    private:
        std::vector<elist_state_t<typename State::u_t>> _elist_states_free;
        std::vector<elist_state_t<typename State::u_t>> _elist_states_edges;
        std::vector<std::vector<size_t>> _candidates;
        std::shared_mutex _move_mutex;
    };
};

} // graph_tool namespace

#endif //DYNAMICS_MCMC_SWAP_HH
