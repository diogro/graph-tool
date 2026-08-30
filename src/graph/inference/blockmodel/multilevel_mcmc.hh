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

#ifndef GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH
#define GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

#include "idx_map.hh"
#include "../loops/multilevel.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_LOOP_STATE_params(State)                                          \
    MULTILEVEL_STATE_params(State)

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCLoopStateBase, MCMC_LOOP_STATE_params(State))

    typedef State::vertex_t vertex_t;
    typedef State::group_t group_t;

    template <class... Ts>
    class MCMCLoopStateImp
        : public MCMCLoopStateBase<Ts...>,
          public MultilevelStateBase
    {
    public:
        GET_PARAMS_USING(MCMCLoopStateBase<Ts...>,
                         MCMC_LOOP_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_LOOP_STATE_params(State))

        template <class... ATs>
        MCMCLoopStateImp(ATs&&... as)
          : MCMCLoopStateBase<Ts...>(std::forward<ATs>(as)...),
            _m_entries(get_num_threads())
        {
            GILRelease gil_release;

            if (_vertices.empty())
            {
                for (auto v : vertices_range(_state._g))
                    _vertices.push_back(v);
            }

            if (_global_moves)
            {
                idx_set<group_t> rs_min, rs_max;
                for (auto v : vertices_range(_state._g))
                {
                    rs_min.insert(_b_min[v]);
                    rs_max.insert(_b_max[v]);
                }
                _has_b_min = rs_min.size() == _B_min;
                _has_b_max = rs_max.size() == _B_max;
            }

            size_t L = _state.get_L();

            if constexpr (std::is_same_v<typename State::coupled_state_t::m_entries_t,
                                         typename State::m_entries_t>)
            {
                for (auto&& [i, mes] : enumerate(_m_entries))
                {
                    mes.resize(L);
                    for (size_t l = 0; l < L - 1; ++l)
                        mes[l]._next = &mes[l+1];
                }
            }
            else
            {
                _m_entries_coupled.resize(get_num_threads());
                for (auto& mes : _m_entries_coupled)
                {
                    mes.resize(L);
                    for (size_t l = 0; l < L - 1; ++l)
                        mes[l]._next = &mes[l+1];
                }

                for (auto&& [i, mes] : enumerate(_m_entries))
                {
                    mes.resize(1);
                    mes[0]._next = &_m_entries_coupled[i][0];
                }
            }

            _state.clear_egroups();
        }

        ~MCMCLoopStateImp() { _state.clear_egroups(); }

        constexpr static group_t null_group = State::null_group;

        constexpr static bool _has_merge = State::spec_t::has_merge;
        constexpr static bool _has_parallel_move = State::spec_t::has_parallel_move;
        constexpr static bool _has_relabel = State::spec_t::has_relabel;
        constexpr static bool _has_parallel_relabel = State::spec_t::has_parallel_relabel;
        constexpr static bool _has_extra_sweep = State::spec_t::has_extra_sweep;

        bool _has_b_max = false;
        bool _has_b_min = false;

        std::vector<std::vector<typename State::m_entries_t>> _m_entries;
        std::vector<std::vector<typename State::coupled_state_t::m_entries_t>> _m_entries_coupled;

        idx_set<group_t> _rs;

        constexpr static group_t _null_group = null_group;

        template <class F>
        void iter_nodes(F&& f)
        {
            auto iter_vs =
                [&](auto&& vs)
                {
                    for (auto v : vs)
                    {
                        if (_state.node_weight(v) > 0)
                            f(v);
                    }
                };

            if (_vertices.empty())
                iter_vs(vertices_range(_state._g));
            else
                iter_vs(_vertices);
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (auto& [r, vs] : _state._groups)
            {
                if (!vs.empty())
                    f(r);
            }
        }

        group_t get_group(vertex_t v)
        {
            return _state._b[v];
        }

        template <class RNG>
        group_t get_new_group(vertex_t v, bool inherit, RNG& rng)
        {
            group_t t = _state.get_empty_group(v, rng, inherit);
            if constexpr (_has_relabel)
                relabel_group(t, sample_group_label(t, rng));
            return t;
        }

        void reserve_empty_group(group_t r)
        {
            _state._empty_groups.erase(r);
        }

        auto& get_m_entries(int tnum)
        {
            if (_parallel)
                return _m_entries[tnum][0];
            else
                return _m_entries[0][0];
        }

        void move_node(vertex_t v, group_t r, bool cache)
        {
            if (cache)
                _state.move_vertex(v, r, get_m_entries(0));
            else
                _state.move_vertex(v, r);
        }

        template <class Vs, class Groups>
        double move_nodes_parallel(Vs&& vs, Groups&& rs)
        {
            if (std::ranges::size(vs) >= _parallel_thres)
            {
                double Sb = entropy();
                _state.move_vertices_parallel(vs, rs);
                return entropy() - Sb;
            }
            else
            {
                double dS = 0;
                for (auto&& [v, nr] : std::views::zip(vs, rs))
                {
                    dS += _state.virtual_move(v, nr, _entropy_args,
                                              _state._m_entries);
                    _state.move_vertex(v, nr, _state._m_entries);
                }
                return dS;
            }
        }

        double virtual_move(vertex_t v, group_t s, int tnum=0)
        {
            return _state.virtual_move(v, s, _entropy_args, get_m_entries(tnum));
        }

        double virtual_merge(group_t r, group_t s, int tnum=0)
        {
            return _state.virtual_merge(r, s, _entropy_args,
                                        _m_entries[tnum][0]);
        }

        void merge_groups(group_t r, group_t s)
        {
            _state.merge_groups(r, s);
        }

        template <class VS>
        size_t get_Bmin(VS&)
        {
            return 1;
        }

        group_t get_b_min(vertex_t v)
        {
            return _b_min[v];
        }

        group_t get_b_max(vertex_t v)
        {
            return _b_max[v];
        }

        template <class RNG>
        group_t sample_group(vertex_t v, bool allow_random, bool allow_empty,
                             bool init_heuristic, RNG& rng)
        {
            return _state.sample_group(v, allow_random ? _c : 0,
                                       allow_empty ? _d : 0, rng,
                                       !(init_heuristic || _fast_proposal));
        }

        double get_move_prob(vertex_t v, group_t r, group_t s, bool allow_random,
                             bool allow_empty, bool reverse, size_t tnum = 0)
        {
            return _state.get_move_prob(v, r, s,
                                        allow_random ? _c : 0,
                                        allow_empty ? _d : 0,
                                        reverse, get_m_entries(tnum));
        }

        auto group_label(group_t r)
        {
            return _state._spec.group_label(r);
        }

        template <class RNG>
        auto sample_group_label(group_t r, RNG& rng)
        {
            return _state._spec.sample_group_label(r, rng);
        }

        double get_group_label_prob(group_t r, auto u)
        {
            return _state._spec.get_group_label_prob(r, u);
        }

        double relabel_group_dS(group_t r, auto u)
        {
            return _state._spec.relabel_group_dS(r, u, _entropy_args);
        }

        void relabel_group(group_t r, auto u)
        {
            _state._spec.relabel_group(r, u);
        }

        template <class Rs, class Us>
        void relabel_groups_parallel(Rs&& rs, Us&& us)
        {
            _state._spec.relabel_groups_parallel(std::forward<Rs>(rs),
                                                 std::forward<Us>(us));
        }

        double entropy()
        {
            auto eargs = _entropy_args;
            eargs.constants = false;
            return _state.entropy(eargs);
        }

        void relax_update(bool relax)
        {
            _state.relax_update(relax);
        }

        double extra_sweep(auto&& rs, double beta, auto& rng)
        {
            return _state._spec.extra_sweep(rs, beta, _entropy_args, rng);
        }

        void put_cache(size_t B) { _state._spec.put_cache(B); }
        void get_cache(size_t B) { _state._spec.get_cache(B); }


        template <class V>
        void push_state(V&& vs) { _state._spec.push_state(vs); }
        void pop_state(bool discard) { _state._spec.pop_state(discard); }
        void swap_state() { _state._spec.swap_state(); }
        void clear_state() { _state._spec.clear_state(); }
    };

    using gmap_t =
        std::conditional_t<std::is_integral_v<vertex_t>,
                           idx_map<group_t, idx_set<vertex_t, {.shared_pos=true}>, {.shared_tpos=true}>,
                           idx_map<group_t, gt_hash_set<vertex_t>>>;

    template <class T>
    using iset = std::conditional_t<std::is_integral_v<T>,
                                    idx_set<T>, gt_hash_set<T>>;

    template <class T, class V>
    using imap = std::conditional_t<std::is_integral_v<T>,
                                    idx_map<T, V>, gt_hash_map<T, V>>;

    template <class... Ts>
    class MCMCLoopState:
        public Multilevel<MCMCLoopStateImp<Ts...>,
                          vertex_t,
                          group_t,
                          iset,
                          imap,
                          iset,
                          imap,
                          gmap_t>
    {
    public:
        template <class... ATs>
        MCMCLoopState(ATs&&... as)
           : Multilevel<MCMCLoopStateImp<Ts...>,
                        vertex_t,
                        group_t,
                        iset,
                        imap,
                        iset,
                        imap,
                        gmap_t>(std::forward<ATs>(as)...)
        {}
    };
};

template <class State>
GEN_DISPATCH(mcmc_loop_state, MCMC<State>::template MCMCLoopState,
             MCMC_LOOP_STATE_params(State))

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MULTILEVEL_MCMC_HH
