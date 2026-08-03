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

#ifndef MULTILEVEL_MCMC_HH
#define MULTILEVEL_MCMC_HH

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
          : MCMCLoopStateBase<Ts...>(std::forward<ATs>(as)...)
        {
            GILRelease gil_release;

            if (_vertices.empty())
            {
                for (auto v : _state.get_vertices())
                    _vertices.push_back(v);
            }

            if (_global_moves)
            {
                idx_set<group_t> rs_min, rs_max;
                for (auto v : _state.get_vertices())
                {
                    rs_min.insert(_b_min[v]);
                    rs_max.insert(_b_max[v]);
                }
                _has_b_min = rs_min.size() == _B_min;
                _has_b_max = rs_max.size() == _B_max;
            }
        }

        constexpr static group_t null_group = State::null_group;

        constexpr static bool _has_merge = State::_has_merge;
        constexpr static bool _has_parallel_move = State::_has_parallel_move;
        constexpr static bool _has_parallel_virtual_move = State::_has_parallel_virtual_move;

        bool _has_b_max = false;
        bool _has_b_min = false;

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
                iter_vs(_state.get_vertices());
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
            return _state.get_empty_group(v, rng, inherit);
        }

        void reserve_empty_group(group_t r)
        {
            _state._empty_groups.erase(r);
        }

        void move_node(vertex_t v, group_t r, bool)
        {
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
                    dS += _state.virtual_move(v, nr, _entropy_args);
                    _state.move_vertex(v, nr);
                }
                return dS;
            }
        }

        double virtual_move(vertex_t v, group_t s, int=0)
        {
            return _state.virtual_move(v, s, _entropy_args);
        }

        double virtual_merge(group_t r, group_t s, int=0)
        {
            return _state.virtual_merge(r, s, _entropy_args);
        }

        void merge_groups(group_t r, group_t s)
        {
            _state.merge_groups(r, s);
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
            if (!init_heuristic)
                return _state.sample_group(v, allow_random ? _c : 0,
                                           allow_empty ? _d : 0, rng);
            else
                return _state.sample_group_local(v, rng);
        }

        double get_move_prob(vertex_t v, group_t r, group_t s, bool allow_random,
                             bool allow_empty, bool reverse, size_t = 0)
        {
            return _state.get_move_prob(v, r, s,
                                        allow_random ? _c : 0,
                                        allow_empty ? _d : 0,
                                        reverse);
        }

        double entropy()
        {
            if constexpr (requires { _entropy_args.constants; })
            {
                auto eargs = _entropy_args;
                eargs.constants = false;
                return _state.entropy(eargs);
            }
            return _state.entropy(_entropy_args);
        }

        void relax_update(bool relax) { _state.relax_update(relax); }

        template <class V>
        void push_state(V&& vs) { _state.push_state(vs); }

        void pop_state() { _state.pop_state(); }
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

#endif //MULTILEVEL_MCMC_HH
