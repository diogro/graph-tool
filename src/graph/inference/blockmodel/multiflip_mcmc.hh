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
#include <algorithm>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

#include "idx_map.hh"
#include "../loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_LOOP_STATE_params(State)                                          \
     MERGE_SPLIT_params(State)

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCLoopStateBase, MCMC_LOOP_STATE_params(State))

    typedef State::vertex_t vertex_t;
    typedef State::group_t group_t;

    template <class... Ts>
    class MCMCLoopStateImp
        : public MCMCLoopStateBase<Ts...>,
          public MergeSplitStateBase
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
            _state.clear_egroups();
            if (!std::isinf(_c))
                _state.init_egroups();

            if (_vertices.empty())
            {
                for (auto v : vertices_range(_state._g))
                    _vertices.push_back(v);
            }
        }

        ~MCMCLoopStateImp() { _state.clear_egroups(); }

        constexpr static group_t _null_group = State::_null_group;

        constexpr static double _pmovelabel = State::spec_t::has_relabel;

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 1;
        constexpr static double _pscoalesce = 1;

        constexpr static bool _has_merge = State::spec_t::has_merge;
        constexpr static bool _has_parallel_move = State::spec_t::has_parallel_move;
        constexpr static bool _has_relabel = State::spec_t::has_relabel;

        template <class F>
        void iter_nodes(F&& f)
        {
            auto iter_vs =
                [&](auto&& vs)
                {
                    for (auto v : vs)
                    {
                        if (_state.node_weight(v) == 0)
                            continue;
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
            // only the groups of the given vertices (which may be a subset)
            // take part in the moves, since only these vertices are tracked
            // by the loop
            gt_hash_set<group_t> rs;
            for (auto v : _vertices)
            {
                if (_state.node_weight(v) == 0)
                    continue;
                group_t r = _state._b[v];
                if (rs.insert(r).second)
                    f(r);
            }
        }

        auto get_group(vertex_t v)
        {
            return _state._b[v];
        }

        template <bool sample_branch=true, class RNG, class VS = std::array<group_t,0>>
        group_t sample_new_group(vertex_t v, RNG& rng, VS&& except = VS())
        {
            for (auto s : except)
            {
                if (_state._empty_groups.find(s) != _state._empty_groups.end())
                    _state._empty_groups.erase(s);
            }

            auto t = _state.get_empty_group(v, rng);

            if constexpr (_has_relabel)
                relabel_group(t, sample_group_label(t, rng));

            for (auto s : except)
            {
                if (s != _null_group && _state._wr[s] == 0)
                    _state._empty_groups.insert(s);
            }

            auto r = _state._b[v];

            _state.visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     if constexpr (sample_branch)
                     {
                         cs.sample_branch(t, r, rng);
                     }
                     else
                     {
                         auto& bh = cs._b;
                         bh[t] = bh[r];
                     }
                 });
            assert(_state.allow_move(r, t));
            assert(_state._wr[t] == 0);
            return t;
        }

        void move_node(vertex_t v, group_t r, bool cache)
        {
            if (cache)
                _state.move_vertex(v, r, _state._m_entries);
            else
                _state.move_vertex(v, r);
        }

        template <class Vs, class Groups>
        void move_nodes_parallel(Vs&& vs, Groups&& rs)
        {
            if (std::ranges::size(vs) >= _parallel_thres)
                _state.move_vertices_parallel(vs, rs);
            else
                _state.move_vertices(vs, rs);
        }

        void reserve_empty_groups(size_t nB)
        {
            if (_state._empty_groups.size() < nB)
                _state.add_group(nB - _state._empty_groups.size());
        }

        bool allow_move(group_t r, group_t s)
        {
            return _state.allow_move(r, s);
        }

        double virtual_move(vertex_t v, group_t s)
        {
            return _state.virtual_move(v, s, _entropy_args);
        }

        double virtual_merge(group_t r, group_t s)
        {
            return _state.virtual_merge(r, s, _entropy_args);
        }

        void merge_groups(group_t r, group_t s)
        {
            _state.merge_groups(r, s);
        }

        template <class RNG>
        group_t sample_group(vertex_t v, bool allow_empty, RNG& rng)
        {
            return _state.sample_group(v, _c, allow_empty ? _d : 0, rng);
        }

        double get_move_prob(vertex_t v, group_t r, group_t s, bool allow_empty,
                             bool reverse)
        {
            return _state.get_move_prob(v, r, s, _c, allow_empty ? _d : 0, reverse);
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

        double new_group_place_lprob(group_t r)
        {
            return _state.get_coupled_branch_log_prob(r, true);
        }

        bool can_swap(group_t r, group_t s)
        {
            if (!_state.has_coupled())
                return _state._bclabel[r] == _state._bclabel[s];
            return _state.visit_coupled
                ([&](auto&& cs, auto&)
                 {
                     auto& hb = cs._b;
                     auto rr = hb[r];
                     auto ss = hb[s];
                     return (rr == ss) && (_state._bclabel[r] == _state._bclabel[s]);
                 });
        }

        void relax_update(bool relax)
        {
            _state.relax_update(relax);
        }

        void clear_next_state()
        {
            _state._egroups.check(_state._bg, _state._mrs);
        }
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
        public MergeSplit<MCMCLoopStateImp<Ts...>,
                          vertex_t,
                          group_t,
                          iset,
                          imap,
                          iset,
                          gmap_t, false, false>
    {
    public:
        template <class... ATs>
        MCMCLoopState(ATs&&... as)
           : MergeSplit<MCMCLoopStateImp<Ts...>,
                        vertex_t,
                        group_t,
                        iset,
                        imap,
                        iset,
                        gmap_t, false, false>(std::forward<ATs>(as)...)
        {}
    };
};

template <class State>
GEN_DISPATCH(mcmc_loop_state, MCMC<State>::template MCMCLoopState,
             MCMC_LOOP_STATE_params(State))

} // graph_tool namespace
