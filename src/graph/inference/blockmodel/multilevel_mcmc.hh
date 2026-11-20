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
                for (auto v : vertices_range(_state._g))
                    _vertices.push_back(v);
            }

            if (_global_moves)
            {
                // the labels of b_min and b_max are mapped to groups: an
                // occupied group with the same label is reused if it has the
                // clabel of the vertices moved to it, otherwise a new group is
                // created, which takes the labels and the place in the
                // hierarchy of the group of one of these vertices. Partitions
                // that are inconsistent with clabel are not used.
                size_t B = num_vertices(_state._bg);
                size_t nB = B;
                std::vector<std::tuple<group_t, vertex_t>> new_rs;
                auto translate = [&](auto& b, auto& rmap, size_t B_x)
                {
                    // one vertex of each label
                    gt_hash_map<group_t, vertex_t> us;
                    for (auto v : vertices_range(_state._g))
                    {
                        if (_state.node_weight(v) == 0)
                            continue;
                        auto [iter, inserted] = us.try_emplace(b[v], v);
                        if (!inserted &&
                            _state._clabel[iter->second] != _state._clabel[v])
                            return false;
                    }
                    if (us.size() != B_x)
                        return false;
                    for (auto& [r, v] : us)
                    {
                        if (size_t(r) < B && _state._wr[r] != 0 &&
                            _state._bclabel[r] == _state._clabel[v])
                        {
                            rmap[r] = r;
                            continue;
                        }
                        rmap[r] = nB;
                        new_rs.emplace_back(nB++, v);
                    }
                    return true;
                };
                _has_b_min = translate(_b_min, _rb_min, _B_min);
                _has_b_max = translate(_b_max, _rb_max, _B_max);
                if (nB > B)
                    _state.add_group(nB - B);
                for (auto& [s, v] : new_rs)
                    _state.inherit_group(s, v);
            }

            resize_m_entries();

            _state.clear_egroups();
        }

        ~MCMCLoopStateImp()
        {
            _state.clear_egroups();
            _state._m_entries_pool.clear();

            auto clear_m_entries =
                [&](this auto&& self, auto& state) -> void
                {
                    state._m_entries_pool.clear();
                    state._m_entries_pool.shrink_to_fit();
                    state.visit_coupled_if
                        ([&](auto& cs, auto&)
                         {
                             self(cs);
                         });
                };

            clear_m_entries(_state);
        }

        constexpr static group_t null_group = State::null_group;

        constexpr static bool _has_merge = State::spec_t::has_merge;
        constexpr static bool _has_parallel_move = State::spec_t::has_parallel_move;
        constexpr static bool _has_relabel = State::spec_t::has_relabel;
        constexpr static bool _has_parallel_relabel = State::spec_t::has_parallel_relabel;
        constexpr static bool _has_extra_sweep = State::spec_t::has_extra_sweep;

        bool _has_b_max = false;
        bool _has_b_min = false;
        gt_hash_map<group_t, group_t> _rb_min, _rb_max; // label -> group

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

        void resize_m_entries()
        {
            // one entry set per thread, also for serial sweeps, since
            // candidate merges are always evaluated in parallel
            auto resize =
                [&](this auto&& self, auto& state) -> void
                {
                    state._m_entries_pool.resize(get_num_threads());
                    state.visit_coupled_if
                        ([&](auto& cs, auto&)
                         {
                             cs._m_entries_pool.resize(state._m_entries_pool.size());
                             // set_next() also shares the accumulators that
                             // some entry sets keep across levels, so the
                             // levels are linked top-down
                             for (auto&& [i, mes] : enumerate(state._m_entries_pool))
                                 mes.set_next(cs._m_entries_pool[i]);
                             self(cs);
                         });
                };

            resize(_state);
        }

        auto& get_m_entries(int tnum)
        {
            return _state._m_entries_pool[tnum];
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
                _state.template move_vertices_parallel<false>(vs, rs);
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
            return _state.virtual_merge(r, s, _entropy_args, get_m_entries(tnum));
        }

        void merge_groups(group_t r, group_t s)
        {
            _state.merge_groups(r, s);
        }

        // groups with different clabels cannot be merged
        template <class VS>
        size_t get_Bmin(VS& vs)
        {
            _rs.clear();
            for (auto& v : vs)
                _rs.insert(_state._bclabel[_state._b[v]]);
            return _rs.size();
        }

        // A group of a partition stored during the sweep (or to which a label
        // of b_min or b_max was mapped at its start) may have been vacated
        // in the meantime, and taken by vertices of another clabel, which
        // also places it in their part of the hierarchy. Vertex v is then
        // moved instead to a new group, which takes the labels and the place
        // in the hierarchy of the group of v (as in the constructor), and is
        // used for the other vertices of the same stored group (via remap).
        group_t restore_group(vertex_t v, group_t s,
                              gt_hash_map<group_t, group_t>& remap)
        {
            if (_state._bclabel[s] == _state._clabel[v])
                return s;
            auto iter = remap.find(s);
            if (iter != remap.end())
                return iter->second;
            group_t t = _state.add_group();
            _state.inherit_group(t, v);
            remap[s] = t;
            return t;
        }

        group_t get_label_group(gt_hash_map<group_t, group_t>& rmap,
                                group_t label, vertex_t v)
        {
            auto& r = rmap.find(label)->second;
            if (_state._bclabel[r] != _state._clabel[v])
            {
                r = _state.add_group();
                _state.inherit_group(r, v);
            }
            return r;
        }

        group_t get_b_min(vertex_t v)
        {
            return get_label_group(_rb_min, _b_min[v], v);
        }

        group_t get_b_max(vertex_t v)
        {
            return get_label_group(_rb_max, _b_max[v], v);
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

        double extra_sweep(auto&& rs, double beta, bool parallel, auto& rng)
        {
            return _state._spec.extra_sweep(rs, beta, _entropy_args, parallel, rng);
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
