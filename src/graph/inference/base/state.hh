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

#ifndef STATE_HH
#define STATE_HH

#include "config.h"

#include <vector>

#include "samplers.hh"
#include "idx_map.hh"
#include "parallel_util.hh"

#include "../support/graph_state.hh"

#include "spec.hh"

#ifndef STATE_NAME
#error "STATE_NAME not defined"
#endif

#ifndef MODULE_PRIO
#define MODULE_PRIO 0
#endif

#ifndef STATE_SPEC
#define STATE_SPEC StateSpec
#endif

namespace graph_tool
{

#define BASE_STATE_params                                                      \
    ((b, STATE_SPEC::vmap_t, 0))                                               \
    ((clabel, STATE_SPEC::vmap_t, 0))

#define STATE_params BASE_STATE_params SPEC_BASE_params SPEC_params

namespace STATE_NAME
{

GEN_STATE_BASE(StateBase, STATE_params)

template <class... Ts>
class State
    : public StateBase<Ts...>
{
public:
    GET_PARAMS_USING(StateBase<Ts...>, STATE_params)
    GET_PARAMS_TYPEDEF(Ts, STATE_params)

    template <class... ATs>
    State(ATs&&... args)
        : StateBase<Ts...>(std::forward<ATs>(args)...),
          _N(0),
          _spec(*this)
    {
        GILRelease gil_release;

        size_t N = 0;
        for (auto v : _spec.get_vertices())
        {
            auto r = _b[v];
            if (std::cmp_greater_equal(r, _bclabel.size()))
                _bclabel.resize(r + 1);
            _bclabel[r] = _clabel[v];
            _groups[r].insert(v);
            N++;
            _max_r = std::max(size_t(r+1), _max_r);
        }
        _N = N;
        _max_r = std::max(N, _max_r);
        _bclabel.resize(_max_r);

        for (size_t r = 0; r < _max_r; ++r)
        {
            if (_groups[r].empty())
                _empty_groups.insert(r);
            else
                _candidate_groups.insert(r);
        }

        _spec.post_init();
    }

    size_t _N;
    size_t _max_r = 0;

    STATE_SPEC::Imp<State> _spec;

    typedef STATE_SPEC::group_t group_t;
    typedef STATE_SPEC::vertex_t vertex_t;
    static constexpr group_t null_group = std::numeric_limits<group_t>::max();

    typedef STATE_SPEC::vmap_t vmap_t;
    typedef STATE_SPEC::bmap_t bmap_t;

    typedef STATE_SPEC::entropy_args_t entropy_args_t;

    idx_set<group_t> _empty_groups;
    idx_set<group_t> _candidate_groups;

    std::vector<group_t> _bclabel;

    idx_map<group_t, idx_set<size_t, {.shared_pos=true}>, {.shared_tpos=true}> _groups;

    vprop_map_t<group_t>::unchecked_t _nb;

    constexpr static bool _has_merge = STATE_SPEC::has_merge;
    constexpr static bool _has_parallel_move = STATE_SPEC::has_parallel_move;
    constexpr static bool _has_parallel_virtual_move = STATE_SPEC::has_parallel_virtual_move;

    decltype(auto) get_vertices()
    {
        return _spec.get_vertices();
    }

    // =========================================================================
    // State modification
    // =========================================================================

    bool allow_move(group_t r, group_t nr)
    {
        return _bclabel[r] == _bclabel[nr];
    }

    void move_vertex(size_t v, group_t nr)
    {
        group_t r = _b[v];

        if (nr == r)
            return;

        if (!allow_move(r, nr))
            throw ValueException("cannot move vertex across clabel barriers");

        _spec.move_vertex(v, r, nr);

        _groups[r].erase(v);
        _groups[nr].insert(v);

        if (_groups[r].empty())
            vacate_group(r);

        if (_groups[nr].size() == 1)
            occupy_group(nr);

        _b[v] = nr;
    }

    void vacate_group(group_t r)
    {
        _empty_groups.insert(r);
        _candidate_groups.erase(r);
    }

    void occupy_group(group_t r)
    {
        _empty_groups.erase(r);
        _candidate_groups.insert(r);
    }

    template <class Vs, class NRs>
    void move_vertices_parallel(Vs&& vs, NRs&& nrs)
    {
        static_assert(std::ranges::random_access_range<Vs>);
        static_assert(std::ranges::random_access_range<NRs>);

        _nb.reserve(_max_r, null_group);

        parallel_loop(std::views::zip(vs, nrs),
                      [&](auto, auto&& vnr)
                      {
                          auto&& [v, nr] = vnr;
                          _nb[v] = nr;
                      });

        auto nb = make_adhoc_prop<vertex_t>
            ([&](auto v)
             {
                 auto r = _nb[v];
                 if (r == null_group)
                     return _b[v];
                 return r;
             });

        _spec.move_vertices_parallel(vs, nrs, nb);

        for (auto&& [v, nr] : std::views::zip(vs, nrs))
        {
            _nb[v] = null_group;
            auto r = _b[v];
            if (r != nr)
            {
                _groups[r].erase(v);
                _groups[nr].insert(v);
                _b[v] = nr;

                if (_groups[r].empty())
                    vacate_group(r);

                if (_groups[nr].size() == 1)
                    occupy_group(nr);
            }
        }
    }

    void merge_groups(group_t s, group_t t)
    {
        if (s == t)
            return;

        _spec.merge_groups(s, t);

        if (!_groups[s].empty())
            vacate_group(s);
        if (_groups[t].empty())
            occupy_group(t);

        std::vector<vertex_t> vs(_groups[s].begin(), _groups[s].end());
        for (auto v : vs)
        {
            _groups[s].erase(v);
            _groups[t].insert(v);
            _b[v] = t;
        }
    }

    void add_group(size_t n = 1)
    {
        size_t B = _groups.size();
        for (size_t i = 0; i < n; ++i)
        {
            _groups[B + i];
            _empty_groups.insert(B + i);
        }
        _bclabel.resize(B + n);
        _spec.add_group(n);
    }

    size_t virtual_remove_size(size_t v)
    {
        return _groups[_b[v]].size() - 1;
    }

    double virtual_move(size_t v, group_t nr,
                        const entropy_args_t& ea)
    {
        auto r = _b[v];
        if (r == nr)
            return 0;
        if (!allow_move(r, nr))
            return std::numeric_limits<double>::infinity();
        return _spec.virtual_move(v, r, nr, ea);
    }

    double virtual_merge(size_t s, size_t t, const entropy_args_t& ea)
    {
        if (s == t)
            return 0;
        if (!allow_move(s, t))
            return std::numeric_limits<double>::infinity();
        return _spec.virtual_merge(s, t, ea);
    }

    group_t sample_group(size_t v, double c, double d, rng_t& rng)
    {
        return _spec.sample_group(v, c, d, rng);
    }

    group_t sample_group_local(size_t v, rng_t& rng)
    {
        return _spec.sample_group_local(v, rng);
    }

    double get_move_prob(size_t v, group_t r, group_t s, double c, double d,
                         bool reverse)
    {
        return _spec.get_move_prob(v, r, s, c, d, reverse);
    }

    double entropy(const entropy_args_t& ea)
    {
        return _spec.entropy(ea);
    }

    constexpr size_t node_weight(auto)
    {
        return 1;
    }

    bool is_last(size_t v)
    {
        return _groups[_b[v]].size() == 1;
    }

    auto& get_candidate_groups(size_t)
    {
        return _candidate_groups;
    }

    auto get_N()
    {
        return _N;
    }

    void relax_update(bool relax) { _spec.relax_update(relax); }

    template <class V>
    void push_state(V&& vs) { _spec.push_state(vs); }

    void pop_state() { _spec.pop_state(); }

    void swap_state() { _spec.swap_state(); }

    void clear_state() { _spec.clear_state(); }
};

GEN_DISPATCH(STATE_NAME, State, STATE_params)

} // STATE_NAME

} // graph_tool namespace

#endif //STATE_HH
