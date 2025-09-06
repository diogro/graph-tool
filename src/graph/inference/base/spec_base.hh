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

#ifndef SPEC_BASE_HH
#define SPEC_BASE_HH

#ifndef STATE_NAME
#error "STATE_NAME undefined"
#endif

#ifndef SPEC_BASE_params
#define SPEC_BASE_params
#endif

#include "../support/util.hh"

namespace graph_tool
{

#define EARGS_NAME BOOST_PP_CAT(STATE_NAME, _entropy_args_t)

namespace STATE_NAME
{

DEFINE_STRUCT(EARGS_NAME, EARGS_params);

struct SpecBase
{
    using vmap_t = vprop_map_t<int64_t>;
    using bmap_t = vmap_t;
    using vertex_t = size_t;
    using group_t = int64_t;

    using entropy_args_t = EARGS_NAME;

    static constexpr bool has_merge = false;
    static constexpr bool has_parallel_move = false;
    static constexpr bool has_parallel_virtual_move = true;

    static void python_export() {}
    template <class>
    static void python_export_state(auto&) {}

    template <class State>
    struct Imp
    {
        Imp(State& state) : _state(state) {}
        State& _state;

        decltype(auto) get_vertices()
        {
            return std::views::iota(size_t(0), _state._N);
        }

        void move_vertex(auto, auto, auto) { static_assert(false, "needs to be overloaded"); }
        double virtual_move(auto, auto, auto, auto& ea) { static_assert(false, "needs to be overloaded"); return 0; }
        double entropy(auto&) { static_assert(false, "needs to be overloaded"); return 0; }
        void add_group(size_t) {}

        group_t sample_group(size_t, double, double d, rng_t& rng)
        {
            std::bernoulli_distribution new_r(d);
            if (d > 0 && !_state._empty_groups.empty() && new_r(rng))
                return uniform_sample(_state._empty_groups, rng);
            return uniform_sample(_state._candidate_groups, rng);
        }

        group_t sample_group_local(size_t v, rng_t& rng)
        {
            return sample_group(v, 0, 0, rng);
        }

        double get_move_prob(size_t, group_t r, group_t s, double, double d,
                             bool reverse)
        {
            size_t B = _state._candidate_groups.size();
            if (reverse)
            {
                if (_state._groups[s].size() == 1)
                    return log(d);
                if (_state._groups[r].size() == 0)
                    B++;
            }
            else
            {
                if (_state._groups[s].size() == 0)
                    return log(d);
            }

            if (B == _state._N)
                d = 0;

            return log(1. - d) - safelog_fast(B);
        }

        // optional
        constexpr static bool _has_merge = false;
        void merge_groups(auto, auto) {}
        double virtual_merge(auto, auto, auto&) { return 0; }

        constexpr static bool _has_parallel_move = false;
        void move_vertices_parallel(auto&&, auto&&, auto&) {}

        void post_init() {}

        void relax_update(bool) {}
        template <class V>
        void push_state(V&&) {}
        void pop_state() {}
        void swap_state() {}
        void clear_state() {}
    };
};

} // STATE_NAME

} // graph_tool namespace

#endif //SPEC_BASE_HH
