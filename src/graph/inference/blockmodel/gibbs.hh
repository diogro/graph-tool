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

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "../loops/mcmc_loop.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_LOOP_STATE_params(State)                                         \
    ((state, State&, 0))                                                      \
    ((vlist, typename vertex_list_t<State>::type, 0))                         \
    ((beta, double, 0))                                                       \
    ((entropy_args, typename State::entropy_args_t, 0))                       \
    ((allow_new_group, bool, 0))                                              \
    ((sequential, bool, 0))                                                   \
    ((deterministic, bool, 0))                                                \
    ((verbose, bool, 0))                                                      \
    ((niter, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCLoopStateBase, MCMC_LOOP_STATE_params(State))

    template <class... Ts>
    class MCMCLoopState
        : public MCMCLoopStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCLoopStateBase<Ts...>,
                         MCMC_LOOP_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_LOOP_STATE_params(State))

        template <class... ATs>
        MCMCLoopState(ATs&&... as)
           : MCMCLoopStateBase<Ts...>(std::forward<ATs>(as)...)
        {
            GILRelease gil_release;
            _state.clear_egroups();
            if (_vlist.empty())
            {
                for (auto v : vertices_range(_state._g))
                    _vlist.push_back(v);
            }
        }

        typedef State::vertex_t vertex_t;
        typedef State::group_t group_t;

        constexpr static auto _null_group = State::_null_group;

        double _c = numeric_limits<double>::infinity();

        std::vector<group_t> _occupied_groups;

        // Places of a new group in the hierarchy (see get_branch_places()),
        // which are all part of the heat-bath, as the moves _null_group - i;
        // without a hierarchy, there is a single new group, _null_group.
        std::vector<std::tuple<size_t, size_t>> _places;

        auto& get_moves(vertex_t v)
        {
            _occupied_groups.clear();
            const auto& c = _state.get_occupied_groups(v);
            _occupied_groups.insert(_occupied_groups.end(), c.begin(), c.end());

            _places.clear();
            if (_allow_new_group)
            {
                auto r = _state._b[v];
                bool vacate = _state.virtual_remove_size(v) == 0;
                bool coupled = false;
                _state.visit_coupled_if
                    ([&](auto& cs, auto&)
                     {
                         coupled = true;
                         cs.get_branch_places(r, vacate, _places);
                     });
                // a new group for the sole member of a group would not
                // change the state
                if (!coupled && !vacate)
                    _places.emplace_back(0, _null_group);
            }
            for (size_t i = 0; i < _places.size(); ++i)
                _occupied_groups.push_back(_null_group - group_t(i));
            return _occupied_groups;
        }

        bool is_new_group(group_t nr)
        {
            return nr > _null_group - group_t(_places.size());
        }

        template <class RNG>
        group_t place_new_group(vertex_t v, group_t nr, RNG& rng)
        {
            auto r = _state._b[v];
            auto s = _state.get_empty_group(v, rng);
            auto& [k, t] = _places[size_t(_null_group - nr)];
            _state.visit_coupled_if
                ([&](auto& cs, auto&)
                 {
                     cs.set_branch(s, r, k, t, rng);
                 });
            return s;
        }

        group_t node_state(vertex_t v)
        {
            return _state._b[v];
        }

        group_t node_weight(vertex_t v)
        {
            return _state.node_weight(v);
        }

        rng_t* _rng = nullptr;
        double virtual_move(vertex_t v, group_t nr, rng_t& rng)
        {
            group_t r = _state._b[v];
            if (!_allow_new_group && nr != r && _state.virtual_remove_size(v) == 0)
                return numeric_limits<double>::infinity();
            _rng = &rng;
            if (is_new_group(nr))
                nr = place_new_group(v, nr, rng);
            return _state.virtual_move(v, nr, _entropy_args);
        }

        void perform_move(vertex_t v, group_t nr)
        {
            if (is_new_group(nr))
                nr = place_new_group(v, nr, *_rng);
            _state.move_vertex(v, nr);
        }
    };
};

template <class State>
GEN_DISPATCH(mcmc_loop_state, MCMC<State>::template MCMCLoopState,
             MCMC_LOOP_STATE_params(State))

} // graph_tool namespace
