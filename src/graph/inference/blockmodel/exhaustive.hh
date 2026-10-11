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
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define EXHAUSTIVE_BLOCK_STATE_params(State)                                   \
    ((state, State&, 0))                                                       \
    ((S, double, 0))                                                           \
    ((vlist, std::vector<typename State::vertex_t>&, 0))                       \
    ((entropy_args, typename State::entropy_args_t, 0))                        \
    ((b_min, typename State::bmap_t, 0))                                       \
    ((max_iter, size_t, 0))


template <class State>
struct Exhaustive
{
    GEN_STATE_BASE(ExhaustiveBlockStateBase,
                   EXHAUSTIVE_BLOCK_STATE_params(State))

    template <class... Ts>
    class ExhaustiveBlockState
        : public ExhaustiveBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(ExhaustiveBlockStateBase<Ts...>,
                         EXHAUSTIVE_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, EXHAUSTIVE_BLOCK_STATE_params(State))

        typedef State::vertex_t vertex_t;
        typedef State::group_t group_t;

        template <class... ATs>
        ExhaustiveBlockState(ATs&&... as)
          : ExhaustiveBlockStateBase<Ts...>(std::forward<ATs>(as)...),
            _g(_state._g), _S_min(_S)
        {
            _state.clear_egroups();
            if (_vlist.empty())
            {
                for (auto v : vertices_range(_g))
                    _vlist.push_back(v);
            }
        }

        typename State::g_t& _g;
        double _S_min;

        size_t get_B()
        {
            return num_vertices(_state._bg);
        }

        group_t node_state(vertex_t v)
        {
            return _state._b[v];
        }

        double virtual_move(vertex_t v, group_t nr)
        {
            return _state.virtual_move(v, nr, _entropy_args);
        }

        void perform_move(vertex_t v, group_t nr)
        {
            _state.move_vertex(v, nr);
        }
    };
};


} // graph_tool namespace
