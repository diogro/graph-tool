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

#ifndef EXHAUSTIVE_LOOP_HH
#define EXHAUSTIVE_LOOP_HH

#include "config.h"
#include "python_interface.hh"

namespace graph_tool
{

template <class ExhaustiveState, class Callback>
void exhaustive_sweep(ExhaustiveState& state, Callback&& callback)
{
    GILRelease gil;

    auto& vlist = state._vlist;

    double& S = state._S;
    double& S_min = state._S_min;

    size_t pos = 0;
    size_t B = state.get_B();
    size_t count = 0;

    callback(state);
    while (pos < vlist.size())
    {
        auto v = vlist[pos];
        size_t r = state.node_state(v);
        if (r < B - 1)
        {
            S += state.virtual_move(v, r + 1);
            state.perform_move(v, r + 1);
            if (S < S_min)
            {
                S_min = S;
                parallel_vertex_loop
                    (state._g,
                     [&](auto u){ state._b_min[u] = state.node_state(u); });
            }
            pos = 0;
            callback(state);
            count++;
            if (count == state._max_iter)
                break;
        }
        else
        {
            S += state.virtual_move(v, 0);
            state.perform_move(v, 0);
            pos++;
        }
    }
}

} // graph_tool namespace

#endif //EXHAUSTIVE_LOOP_HH
