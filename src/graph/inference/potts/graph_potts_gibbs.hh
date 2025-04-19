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

#ifndef GRAPH_POTTS_GIBBS_HH
#define GRAPH_POTTS_GIBBS_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define GIBBS_POTTS_STATE_params(State)                                        \
    ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                \
    ((state, &, State&, 0))                                                    \
    ((vlist,&, std::vector<size_t>&, 0))                                       \
    ((beta,, double, 0))                                                       \
    ((entropy_args,, typename State::_entropy_args_t, 0))                      \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((verbose,, bool, 0))                                                      \
    ((niter,, size_t, 0))

template <class State>
struct Gibbs
{
    GEN_STATE_BASE(GibbsPottsStateBase, GIBBS_POTTS_STATE_params(State))

    template <class... Ts>
    class GibbsPottsState
        : public GibbsPottsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(GibbsPottsStateBase<Ts...>,
                         GIBBS_POTTS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, GIBBS_POTTS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        GibbsPottsState(ATs&&... as)
           : GibbsPottsStateBase<Ts...>(as...)
        {
            for (size_t r = 0; r < _state._q; ++r)
                _candidate_groups.push_back(r);
        }

        std::vector<group_t> _candidate_groups;

        auto& get_moves(size_t)
        {
            return _candidate_groups;
        }

        group_t node_state(size_t v)
        {
            return _state._b[v];
        }

        constexpr group_t node_weight(size_t)
        {
            return 1;
        }

        double virtual_move_dS(size_t v, group_t nr, rng_t&)
        {
            return _state.virtual_move(v, _state._b[v], nr, _entropy_args);
        }

        void perform_move(size_t v, group_t nr)
        {
            _state.move_vertex(v, nr);
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_POTTS_GIBBS_HH
