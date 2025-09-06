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

#ifndef GRAPH_BLOCKMODEL_GIBBS_HH
#define GRAPH_BLOCKMODEL_GIBBS_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_LOOP_STATE_params(State)                                         \
    ((state, State&, 0))                                                      \
    ((vlist, std::vector<typename State::vertex_t>&, 0))                      \
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

        constexpr static auto _null_group = State::null_group;

        double _c = numeric_limits<double>::infinity();

        std::vector<group_t> _candidate_groups;

        auto& get_moves(vertex_t v)
        {
            _candidate_groups.clear();
            const auto& c = _state.get_candidate_groups(v);
            _candidate_groups.insert(_candidate_groups.end(), c.begin(), c.end());
            _candidate_groups.push_back(_null_group);
            return _candidate_groups;
        }

        group_t node_state(vertex_t v)
        {
            return _state._b[v];
        }

        group_t node_weight(vertex_t v)
        {
            return _state.node_weight(v);
        }

        group_t _nr;
        double virtual_move_dS(vertex_t v, group_t nr, rng_t& rng)
        {
            group_t r = _state._b[v];
            if (!_allow_new_group && nr != r && _state.virtual_remove_size(v) == 0)
                return numeric_limits<double>::infinity();
            if (nr == _null_group)
            {
                if (!_allow_new_group ||
                    _state._candidate_groups.size() == num_vertices(_state._g) ||
                    _state.virtual_remove_size(v) == 0)
                    return numeric_limits<double>::infinity();
                _state.get_empty_group(v);
                _nr = nr = uniform_sample(_state._empty_groups, rng);
                auto cstate = _state._coupled_state;
                if (cstate != nullptr)
                    cstate->sample_branch(nr, r, rng);
                _state._bclabel[nr] = _state._bclabel[r];

            }
            return _state.virtual_move(v, nr, _entropy_args);
        }

        void perform_move(vertex_t v, group_t nr)
        {
            if (nr == _null_group)
                nr = _nr;
            _state.move_vertex(v, nr);
        }
    };
};

template <class State>
GEN_DISPATCH(mcmc_loop_state, MCMC<State>::template MCMCLoopState,
             MCMC_LOOP_STATE_params(State))

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_GIBBS_HH
