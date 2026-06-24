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

#ifndef GRAPH_BLOCKMODEL_MCMC_HH
#define GRAPH_BLOCKMODEL_MCMC_HH

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

template <class State>
struct vlist_t
{
    typedef std::conditional_t<std::is_scalar_v<typename State::vertex_t>,
                               std::vector<typename State::vertex_t>&,
                               std::vector<typename State::vertex_t>> type;
};

#define MCMC_LOOP_STATE_params(State)                                          \
    ((state, State&, 0))                                                       \
    ((vertices, typename vlist_t<State>::type, 0))                             \
    ((beta, double, 0))                                                        \
    ((c, double, 0))                                                           \
    ((d, double, 0))                                                           \
    ((entropy_args, typename State::entropy_args_t, 0))                        \
    ((allow_vacate, bool, 0))                                                  \
    ((sequential, bool, 0))                                                    \
    ((deterministic, bool, 0))                                                 \
    ((verbose, int, 0))                                                        \
    ((niter, size_t, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCLoopStateBase, MCMC_LOOP_STATE_params(State))

    template <class... Ts>
    class MCMCLoopState
        : public MCMCLoopStateBase<Ts...>,
          public MetropolisStateBase
    {
    public:
        GET_PARAMS_USING(MCMCLoopStateBase<Ts...>,
                         MCMC_LOOP_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_LOOP_STATE_params(State))

        template <class... ATs>
        MCMCLoopState(ATs&&... as)
           : MCMCLoopStateBase<Ts...>(std::forward<ATs>(as)...),
            _g(_state._g)
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

        ~MCMCLoopState() { _state.clear_egroups(); }

        typedef State::vertex_t vertex_t;
        typedef State::group_t group_t;

        typename state_t::g_t& _g;
        constexpr static group_t _null_move = State::null_group;

        group_t node_state(vertex_t v)
        {
            return _state._b[v];
        }

        bool skip_node(vertex_t v)
        {
            return _state.node_weight(v) == 0;
        }

        template <class RNG>
        group_t move_proposal(vertex_t v, RNG& rng)
        {
            if (!_allow_vacate && _state.is_last(v))
                return _null_move;
            group_t s = _state.sample_group(v, _c, _d, rng);
            if (s == node_state(v))
                return _null_move;
            return s;
        }

        std::tuple<double, double>
        virtual_move_dS(vertex_t v, group_t nr)
        {
            group_t r = _state._b[v];
            if (r == nr)
                return std::make_tuple(0., 0.);

            double dS = _state.virtual_move(v, nr, _entropy_args);
            double a = 0;
            if (!std::isinf(_beta))
            {
                double pf = _state.get_move_prob(v, r, nr, _c, _d, false);
                double pb = _state.get_move_prob(v, nr, r, _c, _d, true);
                a = pb - pf;
            }
            return std::make_tuple(dS, a);
        }

        void perform_move(vertex_t v, group_t nr)
        {
            _state.move_vertex(v, nr, _state._m_entries);
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        auto& get_vlist()
        {
            return _vertices;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }
    };
};

template <class State>
GEN_DISPATCH(mcmc_loop_state, MCMC<State>::template MCMCLoopState,
             MCMC_LOOP_STATE_params(State))

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MCMC_HH
