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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#define GRAPH_VIEWS decltype(never_filtered_never_reversed)
#include "../../blockmodel/state.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "dynamics.hh"
#include "dynamics_mcmc_theta.hh"
#include "../../support/graph_state.hh"
#include "../../loops/parallel_pseudo_mcmc.hh"

using namespace boost;
using namespace graph_tool;
using namespace block_model;

template <class BaseState>
struct Dyn : Dynamics<BaseState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_dynamics_state, MCMCTheta<State>::template MCMCDynamicsState,
             MCMC_DYNAMICS_STATE_params(State))

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    graph_tool::block_model::block_model::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             dynamics_state<block_state_t>::dispatch
                 ([&](auto* s)
                  {
                      typedef std::decay_t<decltype(*s)> state_t;
                      auto& c = __MOD__::get_class<state_t, bases<>, std::shared_ptr<state_t>,
                                                   boost::noncopyable>();
                      c.def("pseudo_mcmc_theta_sweep",
                            +[](state_t&, python::object omcmc_state, rng_t& rng)
                             {
                                 boost::python::tuple ret;
                                 mcmc_dynamics_state<state_t>::make_dispatch
                                     (omcmc_state,
                                      [&](auto& s)
                                      {
                                          auto ret_ = pseudo_mcmc_sweep(*s, rng);
                                          ret = std::apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
                                      });
                                 return ret;
                             });
                  });
             });
}, MODULE_PRIO + 1);
