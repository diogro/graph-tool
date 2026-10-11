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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "state.hh"
#include "exhaustive.hh"
#include "../loops/exhaustive_loop.hh"

#include "numpy_bind.hh"
#include "coroutine.hh"
#include "python_interface.hh"

using namespace boost;

namespace graph_tool
{
namespace STATE_NAME
{
using namespace graph_tool;
using namespace STATE_NAME;

template <class State>
GEN_DISPATCH(exhaustive_block_model,
             Exhaustive<State>::template ExhaustiveBlockState,
             EXHAUSTIVE_BLOCK_STATE_params(State))

} // namespace graph_tool
} // namespace STATE_NAME

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace STATE_NAME;
    using namespace __MOD__;
    using namespace boost::python;

    graph_tool::STATE_NAME::STATE_NAME::dispatch
        ([&](auto* s)
         {
             typedef std::decay_t<decltype(*s)> state_t;
             auto& c = get_class<state_t, std::shared_ptr<state_t>, boost::noncopyable>();
             c.def("exhaustive_sweep",
                   +[](state_t&, python::object omcmc_state,
                       python::object callback)
                    {
                        exhaustive_block_model<state_t>::make_dispatch
                            (omcmc_state,
                             [&](auto& s_)
                             {
                                 exhaustive_sweep(*s_,
                                                  [&](auto& state)
                                                  {
                                                      callback(state._S,
                                                               state._S_min);
                                                  });
                             });
                    });

             c.def("exhaustive_sweep_iter",
                   +[](state_t&, python::object omcmc_state)
                   {
#ifdef HAVE_BOOST_COROUTINE
                       auto coro_dispatch =
                           [&](auto& yield)
                           {
                               exhaustive_block_model<state_t>::make_dispatch
                                   (omcmc_state,
                                    [&](auto& s_)
                                    {
                                        exhaustive_sweep(*s_,
                                                         [&](auto& state)
                                                         {
                                                             yield(python::make_tuple(state._S,
                                                                                      state._S_min));
                                                         });
                                    });
                           };
                       return python::object(CoroGenerator(coro_dispatch));
#else
                       throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif // HAVE_BOOST_COROUTINE
                   });

             c.def("exhaustive_sweep_dens",
                   +[](state_t&, python::object omcmc_state,
                       double S_min, double S_max,
                       python::object ohist)
                    {
                        multi_array_ref<uint64_t, 1> hist = get_array<uint64_t, 1>(ohist);
                        int N = hist.shape()[0];
                        double dS = S_max - S_min;

                        exhaustive_block_model<state_t>::make_dispatch
                            (omcmc_state,
                             [&](auto& s_)
                             {
                                 exhaustive_sweep(*s_,
                                                  [&](auto& state)
                                                  {
                                                      auto S = state._S;
                                                      int i = round((N - 1) * (S - S_min) / dS);
                                                      if (i >= 0 && i < N)
                                                          hist[i]++;
                                                  });
                             });
                    });
         });

}, MODULE_PRIO + 1);
