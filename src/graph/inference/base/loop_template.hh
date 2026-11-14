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

#ifndef LOOP_NAME
#define LOOP_NAME METH_NAME
#endif

#ifndef LOOP_DISPATCH
#define LOOP_DISPATCH mcmc_loop_state
#endif

#ifndef MODULE_PRIO
#error "MODULE_PRIO not defined!"
#endif

// The state types can be split among GT_DISPATCH_SLICES translation units,
// which define GT_DISPATCH_SLICE = 0, ..., GT_DISPATCH_SLICES - 1 before
// including this file (see StateWrap::dispatch()), to reduce the memory
// needed to compile them.
#ifndef GT_DISPATCH_SLICES
#define GT_DISPATCH_SLICES 1
#define GT_DISPATCH_SLICE 0
#endif

using namespace boost;

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace STATE_NAME;
    using namespace boost::python;
    using namespace __MOD__;
    graph_tool::STATE_NAME::STATE_NAME::template dispatch<GT_DISPATCH_SLICES,
                                                          GT_DISPATCH_SLICE>
        ([&](auto* s)
         {
             typedef std::decay_t<decltype(*s)> state_t;
             auto& c = get_class<state_t, std::shared_ptr<state_t>,
                                 boost::noncopyable>();
             c.def(BOOST_PP_STRINGIZE(METH_NAME),
                   +[](state_t&, python::object omcmc_state, rng_t& rng)
                    {
                        python::object ret;
                        LOOP_DISPATCH<state_t>::make_dispatch
                            (omcmc_state,
                             [&](auto& state)
                             {
                                 auto ret_ = LOOP_NAME(*state, rng);
                                 ret = std::apply([&](auto&... args)
                                                  {
                                                      return python::make_tuple(args...);
                                                  }, ret_);
                             });
                        return ret;
                    });
         });
}, MODULE_PRIO + 1);
