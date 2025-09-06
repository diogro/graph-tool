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
#include "../../../blockmodel/state.hh"

#include "../dynamics.hh"
#include "../dynamics_base.hh"
#include "../../../support/graph_state.hh"

#include "test.hh"

#define BASE_NAME test

using namespace boost;
using namespace graph_tool;
using namespace block_model;

template <class BaseState>
struct Dyn : Dynamics<BaseState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

#define MAKE_STATE_(X) mcmc_##X##_sweep
#define MAKE_STATE(X) MAKE_STATE_(X)

python::object MAKE_STATE(BASE_NAME)(GraphInterface& gi, python::object odynamics_state,
                                     python::dict params)
{
    python::object state;

    gt_dispatch<{.gil_release=false}>()
        ([&](auto& g)
         {
             typedef typename std::remove_reference<decltype(g)>::type g_t;
             graph_tool::block_model::block_model::dispatch
                 ([&](auto* bs)
                  {
                      typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;
                      dynamics_state<block_state_t>::dispatch
                          (odynamics_state,
                           [&](auto& s)
                           {
                               typedef std::decay_t<decltype(s)> state_t;
                               using DState = graph_tool::TestStateBase<g_t,state_t>;
                               state = python::object(std::make_shared<DState>(g, s, params));
                           }, false);
                  });
         }, all_graph_views)
        (gi.get_graph_view());

    return state;
}

#define STRINGIFY_(X) #X
#define STRINGIFY(X) STRINGIFY_(X)

struct add_ptr
{
    template <class T>
    struct apply
    {
        typedef typename std::add_pointer<T>::type type;
    };
};

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_" STRINGIFY(BASE_NAME) "_state", &MAKE_STATE(BASE_NAME));

    hana::for_each
        (all_graph_views,
         ([](auto g)
          {
              typedef typename decltype(+g)::type g_t;
              graph_tool::block_model::block_model::dispatch
                  ([&](auto* bs)
                   {
                       typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;
                       dynamics_state<block_state_t>::dispatch
                           ([&](auto* s)
                            {
                                typedef std::decay_t<decltype(*s)> state_t;
                                using DState = graph_tool::TestStateBase<g_t,state_t>;
                                auto name = get_type_name<DState>();
                                class_<DState, bases<DStateBase>, std::shared_ptr<DState>,
                                       boost::noncopyable>(name.c_str(), no_init);
                            });
                   });
          }));
}, MODULE_PRIO + 1);
