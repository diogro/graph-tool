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

#include "../blockmodel/state.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "uncertain.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;
using namespace block_model;

template <class BaseState>
GEN_DISPATCH(uncertain_state, Uncertain<BaseState>::template UncertainState,
             UNCERTAIN_STATE_params)

python::object make_uncertain_state(boost::python::object oblock_state,
                                    boost::python::object ouncertain_state)
{
    python::object state;
    auto dispatch = [&](auto& block_state)
        {
            typedef typename std::remove_reference<decltype(block_state)>::type
            state_t;

            uncertain_state<state_t>::make_dispatch
                (ouncertain_state,
                 [&](auto& s)
                 {
                     state = python::object(s);
                 },
                 block_state);
        };
    graph_tool::block_model::block_model::dispatch(oblock_state, dispatch);
    return state;
}


#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;

    class_<uentropy_args_t, bases<BlockStateSpec::entropy_args_t>>
        ("uentropy_args", init<BlockStateSpec::entropy_args_t>())
        .def_readwrite("latent_edges", &uentropy_args_t::latent_edges)
        .def_readwrite("density", &uentropy_args_t::density)
        .def_readwrite("aE", &uentropy_args_t::aE)
        .def_readwrite("sbm", &uentropy_args_t::sbm);

    def("make_uncertain_state", &make_uncertain_state);

    graph_tool::block_model::block_model::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             uncertain_state<block_state_t>::dispatch
                 ([&](auto* s)
                  {
                      typedef std::decay_t<decltype(*s)> state_t;

                      class_<state_t, bases<>, std::shared_ptr<state_t>, boost::noncopyable>
                          c(get_type_name<state_t>().c_str(),
                            no_init);
                      c.def("remove_edge", &state_t::remove_edge)
                          .def("add_edge", &state_t::add_edge)
                          .def("set_state",
                               +[](state_t& state, GraphInterface& gi,
                                   std::any aw)
                                {
                                    typedef eprop_map_t<int32_t> emap_t;
                                    auto w = std::any_cast<emap_t>(aw).get_unchecked();
                                    gt_dispatch<>()
                                        ([&](auto& g)
                                         { set_state(state, g, w); },
                                         all_graph_views)
                                        (gi.get_graph_view());
                                })
                          .def("gen_eargs",
                               +[](state_t&)
                                {
                                    return uentropy_args_t(BlockStateSpec::entropy_args_t());
                                })
                          .def("remove_edge_dS", &state_t::remove_edge_dS)
                          .def("add_edge_dS", &state_t::add_edge_dS)
                          .def("entropy", &state_t::entropy)
                          .def("set_q_default", &state_t::set_q_default)
                          .def("set_S_const", &state_t::set_S_const)
                          .def("get_edge_prob",
                               +[](state_t& state, size_t u, size_t v,
                                   const uentropy_args_t& ea, double epsilon)
                                {
                                    return get_edge_prob(state, u, v, ea,
                                                         epsilon);
                                })
                          .def("get_edges_prob",
                               +[](state_t& state, python::object edges,
                                   python::object probs, const uentropy_args_t& ea,
                                   double epsilon)
                                {
                                    get_edges_prob(state, edges, probs, ea,
                                                   epsilon);
                                });

                  });
         });

}, MODULE_PRIO);
