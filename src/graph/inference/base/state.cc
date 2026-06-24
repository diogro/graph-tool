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

#include <boost/python.hpp>

#include "state.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;
using namespace STATE_NAME;

#define MAKE_STATE BOOST_PP_CAT(make_,BOOST_PP_CAT(STATE_NAME,_state))

python::object MAKE_STATE(boost::python::object ostate)
{
    python::object state;
    graph_tool::STATE_NAME::STATE_NAME::make_dispatch
        (ostate,
         [&](auto& s){state = python::object(s);});
    return state;
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace STATE_NAME;
    using namespace boost::python;
    using namespace __MOD__;

    STATE_SPEC::python_export();

    def(BOOST_PP_STRINGIZE(MAKE_STATE), &MAKE_STATE);

    EXPORT_STRUCT(EARGS_NAME, EARGS_params)

    graph_tool::STATE_NAME::STATE_NAME::dispatch
        ([&](auto* s)
         {
             typedef std::decay_t<decltype(*s)> state_t;

             auto& c = get_class<state_t, std::shared_ptr<state_t>, boost::noncopyable>
                 (get_type_name<state_t>().c_str(), no_init);

             c.def("move_vertex", &state_t::move_vertex)
                 .def("virtual_move", &state_t::virtual_move)
                 .def("entropy", &state_t::entropy)
                 .def("get_N", &state_t::get_N)
                 .def("gen_eargs",
                      +[](state_t&)
                      {
                          return typename state_t::entropy_args_t();
                      });

             STATE_SPEC::python_export_state<state_t>(c);
         });

}, MODULE_PRIO);
