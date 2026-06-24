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

using namespace boost;

double spence(double);

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace boost::python;

    def("init_q_cache", init_q_cache);
    def("clear_q_cache", clear_q_cache);
    def("log_q", log_q<size_t,size_t>);
    def("q_rec", q_rec);
    def("q_rec_memo", q_rec_memo);
    def("log_q_approx", log_q_approx);
    def("log_q_approx_big", log_q_approx_big);
    def("log_q_approx_small", log_q_approx_small);
    def("spence", spence);
}, -1);
