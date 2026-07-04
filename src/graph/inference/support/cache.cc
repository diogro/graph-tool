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
#include "cache.hh"

namespace graph_tool
{

std::vector<double> __safelog_cache;
std::vector<double> __xlogx_cache;
std::vector<double> __lgamma_cache;

void clear_safelog()
{
    __safelog_cache.clear();
}

void clear_xlogx()
{
    __xlogx_cache.clear();
}

void clear_lgamma()
{
    __lgamma_cache.clear();
}

} // namespace graph_tool

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace boost::python;
    def("lgamma_fast", lgamma_fast<size_t>);
    def("init_lgamma_fast", init_lgamma_fast<size_t>);
    def("safelog_fast", safelog_fast<size_t>);
    def("init_safelog_fast", init_safelog_fast<size_t>);
    def("xlogx_fast", xlogx_fast<size_t>);
    def("init_xlogx_fast", init_xlogx_fast<size_t>);
    def("clear_safelog", clear_safelog);
    def("clear_xlogx", clear_xlogx);
    def("clear_lgamma", clear_lgamma);
}, -1);
