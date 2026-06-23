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

#include "graph_filtering.hh"
#include "graph.hh"
#include "graph_selectors.hh"
#include <boost/variant/get.hpp>

using namespace graph_tool;
using namespace boost;

// retrieves the appropriate degree selector
std::any graph_tool::degree_selector(GraphInterface::deg_t deg)
{
    std::any sel;

    try
    {
        GraphInterface::degree_t d = std::get<GraphInterface::degree_t>(deg);
        hana::for_each(degree_selectors,
                       [&](auto s)
                       {
                           return get_degree_selector()
                               (typename decltype(+s)::type(), d, sel);
                       });
    }
    catch (std::bad_variant_access const&)
    {
        std::any d = std::get<std::any>(deg);
        bool found = false;
        hana::for_each(vertex_properties,
                       [&](auto s)
                       {
                           return get_scalar_selector()
                               (typename decltype(+s)::type(), d, sel, found);
                       });
        if (!found)
            throw ValueException("invalid degree selector");
    }
    return sel;
}
