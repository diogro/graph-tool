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

#include "graph.hh"
#include "graph_filtering.hh"

#include "merge.hh"

using namespace graph_tool;
using namespace boost;

boost::python::tuple graph_merge_dispatch_imp(GraphInterface& ugi, GraphInterface& gi,
                                              std::any avmap, std::any aw1,
                                              std::any aw2, bool multiset,
                                              bool diff, bool sym_diff, bool intersect,
                                              bool symmetric, bool simple)
{
    return graph_merge_dispatch_views(never_directed, never_directed, ugi, gi,
                                      avmap, aw1, aw2, multiset, diff,
                                      sym_diff, intersect, symmetric, simple);
}
