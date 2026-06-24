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

#ifndef GRAPH_WEIGHTED_OVERLAP_BLOCK_MODEL_BASE_HH
#define GRAPH_WEIGHTED_OVERLAP_BLOCK_MODEL_BASE_HH


// weighted spec
#include "../weighted/spec_base.hh"

// overlaping spec
#ifndef COUPLED_BLOCK_MODEL
#define COUPLED_BLOCK_MODEL weighted_block_model::BlockState
#endif
#include "../overlap/spec_base.hh"

#define WEIGHTED_OVERLAP_SPEC_params WEIGHTED_SPEC_params OVERLAP_SPEC_params

#include "types.hh"
#include "entropy.hh"

namespace graph_tool
{
using namespace std;

template <size_t WN = WVEC_N>
struct WeightedOverlapBlockStateSpec
    : public WeightedBlockStateSpec<OverlapBlockStateSpec<>, WN>
{
};

template <class Graph, class CProp, class Props, size_t N>
auto in_degree(const auto& v, const half_edge_adaptor<Graph>& g,
               const WeightsPropMap<CProp, Props, N>& ew)
{
    odeg_t k = {v.get_endpoint(), 0, true};
    for (auto e : in_edges_range(v, g))
        k += ew[e];
    return k;
}

template <class Graph, class CProp, class Props, size_t N>
auto out_degree(const auto& v, const half_edge_adaptor<Graph>& g,
               const WeightsPropMap<CProp, Props, N>& ew)
{
    odeg_t k = {v.get_endpoint(), 0, false};
    for (auto e : out_edges_range(v, g))
        k += ew[e];
    return k;
}

} // namespace boost

#endif // GRAPH_WEIGHTED_OVERLAP_BLOCK_MODEL_BASE_HH
