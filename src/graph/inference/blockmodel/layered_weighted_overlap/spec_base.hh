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

#pragma once

#ifndef COUPLED_BLOCK_MODEL
#define COUPLED_BLOCK_MODEL layered_weighted_block_model::BlockState
#endif

// weighted kayered spec
#include "../layered_weighted/spec_base.hh"

// weighted overlap
#include "../weighted_overlap/spec_base.hh"

// layered overlaping spec
#include "../layered_overlap/spec_base.hh"

#define LAYERED_WEIGHTED_OVERLAP_SPEC_params  LAYERED_SPEC_params WEIGHTED_SPEC_params OVERLAP_SPEC_params

#include "types.hh"
#include "entropy.hh"

namespace graph_tool
{
using namespace std;

struct LayeredWeightedOverlapBlockStateSpec
    : public LayeredWeightedBlockStateSpec<WeightedOverlapBlockStateSpec<LWVEC_N>>,
             LayeredOverlapSpecBase
{
    using LayeredOverlapSpecBase::cvweight_t;
    using LayeredWeightedBlockStateSpec<WeightedOverlapBlockStateSpec>::init_vweight;
    using LayeredOverlapSpecBase::init_vweight;
    using LayeredOverlapSpecBase::visit_group_vs;
    using LayeredOverlapSpecBase::get_evweight;

    template <class State>
    using coupled_state_t =
        OverlapBlockStateSpec<>::coupled_state_t<State, VLayerCount>;
};

} // namespace boost
