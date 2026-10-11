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

#include "../layered/spec.hh"
#include "../state.hh"
#undef STATE_NAME
#undef STATE_NAME
#undef STATE_SPEC
#undef MODULE_PRIO
#undef BLOCKMODEL_HH

#include "spec_base.hh"

#define MODULE_PRIO 3
#define STATE_NAME layered_overlap_block_model
#define STATE_SPEC LayeredOverlapBlockStateSpec

#undef BLOCK_STATE_SPEC_params
#define BLOCK_STATE_SPEC_params LAYERED_OVERLAP_SPEC_params
