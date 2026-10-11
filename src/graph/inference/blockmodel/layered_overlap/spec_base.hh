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

// layered spec
#include "../layered/spec_base.hh"

// overlaping spec
#ifndef COUPLED_BLOCK_MODEL
#define COUPLED_BLOCK_MODEL layered_block_model::BlockState
#endif
#include "../overlap/spec_base.hh"

#define LAYERED_OVERLAP_SPEC_params LAYERED_SPEC_params OVERLAP_SPEC_params

#include "types.hh"
#include "entropy.hh"

namespace graph_tool
{

struct LayeredOverlapSpecBase
{
    typedef OverlapBlockStateSpec<>::vprop_t<ovweight_t> cvweight_t;

    template <LayerCounts LC>
    static void init_vweight(ovweight_t& vw, size_t v, LC&& in_deg, LC&& out_deg)
    {
        vw[v] = {int64_t(in_deg), int64_t(out_deg)};
    }

    template <LayerCounts LC, LayerCounts DC>
    static void init_vweight(LC& vw, size_t v, DC&& in_deg, DC&& out_deg)
    {
        for (auto [l, c] : in_deg._h)
            get<0>(vw._h[l][v]) += c;
        for (auto [l, c] : out_deg._h)
            get<1>(vw._h[l][v]) += c;
    }

    template <class WR>
    static void visit_group_vs(const WR& wr, auto&& f)
    {
        if constexpr (std::is_same_v<WR, ovweight_t>)
        {
            for (auto&& [v, ds] : wr)
                f(v);
        }
        else
        {
            gt_hash_set<int64_t> vs;
            for (auto&& [l, c] : wr._h)
                for (auto&& [v, ds] : c)
                {
                    if (vs.find(v) != vs.end())
                        continue;
                    f(v);
                    vs.insert(v);
                }
        }
    }

    template <class State>
    static auto get_evweight(auto v, auto& e, State& state)
    {
        using vw_t = State::vw_t;
        vw_t vw;
        auto w = state._eweight[e];
        for (auto&& [l, c] : w._h)
        {
            ovweight_t ov;
            ov[v.get_endpoint()] = {in_degree(v, state._g) * c,
                                    out_degree(v, state._g) * c};
            vw._h[l] = ov;
        }
        return vw;
    }
};

struct LayeredOverlapBlockStateSpec
    : public LayeredBlockStateSpec<OverlapBlockStateSpec<>>,
             LayeredOverlapSpecBase
{
    using cvweight_t = LayeredOverlapSpecBase::cvweight_t;
    using LayeredOverlapSpecBase::init_vweight;
    using LayeredOverlapSpecBase::visit_group_vs;
    using LayeredOverlapSpecBase::get_evweight;

    template <class State>
    using coupled_state_t =
        OverlapBlockStateSpec<>::coupled_state_t<State, VLayerCount>;
};

template <class Graph, class EC, class W, bool layered, class MMap>
auto in_degree(const auto& v, const half_edge_adaptor<Graph>& g,
               const LayersPropMap<EC, W, layered, MMap>& ew)
{
    LayerCount<odeg_t, false, false> k;
    for (auto e : in_edges_range(v, g))
    {
        for (auto&& [l, c] : ew[e]._h)
            k._h[l] += odeg_t({v.get_endpoint(), c, true});
    }
    return k;
}

template <class Graph, class EC, class W, bool layered, class MMap>
auto out_degree(const auto& v, const half_edge_adaptor<Graph>& g,
               const LayersPropMap<EC, W, layered, MMap>& ew)
{
    LayerCount<odeg_t, false, false> k;
    for (auto e : out_edges_range(v, g))
    {
        for (auto&& [l, c] : ew[e]._h)
            k._h[l] += odeg_t({v.get_endpoint(), c, false});
    }
    return k;
}

template <>
struct extract_param<VPropAdaptor<eprop_map_t<std::vector<VLayerCount<ovweight_t>>>>>
{
    auto operator()(boost::python::object)
    {
        return VPropAdaptor<eprop_map_t<std::vector<VLayerCount<ovweight_t>>>>();
    }
};

template <>
struct extract_param<vprop_map_t<VLayerCount<ovweight_t>>>
{
    auto operator()(boost::python::object)
    {
        return vprop_map_t<VLayerCount<ovweight_t>>();
    }
};

} // namespace boost
