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

#ifndef SPEC_GRAPH_BASE_HH
#define SPEC_GRAPH_BASE_HH

#define SPEC_BASE_params                                                       \
    ((g, STATE_SPEC::g_tr, 1))                                                 \
    ((eweight, STATE_SPEC::eweight_tr, 1))

#include "spec_base.hh"

namespace graph_tool
{

namespace STATE_NAME
{

struct GraphSpecBase
    : public SpecBase
{
    using g_tr = decltype(never_directed);
    using eweight_tr = decltype(scalar_edge_properties);

    template <class State>
    struct Imp:
        public SpecBase::Imp<State>
    {
        GET_PARAMS_SPEC_TYPEDEF(SPEC_BASE_params)
        GET_PARAMS_SPEC_DEFINE(SPEC_BASE_params)

        Imp(State& state)
          : SpecBase::Imp<State>(state),
            GET_PARAMS_SPEC_INIT(SPEC_BASE_params)
            _state(state)
        {}

        State& _state;

        decltype(auto) get_vertices()
        {
            return vertices_range(_g);
        }

        // assortative default move proposals
        auto sample_group(size_t v, double c, double d, rng_t& rng)
        {
            std::bernoulli_distribution new_r(d);
            if (d > 0 && !_state._empty_groups.empty() && new_r(rng))
                return uniform_sample(_state._empty_groups, rng);
            c = std::max(std::min(c, 1.), 0.);
            std::bernoulli_distribution adj(1.-c);
            auto iter = out_neighbors(v, _g);
            if (iter.first != iter.second && (c == 0 || adj(rng)))
            {
                auto w = uniform_sample(iter.first, iter.second, rng);
                return _state._b[w];
            }
            return uniform_sample(_state._candidate_groups, rng);
        }

        auto sample_group_local(size_t v, rng_t& rng)
        {
            return sample_group(v, 0, 0, rng);
        }

        // Computes the move proposal probability
        double get_move_prob(size_t v, auto r, auto s, double c, double d,
                             bool reverse)
        {
            size_t B = _state._candidate_groups.size();
            if (reverse)
            {
                if (_state._groups[s].size() == 1)
                    return std::log(d);
                if (_state._groups[r].size() == 0)
                    B++;
            }
            else
            {
                if (_state._groups[s].empty())
                    return std::log(d);
            }

            size_t k_s = 0;
            size_t k = 0;
            for (auto w : out_neighbors_range(v, _g))
            {
                if (_state._b[w] == s)
                    k_s++;
                k++;
            }

            if (B == _state._N)
                d = 0;

            if (k > 0)
            {
                double p = k_s / double(k);
                c = 1 - std::max(std::min(c, 1.), 0.);
                return std::log1p(-d) + std::log(c * p + (1. - c)/B);
            }
            else
            {
                return std::log1p(-d) - std::log(B);
            }
        }
    };
};

} // STATE_NAME

template <>
struct extract_param<eprop_map_t<int64_t>>
{
    auto operator()(boost::python::object oemap)
    {
        try
        {
            auto a = python::extract<std::any>(oemap.attr("_get_any")())();
            return std::any_cast<eprop_map_t<int64_t>>(a);
        }
        catch (std::bad_any_cast&)
        {
            oemap = oemap.attr("copy")("int64_t");
            auto a = python::extract<std::any>(oemap.attr("_get_any")())();
            return std::any_cast<eprop_map_t<int64_t>>(a);
        }
    }
};

} // graph_tool namespace

#endif //SPEC_GRAPH_BASE_HH
