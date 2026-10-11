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

#ifndef GRAPH_POTTS_HH
#define GRAPH_POTTS_HH

#define STATE_NAME potts

#include "properties.hh"

#define SPEC_params                                                            \
   ((theta, tmap_t, 0))                                                        \
   ((f, f_t, 0))

#define EARGS_params

#include "../base/graph_spec_base.hh"

namespace graph_tool
{

typedef vprop_map_t<std::vector<double>> tmap_t;
typedef boost::multi_array_ref<double, 2> f_t;

namespace STATE_NAME
{

struct StateSpec
    : public GraphSpecBase
{
    constexpr static bool has_merge = false;
    constexpr static bool has_parallel_move = false;

    using GraphSpecBase::entropy_args_t;

    template <class State>
    struct Imp
        : public GraphSpecBase::Imp<State>
    {
        GET_PARAMS_SPEC_TYPEDEF(SPEC_params)
        GET_PARAMS_SPEC_DEFINE(SPEC_params)

        Imp(State& state)
            : GraphSpecBase::Imp<State>(state),
              GET_PARAMS_SPEC_INIT(SPEC_params)
              _state(state),
              _g(state._g),
              _b(state._b),
              _eweight(state._eweight),
              _q(_f.shape()[0])
        {
            GILRelease gil_release;

            for (auto v : vertices_range(_g))
                _theta[v].resize(_q);
        }

        State& _state;
        typename State::g_t& _g;
        typename State::b_t& _b;
        typename State::eweight_t& _eweight;

        typedef typename GraphSpecBase::group_t group_t;
        typedef typename GraphSpecBase::vertex_t vertex_t;
        static constexpr group_t null_group = State::null_group;

        size_t _q;

        // =====================================================================
        // State modification
        // =====================================================================

        double f(group_t r, group_t s)
        {
            return _f[r % _q][s % _q];
        }

        double virtual_move(vertex_t v, group_t r, group_t nr, auto&)
        {
            double dS = 0;
            for (auto e : out_edges_range(v, _g))
            {
                auto s = _b[target(e, _g)];
                auto w = _eweight[e];
                if (target(e, _g) == v)
                    dS += (f(r, r) - f(nr, nr)) * w;
                else
                    dS += (f(r, s) - f(nr, s)) * w;
            }
            dS += _theta[v][r % _q];
            dS -= _theta[v][nr % _q];
            return dS;
        }

        void move_vertex(vertex_t, group_t, group_t)
        {
        }

        group_t sample_group(vertex_t, auto, auto, rng_t& rng)
        {
            std::uniform_int_distribution<group_t> sample(0, _q-1);
            return sample(rng);
        }

        auto sample_group_local(vertex_t v, rng_t& rng)
        {
            return sample_group(v, 0, 0, rng);
        }

        constexpr double get_move_lprob(vertex_t, auto, auto, double, double, bool)
        {
            return 0;
        }

        double entropy(const auto&)
        {
            double S = 0;

            #pragma omp parallel reduction(+:S)
            parallel_edge_loop_no_spawn
                (_g,
                 [&](auto e)
                 {
                     auto r = _b[source(e, _g)];
                     auto s = _b[target(e, _g)];
                     S -= f(r, s) * _eweight[e];
                 });

            #pragma omp parallel reduction(+:S)
            parallel_vertex_loop_no_spawn
                (_g,
                 [&](auto v)
                 {
                     S -= _theta[v][_b[v] % _q];
                 });

            return S;
        }

    };
};

} // STATE_NAME

} // graph_tool namespace

#endif //GRAPH_POTTS_HH
