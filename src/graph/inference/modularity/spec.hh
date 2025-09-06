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

#ifndef SPEC_HH
#define SPEC_HH

#define STATE_NAME modularity

#define SPEC_params

#define EARGS_params                                                            \
    ((gamma, double))

#include "../base/graph_spec_base.hh"

namespace graph_tool
{

namespace STATE_NAME
{

struct StateSpec
    : public GraphSpecBase
{
    constexpr static bool has_merge = true;
    constexpr static bool has_parallel_move = true;

    using GraphSpecBase::entropy_args_t;

    template <class State>
    struct Imp
        : public GraphSpecBase::Imp<State>
    {
        GET_PARAMS_SPEC_TYPEDEF(SPEC_params)
        GET_PARAMS_SPEC_DEFINE(SPEC_params)

        using GraphSpecBase::Imp<State>::_g;
        using GraphSpecBase::Imp<State>::_eweight;

        Imp(State& state)
            : GraphSpecBase::Imp<State>(state),
              GET_PARAMS_SPEC_INIT(SPEC_params)
              _state(state),
              _b(state._b)
        {
            GILRelease gil_release;

            _er.resize(num_vertices(_g), 0);

            for (auto v : vertices_range(_g))
            {
                auto r = _b[v];
                _er[r] += out_degree(v, _g, _eweight);
            }

            parallel_edge_loop
                (_g,
                 [&](auto&& e)
                 {
                     if (_b[source(e, _g)] == _b[target(e, _g)])
                         atomic_inc(_Err, 2 * _eweight[e]);
                     atomic_inc(_E, _eweight[e]);
                 });
        }

        State& _state;
        typename State::b_t& _b;

        typedef typename GraphSpecBase::group_t group_t;
        typedef typename GraphSpecBase::vertex_t vertex_t;
        static constexpr group_t null_group = State::null_group;

        typedef typename property_traits<typename State::eweight_t>::value_type w_t;

        w_t _E = 0;

        std::vector<group_t> _bclabel;
        std::vector<w_t> _er;
        w_t _Err = 0;

        // =====================================================================
        // State modification
        // =====================================================================

        auto get_delta(vertex_t v, group_t r, group_t nr, auto&& nb, auto&& skip)
        {
            w_t dErr = 0;
            w_t k = 0;
            for (auto e : out_edges_range(v, _g))
            {
                auto ew = _eweight[e];
                k += ew;
                auto u = target(e, _g);
                if (u == v || skip(u))
                    continue;
                auto s = _b[u];
                auto ns = nb[u];
                if (s == r)
                    dErr -= ew;
                if (ns == nr)
                    dErr += ew;
            }
            return std::make_tuple(k, 2 * dErr);
        }

        [[gnu::flatten]]
        auto get_delta(vertex_t v, group_t r, group_t nr)
        {
            return get_delta(v, r, nr, _b, [](auto){ return false; });
        }

        template <bool atomic=false>
        void apply_delta(group_t r, group_t nr, auto&& delta)
        {
            auto&& [k, dErr] = delta;
            atomic_inc<atomic>(_Err, dErr);
            atomic_dec<atomic>(_er[r], k);
            atomic_inc<atomic>(_er[nr], k);
        }

        void move_vertex(size_t v, group_t r, group_t nr)
        {
            auto delta = get_delta(v, r, nr);
            apply_delta(r, nr, delta);
        }

        template <class Vs, class NRs>
        void move_vertices_parallel(Vs&& vs, NRs&& nrs, auto&& nb)
        {
            static_assert(std::ranges::random_access_range<Vs>);
            static_assert(std::ranges::random_access_range<NRs>);

            parallel_loop(std::views::zip(vs, nrs),
                          [&](auto, auto&& vnr)
                          {
                              auto&& [v, nr] = vnr;
                              auto r = _b[v];
                              auto delta =
                                  get_delta(v, r, nr, nb,
                                            [&](auto u)
                                            {
                                                return ((v > u) && (_state._nb[u] != null_group));
                                            });
                              apply_delta<true>(r, nr, delta);
                          }, 0);
        }

        auto get_merge_delta(group_t s, group_t t)
        {
            w_t k = _er[s];

            auto get_mrs =
                [&](auto x, auto y)
                {
                    w_t m = 0;
                    for (auto v : _state._groups[x])
                    {
                        for (auto e : out_edges_range(v, _g))
                        {
                            if (_b[target(e, _g)] == y)
                                m += _eweight[e];
                        }
                    }
                    return m;
                };

            w_t dErr = 2 * ((_er[s] < _er[t]) ?  get_mrs(s, t) : get_mrs(t, s));

            return std::make_tuple(k, dErr);
        }

        void merge_groups(group_t s, group_t t)
        {
            auto delta = get_merge_delta(s, t);
            apply_delta(s, t, delta);
        }

        void add_group(size_t n = 1)
        {
            size_t N = _er.size();
            _er.resize(N + n, 0);
        }

        // =====================================================================
        // Virtual changes
        // =====================================================================

        double get_delta_dS(group_t r, group_t nr, auto&& delta,
                            const entropy_args_t& ea)
        {
            auto&& [k, dErr] = delta;

            double Qb = 0;
            double Qa = 0;

            double M = 2 * _E;

            Qb -= ea.gamma * _er[r] * (_er[r] / M);
            Qb -= ea.gamma * _er[nr] * (_er[nr] / M);

            Qa += dErr;
            Qa -= ea.gamma * (_er[r] - k) * ((_er[r] - k) / M);
            Qa -= ea.gamma * (_er[nr] + k) * ((_er[nr] + k) / M);

            double dS = -(Qa - Qb);
            return dS;
        }

        double virtual_move(size_t v, group_t r, group_t nr,
                            const entropy_args_t& ea)
        {
            auto delta = get_delta(v, r, nr);
            return get_delta_dS(r, nr, delta, ea);
        }

        double virtual_merge(size_t s, size_t t, const entropy_args_t& ea)
        {
            auto delta = get_merge_delta(s, t);
            return get_delta_dS(s, t, delta, ea);
        }

        double entropy(const entropy_args_t& ea)
        {
            double M = 2 * _E;
            double Q = _Err;
            for (auto r : _state._candidate_groups)
                Q -= ea.gamma * _er[r] * (_er[r] / M);
            return -Q;
        }
    };
};

} // STATE_NAME

} // graph_tool namespace

#endif //SPEC_HH
