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

#ifndef SPEC_HH
#define SPEC_HH

#define STATE_NAME norm_cut

#define SPEC_params

#define EARGS_params

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

        Imp(State& state)
            : GET_PARAMS_SPEC_INIT(SPEC_params)
              GraphSpecBase::Imp<State>(state),
              _state(state),
              _b(state._b),
              _er(num_vertices(_g)),
              _err(num_vertices(_g))
        {
            GILRelease gil_release;

            for (auto v : vertices_range(_g))
            {
                auto r = _b[v];
                _er[r] += out_degree(v, _g, _eweight);
            }

            for (auto e : edges_range(_g))
            {
                auto r = _b[source(e, _g)];
                auto s = _b[target(e, _g)];
                if (r == s)
                    _err[r] += 2 * _eweight[e];
            }
        }

        State& _state;
        using GraphSpecBase::Imp<State>::_g;
        using GraphSpecBase::Imp<State>::_eweight;
        typename State::b_t& _b;

        typedef typename GraphSpecBase::group_t group_t;
        typedef typename GraphSpecBase::vertex_t vertex_t;
        static constexpr group_t null_group = State::null_group;

        typedef typename property_traits<typename State::eweight_t>::value_type wval_t;
        typedef decltype(-wval_t(1)) w_t;

        std::vector<w_t> _er;
        std::vector<w_t> _err;

        // =====================================================================
        // State modification
        // =====================================================================

        auto get_delta(vertex_t v, group_t r, group_t nr, auto&& nb, auto&& skip)
        {
            std::array<w_t, 2> derr{};
            w_t k = 0;
            for (auto e : out_edges_range(v, _g))
            {
                auto ew = _eweight[e];
                k += ew;
                auto u = target(e, _g);
                if (u == v)
                {
                    derr[0] -= ew;
                    derr[1] += ew;
                    continue;
                }
                if (skip(u))
                    continue;
                auto s = _b[u];
                auto ns = nb[u];
                if (s == r)
                    derr[0] -= 2 * ew;
                if (ns == nr)
                    derr[1] += 2 * ew;
            }
            return std::make_tuple(k, derr);
        }

        [[gnu::flatten]]
        auto get_delta(vertex_t v, group_t r, group_t nr)
        {
            return get_delta(v, r, nr, _b, [](auto){ return false; });
        }

        template <bool atomic=false>
        void apply_delta(group_t r, group_t nr, auto&& delta)
        {
            auto&& [k, derr] = delta;
            atomic_inc<atomic>(_err[r], derr[0]);
            atomic_inc<atomic>(_err[nr], derr[1]);
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

            w_t derr = 2 * ((_er[s] < _er[t]) ?  get_mrs(s, t) : get_mrs(t, s));
            return std::make_tuple(_er[s], std::array<w_t,2>{-_err[s], _err[s] + derr});
        }

        void merge_groups(group_t s, group_t t)
        {
            auto delta = get_merge_delta(s, t);
            apply_delta(s, t, delta);
        }


        // =====================================================================
        // Virtual changes
        // =====================================================================

        double get_delta_dS(group_t r, group_t nr, auto&& delta, size_t n,
                            const entropy_args_t&)
        {
            auto&& [k, derr] = delta;

            double Cb = 0;
            double Ca = 0;

            Cb -= (_er[r] > 0) ? _err[r]/double(_er[r]) : 0;
            Cb -= (_er[nr] > 0) ? _err[nr]/double(_er[nr]) : 0;

            Ca -= (_er[r] - k > 0) ? (_err[r] + derr[0])/double(_er[r] - k) : 0;
            Ca -= (_er[nr] + k > 0) ? (_err[nr] + derr[1])/double(_er[nr] + k) : 0;

            int64_t dB = 0;
            if (_state._groups[r].size() == n)
                dB--;
            if (_state._groups[nr].empty())
                dB++;

            Cb += _state._occupied_groups.size();
            Ca += _state._occupied_groups.size() + dB;

            double dS = (Ca - Cb);
            return dS;
        }

        double virtual_move(size_t v, group_t r, group_t nr,
                            const entropy_args_t& ea)
        {
            auto delta = get_delta(v, r, nr);
            return get_delta_dS(r, nr, delta, 1, ea);
        }

        double virtual_merge(size_t s, size_t t,
                             const entropy_args_t& ea)
        {
            auto delta = get_merge_delta(s, t);
            return get_delta_dS(s, t, delta, _state._groups[s].size(), ea);
        }

        double entropy(const entropy_args_t&)
        {
            size_t B = _state._occupied_groups.size();
            double C = B;
            for (auto r : _state._occupied_groups)
                C -= (_er[r] > 0) ? _err[r] / double(_er[r]) : 0;
            return C;
        }
    };
};

} // STATE_NAME

} // graph_tool namespace

#endif //SPEC_HH
