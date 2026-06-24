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

#include "../blockmodel/partition.hh"
#include "../blockmodel/deg_partition.hh"
#include "../blockmodel/entropy.hh"

#define STATE_NAME planted_partition

#define SPEC_params                                                            \
    ((uniform, bool, 0))                                                       \
    ((deg_corr, bool, 0))                                                      \
    ((pclabel, STATE_SPEC::vmap_t, 0))

#define EARGS_params                                                           \
    ((adjacency, bool))                                                        \
    ((parallel_entropy, bool))                                                 \
    ((degree_entropy, bool))                                                   \
    ((partition_dl, bool))                                                     \
    ((degree_dl, bool))                                                        \
    ((edges_dl, bool))                                                         \
    ((degree_dl_kind, deg_dl_kind))                                            \
    ((constants, bool))

#include "../base/graph_spec_base.hh"

namespace graph_tool
{

namespace STATE_NAME
{

struct StateSpec
    : public GraphSpecBase
{
    using g_tr = decltype(all_graph_views);
    using eweight_tr = decltype(integer_edge_properties);

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
            : GraphSpecBase::Imp<State>(state),
              GET_PARAMS_SPEC_INIT(SPEC_params)
              _state(state),
              _g(state._g),
              _b(state._b),
              _eweight(state._eweight),
              _N(hard_num_vertices(_g))
        {}

        void post_init()
        {
            _erp.resize(_state._max_r, 0);
            if constexpr (is_directed_v<g_t>)
                _erm.resize(_state._max_r, 0);
            _err.resize(_state._max_r, 0);
            _eio.resize(2, 0);
            _degs.resize(num_vertices(_g));

            for (auto v : vertices_range(_g))
            {
                auto r = _b[v];
                _degs[v] = {in_degree(v, _g, _eweight),
                            out_degree(v, _g, _eweight)};
                _erp[r] += std::get<1>(_degs[v]);
                if constexpr (is_directed_v<g_t>)
                    _erm[r] += std::get<0>(_degs[v]);
            }

            for (auto e : edges_range(_g))
            {
                auto w = _eweight[e];
                auto r = _b[source(e, _g)];
                auto s = _b[target(e, _g)];
                if (r == s)
                {
                    _err[r] += w;
                    _eio[0] += w;
                }
                else
                {
                    _eio[1] += w;
                }
                _E += w;
            }

            init_partition_stats();

            if (_deg_corr)
            {
                _deg_stats.init([&](auto&& f)
                                {
                                    for (auto v : vertices_range(_g))
                                    {
                                        f(_degs[v], _b[v]);
                                    }
                                });
                _deg_stats.add_group(_state._empty_groups.size());
                init_q_cache(std::min(std::max(2 * int(_E), 100), 10000));
            }

            init_safelog_fast(2 * _E);
            init_lgamma_fast(2 * _E);
        }

        State& _state;
        using g_t = typename State::g_t;
        g_t& _g;
        typename State::b_t& _b;
        typename State::eweight_t& _eweight;

        typedef typename GraphSpecBase::group_t group_t;
        typedef typename GraphSpecBase::vertex_t vertex_t;
        static constexpr group_t null_group = State::null_group;

        typedef typename property_traits<typename State::eweight_t>::value_type w_t;

        size_t _N = 0;
        size_t _E = 0;

        std::vector<int64_t> _erp;
        std::vector<int64_t> _erm;
        std::vector<int64_t> _err;
        std::vector<int64_t> _eio;
        std::vector<std::tuple<w_t, w_t>> _degs;

        std::vector<PartitionStats<size_t>> _partition_stats;
        DegStats<is_directed_v<typename State::g_t>, int64_t, group_t> _deg_stats;

        std::vector<std::vector<vertex_t>> _groups_in, _groups_out;

        void init_partition_stats()
        {
#ifndef _LIBCPP_VERSION
            auto vi = std::max_element(vertices(_g).first,
                                       vertices(_g).second,
#else // libc++ is stupid
            auto vi = boost::first_max_element(vertices(_g).first,
                                               vertices(_g).second,
#endif
                                       [&](auto u, auto v)
                                       { return (_pclabel[u] < _pclabel[v]); });

            size_t C = _pclabel[*vi] + 1;

            vector<vector<vertex_t>> vcs(C);
            for (auto v : vertices_range(_g))
                vcs[_pclabel[v]].push_back(v);

            for (size_t c = 0; c < C; ++c)
            {
                _partition_stats.emplace_back([&](auto&& f)
                                              {
                                                  for (auto v : vcs[c])
                                                      f(_b[v]);
                                              });
                _partition_stats.back().add_group(_state._empty_groups.size());
            }
        }

        auto& get_partition_stats(vertex_t v)
        {
           size_t r = _pclabel[v];
           return _partition_stats[r];
        }

        // =====================================================================
        // State modification
        // =====================================================================

        auto get_delta(vertex_t v, group_t r, group_t nr, auto&& nb, auto&& skip)
        {
            std::array<int64_t, 2> derr{};
            std::array<int64_t, 2> dio{};
            w_t kin = 0;
            w_t kout = 0;
            w_t m = 0;

            auto get_e_delta =
                [&](auto u, auto w)
                {
                    if (skip(u))
                        return;

                    auto s = _b[u];
                    auto ns = nb[u];

                    if (s == r)
                    {
                        derr[0] -= w;
                        dio[0] -= w;
                    }
                    else
                    {
                        dio[1] -= w;
                    }

                    if (ns == nr)
                    {
                        derr[1] +=  w;
                        dio[0] += w;
                    }
                    else
                    {
                        dio[1] += w;
                    }
                };

            for (auto e : out_edges_range(v, _g))
            {
                auto w = _eweight[e];
                kout += w;
                auto u = target(e, _g);
                if (u == v)
                {
                    m += w;
                    continue;
                }
                get_e_delta(u, w);
            }

            if constexpr (is_directed_v<g_t>)
            {
                derr[0] -= m;
                derr[1] += m;
            }
            else
            {
                derr[0] -= m / 2;
                derr[1] += m / 2;
            }

            for (auto e : in_edges_range(v, _g))
            {
                auto w = _eweight[e];
                kin += w;
                auto u = source(e, _g);
                if (u == v)
                    continue;
                get_e_delta(u, w);
            }

            return std::make_tuple(kin, kout, derr, dio);
        }

        [[gnu::flatten]]
        auto get_delta(vertex_t v, group_t r, group_t nr)
        {
            return get_delta(v, r, nr, _b, [](auto){ return false; });
        }

        template <bool atomic=false>
        void apply_delta(group_t r, group_t nr, auto&& delta)
        {
            auto&& [kin, kout, derr, dio] = delta;
            if constexpr (is_directed_v<g_t>)
            {
                atomic_dec<atomic>(_erm[r], kin);
                atomic_inc<atomic>(_erm[nr], kin);
            }
            atomic_dec<atomic>(_erp[r], kout);
            atomic_inc<atomic>(_erp[nr], kout);
            atomic_inc<atomic>(_err[r], derr[0]);
            atomic_inc<atomic>(_err[nr], derr[1]);
            atomic_inc<atomic>(_eio[0], dio[0]);
            atomic_inc<atomic>(_eio[1], dio[1]);
        }

        void move_vertex(vertex_t v, group_t r, group_t nr)
        {
            auto delta = get_delta(v, r, nr);
            apply_delta(r, nr, delta);
            auto& ps = get_partition_stats(v);
            ps.remove_item(r);
            ps.add_item(nr);
            if (_deg_corr)
            {
                _deg_stats.remove_vertex(_degs[v], r);
                _deg_stats.add_vertex(_degs[v], nr);
            }
        }

        template <class Vs, class NRs>
        void move_vertices_parallel(Vs&& vs, NRs&& nrs, auto&& nb)
        {
            static_assert(std::ranges::random_access_range<Vs>);
            static_assert(std::ranges::random_access_range<NRs>);

            _groups_in.clear();
            _groups_out.clear();
            _groups_in.resize(_state._groups.size());
            _groups_out.resize(_state._groups.size());

            for (const auto& [v, nr] : std::views::zip(vs, nrs))
            {
                auto r = _b[v];
                _groups_out[r].push_back(v);
                _groups_in[nr].push_back(v);
            }

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

            parallel_loop
                (_groups_in,
                 [&](auto nr, auto& vs)
                 {
                     for (auto v : vs)
                     {
                         auto& ps = get_partition_stats(v);
                         ps.add_item(nr);
                         if (_deg_corr)
                             _deg_stats.add_vertex(_degs[v], nr);
                     }
                 }, 0);

            parallel_loop
                (_groups_out,
                 [&](auto r, auto& vs)
                 {
                     for (auto v : vs)
                     {
                         auto& ps = get_partition_stats(v);
                         ps.remove_item(r);
                         if (_deg_corr)
                             _deg_stats.remove_vertex(_degs[v], r);
                     }
                 }, 0);
        }

        auto get_merge_delta(group_t s, group_t t)
        {
            w_t kout = _erp[s];
            w_t kin = 0;
            if constexpr (is_directed_v<g_t>)
                kin = _erm[s];

            auto get_mrs =
                [&](auto r, auto s)
                {
                    w_t m = 0;
                    for (auto v : _state._groups[r])
                    {
                        for (auto e : out_edges_range(v, _g))
                        {
                            auto u = target(e, _g);
                            if (_b[u] == s)
                                m += _eweight[e];
                        }

                        for (auto e : in_edges_range(v, _g))
                        {
                            auto u = source(e, _g);
                            if (_b[u] == s)
                                m += _eweight[e];
                        }
                    }
                    return m;
                };

            auto es = is_directed_v<g_t> ? _erm[s] + _erp[s] : _erp[s];
            auto et = is_directed_v<g_t> ? _erm[t] + _erp[t] : _erp[t];

            int64_t d = ((es < et) ? get_mrs(s, t) : get_mrs(t, s));

            assert(get_mrs(s, t) == get_mrs(t, s));

            std::array<int64_t, 2> derr = {-_err[s], _err[s] + d};
            std::array<int64_t, 2> dio = {d, -d};

            return std::make_tuple(kin, kout, derr, dio);
        }

        void merge_groups(group_t s, group_t t)
        {
            auto delta = get_merge_delta(s, t);
            apply_delta(s, t, delta);
            auto& ps = _partition_stats[_state._bclabel[s]];
            ps.remove_item(s, _state._groups[s].size());
            ps.add_item(t, _state._groups[s].size());
            if (_deg_corr)
                _deg_stats.merge_groups(s, t);
        }

        size_t add_group(size_t n = 1)
        {
            size_t N = _state._groups.size();
            _erp.resize(N + n, 0);
            if constexpr (is_directed_v<g_t>)
                _erm.resize(N + n, 0);
            _err.resize(N + n, 0);
            for (auto& ps : _partition_stats)
                ps.add_group(n);
            if (_deg_corr)
                _deg_stats.add_group(n);
            return N;
        }

        // =====================================================================
        // Virtual changes
        // =====================================================================

        double get_delta_dS(group_t r, group_t nr, auto&& delta, size_t n,
                            const entropy_args_t& ea)
        {
            auto&& [kin, kout, derr, dio] = delta;

            double Sb = 0;
            double Sa = 0;

            size_t B = _state._candidate_groups.size();

            int64_t dB = 0;
            if (_state._groups[r].size() == n)
                dB--;
            if (_state._groups[nr].size() == 0)
                dB++;

            if (ea.adjacency)
            {
                if (_deg_corr)
                {
                    Sb += lgamma_fast(_erp[r] + 1);
                    Sb += lgamma_fast(_erp[nr] + 1);
                    Sa += lgamma_fast(_erp[r] - kout + 1);
                    Sa += lgamma_fast(_erp[nr] + kout + 1);
                    if constexpr (is_directed_v<g_t>)
                    {
                        Sb += lgamma_fast(_erm[r] + 1);
                        Sb += lgamma_fast(_erm[nr] + 1);
                        Sa += lgamma_fast(_erm[r] - kin + 1);
                        Sa += lgamma_fast(_erm[nr] + kin + 1);
                    }
                }
                else
                {
                    if constexpr (is_directed_v<g_t>)
                    {
                        Sb += (_erp[r] + _erm[r]) * safelog_fast(_state._groups[r].size());
                        Sb += (_erp[nr] + _erm[nr]) * safelog_fast(_state._groups[nr].size());
                        Sa += (_erp[r] + _erm[r] - kout - kin) * safelog_fast(_state._groups[r].size() - n);
                        Sa += (_erp[nr] + _erm[nr] + kout + kin) * safelog_fast(_state._groups[nr].size() + n);
                    }
                    else
                    {
                        Sb += _erp[r] * safelog_fast(_state._groups[r].size());
                        Sb += _erp[nr] * safelog_fast(_state._groups[nr].size());
                        Sa += (_erp[r] - kout) * safelog_fast(_state._groups[r].size() - n);
                        Sa += (_erp[nr] + kout) * safelog_fast(_state._groups[nr].size() + n);
                    }
                }

                if (_uniform)
                {
                    Sb -= lgamma_fast(_eio[0] + 1);
                    Sb -= lgamma_fast(_eio[1] + 1);
                    Sa -= lgamma_fast(_eio[0] + dio[0] + 1);
                    Sa -= lgamma_fast(_eio[1] + dio[1] + 1);

                    if constexpr (is_directed_v<g_t>)
                    {
                        Sb += safelog_fast(B) * _eio[0];
                        Sb += (lbinom_fast(B, size_t(2)) + std::log(2)) * _eio[1];

                        Sa += safelog_fast(B + dB) * (_eio[0] + dio[0]);
                        Sa += (lbinom_fast(B + dB, size_t(2)) + std::log(2)) * (_eio[1] + dio[1]);
                    }
                    else
                    {
                        Sb += (safelog_fast(B) - std::log(2)) * _eio[0];
                        Sb += lbinom_fast(B, size_t(2)) * _eio[1];

                        Sa += (safelog_fast(B + dB) - std::log(2)) * (_eio[0] + dio[0]);
                        Sa += lbinom_fast(B + dB, size_t(2)) * (_eio[1] + dio[1]);
                    }
                }
                else
                {
                    Sb -= lgamma_fast(_eio[1] + 1);
                    Sa -= lgamma_fast(_eio[1] + dio[1] + 1);

                    if constexpr (is_directed_v<g_t>)
                    {
                        Sb -= lgamma_fast(_err[r] + 1);
                        Sb -= lgamma_fast(_err[nr] + 1);
                        Sa -= lgamma_fast(_err[r] + derr[0] + 1);
                        Sa -= lgamma_fast(_err[nr] + derr[1] + 1);
                    }
                    else
                    {
                        Sb -= _err[r] * std::log(2) + lgamma_fast(_err[r] + 1);
                        Sb -= _err[nr] * std::log(2) + lgamma_fast(_err[nr] + 1);
                        Sa -= (_err[r] + derr[0]) * std::log(2) + lgamma_fast(_err[r] + derr[0] + 1);
                        Sa -= (_err[nr] + derr[1]) * std::log(2) + lgamma_fast(_err[nr] + derr[1] + 1);
                    }

                    if constexpr (is_directed_v<g_t>)
                    {
                        Sb += _eio[1] * (lbinom_fast(B, size_t(2)) + std::log(2));
                        Sa += (_eio[1] + dio[1]) * (lbinom_fast(size_t(B + dB), size_t(2)) + std::log(2));
                    }
                    else
                    {
                        Sb += _eio[1] * lbinom_fast(B, size_t(2));
                        Sa += (_eio[1] + dio[1]) * lbinom_fast(size_t(B + dB), size_t(2));
                    }

                    Sb += lbinom_fast(B + _eio[0] - 1, size_t(_eio[0]));
                    Sa += lbinom_fast(B + dB + _eio[0] + dio[0] - 1, size_t(_eio[0] + dio[0]));

                }
            }

            if (ea.edges_dl)
            {
                if (B > 1)
                    Sb += safelog_fast(_E + 1);
                if (B + dB > 1)
                    Sa += safelog_fast(_E + 1);
            }

            return (Sa - Sb);
        }

        double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea)
        {
            auto delta = get_delta(v, r, nr);

            double dS = get_delta_dS(r, nr, delta, 1, ea);

            if (ea.partition_dl)
            {
                auto& ps = get_partition_stats(v);
                dS += ps.get_delta_dl(r, nr);
            }
            if (_deg_corr && ea.degree_dl)
                dS += _deg_stats.get_delta_dl(_degs[v], r, nr, ea.degree_dl_kind);
            return dS;
        }

        double virtual_merge(size_t s, size_t t, const entropy_args_t& ea)
        {
            auto delta = get_merge_delta(s, t);
            double dS = get_delta_dS(s, t, delta, _state._groups[s].size(), ea);
            if (ea.partition_dl)
            {
                auto& ps = _partition_stats[_state._bclabel[s]];
                dS += ps.get_delta_dl(s, t, _state._groups[s].size());
            }
            if (_deg_corr && ea.degree_dl)
                dS += _deg_stats.get_merge_dl(ea.degree_dl_kind, s, t);
            return dS;
        }

        double entropy(const entropy_args_t& ea)
        {
            double S = 0;

            size_t B = _state._candidate_groups.size();

            if (ea.adjacency)
            {
                #pragma omp parallel reduction(+:S)
                parallel_loop_no_spawn
                    (_state._candidate_groups,
                     [&](auto, auto r)
                     {
                         if (_state._groups[r].empty())
                             return;
                         if (_deg_corr)
                         {
                             S += lgamma_fast(_erp[r] + 1);
                             if constexpr (is_directed_v<g_t>)
                                 S += lgamma_fast(_erm[r] + 1);
                         }
                         else
                         {
                             S += _erp[r] * safelog_fast(_state._groups[r].size());
                             if constexpr (is_directed_v<g_t>)
                                 S += _erm[r] * safelog_fast(_state._groups[r].size());
                         }
                     });

                if (_uniform)
                {
                    S -= lgamma_fast(_eio[0] + 1);
                    S -= lgamma_fast(_eio[1] + 1);
                    if constexpr (is_directed_v<g_t>)
                    {
                        S += safelog_fast(B) * _eio[0];
                        S += (lbinom_fast(B, 2) + std::log(2)) * _eio[1];
                    }
                    else
                    {
                        S += (safelog_fast(B) - std::log(2)) * _eio[0];
                        S += lbinom_fast(B, 2) * _eio[1];
                    }
                }
                else
                {
                    S -= lgamma_fast(_eio[1] + 1);
                    if constexpr (is_directed_v<g_t>)
                        S += _eio[1] * (lbinom_fast(B, 2) + std::log(2));
                    else
                        S += _eio[1] * lbinom_fast(B, 2);
                    #pragma omp parallel reduction(+:S)
                    parallel_loop_no_spawn
                        (_state._candidate_groups,
                         [&](auto, auto r)
                         {
                             if constexpr (is_directed_v<g_t>)
                                 S -= lgamma_fast(_err[r] + 1);
                             else
                                 S -= _err[r] * std::log(2) + lgamma_fast(_err[r] + 1);
                         });
                    S += lbinom_fast(B + _eio[0] - 1, size_t(_eio[0]));
                }

                if (ea.constants)
                {
                    if (_deg_corr && ea.degree_entropy)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_vertex_loop_no_spawn
                        (_g,
                         [&](auto v)
                         {
                             S += get_deg_entropy(_degs[v], _g);
                         });
                    }

                    if (ea.parallel_entropy)
                        S += get_parallel_entropy(_g, _eweight);
                }
            }

            if (B > 1 && ea.edges_dl)
                S += safelog_fast(_E + 1);

            if (ea.partition_dl)
            {
                for (auto& ps : _partition_stats)
                    S += ps.get_dl();
            }

            if (_deg_corr && ea.degree_dl)
                S += _deg_stats.get_dl(ea.degree_dl_kind);

            return S;
        }
    };

    template <class State>
    static void python_export_state(auto&& c)
    {
        c.def("get_err",
              +[](State& state)
               {
                   return state._spec._err;
               });
        c.def("get_erm",
              +[](State& state)
               {
                   return state._spec._erm;
               });
        c.def("get_erp",
              +[](State& state)
               {
                   return state._spec._erp;
               });
        c.def("get_eio",
              +[](State& state)
               {
                   return boost::python::make_tuple(state._spec._eio[0],
                                                    state._spec._eio[1]);
               });
    }
};

} // STATE_NAME

} // graph_tool namespace

#endif //SPEC_HH
