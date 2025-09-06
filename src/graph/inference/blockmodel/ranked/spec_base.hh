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

#ifndef SPEC_BASE_HH
#define SPEC_BASE_HH

#include "config.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <print>

#include "../spec.hh"

#include "../../support/graph_state.hh"

#include "../../../parallel_util.hh"

#define REARGS_params                                                          \
    ((label_dl, bool))                                                         \

#define RANKED_SPEC_params                                                     \
    ((distance, bool, 0))                                                      \
    ((unique, bool, 0))                                                        \
    ((M, int64_t, 0))                                                          \
    ((u, std::vector<int64_t>&, 0))                                            \
    ((rng, rng_t&, 0))

#include "types.hh"
#include "entropy.hh"

namespace graph_tool
{

template <class Base = BlockStateSpec>
struct RankedBlockStateSpec
    : public Base
{
    typedef decltype(get_ref_tr(always_directed)) g_tr;

    using group_t = Base::group_t;
    using count_t = Base::count_t;

    constexpr static bool has_merge = true;
    constexpr static bool has_parallel_move = true;
    constexpr static bool has_relabel = true;
    constexpr static bool has_parallel_relabel = true;

    constexpr static bool edges_dl = false;

    template <class State>
    struct m_entries_t
    {
        struct RankedEntrySet:
            public EntrySet<true, group_t, count_t, GraphInterface::edge_t,
                            typename State::weight_t>
        {
            typedef EntrySet<false, group_t, count_t, GraphInterface::edge_t,
                             typename State::weight_t> NextEntrySet;

            void set_next(NextEntrySet& next)
            {
                _next = &next;
            }

            NextEntrySet* _next = nullptr;
        };

        typedef RankedEntrySet type;
    };

    template <class> class dummy {};

    template <class State, template <class> class TVal = dummy>
    struct coupled_state_t
    {
        static inline auto anames = State::__param_names;
        static inline auto fargs_t =
            hana::filter(hana::zip(State::args_t, anames),
                         [](auto x) constexpr
                         {
                             constexpr auto distance = BOOST_HANA_STRING("distance");
                             constexpr auto u = BOOST_HANA_STRING("u");
                             constexpr auto M = BOOST_HANA_STRING("M");
                             constexpr auto unique = BOOST_HANA_STRING("unique");
                             constexpr auto rng = BOOST_HANA_STRING("rng");
                             return std::integral_constant<bool,
                                                           (x[1_c] != distance &&
                                                            x[1_c] != u &&
                                                            x[1_c] != M &&
                                                            x[1_c] != unique &&
                                                            x[1_c] != rng)>{};
                         });
        static inline auto args_t =
            hana::transform(fargs_t,
                            [](auto y) constexpr
                            {
                                auto x = y[0_c];
                                using T = std::decay_t<typename decltype(x)::type>;
                                typedef typename State::g_t g_t;
                                typedef typename State::bg_t bg_t;
                                typedef std::decay_t<decltype(base_graph(std::declval<g_t>()))> u_t;
                                if constexpr (std::is_same_v<T, g_t>)
                                    return hana::type_c<boost::undirected_adaptor<u_t>&>;
                                else if constexpr (std::is_same_v<T, bg_t>)
                                    return hana::type_c<boost::undirected_adaptor<u_t>&>;
                                else
                                    return x;
                            });

        constexpr static inline auto state_t =
            hana::unpack(args_t,
                         [](auto... as)
                         {
                             return hana::type_c<block_model::BlockState<typename decltype(as)::type...>>;
                         });

        typedef decltype(state_t)::type type;
    };

    DEFINE_STRUCT_BASED(entropy_args_t, Base::entropy_args_t, REARGS_params);

    template <class State>
    static void python_export_state(auto& c)
    {
        EXPORT_STRUCT_BASED(entropy_args_t, typename Base::entropy_args_t, REARGS_params);

        Base::template python_export_state<State>(c);
        c.def("get_Es",
              +[](State& state)
               {
                   return python::make_tuple(state._spec._E[0],
                                             state._spec._E[1],
                                             state._spec._E[2],
                                             state._spec._E[3]);
               });
        c.def("get_ed",
              +[](State& state)
               {
                   std::vector<int64_t> edp, edm;
                   for (auto& [ep, em] : state._spec._ed)
                   {
                       edp.push_back(ep);
                       edm.push_back(em);
                   }
                   return python::make_tuple(edp, edm, state._spec._Dp,
                                             state._spec._Dm);
               });
    }

    template <class State>
    struct Imp
        : public Base::template Imp<State>
    {
        using ibase_t = Base::template Imp<State>;
        using spec_t = State::spec_t;
        using group_t = spec_t::group_t;
        using vertex_t = State::vertex_t;

        typedef spec_t sp;

        GET_PARAMS_SPEC_TYPEDEF(RANKED_SPEC_params)
        GET_PARAMS_SPEC_DEFINE(RANKED_SPEC_params)

        Imp(State& state)
            : ibase_t(state),
              GET_PARAMS_SPEC_INIT(RANKED_SPEC_params)
              _state(state),
              _g(state._g),
              _b(state._b),
              _eweight(state._eweight)
        {
        }

        State& _state;
        typename State::g_t& _g;
        typename State::b_t& _b;
        typename State::eweight_t& _eweight;
        using weight_t = property_traits<typename State::eweight_t>::value_type;

        void post_init()
        {
            ibase_t::post_init();
            while (_u.size() < num_vertices(_state._bg))
                _u.emplace_back(sample_group_label(_u.size(), _rng));

            if (_distance)
            {
                if (_unique)
                {
                    for (auto&& [i, r] : std::views::enumerate(_u))
                        r = i;
                }

                _ed.resize(num_vertices(_state._g));

                for (auto e : edges_range(_state._bg))
                {
                    auto r = source(e, _state._bg);
                    auto s = target(e, _state._bg);

                    if (r == s)
                        continue;

                    auto ers = _state._mrs[e];
                    auto esr = get_mrs(s, r);
                    auto mrs = ers + esr;

                    if (_u[r] > _u[s])
                    {
                        if (esr > 0)
                            continue;
                        ers = 0;
                    }

                    auto d = rank_dist(r, s);

                    if (d == 0)
                        continue;

                    get<0>(_ed[d]) += get_edp(ers, mrs);
                    get<1>(_ed[d]) += get_edm(ers, mrs);
                    _Dp += (ers < floor(mrs/2.));
                    _Dm += (ers > ceil(mrs/2.));
                }
            }

            for (auto e : edges_range(_g))
            {
                auto u = source(e, _g);
                auto v = target(e, _g);
                _E[stream_dir(_b[u], _b[v])] += _eweight[e];
                if (_b[u] == _b[v])
                    _E[3] += _eweight[e];
            }
        }

        typedef std::array<weight_t, 4> E_t;
        E_t _E = {};

        std::vector<std::tuple<weight_t,weight_t>> _ed;
        count_t _Dp{}, _Dm{};

        weight_t get_mrs(group_t r, group_t s) const
        {
            auto me = _state._emat.get_me(r, s);
            if (me == _state._emat.get_null_edge())
                return 0;
            return _state._mrs[me];
        }

        int ustream_dir(auto x, auto y)
        {
            if (x < y)
                return 0;   // upstream
            else if (x > y)
                return 2;   // downstream
            else
                return 1;   // lateral
        }

        int stream_dir(group_t r, group_t s)
        {
            auto x = _u[r];
            auto y = _u[s];
            return ustream_dir(x, y);
        }

        int64_t ceil(auto x) const
        {
            return std::ceil(x);
        }

        int64_t floor(auto x) const
        {
            return std::floor(x);
        }

        int64_t abs(auto x) const
        {
            return std::abs(x);
        }

        group_t rank_dist(group_t r, group_t s) const
        {
            return abs(_u[r] - _u[s]);
        }

        weight_t get_edp(weight_t ers, weight_t mrs) const
        {
            return (ers <= floor(mrs/2.)) * ers + (ers >= ceil(mrs/2.)) * (mrs - ers) - (ers == mrs/2.) * ers;
        }

        weight_t get_edm(weight_t ers, weight_t mrs) const
        {
            return (ers <= floor(mrs/2.)) * (floor(mrs/2.) - ers) + (ers >= ceil(mrs/2.)) * (ers - ceil(mrs/2.));
        }

        auto get_dE(auto&& m_entries)
        {
            E_t dE{};
            entries_op(m_entries,
                       [&](auto t, auto u, auto delta)
                       {
                           dE[stream_dir(t, u)] += delta;
                           if (t == u)
                               dE[3] += delta;
                       });
            return dE;
        }

        double get_ed_dS(auto&& ed_d)
        {
            double dS = 0;
            for (auto [d, delta] : ed_d)
            {
                auto& c = _ed[d];
                dS -= get_Sd(std::get<0>(c), std::get<1>(c));
                dS += get_Sd(std::get<0>(c) + std::get<0>(delta),
                             std::get<1>(c) + std::get<1>(delta));
            }
            return dS;
        }

        double get_edges_dl(const E_t& dE = {}, int64_t dB = 0) const
        {
            size_t B = 0;
            for (auto& ps : _state._partition_stats)
                B += ps.get_actual_B();
            B += dB;

            double S = 0;

            if (_state._coupled_state == nullptr)
            {
                auto dE_tot = std::accumulate(dE.begin(), dE.end(), 0);
                S += lbinom_fast((B * (B + 1)) / 2 + _state._E + dE_tot - 1,
                                 _state._E + dE_tot);
            }

            E_t E;
            for (auto&& [Ei, Ei_, dEi] : std::views::zip(E, _E, dE))
                Ei = Ei_ + dEi;

            if (_distance)
            {
                S += lbinom_fast(_state.get_N(), B);
            }
            else
            {
                S += lgamma_fast(E[0] + E[2] + 2);
                S -= lgamma_fast(E[0] + 1) + lgamma_fast(E[2] + 1);
            }

            S += (E[1] - E[3]) * std::log(2);

            return S;
        }

        double get_Sd(int64_t edp, int64_t edm) const
        {
            return lgamma_fast(edp + edm + 2) - lgamma_fast(edp + 1) - lgamma_fast(edm + 1);
        }

        double entropy(const entropy_args_t& ea) const
        {
            double S = ibase_t::entropy(ea);
            double S_dl = 0;

            if (ea.edges_dl || _state._coupled_state != nullptr)
            {
                S_dl += get_edges_dl();

                #pragma omp parallel reduction(+:S_dl)
                parallel_edge_loop_no_spawn
                    (_state._bg,
                     [&](const auto& e)
                     {
                         auto r = source(e, _state._bg);
                         auto s = target(e, _state._bg);

                         if (r == s)
                             return;

                         auto ers = _state._mrs[e];
                         auto esr = get_mrs(s, r);
                         auto mrs = ers + esr;
                         auto d = rank_dist(r, s);

                         if ((_u[r] > _u[s]) || (d == 0 && s > r && esr > 0))
                             return;

                         if (!_distance || _u[r] == _u[s])
                         {
                             S_dl -= lbinom_fast(mrs, ers);
                         }
                         else
                         {
                             if (ers < floor(mrs/2.))
                                 S_dl -= lbinom_fast(floor(mrs/2.), ers);
                             if (ers > ceil(mrs/2.))
                                 S_dl -= lbinom_fast(mrs - ceil(mrs/2.), ers - ceil(mrs/2.));
                         }
                     });

                if (_distance)
                {
                    #pragma omp parallel reduction(+:S_dl)
                    parallel_loop_no_spawn
                        (_ed,
                         [&](auto, auto&& c)
                         {
                             S_dl += get_Sd(std::get<0>(c), std::get<1>(c));
                         });
                    S_dl += get_Sd(_Dp, _Dm);
                }
            }

            if (!_unique && ea.label_dl)
            {
                auto B = _state._actual_B;
                S_dl += lbinom_fast(_M + B - 1, B);
            }

            return S + ea.beta_dl * S_dl;
        }

        template <bool concurrent=false, class Delta = std::monostate>
        void apply_delta(auto& m_entries, Delta&& deltas = std::monostate())
        {
            auto dE = get_dE(m_entries);

            for (auto&& [Ei, dEi] : std::views::zip(_E, dE))
                atomic_inc<concurrent>(Ei, dEi);

            if (!_distance)
                return;

            entries_op(m_entries,
                       [&](auto t, auto u, auto delta)
                       {
                           if (t == u || delta == 0)
                               return;

                           auto d = rank_dist(t, u);
                           if (d == 0)
                               return;

                           auto me = _state._emat.get_me(t, u);
                           int64_t etu = (me != _state._emat.get_null_edge()) ?
                               _state._mrs[me] : 0;
                           int64_t eut = get_mrs(u, t);
                           int64_t delta_r = 0;
                           if constexpr (std::is_same_v<Delta, std::monostate>)
                           {
                               delta_r = m_entries.get_delta(u, t);
                           }
                           else
                           {
                               auto iter = deltas[u].find(t);
                               if (iter != deltas[u].end())
                                   delta_r = iter->second;
                           }
                           int64_t delta_m = delta + delta_r;
                           auto mtu = etu + eut;

                           if (_u[t] > _u[u])
                           {
                               if (delta_r != 0)
                                   return;
                               etu = eut;
                               delta = 0;
                           }

                           auto& ed = _ed[d];

                           atomic_inc<concurrent>
                               (get<0>(ed),
                                get_edp(etu + delta, mtu + delta_m) - get_edp(etu, mtu));

                           assert(get<0>(ed) >= 0);

                           atomic_inc<concurrent>
                               (get<1>(ed),
                                get_edm(etu + delta, mtu + delta_m) - get_edm(etu, mtu));

                           assert(get<1>(ed) >= 0);

                           atomic_inc<concurrent>
                               (_Dp,
                                int((etu + delta) < floor((mtu + delta_m)/2.)) - int(etu < floor(mtu/2.)));

                           atomic_inc<concurrent>
                               (_Dm,
                                int((etu + delta) > ceil((mtu + delta_m)/2.)) - int(etu > ceil(mtu/2.)));
                       });
        }

        double virtual_move_dS(group_t r, group_t nr, const entropy_args_t& ea,
                               auto& m_entries, int n)
        {
            double dS_dl = 0;

            int64_t dB = 0;
            if (_state._wr[r] == n)
                dB--;
            if (_state._wr[nr] == 0)
                dB++;

            if (ea.edges_dl || _state._coupled_state != nullptr)
            {
                int64_t Dp_d = 0, Dm_d = 0;

                gt_hash_map<weight_t, std::tuple<weight_t,weight_t>> ed_d;

                entries_op(m_entries,
                           [&](auto t, auto u, auto delta)
                           {
                               if (delta == 0 || t == u)
                                   return;

                               auto me = _state._emat.get_me(t, u);
                               int64_t etu = (me != _state._emat.get_null_edge()) ?
                                   _state._mrs[me] : 0;
                               int64_t eut = get_mrs(u, t);
                               int64_t delta_r = m_entries.get_delta(u, t);
                               auto mtu = etu + eut;
                               auto delta_m = delta + delta_r;

                               auto d = rank_dist(t, u);

                               if ((delta_r != 0) && (_u[t] > _u[u] || (d == 0 && t > u)))
                                   return;

                               if (!_distance || d == 0)
                               {
                                   dS_dl += lbinom_fast(mtu, etu);
                                   dS_dl -= lbinom_fast(mtu + delta_m, etu + delta);
                               }
                               else
                               {
                                   if (_u[t] > _u[u])
                                   {
                                       etu = eut;
                                       delta = 0;
                                   }

                                   if (etu < floor(mtu/2.))
                                       dS_dl += lbinom_fast(floor(mtu/2.), etu);
                                   if (etu > ceil(mtu/2.))
                                       dS_dl += lbinom_fast(mtu - ceil(mtu/2.), etu - ceil(mtu/2.));

                                   if (etu + delta < floor((mtu + delta_m)/2.))
                                       dS_dl -= lbinom_fast(floor((mtu + delta_m)/2.), etu  + delta);
                                   if (etu + delta > ceil((mtu + delta_m)/2.))
                                       dS_dl -= lbinom_fast((mtu + delta_m) - ceil((mtu + delta_m)/2.), etu + delta - ceil((mtu + delta_m)/2.));

                                   get<0>(ed_d[d]) -= get_edp(etu, mtu);
                                   get<1>(ed_d[d]) -= get_edm(etu, mtu);
                                   get<0>(ed_d[d]) += get_edp(etu + delta, mtu + delta_m);
                                   get<1>(ed_d[d]) += get_edm(etu + delta, mtu + delta_m);

                                   Dp_d -= (etu < floor(mtu/2.));
                                   Dm_d -= (etu > ceil(mtu/2.));
                                   Dp_d += (etu + delta < floor((mtu + delta_m)/2.));
                                   Dm_d += (etu + delta > ceil((mtu + delta_m)/2.));
                               }
                           });

                auto dE = get_dE(m_entries);
                dS_dl -= get_edges_dl();
                dS_dl += get_edges_dl(dE, dB);

                if (_distance)
                {
                    dS_dl += get_ed_dS(ed_d);
                    dS_dl -= get_Sd(_Dp, _Dm);
                    dS_dl += get_Sd(_Dp + Dp_d, _Dm + Dm_d);
                }
            }

            if (!_unique && ea.label_dl)
            {
                auto B = _state._actual_B;
                dS_dl -= lbinom_fast(_M + B - 1, B);
                dS_dl += lbinom_fast(_M + B + dB - 1, B + dB);
            }

            return dS_dl;
        }

        double virtual_move_dS(vertex_t v, group_t nr, const entropy_args_t& ea,
                               auto& m_entries)
        {
            auto r = _state._b[v];
            return virtual_move_dS(r, nr, ea, m_entries, 1);
        }

        double virtual_merge_dS(group_t s, group_t t, const entropy_args_t& ea,
                                auto& m_entries)
        {
            return virtual_move_dS(s, t, ea, m_entries, _state._wr[s]);
        }

        void merge_groups(vertex_t, group_t, auto& m_entries)
        {
            apply_delta(m_entries);
        }

        double modify_edge_dS(vertex_t u, vertex_t v, auto&,
                              int64_t dm, const entropy_args_t& ea)
        {
            double dS_dl = 0;
            if (ea.edges_dl || _state._coupled_state != nullptr)
            {
                auto r = _b[u];
                auto s = _b[v];

                dS_dl -= get_edges_dl();
                E_t dE{};
                dE[stream_dir(r, s)] += dm;
                if (r == s)
                    dE[3] += dm;
                dS_dl += get_edges_dl(dE, 0);

                if (r != s)
                {
                    size_t ers = get_mrs(r, s);
                    size_t esr = get_mrs(s, r);
                    dS_dl += lbinom_fast(ers + esr, ers);
                    dS_dl -= lbinom_fast(ers + esr + dm, ers + dm);
                }
            }
            return  dS_dl * ea.beta_dl;
        }

        template <bool Add, bool Deplete=true>
        void modify_edge(vertex_t u, vertex_t v, auto&, int64_t dm)
        {
            _E[stream_dir(_b[u], _b[v])] += (Add) ? dm : -dm;
            if (_b[u] == _b[v])
                _E[3] += (Add) ? dm : -dm;
        }

        void add_group(size_t n)
        {
            _u.resize(num_vertices(_state._bg));
            if (num_vertices(_state._bg) > _ed.size())
                _ed.resize(num_vertices(_state._bg));
            if (_state._coupled_state != nullptr)
                boost::add_vertex(_state._coupled_state->_g, n);
        }

        void set_concurrent(bool concurrent, size_t M)
        {
            if (_state._coupled_state != nullptr)
            {
                auto& bg = _state._coupled_state->_g;
                auto& mrs = _state._coupled_state->_eweight;
                if (concurrent)
                {
                    M = std::max(base_graph(_g).get_edge_index_range(), M);
                    base_graph(bg).set_concurrent(true);
                    base_graph(bg).reserve_epos(2 * M + 10 * get_num_threads());
                    mrs.reserve(2 * M + 10 * get_num_threads());
                }
                else
                {
                    base_graph(bg).set_concurrent(false);
                    base_graph(bg).shrink_to_fit();
                    mrs.resize(base_graph(bg).get_edge_index_range());
                    mrs.shrink_to_fit();
                }
            }
        }

        template <bool lock>
        void update_rs(auto& me, const auto& delta, auto&&... locks)
        {
            ibase_t::template update_rs<lock>(me, delta, locks...);

            if (_state._coupled_state != nullptr)
            {
                auto r = source(me, _state._bg);
                auto s = target(me, _state._bg);
                auto ew = _state._coupled_state->_eweight.get_checked();

                auto [e, exists] = edge(r, s, _state._coupled_state->_g);

                if (!exists)
                {
                    upgrade_canary_lock<{.ula={.la={.enable=lock}}}>
                        ([&](bool succ)
                         {
                             if (!succ)    [[ unlikely ]]
                                 std::tie(e, exists) = edge(r, s, _state._coupled_state->_g);
                             if (!exists)
                             {
                                 std::tie(e, exists) = add_edge(r, s, _state._coupled_state->_g);
                                 ew[e] = delta;
                             }
                             else
                             {
                                 ew[e] += delta;
                             }
                         }, locks...);
                }
                else
                {
                    auto& w = ew[e];
                    atomic_inc(w, delta);
                    if (w == 0)
                    {
                        upgrade_canary_lock<{.ula={.la={.enable=lock}}}>
                            ([&](bool succ)
                             {
                                 if (!succ)    [[ unlikely ]]
                                     std::tie(e, exists) = edge(r, s, _state._coupled_state->_g);
                                 if (exists && ew[e] == 0)
                                     remove_edge(e, _state._coupled_state->_g);
                             }, locks...);
                    }
                }
            }
        }

        auto get_group_dE(group_t r, int64_t v, auto& u, auto&& skip)
        {
            E_t dE{};

            for (auto e : out_edges_range(r, _state._bg))
            {
                group_t s = target(e, _state._bg);
                if (s == r || skip(s))
                    continue;
                auto w = _state._mrs[e];
                dE[stream_dir(r, s)] -= w;
                dE[ustream_dir(v, u[s])] += w;
            }

            for (auto e : in_edges_range(r, _state._bg))
            {
                group_t s = source(e, _state._bg);
                if (s == r || skip(s))
                    continue;
                auto w = _state._mrs[e];
                dE[stream_dir(s, r)] -= w;
                dE[ustream_dir(u[s], v)] += w;
            }
            return dE;
        }

        auto get_group_dE(group_t r, int64_t v)
        {
            return get_group_dE(r, v, _u, [](auto){ return false; });
        }

        int64_t group_label(group_t r)
        {
            return _u[r];
        }

        template <class RNG>
        int64_t sample_group_label(size_t r, RNG& rng)
        {
            if (_unique)
            {
                if (_distance)
                    return r;
                uniform_int_distribution<int64_t>
                    usample(std::numeric_limits<int64_t>::min(),
                            std::numeric_limits<int64_t>::max());
                return usample(rng);
            }
            uniform_int_distribution<int64_t> usample(0, _M-1);
            return usample(rng);
        }

        constexpr double get_group_label_prob(group_t, int64_t, int64_t, bool)
        {
            return 0;
        }

        double relabel_group_dS(group_t r, int64_t nu, const entropy_args_t& ea)
        {
            if (_u[r] == nu)
                return 0;

            double dS = 0;

            if (ea.edges_dl || _state._coupled_state != nullptr)
            {
                auto dE = get_group_dE(r, nu);
                dS -= get_edges_dl();
                dS += get_edges_dl(dE);
            }

            if (_distance)
            {
                int64_t Dp_d = 0, Dm_d = 0;
                gt_hash_map<weight_t, std::tuple<weight_t,weight_t>> ed_d;

                for (auto me : all_edges_range(r, _state._bg))
                {
                    group_t t = source(me, _state._bg);
                    group_t u = target(me, _state._bg);

                    if (t == u)
                        continue;

                    auto etu = _state._mrs[me];
                    auto eut = get_mrs(u, t);
                    auto mtu = etu + eut;
                    auto d = rank_dist(t, u);

                    if ((_u[t] > _u[u]) || (d == 0 && t > u))
                    {
                        if (eut != 0)
                            continue;
                        etu = 0;
                    }

                    if (d == 0)
                    {
                        dS += lbinom_fast(mtu, etu);
                    }
                    else
                    {
                        if (etu < floor(mtu/2.))
                            dS += lbinom_fast(floor(mtu/2.), etu);
                        if (etu > ceil(mtu/2.))
                            dS += lbinom_fast(mtu - ceil(mtu/2.), etu - ceil(mtu/2.));

                        get<0>(ed_d[d]) -= get_edp(etu, mtu);
                        get<1>(ed_d[d]) -= get_edm(etu, mtu);
                        Dp_d -= (etu < floor(mtu/2.));
                        Dm_d -= (etu > ceil(mtu/2.));
                    }
                }

                for (auto me : all_edges_range(r, _state._bg))
                {
                    group_t t = source(me, _state._bg);
                    group_t u = target(me, _state._bg);

                    if (t == u)
                        continue;

                    auto etu = _state._mrs[me];
                    auto eut = get_mrs(u, t);
                    auto mtu = etu + eut;
                    auto d = (t == r) ? abs(nu - _u[u]) : abs(_u[t] - nu);

                    if (((t == r) && (nu > _u[u])) ||
                        ((u == r) && (_u[t] > nu)) || (d == 0 && t > u))
                    {
                        if (eut != 0)
                            continue;
                        etu = 0;
                    }

                    if (d == 0)
                    {
                        dS -= lbinom_fast(mtu, etu);
                    }
                    else
                    {
                        if (etu < floor(mtu/2.))
                            dS -= lbinom_fast(floor(mtu/2.), etu);
                        if (etu > ceil(mtu/2.))
                            dS -= lbinom_fast(mtu - ceil(mtu/2.), etu - ceil(mtu/2.));

                        get<0>(ed_d[d]) += get_edp(etu, mtu);
                        get<1>(ed_d[d]) += get_edm(etu, mtu);
                        Dp_d += (etu < floor(mtu/2.));
                        Dm_d += (etu > ceil(mtu/2.));
                    }
                }

                if (ea.edges_dl || _state._coupled_state != nullptr)
                {
                    dS += get_ed_dS(ed_d);
                    dS -= get_Sd(_Dp, _Dm);
                    dS += get_Sd(_Dp + Dp_d, _Dm + Dm_d);
                }
            }
            return dS;
        }

        template <bool concurrent=false>
        void relabel_group(group_t r, int64_t nu, auto&& nl, auto&& skip)
        {
            if (_u[r] == nu)
                return;
            auto dE = get_group_dE(r, nu, nl, skip);
            for (auto&& [Ei, dEi] : std::views::zip(_E, dE))
                atomic_inc<concurrent>(Ei, dEi);
            if (_distance)
            {
                for (auto me : all_edges_range(r, _state._bg))
                {
                    group_t t = source(me, _state._bg);
                    group_t u = target(me, _state._bg);

                    if ((t == u) || ((t == r) && skip(u)) || ((u == r) && skip(t)))
                        continue;

                    auto etu = _state._mrs[me];
                    auto eut = get_mrs(u, t);
                    auto mtu = etu + eut;

                    if (_u[t] > _u[u])
                    {
                        if (eut != 0)
                            continue;
                        etu = 0;
                    }

                    auto d = rank_dist(t, u);

                    if (d > 0)
                    {
                        atomic_dec<concurrent>(get<0>(_ed[d]), get_edp(etu, mtu));
                        atomic_dec<concurrent>(get<1>(_ed[d]), get_edm(etu, mtu));
                        atomic_dec<concurrent>(_Dp, int(etu < floor(mtu/2.)));
                        atomic_dec<concurrent>(_Dm, int(etu > ceil(mtu/2.)));
                    }
                }

                for (auto me : all_edges_range(r, _state._bg))
                {
                    group_t t = source(me, _state._bg);
                    group_t u = target(me, _state._bg);

                    if ((t == u) || ((t == r) && skip(u)) || ((u == r) && skip(t)))
                        continue;

                    auto etu = _state._mrs[me];
                    auto eut = get_mrs(u, t);
                    auto mtu = etu + eut;

                    if (((t == r) && (nu > nl[u])) || ((u == r) && (nl[t] > nu)))
                    {
                        if (eut != 0)
                            continue;
                        etu = 0;
                    }

                    auto d = (t == r) ? abs(nu - nl[u]) : abs(nl[t] - nu);

                    if (d > 0)
                    {
                        atomic_inc<concurrent>(get<0>(_ed[d]), get_edp(etu, mtu));
                        atomic_inc<concurrent>(get<1>(_ed[d]), get_edm(etu, mtu));
                        atomic_inc<concurrent>(_Dp, int(etu < floor(mtu/2.)));
                        atomic_inc<concurrent>(_Dm, int(etu > ceil(mtu/2.)));
                    }
                }
            }
        }

        template <bool concurrent=false>
        void relabel_group(group_t r, int64_t nu)
        {
            relabel_group<concurrent>(r, nu, _u, [](auto){ return false; });
            _u[r] = nu;
        }

        template <class Rs, class Us>
        void relabel_groups_parallel(Rs&& rs, Us&& us)
        {
            gt_hash_map<group_t, int64_t> nu_h;
            for (auto [r, u] : std::views::zip(rs, us))
            {
                if (u != _u[r])
                    nu_h[r] = u;
            }

            if (nu_h.empty())
                return;

            auto nu = make_adhoc_prop<group_t>
                ([&](auto r)
                 {
                     auto iter = nu_h.find(r);
                     if (iter == nu_h.end())
                         return _u[r];
                     return iter->second;
                 });

            parallel_loop
                (std::views::zip(rs, us),
                 [&](auto, const auto& ru)
                 {
                     auto [r, u] = ru;
                     if (u == _u[r])
                         return;
                     relabel_group<true>(r, u, nu, [&](auto s) { return (s > r) && (nu_h.find(s) != nu_h.end()); });
                 });

            parallel_loop
                (std::views::zip(rs, us),
                 [&](auto, const auto& ru)
                 {
                     auto [r, u] = ru;
                     _u[r] = u;
                 });
        }

        bool check_edge_counts()
        {
            if (_state._coupled_state != nullptr)
            {
                for (auto me : edges_range(_state._bg))
                {
                    auto r = source(me, _state._bg);
                    auto s = target(me, _state._bg);
                    auto mrs = _state._mrs[me];
                    if (r != s)
                        mrs += get_mrs(s, r);

                    auto [e, exists] = edge(r, s, _state._coupled_state->_g);

                    auto w = (exists) ? _state._coupled_state->_eweight[e] : 0;
                    assert(w == mrs);
                    if (w != mrs)
                        return false;
                }
            }

            E_t E{};
            for (auto e : edges_range(_g))
            {
                auto u = source(e, _g);
                auto v = target(e, _g);
                E[stream_dir(_b[u], _b[v])] += _eweight[e];
            }

            for (auto [x, y] : std::views::zip(_E, E))
            {
                assert(x == y);
                if (x != y)
                    return false;
            }

            return true;
        }
    };
};

} //namespace graph_tool

#endif // SPEC_BASE_HH
