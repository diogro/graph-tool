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

#ifndef GRAPH_RANKED_HH
#define GRAPH_RANKED_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include <boost/container/flat_set.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef vprop_map_t<int64_t> umap_t;

#define RANKED_STATE_params                                                    \
   ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                 \
   ((metric,, bool, 0))                                                        \
   ((u,, umap_t, 0))

GEN_STATE_BASE(RankedStateBase, RANKED_STATE_params)

template <class BState>
class OState
{
public:

    template <class... Ts>
    class RankedState
        : public RankedStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(RankedStateBase<Ts...>, RANKED_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, RANKED_STATE_params)
        using typename RankedStateBase<Ts...>::args_t;
        using RankedStateBase<Ts...>::dispatch_args;

        typedef BState bstate_t;

        typedef typename bstate_t::_entropy_args_t _entropy_args_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        RankedState(BState& ustate, ATs&&... args)
            : RankedStateBase<Ts...>(std::forward<ATs>(args)...),
              _ustate(ustate),
              _g(ustate._g),
              _b(ustate._b),
              _eweight(ustate._eweight),
              _m_entries(num_vertices(ustate._bg)),
              _u_c(_u.get_checked()),
              _args(std::forward<ATs>(args)...)
        {
            GILRelease gil_release;

            _u.resize(num_vertices(ustate._bg));

            for (auto e : edges_range(_g))
            {
                auto u = source(e, _g);
                auto v = target(e, _g);
                _E[stream_dir(_b[u], _b[v])] += _eweight[e];
            }

            if (_metric)
            {
                for (auto r : vertices_range(ustate._bg))
                    _u[r] = r;

                for (auto e : edges_range(ustate._bg))
                {
                    auto r = source(e, ustate._bg);
                    auto s = target(e, ustate._bg);

                    if (r == s)
                        continue;

                    size_t ers = _ustate._mrs[e];
                    size_t esr = get_beprop(s, r, _ustate._mrs, _ustate._emat);
                    size_t mrs = ers + esr;

                    if (r > s)
                    {
                        if (esr > 0)
                            continue;
                        ers = esr;
                    }

                    auto d = rank_dist(r, s);

                    get<0>(_ed[d]) += get_edp(ers, mrs);
                    get<1>(_ed[d]) += get_edm(ers, mrs);
                    _Dp += (ers < floor(mrs/2.));
                    _Dm += (ers > floor(mrs/2.));
                }
            }
        }

        bstate_t& _ustate;
        typename bstate_t::g_t& _g;
        typename bstate_t::b_t& _b;
        typename bstate_t::eweight_t& _eweight;

        class m_entries_t:
            public bstate_t::m_entries_t
        {
        public:
            m_entries_t(size_t B=0)
                : bstate_t::m_entries_t(B)
            {}

            std::array<int64_t, 3> _dE = {0, 0, 0};
        };

        m_entries_t _m_entries;

        std::array<size_t, 3> _E = {0, 0, 0};

        typename u_t::checked_t _u_c;

        idx_map<int64_t, std::tuple<int64_t,int64_t>> _ed, _ed_d;
        int64_t _Dp = 0, _Dm = 0;

        args_t _args;

        int stream_dir(size_t r, size_t s)
        {
            auto x = _u[r];
            auto y = _u[s];
            return ustream_dir(x, y);
        }

        int ustream_dir(int64_t x, int64_t y)
        {
            if (x < y)
                return 0;   // upstream
            else if (x > y)
                return 2;   // downstream
            else
                return 1;   // lateral
        }

        size_t rank_dist(size_t r, size_t s)
        {
            return std::abs(int64_t(r) - int64_t(s));
        }

        int64_t get_edp(int64_t ers, int64_t mrs)
        {
            return (ers <= floor(mrs/2.)) * ers + (ers >= ceil(mrs/2.)) * (mrs - ers) - (ers == mrs/2.) * ers;
        }

        int64_t get_edm(int64_t ers, int64_t mrs)
        {
            return (ers <= floor(mrs/2.)) * (floor(mrs/2.) - ers) + (ers >= ceil(mrs/2.)) * (ers - ceil(mrs/2.));
        }

        void add_block(size_t n)
        {
            _ustate.add_block(n);
        }

        std::array<int64_t, 3>
        get_dE(size_t v, size_t r, size_t nr)
        {
            std::array<int64_t, 3> dE = {0, 0, 0};

            for (auto e : out_edges_range(v, _g))
            {
                auto s = _b[target(e, _g)];
                auto w = _eweight[e];
                dE[stream_dir(r, s)] -= w;
                if (target(e, _g) == v)
                    s = nr;
                dE[stream_dir(nr, s)] += w;
            }

            for (auto e : in_edges_range(v, _g))
            {
                auto s = _b[source(e, _g)];
                auto w = _eweight[e];
                dE[stream_dir(s, r)] -= w;
                if (source(e, _g) == v)
                    s = nr;
                dE[stream_dir(s, nr)] += w;
            }
            return dE;
        }

        double get_ed_dS()
        {
            double dS = 0;
            for (auto [d, delta] : _ed_d)
            {
                auto& c =_ed[d];
                dS -= get_Sd(std::get<0>(c), std::get<1>(c));
                dS += get_Sd(std::get<0>(c) + std::get<0>(delta),
                             std::get<1>(c) + std::get<1>(delta));
            }
            return dS;
        }

        double virtual_move(size_t v, size_t r, size_t nr, entropy_args_t& ea,
                            m_entries_t& m_entries)
        {
            if (r == nr)
                return 0;

            entropy_args_t uea = ea;
            uea.edges_dl = false;

            double dS = _ustate.virtual_move(v, r, nr, uea, m_entries);

            assert(!std::isnan(dS) && !std::isinf(dS));

            if (ea.edges_dl || _ustate._coupled_state != nullptr)
            {
                m_entries._dE = get_dE(v, r, nr);

                int64_t dB = 0;
                if (_ustate._wr[r] == 1)
                    dB--;
                if (_ustate._wr[nr] == 0)
                    dB++;

                dS -= get_edges_dl({0, 0, 0}, 0);
                dS += get_edges_dl(m_entries._dE, dB);

                assert(!std::isnan(dS) && !std::isinf(dS));

                int64_t Dp_d = 0, Dm_d = 0;
                if (_metric)
                    _ed_d.clear();

                entries_op(m_entries, _ustate._emat,
                           [&](auto t, auto u, auto& me, auto delta)
                           {
                               if (delta == 0 || t == u)
                                   return;

                               size_t etu = (me != _ustate._emat.get_null_edge()) ?
                                   _ustate._mrs[me] : 0;
                               size_t eut = get_beprop(u, t, _ustate._mrs,
                                                       _ustate._emat);

                               int64_t delta_r = m_entries.get_delta(u, t);

                               if (delta_r != 0 && t > u)
                                   return;

                               dS += lbinom_fast(etu + eut, etu);
                               dS -= lbinom_fast(etu + delta + eut + delta_r,
                                                 etu + delta);

                               if (_metric && t != u)
                               {
                                   auto d = rank_dist(t, u);
                                   auto mtu = etu + eut;
                                   auto delta_m = delta + delta_r;

                                   if (t > u)
                                   {
                                       etu = eut;
                                       delta = 0;
                                   }

                                   if (etu <= floor(mtu/2.))
                                       dS += lbinom(floor(mtu/2.), etu);
                                   if (etu >= ceil(mtu/2.))
                                       dS += lbinom(mtu - ceil(mtu/2.), etu - ceil(mtu/2.));

                                   if (etu + delta <= floor((mtu + delta_m)/2.))
                                       dS -= lbinom(floor((mtu + delta_m)/2.), etu  + delta);
                                   if (etu + delta >= ceil((mtu + delta_m)/2.))
                                       dS -= lbinom((mtu + delta_m) - ceil((mtu + delta_m)/2.), etu + delta - ceil((mtu + delta_m)/2.));

                                   get<0>(_ed_d[d]) -= get_edp(etu, mtu);
                                   get<1>(_ed_d[d]) -= get_edm(etu, mtu);
                                   get<0>(_ed_d[d]) += get_edp(etu + delta, mtu + delta_m);
                                   get<1>(_ed_d[d]) += get_edm(etu + delta, mtu + delta_m);

                                   Dp_d -= (etu < floor(mtu/2.));
                                   Dm_d -= (etu > floor(mtu/2.));
                                   Dp_d += (etu + delta < floor((mtu + delta_m)/2.));
                                   Dm_d += (etu + delta > floor((mtu + delta_m)/2.));
                               }

                               assert(!std::isnan(dS) && !std::isinf(dS));
                           });

                if (_metric)
                {
                    dS += get_ed_dS();
                    dS -= get_Sd(_Dp, _Dm);
                    dS += get_Sd(_Dp + Dp_d, _Dm + Dm_d);
                }
            }

            assert(!std::isnan(dS) && !std::isinf(dS));
            return dS;
        }

        double virtual_move(size_t v, size_t r, size_t nr, entropy_args_t& ea)
        {
            return virtual_move(v, r, nr, ea, _m_entries);
        }

        template <class ME>
        void move_vertex(size_t v, size_t nr, ME& m_entries)
        {
            size_t r = _b[v];

            if (r == nr)
                return;

            for (size_t i = 0; i < 3; ++i)
                _E[i] += m_entries._dE[i];

            if (_metric)
            {
                entries_op(m_entries, _ustate._emat,
                           [&](auto t, auto u, auto& me, auto delta)
                           {
                               if (t == u || delta == 0)
                                   return;

                               int64_t etu = (me != _ustate._emat.get_null_edge()) ?
                                   _ustate._mrs[me] : 0;

                               int64_t eut = get_beprop(u, t, _ustate._mrs,
                                                       _ustate._emat);

                               int64_t delta_r = m_entries.get_delta(u, t);
                               int64_t delta_m = delta + delta_r;

                               auto d = rank_dist(t, u);
                               auto mtu = etu + eut;

                               if (t > u)
                               {
                                   if (delta_r != 0)
                                       return;
                                   etu = eut;
                                   delta = delta_r;
                               }

                               auto& ed = _ed[d];

                               get<0>(ed) -= get_edp(etu, mtu);
                               get<0>(ed) += get_edp(etu + delta, mtu + delta_m);

                               assert(get<0>(ed) >= 0);

                               get<1>(ed) -= get_edm(etu, mtu);
                               get<1>(ed) += get_edm(etu + delta, mtu + delta_m);

                               assert(get<1>(ed) >= 0);

                               _Dp -= (etu < floor(mtu/2.));
                               _Dp += ((etu + delta) < floor((mtu + delta_m)/2.));

                               _Dm -= (etu > floor(mtu/2.));
                               _Dm += ((etu + delta) > floor((mtu + delta_m)/2.));
                           });
            }

            _ustate.move_vertex(v, nr, m_entries);
        }

        void move_vertex(size_t v, size_t nr)
        {
            size_t r = _b[v];

            if (r == nr)
                return;

            auto dE = get_dE(v, r, nr);

            for (size_t i = 0; i < 3; ++i)
            {
                _E[i] += dE[i];
                _m_entries._dE[i] = 0;
            }

            _ustate.get_move_entries(v, r, nr, _m_entries,
                                     [](auto&) constexpr {return false;});

            move_vertex(v, nr, _m_entries);
        }

        std::array<int64_t, 3>
        get_group_dE(size_t r, int64_t nu)
        {
            std::array<int64_t, 3> dE = {0, 0, 0};

            for (auto e : out_edges_range(r, _ustate._bg))
            {
                auto s = target(e, _g);
                if (s == r)
                    continue;
                auto w = _ustate._mrs[e];
                dE[stream_dir(r, s)] -= w;
                dE[ustream_dir(nu, _u[s])] += w;
            }

            for (auto e : in_edges_range(r, _ustate._bg))
            {
                auto s = source(e, _g);
                if (s == r)
                    continue;
                auto w = _ustate._mrs[e];
                dE[stream_dir(s, r)] -= w;
                dE[ustream_dir(_u[s], nu)] += w;
            }
            return dE;
        }

        double reorder_group_dS(size_t r, int64_t nu)
        {
            if (_u[r] == nu)
                return 0;
            auto dE = get_group_dE(r, nu);
            double dS = 0;
            dS -= get_edges_dl({0, 0, 0}, 0);
            dS += get_edges_dl(dE, 0);
            return dS;
        }

        void reorder_group(size_t r, int64_t nu)
        {
            if (_u[r] == nu)
                return;
            auto dE = get_group_dE(r, nu);
            for (size_t i = 0; i < 3; ++i)
                _E[i] += dE[i];
            _u[r] = nu;
        }

        size_t get_empty_block(size_t v, bool force_add = false)
        {
            return _ustate.get_empty_block(v, force_add);
        }

        int64_t sample_u(size_t r, rng_t& rng)
        {
            if (_metric)
                return r;
            uniform_int_distribution<int64_t>
                usample(std::numeric_limits<int64_t>::min(),
                        std::numeric_limits<int64_t>::max());
            return usample(rng);
        }

        size_t sample_block(size_t v, double c, double d, rng_t& rng)
        {
            auto s = _ustate.sample_block(v, c, d, rng);

            if (_ustate._wr[s] == 0)
                _u_c[s] = sample_u(s, rng);

            return s;
        }

        size_t sample_block_local(size_t v, rng_t& rng)
        {
            return _ustate.sample_block_local(v, rng);
        }

        // Computes the move proposal probability
        double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                             bool reverse)
        {
            return _ustate.get_move_prob(v, r, s, c, d, reverse);
        }

        double get_edges_dl(const std::array<int64_t,3>& dE, int64_t dB,
                            int64_t dE_tot = 0)
        {
            size_t B = 0;
            for (auto& ps : _ustate._partition_stats)
                B += ps.get_actual_B();
            B += dB;

            double S = 0;

            if (_ustate._coupled_state == nullptr)
                S += lbinom_fast<false>((B * (B + 1)) / 2 + _ustate._E + dE_tot - 1,
                                        _ustate._E + dE_tot);

            std::array<size_t, 3> E;
            for (int i = 0; i < 3; ++i)
                E[i] = _E[i] + dE[i];

            if (_metric)
            {
                S += lbinom_fast(_ustate.get_N(), B);
            }
            else
            {
                S += lgamma_fast(E[0] + E[2] + 2);
                S -= lgamma_fast(E[0] + 1) + lgamma_fast(E[2] + 1);
            }

            assert(!std::isnan(S) && !std::isinf(S));
            return S;
        }

        double get_Sd(int64_t edp, int64_t edm)
        {
            return lgamma_fast(edp + edm + 2) -lgamma_fast(edp + 1) - lgamma_fast(edm + 1);
        }

        double entropy(entropy_args_t ea, bool propagate=false)
        {
            double S = 0;

            if (ea.edges_dl || _ustate._coupled_state != nullptr)
            {
                S += get_edges_dl({0, 0, 0}, 0);

                #pragma omp parallel reduction(+:S)
                parallel_edge_loop_no_spawn
                    (_ustate._bg,
                     [&](const auto& e)
                     {
                         auto r = source(e, _ustate._bg);
                         auto s = target(e, _ustate._bg);

                         if (r >= s)
                             return;

                         size_t ers = _ustate._mrs[e];
                         size_t esr = get_beprop(s, r, _ustate._mrs, _ustate._emat);
                         size_t mrs = ers + esr;

                         S -= lbinom_fast(mrs, ers);

                         if (_metric)
                         {
                             if (ers <= floor(mrs/2.))
                                 S -= lbinom(floor(mrs/2.), ers);
                             if (ers >= ceil(mrs/2.))
                                 S -= lbinom(mrs - ceil(mrs/2.), ers - ceil(mrs/2.));
                         }
                     });

                if (_metric)
                {
                    for (auto& [d, c] : _ed)
                        S += get_Sd(std::get<0>(c), std::get<1>(c));
                    S += get_Sd(_Dp, _Dm);
                }
            }

            ea.edges_dl = false;
            S += _ustate.entropy(ea, propagate);

            return S;
        }

        double modify_edge_dS(size_t u, size_t v, const GraphInterface::edge_t& e,
                              int64_t dm, entropy_args_t ea)
        {
            double dS = 0;

            if (ea.edges_dl || _ustate._coupled_state != nullptr)
            {
                auto r = _b[u];
                auto s = _b[v];

                std::array<int64_t, 3> dE = {0, 0, 0};
                dS -= get_edges_dl(dE, 0);
                dE[stream_dir(r, s)] += dm;
                dS += get_edges_dl(dE, 0, dm);

                if (r != s)
                {
                    size_t ers = get_beprop(r, s, _ustate._mrs, _ustate._emat);
                    size_t esr = get_beprop(s, r, _ustate._mrs, _ustate._emat);
                    dS += lbinom_fast(ers + esr, ers);
                    dS -= lbinom_fast(ers + esr + dm, ers + dm);
                }
            }

            ea.edges_dl = false;
            dS += _ustate.modify_edge_dS(u, v, e, dm, ea);
            return dS;
        }

        template <bool Add, bool Deplete=true>
        void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e, int64_t dm)
        {
            _ustate.template modify_edge<Add, Deplete>(u, v, e, dm);
            _E[stream_dir(_b[u], _b[v])] += (Add) ? dm : -dm;
        }

        template <class MCMCState>
        void init_mcmc(MCMCState& state)
        {
            _ustate.init_mcmc(state);
        }

        size_t node_weight(size_t v)
        {
            return _ustate.node_weight(v);
        }

        bool is_last(size_t v)
        {
            return _ustate.is_last(v);
        }

        bool allow_move(size_t v, size_t r)
        {
            return _ustate.allow_move(v, r);
        }

        size_t virtual_remove_size(size_t v)
        {
            return _ustate.virtual_remove_size(v);
        }

        auto& get_candidate_groups(size_t v)
        {
            return _ustate.get_candidate_groups(v);
        }

        template <class V>
        void push_state(V&) {}
        void pop_state() {}
        void store_next_state(size_t) {}
        void clear_next_state() {}

        void relax_update(bool relax)
        {
            _ustate.relax_update(relax);
        }

        void couple_state(BlockStateVirtualBase& us,
                          const entropy_args_t& ea)
        {
            _ustate.couple_state(us, ea);
        }

        void decouple_state()
        {
            _ustate.decouple_state();
        }

        //owned by deep copy
        std::shared_ptr<BState> _ustate_p;

        template <size_t... Is>
        RankedState* deep_copy(index_sequence<Is...>)
        {
            auto u = _u.copy();
            auto args =
                dispatch_args(_args,
                              [&](std::string name, auto& a) -> decltype(auto)
                              {
                                  typedef std::remove_reference_t<decltype(a)> a_t;
                                  if (name == "u")
                                  {
                                      auto& u_ = u; // workaround clang bug
                                      if constexpr (std::is_same_v<a_t, u_t>)
                                          return u_;
                                      return a;
                                  }
                                  return a;
                              });

            auto* ustate = _ustate.deep_copy();
            auto state = new RankedState(*ustate, std::get<Is>(args)...);
            state->_u_c = state->_u.get_checked();
            state->_ustate_p = std::shared_ptr<BState>(ustate);
            return state;
        };

        RankedState* deep_copy()
        {
            return deep_copy(make_index_sequence<sizeof...(Ts)>{});
        }

        void deep_assign(const RankedState& state_)
        {
            const RankedState& state =
                *dynamic_cast<const RankedState*>(&state_);
            _ustate.deep_assign(state._ustate);
            _u.get_storage() = state._u.get_storage();
        }
    };
};
} // graph_tool namespace

#endif //GRAPH_RANKED_HH
