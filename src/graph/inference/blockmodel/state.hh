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

#ifndef BLOCKMODEL_HH
#define BLOCKMODEL_HH

#include "config.h"

#include <print>
#include <vector>
#include <ranges>
#include <mutex>
#include <shared_mutex>
#include <boost/scope/scope_exit.hpp>
#include "idx_map.hh"

#include "../support/graph_state.hh"
#include "../../parallel_util.hh"

#include "spec.hh"
#include "entropy.hh"
#include "partition.hh"
#include "entries.hh"
#include "emat.hh"
#include "elist.hh"

#ifndef STATE_NAME
#define STATE_NAME block_model
#endif

#ifndef STATE_SPEC
#define STATE_SPEC BlockStateSpec
#endif

#ifndef BLOCK_STATE_SPEC_params
#define BLOCK_STATE_SPEC_params
#endif

#ifndef MODULE_PRIO
#define MODULE_PRIO 0
#endif

namespace graph_tool
{
namespace STATE_NAME
{

using namespace boost;

#define BLOCK_MODEL_params_base                                                \
    ((g, STATE_SPEC::g_tr, 1))                                                 \
    ((bg, STATE_SPEC::bg_tr, 1))                                               \
    ((flat_map, STATE_SPEC::flat_map_tr, 1))                                   \
    ((eweight, STATE_SPEC::wmap_tr, 1))                                        \
    ((mrs, STATE_SPEC::bwmap_tr, 1))                                           \
    ((mrp, STATE_SPEC::mrp_tr, 1))                                             \
    ((mrm, STATE_SPEC::mrm_tr, 1))                                             \
    ((vweight, STATE_SPEC::vweight_tr, 1))                                     \
    ((wr, STATE_SPEC::wr_tr, 1))                                               \
    ((b, STATE_SPEC::bmap_t, 0))                                               \
    ((clabel, STATE_SPEC::bmap_t, 0))                                          \
    ((pclabel, STATE_SPEC::bmap_t, 0))                                         \
    ((bfield, STATE_SPEC::bfield_t, 0))                                        \
    ((Bfield, std::vector<double>&, 0))                                        \
    ((deg_corr, bool, 0))

#define BLOCK_MODEL_params BLOCK_MODEL_params_base BLOCK_STATE_SPEC_params

GEN_STATE_BASE(BlockStateBase, BLOCK_MODEL_params)

template <class... Ts>
class BlockState
    : public BlockStateBase<Ts...>
{
public:
    GET_PARAMS_USING(BlockStateBase<Ts...>, BLOCK_MODEL_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_MODEL_params)

    constexpr static inline auto args_t = hana::tuple_t<Ts...>;

    template <class... As>
    using state_t = BlockState<As...>;

    using vertex_t = boost::graph_traits<g_t>::vertex_descriptor;

    typedef STATE_SPEC spec_t;

    typedef spec_t::count_t count_t;
    typedef spec_t::group_t group_t;
    typedef spec_t::vcount_t vcount_t;
    typedef spec_t::bmap_t bmap_t;
    typedef spec_t::entropy_args_t entropy_args_t;
    static constexpr group_t _null_group = spec_t::null_group;

    using weight_t = property_traits<eweight_t>::value_type;
    using deg_t = spec_t::deg_t<BlockState>::type;
    using vw_t = property_traits<vweight_t>::value_type;
    using rw_t = property_traits<wr_t>::value_type;

    using coupled_state_t = spec_t::coupled_state_t<BlockState>::type;
    using coupled_entropy_args_t = spec_t::get_coupled_eargs_t<BlockState>::type;

    template <class... ATs>
    BlockState(ATs&&... args)
        : BlockStateBase<Ts...>(std::forward<ATs>(args)...),
          _bclabel(num_vertices(_bg)),
          _bpclabel(num_vertices(_bg)),
          _spec(*this)
    {
        GILRelease gil_release;

        size_t B = 0;
        for (auto v : vertices_range(_g))
        {
            if (_vweight[v] == 0)
            {
                _b[v] = _null_group;
                _clabel[v] = _null_group;
                _pclabel[v] = _null_group;
                continue;
            }
            auto r = _b[v];
            B = std::max(B, size_t(r + 1));
            _bpclabel.get_checked()[r] = _pclabel[v];
            _bclabel.get_checked()[r] = _clabel[v];
            _wr.get_checked()[r] += _vweight[v];
            if (_groups[r].empty())
                _actual_B++;
            _groups[r].insert(v);
            _N[_clabel[v]] += _vweight[v];
        }

        boost::add_vertex(_bg, B);
        _mrp.reserve(B);
        _mrm.reserve(B);

        for (auto r [[maybe_unused]] : vertices_range(_bg))
            _group_mutex.push_back(std::make_shared<shared_canary_mutex>());

        set_concurrent(true);

        parallel_edge_loop
            (_g,
             [&](const auto& e)
             {
                 atomic_inc(_E, _eweight[e]);
             });

        _emat.add_group(_bg);

        parallel_edge_loop
            (_g,
             [&](const auto& e)
             {
                 auto r = _b[source(e, _g)];
                 auto s = _b[target(e, _g)];

                 auto me = _emat.get_null_edge();

                 do_lock<lock_t::shared, {.pass_lock=true}>
                     (r, s,
                      [&](auto&... locks)
                      {
                          me = _emat.get_me(r, s);
                          if (me == _emat.get_null_edge())
                          {
                              upgrade_canary_lock<{.ula={.relock=false}}>
                                  ([&](bool succ)
                                   {
                                       if (!succ) [[unlikely]]
                                       {
                                           me = _emat.get_me(r, s);
                                           if (me != _emat.get_null_edge())
                                               return;
                                       }
                                       me = boost::add_edge(size_t(r), size_t(s), _bg).first;
                                       _emat.put_me(r, s, me);
                                       _mrs[me] = weight_t{};
                                   }, locks...);
                          }
                      });

                 atomic_inc(_mrs[me], _eweight[e]);
             });
        base_graph(_bg).set_keep_epos(true);

        _m_entries.resize(num_vertices(_bg));

        for (auto r : vertices_range(_bg))
        {
            if (_wr[r] == 0)
                _empty_groups.insert(r);
            else
                _occupied_groups[_bclabel[r]].insert(r);
        }

        _degs.resize(num_vertices(_g));

        parallel_vertex_loop<true>
            (_g,
             [&, this](auto v)
             {
                 if (_vweight[v] == 0)
                     return;
                 auto r = _b[v];
                 deg_t kin = in_degree(v, _g, _eweight);
                 deg_t kout = out_degree(v, _g, _eweight);
                 _degs[v] = {kin, kout};
                 atomic_inc(_mrp[r], kout);
                 if constexpr (is_directed_v<g_t>)
                     atomic_inc(_mrm[r], kin);
                 if constexpr (std::is_scalar_v<deg_t>)
                 {
                     if (kin + kout > 0 && _vweight[v] == 0)
                         throw ValueException("Invalid graph: vweight cannot be zero if total degree is non-zero!");
                 }
             });

        set_concurrent(false);

        for (auto v : vertices_range(_g))
            _ps.template add_item<true>(_b[v], _pclabel[v], _vweight[v]);

        if (_deg_corr)
        {
            for (auto v : vertices_range(_g))
                _deg_stats.add_vertex(_degs[v], _b[v], _vweight[v]);
            init_q_cache(std::min(std::max(2 * int(_E), 100), 10000));
        }

        init_safelog_fast(2 * count_t(_E));
        init_lgamma_fast(2 * count_t(_E));

        _spec.post_init();
    }

    // =========================================================================
    // State modification
    // =========================================================================

    bool allow_move(group_t r, group_t nr)
    {
        if (r == _null_group || nr == _null_group)
            return true;

        if (has_coupled())
        {
            if (!visit_coupled
                    ([&](auto&& s, auto&)
                     {
                         auto& hb = s._b;
                         auto rr = hb[r];
                         auto ss = hb[nr];
                         if (rr != ss && !s.allow_move(rr, ss))
                             return false;
                         return true;
                     }))
                return false;
        }
        return _bclabel[r] == _bclabel[nr];
    }

    template <class MEntries>
    void move_vertex(vertex_t v, group_t r, group_t nr, MEntries& m_entries)
    {
        if (r == nr)
            return;

        if (!allow_move(r, nr))
            throw ValueException("cannot move vertex across clabel barriers");

        _spec.move_vertex(v, r, nr, m_entries);

        apply_delta(*this, m_entries, spec_t::propagate_delta);

        if (r != _null_group)
            BlockState::remove_partition_node(v, r);
        if (nr != _null_group)
            BlockState::add_partition_node(v, nr);
    }

    template <class MEntries>
    void move_vertex(vertex_t v, group_t nr, MEntries& m_entries)
    {
        move_vertex(v, _b[v], nr, m_entries);
    }

    void move_vertex(vertex_t v, group_t r, group_t nr)
    {
        get_move_entries(v, r, nr, _m_entries);
        move_vertex(v, r, nr, _m_entries);
    }

    void move_vertex(vertex_t v, group_t nr)
    {
        auto r = _b[v];
        move_vertex(v, r, nr);
    }

    template <class Vs, class Nrs>
    void move_vertices(Vs&& vs, Nrs&& nrs)
    {
        for (auto&& [v, nr] : std::views::zip(std::forward<Vs>(vs),
                                              std::forward<Nrs>(nrs)))
            move_vertex(v, nr);
    }

    void move_vertices(python::object ovs, python::object ors, bool parallel)
    {
        if constexpr (std::is_scalar_v<vertex_t>)
        {
            multi_array_ref<vertex_t, 1> vs = get_array<vertex_t, 1>(ovs);
            multi_array_ref<group_t, 1> rs = get_array<group_t, 1>(ors);

            if (vs.size() != rs.size())
                throw ValueException("vertex and group lists do not have the same size");

            if (parallel)
                move_vertices_parallel(make_random_access_adaptor_range(vs),
                                       make_random_access_adaptor_range(rs));
            else
                move_vertices(vs, rs);
        }
    }

    enum class lock_t { unique, shared };

    template <lock_t kind, lock_args largs = {}>
    decltype(auto) do_lock(group_t r, group_t s, auto&& dispatch)
    {
        if constexpr (kind == lock_t::unique)
            return do_ulock_pair<largs>(dispatch, r, s, *_group_mutex[r], *_group_mutex[s]);
        else
            return do_slock_pair<largs>(dispatch, r, s, *_group_mutex[r], *_group_mutex[s]);
    }

    template <bool clear_m_entries, class Vs, class NRs>
    void move_vertices_parallel_dispatch(Vs&& vs, NRs&& nrs)
    {
        static_assert(std::ranges::random_access_range<Vs>);
        static_assert(std::ranges::random_access_range<NRs>);

        bool has_egroups = bool(_egroups);
        clear_egroups();
        _vset.reserve(num_vertices(_g), _null_group);
        _groups_in.resize(num_vertices(_bg));
        _groups_out.resize(num_vertices(_bg));
        _group_delta.reserve(num_vertices(_bg));
        _deltas.resize(num_vertices(_bg));

        for (const auto& [v, nr] : std::views::zip(vs, nrs))
        {
            auto r = _b[v];

            if (nr == r)
                continue;

            if (size_t(nr) >= num_vertices(_bg))
                add_group(1 + nr - num_vertices(_bg));

            _group_delta[r] -= _vweight[v];
            _group_delta[nr] += _vweight[v];

            _rset.insert(r);
            _rset.insert(nr);

            _vset[v] = nr;

            _groups_out[r].push_back(v);
            _groups_in[nr].push_back(v);
        }

        set_concurrent(true);

        auto lguard =
            boost::scope::make_scope_exit
            ([&]()
             {
                 set_concurrent(false);
             });

        auto nb = make_adhoc_prop<vertex_t>
            ([&](auto v)
             {
                 auto r = _vset[v];
                 if (r == _null_group)
                     return _b[v];
                 return r;
             });

        auto insert_delta =
            [&](auto& state, auto& m_entries, auto& deltas)
            {
                typedef std::decay_t<decltype(state)> cstate_t;

                entries_op(m_entries,
                           [&](auto r, auto s, const auto& delta)
                           {
                               if (delta == 0)
                                   return;

                               if (!is_directed_v<typename cstate_t::g_t> && r > s)
                                   std::swap(r, s);

                               auto& delta_r = deltas[r];

                               do_slock<{.pass_lock=true}>
                                   ([&](auto& lock)
                                    {
                                        auto iter = delta_r.find(s);
                                        if (iter == delta_r.end())
                                        {
                                            upgrade_canary_lock<{.ula={.relock=false}}>
                                                ([&](bool succ)
                                                 {
                                                     if (!succ) [[unlikely]]  // lost race
                                                         iter = delta_r.find(s);
                                                     if (iter == delta_r.end())
                                                     {
                                                         delta_r[s] = delta;
                                                         _spec.set_concurrent_delta(delta_r[s], true);
                                                     }
                                                     else
                                                     {
                                                         iter->second += delta;
                                                     }
                                                 }, lock);
                                        }
                                        else
                                        {
                                            atomic_inc(iter->second, delta);
                                        }
                                    }, *state._group_mutex[r]);
                           });
            };

        _m_entries_pool.resize(get_num_threads());

        parallel_loop_tnum
            (std::views::zip(vs, nrs),
             [&](auto, const auto& vnr, auto tid)
             {
                 const auto& [v, nr] = vnr;
                 auto r = _b[v];

                 if (nr == r)
                     return;

                 auto& m_entries = _m_entries_pool[tid];

                 modify_entries
                     (v, r, nr, _b, nb, _g, _eweight, num_vertices(_bg), m_entries,
                      [&](auto& e, bool out)
                      {
                          if constexpr (is_directed_v<g_t>)
                          {
                              auto u = (out) ? target(e, _g) : source(e, _g);
                              return (!out && (_vset[u] != _null_group));
                          }
                          else
                          {
                              auto u = target(e, _g);
                              return ((v > u) && (_vset[u] != _null_group));
                          }
                      });

                 insert_delta(*this, m_entries, _deltas);
                 _spec.move_vertex_concurrent(v, r, nr, nb,
                                              [&](auto u){ return (v > u) && (_vset[u] != _null_group); });
             });

        _spec.move_vertices_parallel(vs, nrs);

        auto dispatch_coupled =
            [&]<class cstate_t>(this auto&& self, cstate_t& state) -> void
            {
                state._m_entries_pool.resize(get_num_threads());

                state.visit_coupled_if
                    ([&](auto& s, auto&)
                     {
                         s._m_entries_pool.resize(get_num_threads());

                         for (auto&& [i, mes] : enumerate(state._m_entries_pool))
                             mes.set_next(s._m_entries_pool[i]);

                         s._deltas.resize(num_vertices(s._bg));
                     });

                auto clear =
                    [&](bool coupled)
                    {
                        if constexpr (clear_m_entries)
                        {
                            state._m_entries_pool.clear();
                            state._m_entries_pool.shrink_to_fit();
                        }
                        state._deltas.clear();
                        state._deltas.shrink_to_fit();
                        if (coupled)
                        {
                            state.visit_coupled_if
                                ([&](auto& s, auto&)
                                 {
                                     if constexpr (clear_m_entries)
                                     {
                                         s._m_entries_pool.clear();
                                         s._m_entries_pool.shrink_to_fit();
                                     }
                                     s._deltas.clear();
                                     s._deltas.shrink_to_fit();
                                 });
                        }
                    };

                auto lguard_ = boost::scope::make_scope_exit([&]() { clear(true); });

                bool stop = true;

                auto dispatch_delta =
                    [&](auto&& f)
                    {
                        parallel_vertex_loop_tnum
                            (state._bg,
                             [&](auto r, auto tid)
                             {
                                 if (state._deltas[r].empty())
                                     return;

                                 auto& m_entries = state._m_entries_pool[tid];

                                 m_entries.set_move(r, state._null_group, num_vertices(state._bg));

                                 for (auto&& [s, delta] : state._deltas[r])
                                 {
                                     if (delta == 0)
                                         continue;
                                     m_entries.template insert_delta_rnr<true, true, true>(r, s, delta);
                                 }

                                 if (m_entries.empty())
                                      return;

                                 stop = false;

                                 f(m_entries);
                             });
                    };

                if constexpr (!std::is_same_v<decltype(state._spec.template apply_delta<true>(state._m_entries_pool[0], state._deltas)), bool>)
                {
                    dispatch_delta([&](auto& m_entries)
                                   {
                                       state._spec.template apply_delta<true>(m_entries, state._deltas);
                                   });
                }

                dispatch_delta([&](auto& m_entries)
                               {
                                   if constexpr (cstate_t::spec_t::propagate_delta)
                                   {
                                       state.visit_coupled_if
                                           ([&](auto& s, auto&)
                                            {
                                                s.propagate_entries(state, m_entries, false);
                                                insert_delta(s, *m_entries._next, s._deltas);
                                            });
                                   }
                                   apply_delta<true>(state, m_entries, false);
                               });

                 if (stop || !state.has_coupled() || !cstate_t::spec_t::propagate_delta)
                     return;

                 clear(false);
                 state.visit_coupled
                     ([&](auto& s, auto&)
                      {
                          self(s);
                      });
            };

        dispatch_coupled(*this);

        for (auto r : _rset)
        {
            const auto& d = _group_delta[r];
            if (_groups[r].empty())
                occupy_group(r, d);
            for (auto v : _groups_out[r])
                _groups[r].erase(v);
            if (d == 0)
                continue;
            if (d > 0)
                _ps.add_item(r, _bpclabel[r], d);
            else
                _ps.remove_item(r, _bpclabel[r], -d);
            if (!_groups[r].empty())
                _spec.update_group_weight(r, d);
            _wr[r] += d;
        }

        for (auto r : _rset)
        {
            for (auto v : _groups_in[r])
                _groups[r].insert(v);
            if (_groups[r].empty())
                vacate_group(r);
            _group_delta[r] = 0;
        }

        if (_deg_corr)
        {
            parallel_loop
                (_rset,
                 [&](auto, auto r)
                 {
                     for (auto v : _groups_out[r])
                         _deg_stats.remove_vertex(_degs[v], r, _vweight[v]);
                     for (auto v : _groups_in[r])
                         _deg_stats.add_vertex(_degs[v], r, _vweight[v]);
                 }, 0);
        }

        parallel_loop(std::views::zip(vs, nrs),
                      [&](auto&&, const auto& vnr)
                      {
                          const auto& [v, nr] = vnr;
                          _b[v] = nr;
                          _vset[v] = _null_group;
                      }, 0);

        _rset.clear();
        _groups_in.clear();
        _groups_out.clear();
        _deltas.clear();
        _vset.shrink_to_fit();
        _rset.shrink_to_fit();
        _groups_in.shrink_to_fit();
        _groups_out.shrink_to_fit();
        _deltas.shrink_to_fit();
        if constexpr (clear_m_entries)
        {
            _m_entries_pool.clear();
            _m_entries_pool.shrink_to_fit();
        }
        if (has_egroups)
            init_egroups();
    }

    template <bool clear_m_entries=true>
    void move_vertices_parallel(auto&& vs, auto&& nrs)
    {
        if constexpr (std::ranges::random_access_range<std::decay_t<decltype(vs)>>)
        {
            move_vertices_parallel_dispatch<clear_m_entries>(vs, nrs);
        }
        else
        {
            std::vector<std::decay_t<decltype(*std::ranges::begin(vs))>>
                vs_(std::from_range_t(), vs);
            if constexpr (std::ranges::random_access_range<std::decay_t<decltype(nrs)>>)
            {
                move_vertices_parallel_dispatch<clear_m_entries>(vs_, nrs);
            }
            else
            {
                std::vector<std::decay_t<decltype(*std::ranges::begin(nrs))>>
                    nrs_(std::from_range_t(), nrs);
                move_vertices_parallel_dispatch<clear_m_entries>(vs_, nrs_);
            }
        }
    }

    template <class VMap>
    void set_partition(VMap& b, bool parallel)
    {
        visit_coupled
            ([&](auto&& s, auto&)
             {
                 std::decay_t<decltype(s._b)> hb;
                 if (this->has_coupled())
                     hb = s._b;

                 for (auto v : vertices_range(_g))
                 {
                     group_t r = b[v];
                     if (size_t(r) >= num_vertices(_bg))
                         add_group(r + 1 - num_vertices(_bg));
                     if (_wr[r] == 0)
                     {
                         if (this->has_coupled())
                             hb[r] = hb[_b[v]];
                         _bclabel[r] = _bclabel[_b[v]];
                         _bpclabel[r] = _bpclabel[_b[v]];
                     }
                 }
             });

        if (parallel)
        {
            move_vertices_parallel(vertices_range(_g),
                                   std::ranges::transform_view(vertices_range(_g),
                                                               [&](auto v){ return b[v]; }));
        }
        else
        {
            move_vertices(vertices_range(_g),
                          std::ranges::transform_view(vertices_range(_g),
                                                      [&](auto v){ return b[v]; }));
        }
    }

    void set_partition(std::any& ab, bool parallel)
    {
        bmap_t& b = std::any_cast<bmap_t&>(ab);
        set_partition<bmap_t>(b, parallel);
    }

    void remove_partition_node(vertex_t v, group_t r)
    {
        auto& w = _vweight[v];

        assert(w != 0);

        _ps.remove_item(r, _pclabel[v], w);

        if (w != 0 && _wr[r] == w)
            vacate_group(r);
        else
            _spec.update_group_weight(r, -w);

        _wr[r] -= w;

        if (_deg_corr)
            _deg_stats.remove_vertex(_degs[v], r, w);

        _groups[r].erase(v);
    }

    void add_partition_node(vertex_t v, group_t r)
    {
        auto& w = _vweight[v];

        assert(w != 0);

        if (w != 0 && _wr[r] == 0)
            occupy_group(r, w);
        else
            _spec.update_group_weight(r, w);

        _wr[r] += w;

        _ps.add_item(r, _pclabel[v], w);

        if (_deg_corr)
            _deg_stats.add_vertex(_degs[v], r, w);

        _groups[r].insert(v);

        _b[v] = r;
    }

    void vacate_group(group_t r)
    {
        _empty_groups.insert(r);
        _occupied_groups[_bclabel[r]].erase(r);
        _actual_B--;
        _ps.remove_item(r, _bpclabel[r], rw_t{});

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& s, auto&)
                 {
                     auto& hb = s._b;
                     s.remove_partition_node(r, hb[r]);
                     s.set_vertex_weight(r, rw_t{});
                 });
        }
    }

    void occupy_group(group_t r, const rw_t& w)
    {
        _empty_groups.erase(r);
        _occupied_groups[_bclabel[r]].insert(r);
        _actual_B++;
        _ps.add_item(r, _bpclabel[r], rw_t{});

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& s, auto&)
                 {
                     auto& hb = s._b;
                     auto rw = w;
                     rw = 1; // NOTE: setting this way is important for the layered model
                     s.set_vertex_weight(r, rw);
                     s.add_partition_node(r, hb[r]);
                 });
        }
    }

    template <class MEntries>
    void merge_groups(group_t s, group_t t, MEntries& m_entries)
    {
        if (s == t)
            return;

        _spec.merge_groups(s, t, m_entries);

        apply_delta(*this, m_entries, spec_t::propagate_delta);

        auto& ws = _wr[s];

        if (_deg_corr)
            _deg_stats.merge_groups(s, t);

        if (_wr[t] == 0)
            occupy_group(t, ws);
        else
            _spec.update_group_weight(t, ws);

        _wr[t] += ws;

        _ps.remove_item(s, _bpclabel[s], ws);
        _ps.add_item(t, _bpclabel[t], ws);

        ws = 0;
        vacate_group(s);

        std::vector<vertex_t> vs(_groups[s].begin(), _groups[s].end());
        for (auto v : vs)
        {
            _groups[s].erase(v);
            _groups[t].insert(v);
            _b[v] = t;
        }
    }

    void merge_groups(group_t s, group_t t)
    {
        get_move_entries(s, s, t, get(vertex_index_t(), _bg),
                         get(vertex_index_t(), _bg), _bg, _mrs,
                         num_vertices(_bg), _m_entries);
        merge_groups(s, t, _m_entries);
    }

    group_t add_group(size_t n = 1)
    {
        _wr.soft_reserve(num_vertices(_bg) + n);
        _mrm.soft_reserve(num_vertices(_bg) + n);
        _mrp.soft_reserve(num_vertices(_bg) + n);
        _bclabel.soft_reserve(num_vertices(_bg) + n);
        _bpclabel.soft_reserve(num_vertices(_bg) + n);
        if (_deg_corr)
            _deg_stats.add_group(n);
        group_t r = _null_group;
        for (size_t i = 0; i < n; ++i)
        {
            r = boost::add_vertex(_bg);
            _wr[r] = 0;
            _mrm[r] = _mrp[r] = 0;
            _empty_groups.insert(r);
            if (_egroups)
                _egroups->add_group();
            _group_mutex.push_back(std::make_shared<shared_canary_mutex>());
        }
        _emat.add_group(_bg);
        _spec.add_group(n);
        visit_coupled_if
            ([&](auto&& s, auto&)
             {
                 s.coupled_resize_vertices();
             });
        return r;
    }

    // =========================================================================
    // Nested states
    // =========================================================================

    bool has_coupled()
    {
        if constexpr (std::is_pointer_v<coupled_state_t>)
            return _coupled_state != nullptr;
        else
            return _coupled_state.index() != 0;
    }

    template <bool mono = false>
    decltype(auto) visit_coupled(auto&& f)
    {
        if constexpr (std::is_pointer_v<coupled_state_t>)
        {
            if constexpr (mono)
            {
                if (_coupled_state != nullptr)
                    return f(*_coupled_state, _coupled_entropy_args);
                else
                    return f(std::monostate(), _coupled_entropy_args);
            }
            else
            {
                return f(*_coupled_state, _coupled_entropy_args);
            }
        }
        else
        {
            return std::visit
                ([&](auto s)
                 {
                     if constexpr (std::is_pointer_v<decltype(s)>)
                     {
                         return std::visit
                             ([&](auto& ea)
                              {
                                  return f(*s, ea);
                              }, _coupled_entropy_args);
                     }
                     else
                     {
                         if constexpr (mono)
                         {
                             return std::visit
                                 ([&](auto&& ea)
                                  {
                                      return f(s, ea);
                                  }, _coupled_entropy_args);
                         }
                         else
                         {
                             // will result in error; needed for return type deduction
                             std::variant_alternative_t<1, coupled_state_t> st{};
                             return std::visit
                                 ([&](auto& ea)
                                  {
                                      return f(*st, ea);
                                  }, _coupled_entropy_args);
                         }
                     }
                 }, _coupled_state);
        }
    }

    template <bool mono = false>
    void visit_coupled_if(auto&& f)
    {
        visit_coupled<true>
            ([&](auto&& s, auto& ea)
             {
                 if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                                               std::monostate>)
                     f(s, ea);
             });
    }

    void couple_state(coupled_state_t s, const coupled_entropy_args_t& ea)
    {
        _coupled_state = s;
        _coupled_entropy_args = ea;
        visit_coupled
            ([&](auto& cs, auto&)
             {
                 _m_entries.set_next(cs.get_m_entries());
             });
        _spec.couple_state(s);
    }

    void decouple_state()
    {
        if constexpr (std::is_pointer_v<coupled_state_t>)
            _coupled_state = nullptr;
        else
            _coupled_state = std::monostate();
    }

    template <class State>
    size_t count_L(State& cstate)
    {
        if (cstate.has_coupled())
            return 1 + cstate.visit_coupled([&](auto& s, auto&) { return count_L(s); });
        return 1;
    }

    size_t get_L()
    {
        return count_L(*this);
    }

    void set_concurrent(bool concurrent, size_t M = 0)
    {
        if (concurrent)
        {
            M = std::max(base_graph(_g).get_edge_index_range(), M);
            base_graph(_bg).set_concurrent(true);
            base_graph(_bg).reserve_epos(2 * M + 10 * get_num_threads());
            _mrs.reserve(2 * M + 10 * get_num_threads());
        }
        else
        {
            base_graph(_bg).set_concurrent(false);
            base_graph(_bg).shrink_to_fit();
            _mrs.resize(base_graph(_bg).get_edge_index_range());
            _mrs.shrink_to_fit();
        }

        _spec.set_concurrent(concurrent, M);

        visit_coupled_if
            ([&](auto& s, auto&)
             {
                 s.set_concurrent(concurrent, M / 2);
             });
    }

    void coupled_resize_vertices()
    {
        size_t N = num_vertices(_g);
        _b.soft_reserve(N, _null_group);
        _bfield.soft_reserve(N);
        _vweight.soft_reserve(N);
        _clabel.soft_reserve(N, _null_group);
        _pclabel.soft_reserve(N, _null_group);
        _degs.resize(N);
        _spec.coupled_resize_vertices(N);
    }

    void set_vertex_weight(vertex_t v, const vw_t& w)
    {
        //assert(total_degree(v, _g, _eweight) == 0 || w != 0);
        auto r = _b[v];
        auto c = _bclabel[r];
        _N[c] -= _vweight[v];
        _N[c] += w;
        _vweight[v] = w;
        _spec.set_vertex_weight(v, w);
    }

    // =========================================================================
    // Virtual state modification
    // =========================================================================

    void get_move_entries(auto v, group_t r, group_t nr, auto&& b, auto&& nb,
                          auto& g, auto&& eweight, size_t B, auto& m_entries,
                          auto&& efilt)
    {
        modify_entries(v, r, nr, b, nb, g, eweight, B, m_entries, efilt);

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& s, auto&)
                 {
                     s.propagate_entries(*this, m_entries);
                 });
        }
    }

    void get_move_entries(auto v, group_t r, group_t nr, auto&& b, auto&& nb,
                          auto& g, auto&& eweight, size_t B, auto& m_entries)
    {
        get_move_entries(v, r, nr, b, nb, g, eweight, B, m_entries,
                         [](auto&&...) { return false; });
    }

    template <class MEntries, class Efilt>
    void get_move_entries(vertex_t v, group_t r, group_t nr, MEntries& m_entries,
                          Efilt&& efilt)
    {
        get_move_entries(v, r, nr, _b, _b, _g, _eweight, num_vertices(_bg),
                         m_entries, efilt);
    }

    template <class MEntries>
    void get_move_entries(vertex_t v, group_t r, group_t nr, MEntries& m_entries)
    {
        get_move_entries(v, r, nr, m_entries, [](auto&&...) { return false; });
    }

    void propagate_entries(auto& below, auto& m_entries_below, bool propagate=true)
    {
        auto& m_entries = *m_entries_below._next;

        auto [u, v] = m_entries_below.get_move();
        group_t r = (u != _null_group) ? _b[u] : _null_group;
        group_t s = (v != _null_group) ? _b[v] : _null_group;

        m_entries.set_move(r, s, num_vertices(_bg));

        entries_op(m_entries_below,
                   [&](auto t, auto w, auto delta)
                   {
                       _spec.propagate_delta(below, t, w, delta);
                       if (delta == 0)
                           return;
                       m_entries.template insert_delta<true>
                           (_b[t], _b[w], delta);
                   });

        if (propagate)
        {
            visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     cs.propagate_entries(*this, m_entries);
                 });
        }
    }

    // obtain the entropy difference given a set of entries in the e_rs matrix
    double entries_dS(group_t r, group_t nr, const rw_t& dr, const rw_t& dnr,
                      const auto& dkin, const auto& dkout,
                      const entropy_args_t& ea, auto& m_entries)
    {
        double dS = 0;

        if (ea.adjacency)
        {
            if (ea.dense)
            {
                if (_deg_corr)
                    throw GraphException("Dense entropy for degree corrected model not implemented!");

                auto get_dS =
                    [&](group_t t, group_t u, const auto& me, const auto& d)
                    {
                        weight_t etu{};
                        if (me != _emat.get_null_edge())
                            etu = _mrs[me];

                        double ddS = 0;

                        auto wt = _wr[t];
                        auto wu = _wr[u];

                        ddS -= eterm_dense(t, u, etu, wt, wu, ea.multigraph, _bg);

                        if (t == r)
                            wt += dr;
                        if (t == nr)
                            wt += dnr;

                        if (u == r)
                            wu += dr;
                        if (u == nr)
                            wu += dnr;

                        ddS += eterm_dense(t, u, etu + d, wt, wu, ea.multigraph, _bg);

                        assert(std::isfinite(ddS));
                        return ddS;
                    };

                entries_op(m_entries,
                           [&](auto t, auto u, const auto& delta)
                           {
                               if (delta == 0)
                                   return;
                               auto me = _emat.get_me(t, u);
                               dS += get_dS(t, u, me, delta);
                           });

                for (auto t : std::array<group_t, 2>({r, nr}))
                {
                    if (t == _null_group)
                        continue;
                    if ((t == r && dr != 0) || (t == nr && dnr != 0))
                    {
                        for (auto me : out_edges_range(t, _bg))
                        {
                            group_t u = target(me, _bg);
                            if (u == r || u == nr || m_entries.get_delta(t, u) != 0)
                                continue;
                            dS += get_dS(t, u, me, weight_t{});
                            assert(std::isfinite(dS));
                        }

                        for (auto me : in_edges_range(t, _bg))
                        {
                            group_t u = source(me, _bg);
                            if (u == r || u == nr || m_entries.get_delta(u, t) != 0)
                                continue;
                            dS += get_dS(u, t, me, weight_t{});
                            assert(std::isfinite(dS));
                        }
                    }

                    if (r != _null_group && m_entries.get_delta(t, r) == 0)
                        dS += get_dS(t, r, _emat.get_me(t, r), weight_t{});

                    assert(std::isfinite(dS));

                    if (r == nr)
                        break;

                    if (nr != _null_group && (is_directed_v<bg_t> || t != r) &&
                        m_entries.get_delta(t, nr) == 0)
                        dS += get_dS(t, nr, _emat.get_me(t, nr), weight_t{});

                    assert(std::isfinite(dS));
                };
            }
            else
            {
                entries_op(m_entries,
                           [&](auto t, auto w, const auto& d)
                           {
                               auto me = _emat.get_me(t, w);
                               weight_t ers{};
                               if (me != _emat.get_null_edge())
                                   ers = _mrs[me];
                               dS += eterm(t, w, ers + d, _bg) - eterm(t, w, ers, _bg);
                           });

                auto vt = [&](auto&& mrp_, auto&& mrm_, auto&& nr_)
                          {
                              return vterm(mrp_, mrm_, nr_, _deg_corr, _bg);
                          };

                if (r != _null_group)
                {
                    auto&& mrp_r = _mrp[r];
                    auto&& mrm_r = _mrm[r];
                    auto&& wr_r = _wr[r];
                    dS += vt(mrp_r - dkout, mrm_r - dkin, wr_r + dr);
                    dS -= vt(mrp_r        , mrm_r       , wr_r     );
                }

                if (nr != _null_group)
                {
                    auto&& mrp_nr = _mrp[nr];
                    auto&& mrm_nr = _mrm[nr];
                    auto&& wr_nr = _wr[nr];
                    dS += vt(mrp_nr + dkout, mrm_nr + dkin, wr_nr + dnr);
                    dS -= vt(mrp_nr        , mrm_nr       , wr_nr      );
                }
            }
        }

        dS += _spec.entries_dS(ea, m_entries);

        double dS_dl = 0;

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& cs, auto& cea)
                 {
                    using crw_t = typename std::decay_t<decltype(cs)>::rw_t;
                    crw_t ddr{}, ddnr{};

                    if (r != nr)
                    {
                        if (r != _null_group)
                        {
                            crw_t wr_r = _wr[r];
                            crw_t n_wr_r = _wr[r] + dr;
                            ddr = -thresh_diff(n_wr_r, wr_r);
                        }
                        if (nr != _null_group)
                        {
                            crw_t wr_nr = _wr[nr];
                            crw_t n_wr_nr = _wr[nr] + dnr;
                            ddnr = thresh_diff(wr_nr, n_wr_nr);
                        }
                    }
                    else
                    {
                        ddr = ddnr = 0;
                    }

                    if (!m_entries._next->empty() || ddr != 0 || ddnr != 0)
                        dS_dl += cs.propagate_entries_dS(r, nr, ddr, ddnr,
                                                         *m_entries._next,
                                                         cea);
                 });
        }

        return dS + ea.beta_dl * dS_dl;
    }

    double propagate_entries_dS(group_t u, group_t v, const rw_t& du,
                                const rw_t& dv, auto& m_entries,
                                const entropy_args_t& ea)
    {
        group_t r = (u != _null_group) ? _b[u] : _null_group;
        group_t s = (v != _null_group) ? _b[v] : _null_group;

        typedef std::decay_t<decltype(_mrm[r])> mr_t;
        double dS = entries_dS(r, s, du, dv,
                               (r != _null_group) ? _mrm[r] : mr_t{},
                               (s != _null_group) ? _mrp[s] : mr_t{},
                               ea, m_entries);

        if (spec_t::edges_dl && ea.edges_dl && r != s)
        {
            count_t dr = 0, ds = 0;
            if (r != _null_group)
                dr = (_wr[r] + du == 0 && _wr[r] != 0) ? -1 : 0;
            if (r != _null_group)
                ds = (_wr[s] == 0 && dv != 0) ? 1 : 0;
            dS -= get_edges_dl(_actual_B, _E, _g);
            dS += get_edges_dl(_actual_B + (dr + ds), _E, _g);
        }

        assert(std::isfinite(dS));
        return dS;
    }

    template <bool merge=false, class MEntries>
    double virtual_move_groups(group_t s, group_t t, const entropy_args_t& ea,
                               MEntries& m_entries, auto&& deg, auto&& n,
                               vertex_t v = vertex_t{})
    {
        if constexpr (!merge)
            get_move_entries(v, s, t, m_entries);
        else
            get_move_entries(s, s, t, get(vertex_index_t(), _bg),
                             get(vertex_index_t(), _bg), _bg,
                             _mrs, num_vertices(_bg), m_entries);

        if (s == t || n == 0)
            return 0;

        assert(s == _null_group || _wr[s] != 0);

        if (!allow_move(s, t))
            return std::numeric_limits<double>::infinity();

        double dS = entries_dS(s, t, -n, n, get<0>(deg), get<1>(deg), ea, m_entries);

        double dS_dl = get_delta_partition_dl<merge>(v, s, t, ea, n);

        if (_deg_corr && ea.degree_dl)
        {
            if constexpr (!merge)
                dS_dl += _deg_stats.get_delta_dl(deg, s, t, ea.degree_dl_kind, n);
            else
                dS_dl += _deg_stats.get_merge_dl(ea.degree_dl_kind, s, t);
        }

        count_t dB = 0;
        if (s != _null_group && _wr[s] == n)
            dB--;
        if (t != _null_group && _wr[t] == 0)
            dB++;

        if (spec_t::edges_dl && ea.edges_dl && dB != 0)
        {
            dS_dl -= get_edges_dl(_actual_B, _E, _g);
            dS_dl += get_edges_dl(_actual_B + dB, _E, _g);
        }

        if (!_Bfield.empty() && ea.Bfield && dB != 0)
        {
            size_t B = _actual_B;
            dS_dl += (B < _Bfield.size()) ? _Bfield[B] : _Bfield.back();
            B += dB;
            dS_dl -= (B < _Bfield.size()) ? _Bfield[B] : _Bfield.back();
        }

        if (!merge)
            dS += _spec.virtual_move(v, s, t, ea, m_entries);
        else
            dS += _spec.virtual_merge(s, t, ea, m_entries);

        return dS + ea.beta_dl * dS_dl;
    }


    template <class MEntries>
    double virtual_move(vertex_t v, group_t r, group_t nr,
                        const entropy_args_t& ea, MEntries& m_entries)
    {
        return virtual_move_groups<false>(r, nr, ea, m_entries, _degs[v],
                                          _vweight[v], v);
    }

    template <class MEntries>
    double virtual_move(vertex_t v, group_t nr, const entropy_args_t& ea,
                        MEntries& m_entries)
    {
        return virtual_move(v, _b[v], nr, ea, m_entries);
    }

    double virtual_move(vertex_t v, group_t r, group_t nr, const entropy_args_t& ea)
    {
        return virtual_move(v, r, nr, ea, _m_entries);
    }

    double virtual_move(vertex_t v, group_t nr, const entropy_args_t& ea)
    {
        return virtual_move(v, nr, ea, _m_entries);
    }

    template <bool merge=false>
    double get_delta_partition_dl(vertex_t v, group_t r, group_t nr,
                                  const entropy_args_t& ea, vw_t vw = vw_t{})
    {
        if (r == nr)
            return 0;

        double dS = 0;

        if constexpr (!merge)
        {
            vw = _vweight[v];
            auto& f = _bfield[v];
            if (!f.empty())
            {
                if (nr != _null_group)
                    dS -= (size_t(nr) < f.size()) ? f[nr] : f.back();
                if (r != _null_group)
                    dS += (size_t(r) < f.size()) ? f[r] : f.back();
            }
        }

        if (ea.partition_dl)
        {
            if (r == _null_group)
            {
                if constexpr (!merge)
                    dS += _ps.get_delta_dl(r, nr, _bpclabel[nr]);
                else
                    dS += _ps.get_delta_dl(r, nr, _bpclabel[nr], vw);
            }
            else
            {
                dS += _ps.get_delta_dl(r, nr, _bpclabel[r], vw);
            }
        }

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto& cs, auto& cea)
                 {
                     bool r_vacate = (r != _null_group && _wr[r] == vw);
                     bool nr_occupy = (nr != _null_group && _wr[nr] == 0);

                     auto& bh = cs._b;
                     if (r_vacate && nr_occupy)
                     {
                         dS += cs.get_delta_partition_dl(r, bh[r], bh[nr], cea);
                     }
                     else
                     {
                         if (r_vacate)
                             dS += cs.get_delta_partition_dl(r, bh[r],
                                                             _null_group, cea);
                         if (nr_occupy)
                             dS += cs.get_delta_partition_dl(nr, _null_group,
                                                             bh[nr], cea);
                     }
                 });
        }
        return dS;
    }

    template <class MEntries>
    double virtual_merge(group_t s, group_t t, const entropy_args_t& ea,
                         MEntries& m_entries)
    {
        return virtual_move_groups<true>(s, t, ea, m_entries,
                                         std::forward_as_tuple(_mrm[s], _mrp[s]),
                                         _wr[s]);
    }

    double virtual_merge(group_t s, group_t t, const entropy_args_t& ea)
    {
        return virtual_merge(s, t, ea, _m_entries);
    }

    size_t virtual_remove_size(vertex_t v)
    {
        return _wr[_b[v]] - _vweight[v];
    }

    // =========================================================================
    // Move proposals
    // =========================================================================

    template <class RNG>
    group_t get_empty_group(vertex_t v, RNG& rng, bool inherit = true)
    {
        group_t s;
        if (_empty_groups.empty())
            s = add_group();
        else
            s = uniform_sample(_empty_groups, rng);

        if (inherit)
        {
            auto r = _b[v];
            _bclabel[s] = _bclabel[r];
            _bpclabel[s] = _bpclabel[r];
            assert(_clabel[v] == _bclabel[r]);

            visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     auto& hb = cs._b;
                     auto& hpclabel = cs._pclabel;
                     auto& hclabel = cs._clabel;
                     hb[s] = hb[r];
                     hpclabel[s] = hclabel[s] = _pclabel[v];
                 });
        }

        return s;
    }

    template <class RNG>
    void sample_branch(vertex_t v, vertex_t u, RNG& rng)
    {
        group_t s;
        auto r = _b[u];
        auto& occupied_groups = _occupied_groups[_bclabel[r]];
        std::bernoulli_distribution new_r(1./(occupied_groups.size()+1));
        if (occupied_groups.size() < _N[_bclabel[r]] && new_r(rng))
        {
            s = get_empty_group(u, rng);
            visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     cs.sample_branch(s, r, rng);
                 });
        }
        else
        {
            s = uniform_sample(occupied_groups, rng);
        }
        _b[v] = s;
    }

    // Sample node placement
    template <class RNG>
    group_t sample_group(vertex_t v, double c, double d, RNG& rng,
                         bool egroups = true)
    {
        auto r = _b[v];

        size_t B = _occupied_groups[_bclabel[r]].size();

        // attempt new group
        std::bernoulli_distribution new_r(d);
        if (d > 0 && B < _N[_bclabel[r]] && new_r(rng))
        {
            auto s = get_empty_group(v, rng);
            if constexpr (spec_t::has_relabel)
                _spec.relabel_group(s, _spec.sample_group_label(s, rng));
            return s;
        }

        group_t s;
        if (!std::isinf(c) && total_degree(v, _g) > 0) // FIXME: use weight?
        {
            auto u = graph_tool::random_neighbor(v, _g, rng);
            group_t t = _b[u];
            double p_rand = 0;
            if (c > 0)
            {
                if (is_directed_v<g_t>)
                    p_rand = c * B / double(int64_t(_mrp[t]) + int64_t(_mrm[t]) + c * B);
                else
                    p_rand = c * B / double(int64_t(_mrp[t]) + c * B);
            }

            std::bernoulli_distribution rand(p_rand);
            if (c == 0 || !rand(rng))
            {
                if (egroups)
                {
                    if (!_egroups)
                        init_egroups();
                    s = _egroups->sample_edge(t, rng);
                }
                else
                {
                    u = uniform_sample(_groups[t], rng);
                    if (total_degree(u, _g) > 0)
                        u = graph_tool::random_neighbor(u, _g, rng);
                    s = _b[u];
                }
            }
            else
            {
                s = uniform_sample(_occupied_groups[_bclabel[r]], rng);
            }
        }
        else
        {
            s = uniform_sample(_occupied_groups[_bclabel[r]], rng);
        }

        return s;
    }

    template <class RNG>
    vertex_t random_neighbor(vertex_t v, RNG& rng)
    {
        if (count_t(total_degree(v, _g)) == 0)
            return v;
        return graph_tool::random_neighbor(v, _g, rng);
    }

    // Computes the move proposal probability
    template <class MEntries>
    double get_move_prob(vertex_t v, group_t r, group_t s, double c, double d,
                         bool reverse, MEntries& m_entries)
    {
        size_t B = _occupied_groups[_bclabel[r]].size();

        if (r == s)
            reverse = false;

        if (reverse)
        {
            if (_wr[s] == _vweight[v])
                return log(d);

            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return log(d);
        }

        if (B == _N[_bclabel[r]])
            d = 0;

        if (std::isinf(c))
            return log(1. - d) - safelog_fast(B);

        double p = 0;
        size_t w = 0;

        size_t kin, kout;
        std::tie(kin, kout) = _degs[v];

        m_entries.get_mes(_emat);

        auto sum_prob = [&](auto&& range)
            {
                for (auto u : range)
                {
                    group_t t = _b[u];
                    if (u == v)
                        t = r;
                    w++;

                    count_t mts = 0;
                    const auto& me = m_entries.get_me(t, s, _emat);
                    if (me != _emat.get_null_edge())
                        mts = _mrs[me];
                    count_t mtp = _mrp[t];
                    count_t mst = mts;
                    count_t mtm = mtp;

                    if constexpr (is_directed_v<g_t>)
                    {
                        mst = 0;
                        const auto& me_ = m_entries.get_me(s, t, _emat);
                        if (me_ != _emat.get_null_edge())
                            mst = _mrs[me_];
                        mtm = _mrm[t];
                    }

                    if (reverse)
                    {
                        count_t dts = m_entries.get_delta(t, s);
                        count_t dst = dts;
                        if constexpr (is_directed_v<g_t>)
                            dst = m_entries.get_delta(s, t);

                        mts += dts;
                        mst += dst;

                        if (t == s)
                        {
                            mtp -= kout;
                            mtm -= kin;
                        }

                        if (t == r)
                        {
                            mtp += kout;
                            mtm += kin;
                        }
                    }

                    if constexpr (is_directed_v<g_t>)
                    {
                        auto T = (mtp + mtm + c * B);
                        if (T > 0)
                            p += (mts + mst + c) / T;
                    }
                    else
                    {
                        if (t == s)
                            mts *= 2;
                        auto T = (mtp + c * B);
                        if (T > 0)
                            p += (mts + c) / T;
                    }
                }
            };

        sum_prob(_spec.random_neighbors_range(v, _g));

        if (w > 0)
            return log(1. - d) + log(p) - log(w);
        else
            return log(1. - d) - safelog_fast(B);
    }

    double get_move_prob(vertex_t v, group_t r, group_t s, double c, double d,
                         bool reverse)
    {
        get_move_entries(v, _b[v], (reverse) ? r : s, _m_entries);
        return get_move_prob(v, r, s, c, d, reverse, _m_entries);
    }

    bool is_last(vertex_t v)
    {
        return bool(_vweight[v] != 0) && bool(_wr[_b[v]] == _vweight[v]);
    }

    auto node_weight(vertex_t v)
    {
        return _vweight[v];
    }

    // =========================================================================
    // Entropy computation
    // =========================================================================

    double entropy(const entropy_args_t& ea)
    {
        double S = 0, S_dl = 0;

        if (ea.adjacency)
        {
            if (!ea.dense)
                S = sparse_entropy(_g, _eweight, _degs, _bg, _mrs, _mrp, _mrm,
                                   _wr, _deg_corr, ea);
            else
                S = dense_entropy(_bg, _mrs, _wr, _deg_corr, ea);

            assert(!isnan(S) && !isnan(S_dl));
        }

        if (ea.partition_dl)
            S_dl += _ps.get_dl();

        assert(!isnan(S) && !isnan(S_dl));

        if (_deg_corr && ea.degree_dl)
            S_dl += _deg_stats.get_dl(ea.degree_dl_kind);

        assert(!isnan(S) && !isnan(S_dl));

        if (spec_t::edges_dl && ea.edges_dl)
            S_dl += get_edges_dl(_actual_B, _E, _g);

        assert(!isnan(S) && !isnan(S_dl));

        #pragma omp parallel reduction(+:S_dl)
        parallel_vertex_loop_no_spawn
            (_g,
             [&](auto v)
             {
                 auto& f = _bfield[v];
                 if (f.empty())
                     return;
                 size_t r = _b[v];
                 S_dl -= (r < f.size()) ? f[r] : f.back();
             });

        assert(!isnan(S) && !isnan(S_dl));

        if (!_Bfield.empty() && ea.Bfield)
        {
            S_dl -= (_actual_B < _Bfield.size()) ?
                _Bfield[_actual_B] : _Bfield.back();
        }
        assert(!isnan(S) && !isnan(S_dl));

        S += _spec.entropy(ea);

        if (ea.propagate && ea.beta_dl != 0)
        {
            visit_coupled_if
                ([&](auto&& s, auto& cea)
                 {
                     S_dl += s.entropy(cea);
                 });
        }

        assert(!isnan(S) && !isnan(S_dl));

        return S + S_dl * ea.beta_dl;
    }

    // =========================================================================
    // Edge modification
    // =========================================================================

    template <bool Add, bool Deplete=true>
    void modify_edge(vertex_t u, vertex_t v, GraphInterface::edge_t& e, const weight_t& dm)
    {
        if (dm == weight_t{})
            return;

        group_t r = _b[u];
        group_t s = _b[v];

        if (_deg_corr)
        {
            _deg_stats.remove_vertex(_degs[u], r, _vweight[u]);
            if (u != v)
                _deg_stats.remove_vertex(_degs[v], s, _vweight[v]);
        }

        auto me = _emat.get_me(r, s);
        if constexpr (Add)
        {
            if (me == _emat.get_null_edge())
            {
                me = boost::add_edge(size_t(r), size_t(s), _bg).first;
                _emat.put_me(r, s, me);
                _mrs.get_checked()[me] = weight_t{};
            }

            if (!has_coupled() || !spec_t::propagate_delta)
                _mrs[me] += dm;

            _mrp[r] += dm;
            _mrm[s] += dm;
        }
        else
        {
            assert(me != _emat.get_null_edge());
            if (!has_coupled() || !spec_t::propagate_delta)
            {
                _mrs[me] -= dm;
                if (int64_t(_mrs[me]) == 0)
                {
                    _emat.remove_me(me, _bg);
                    boost::remove_edge(me, _bg);
                }
            }
            else
            {
                weight_t mrs = _mrs[me];
                if (int64_t(mrs - dm) == 0)
                    _emat.remove_me(me, _bg);
            }
            _mrp[r] -= dm;
            _mrm[s] -= dm;
        }

        if constexpr (Add)
        {
            if (e == GraphInterface::edge_t())
            {
                e = boost::add_edge(u, v, _g).first;
                _eweight.get_checked()[e] = dm;
            }
            else
            {
                _eweight[e] += dm;
            }

            get<1>(_degs[u]) += dm;
            if constexpr (is_directed_v<g_t>)
                get<0>(_degs[v]) += dm;
            else
                get<1>(_degs[v]) += dm;

            _E += dm;
        }
        else
        {
            _eweight[e] -= dm;
            if (int64_t(_eweight[e]) == 0 && Deplete)
            {
                boost::remove_edge(e, _g);
                e = GraphInterface::edge_t();
            }

            get<1>(_degs[u]) -= dm;
            if constexpr (is_directed_v<g_t>)
                get<0>(_degs[v]) -= dm;
            else
                get<1>(_degs[v]) -= dm;

            _E -= dm;
        }

        if (_deg_corr)
        {
            _deg_stats.add_vertex(_degs[u], r, _vweight[u]);
            if (u != v)
                _deg_stats.add_vertex(_degs[v], s, _vweight[v]);
        }

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     cs.template modify_edge<Add, Deplete>(r, s, me, dm);
                 });
        }

        _spec.template modify_edge<Add, Deplete>(u, v, e, dm);

        clear_egroups();
    }


    double modify_edge_dS(vertex_t u, vertex_t v, const GraphInterface::edge_t& e,
                          const weight_t& dm, const entropy_args_t& ea)
    {
        if (dm == 0)
            return 0;

        double S = 0, S_dl = 0;
        group_t r = _b[u];
        group_t s = _b[v];

        if (ea.degree_dl && _deg_corr)
        {
            deg_t null = get<0>(_degs[u]);
            null = numeric_limits<int64_t>::max(); // FIXME: check if valid for layered case

            if (r != s || u == v)
            {
                std::array<std::tuple<deg_t, count_t>, 2> kins, kouts;

                auto [kin, kout] = _degs[u];

                if constexpr (is_directed_v<g_t>)
                    kins = {make_tuple(kin, 0), make_tuple(null, 0)};
                kouts = {make_tuple(kout, 0), make_tuple(null, 0)};

                if (u != v)
                {
                    if constexpr (is_directed_v<g_t>)
                        kins[0] = {null, 0};
                    kouts[1] = {deg_t(kout + dm), 0};
                }
                else
                {
                    if constexpr (!is_directed_v<g_t>)
                    {
                        kouts[1] = {kout + 2 * dm, 0};
                    }
                    else
                    {
                        kins[1] = {deg_t(kin + dm), 0};
                        kouts[1] = {deg_t(kout + dm), 0};
                    }
                }

                auto& dsu = _deg_stats;

                S_dl -= dsu.get_dl(ea.degree_dl_kind,
                                   std::array<group_t,1>({r}),
                                   kins, kouts);

                if constexpr (is_directed_v<g_t>)
                    get<1>(kins[0]) -= 1;
                get<1>(kouts[0]) -= 1;
                if constexpr (is_directed_v<g_t>)
                    get<1>(kins[1]) += 1;
                get<1>(kouts[1]) += 1;

                S_dl += dsu.get_dl(ea.degree_dl_kind,
                                   std::array<group_t,1>({r}),
                                   kins, kouts);

                if (u != v) // r != s
                {
                    auto [ki, ko] = _degs[v];

                    if constexpr (is_directed_v<g_t>)
                        kins = {make_tuple(ki, 0), make_tuple(null, 0)};
                    kouts = {make_tuple(ko, 0), make_tuple(null, 0)};

                    if constexpr (!is_directed_v<g_t>)
                    {
                        kouts[1] = {ko + dm, 0};
                    }
                    else
                    {
                        kins[1] = {deg_t(ki + dm), 0};
                        kouts[0] = {null, 0};
                    }

                    auto& dsv = _deg_stats;

                    S_dl -= dsv.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({s}),
                                       kins, kouts);

                    if constexpr (is_directed_v<g_t>)
                        get<1>(kins[0]) -= 1;
                    get<1>(kouts[0]) -= 1;
                    if constexpr (is_directed_v<g_t>)
                        get<1>(kins[1]) += 1;
                    get<1>(kouts[1]) += 1;

                    S_dl += dsv.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({s}),
                                       kins, kouts);
                }
            }
            else // r == s && u != v
            {
                auto& dsu = _deg_stats;

                if constexpr (!is_directed_v<g_t>)
                {
                    std::array<std::tuple<deg_t, count_t>, 0> kins;
                    std::array<std::tuple<deg_t, count_t>, 4> kouts;

                    auto [ukin, ukout] = _degs[u];
                    auto [vkin, vkout] = _degs[v];

                    kouts = {make_tuple(ukout,     -1),
                             make_tuple(ukout + dm, 1),
                             make_tuple(vkout,     -1),
                             make_tuple(vkout + dm, 1)};

                    for (size_t i = 0; i < 2; ++i)
                    {
                        if (get<0>(kouts[2]) == get<0>(kouts[i]))
                        {
                            get<1>(kouts[i]) += get<1>(kouts[2]);
                            kouts[2] = {null, 0};
                        }

                        if (get<0>(kouts[3]) == get<0>(kouts[i]))
                        {
                            get<1>(kouts[i]) += get<1>(kouts[3]);
                            kouts[3] = {null, 0};
                        }
                    }

                    S_dl += dsu.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({r}),
                                       kins, kouts);

                    for (size_t i = 0; i < 4; ++i)
                        get<1>(kouts[i]) = 0;

                    S_dl -= dsu.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({r}),
                                       kins, kouts);
                }
                else
                {
                    std::array<std::tuple<deg_t, count_t>, 2> kins;
                    std::array<std::tuple<deg_t, count_t>, 2> kouts;

                    auto [kin, kout] = _degs[u];

                    kouts = {make_tuple(kout, -1), make_tuple(kout + dm, 1)};

                    std::tie(kin, kout) = _degs[v];

                    kins = {make_tuple(kin, -1), make_tuple(kin + dm, 1)};

                    S_dl += dsu.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({r}),
                                       kins, kouts);

                    for (size_t i = 0; i < 2; ++i)
                        get<1>(kins[i]) = get<1>(kouts[i]) = 0;

                    S_dl -= dsu.get_dl(ea.degree_dl_kind,
                                       std::array<group_t,1>({r}),
                                       kins, kouts);
                }
            }
        }

        auto me = _emat.get_me(r, s);
        weight_t mrs{};
        if (me != _emat.get_null_edge())
            mrs = _mrs[me];

        if (ea.adjacency)
        {
            if (ea.dense)
            {
                S -= eterm_dense(r, s, mrs, _wr[r], _wr[s], ea.multigraph, _bg);
                S += eterm_dense(r, s, mrs + dm, _wr[r], _wr[s], ea.multigraph, _bg);
            }
            else
            {
                S -= eterm(r, s, mrs, _bg);
                S += eterm(r, s, mrs + dm, _bg);
                if (s != r)
                {
                    S -= vterm(_mrp[r], _mrm[r], _wr[r], _deg_corr, _bg);
                    S -= vterm(_mrp[s], _mrm[s], _wr[s], _deg_corr, _bg);
                    if constexpr (is_directed_v<g_t>)
                    {
                        S += vterm(_mrp[r] + dm, _mrm[r],      _wr[r], _deg_corr, _bg);
                        S += vterm(_mrp[s],      _mrm[s] + dm, _wr[s], _deg_corr, _bg);
                    }
                    else
                    {
                        S += vterm(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                        S += vterm(_mrp[s] + dm, _mrm[s] + dm, _wr[s], _deg_corr, _bg);
                    }
                }
                else
                {
                    S -= vterm(_mrp[r], _mrm[r], _wr[r], _deg_corr, _bg);
                    if constexpr (is_directed_v<g_t>)
                        S += vterm(_mrp[r] + dm, _mrm[r] + dm, _wr[r], _deg_corr, _bg);
                    else
                        S += vterm(_mrp[r] + 2 * dm, _mrm[r] + 2 * dm, _wr[r], _deg_corr, _bg);
                }

                if (ea.multigraph && ea.parallel_entropy)
                {
                    weight_t m{};
                    if (e != GraphInterface::edge_t())
                        m = _eweight[e];
                    if (u == v && !is_directed_v<g_t>)
                    {
                        S -= parallel_term(u, v, 2 * m, _g);
                        S += parallel_term(u, v, 2 * (m + dm), _g);
                    }
                    else
                    {
                        S -= parallel_term(u, v, m, _g);
                        S += parallel_term(u, v, m + dm, _g);
                    }
                }

                if (_deg_corr)
                {
                    auto get_deg_ent =
                        [&](auto w, const std::array<deg_t,2>& delta = std::array<deg_t,2>{})
                        {
                            return get_deg_entropy(_degs[w], _g, delta);
                        };

                    deg_t zero_in = get<0>(_degs[v]);
                    deg_t zero_out = get<1>(_degs[v]);
                    deg_t dm_in = get<0>(_degs[v]);
                    deg_t dm_out = get<1>(_degs[v]);

                    zero_in = 0;
                    zero_out = 0;
                    dm_in = dm;
                    dm_out = dm;

                    if (u != v)
                    {
                        S -= get_deg_ent(u);
                        S += get_deg_ent(u, std::array<deg_t,2>{zero_in, dm_out});
                        S -= get_deg_ent(v);
                        if constexpr (is_directed_v<g_t>)
                            S += get_deg_ent(v, {dm_in, zero_out});
                        else
                            S += get_deg_ent(v, {zero_in, dm_out});
                    }
                    else
                    {
                        S -= get_deg_ent(u);
                        if constexpr (is_directed_v<g_t>)
                            S += get_deg_ent(u, {dm_in, dm_out});
                        else
                            S += get_deg_ent(u, {zero_in, 2 * dm_out});
                    }
                }
            }
        }

        if constexpr (spec_t::propagate_delta)
        {
            visit_coupled_if
                ([&](auto&& cs, auto& cea)
                 {
                     S_dl += cs.modify_edge_dS(r, s, me, dm, cea);
                 });
        }
        else
        {
            if (spec_t::edges_dl && ea.edges_dl)
            {
                S_dl -= get_edges_dl(_actual_B, _E, _g);
                S_dl += get_edges_dl(_actual_B, _E + dm, _g);
            }
        }

        S += _spec.modify_edge_dS(u, v, e, dm, ea);

        return S + S_dl * ea.beta_dl;
    }


    // =========================================================================
    // Initializetio and setup
    // =========================================================================

    auto& get_m_entries()
    {
        return _m_entries;
    }

    auto get_coupled_state()
    {
        return _coupled_state;
    }

    void init_egroups()
    {
        #pragma omp critical (egroups)
        {
            if (!_egroups)
                _egroups = std::make_unique<EGroups<flat_map_t::value>>(_bg, _mrs);
        }
    }

    void clear_egroups()
    {
        _egroups.reset();
    }

    void sync_emat()
    {
        _emat.sync(_bg);
    }

    size_t get_N()
    {
        size_t N = 0;
        for (auto& [k, n] : _N)
            N += n;
        return N;
    }

    size_t get_E()
    {
        return _E;
    }

    size_t get_actual_B()
    {
        return _actual_B;
    }

    bool check_edge_counts(bool emat=true, bool propagate=true)
    {
        if (!_spec.check_edge_counts())
        {
            assert(false);
            return false;
        }

        for (auto r : _empty_groups)
        {
           if (_wr[r] != 0)
           {
               std::println("empty group not empty: {} {}", r, _wr[r]);
               assert(false);
               return false;
           }
        }

        for (auto v : vertices_range(_g))
        {
            if (total_degree(v, _g, _eweight) != 0 && _vweight[v] == 0)
            {
                std::println("zero weight node has positive degree: {} {}", v, total_degree(v, _g, _eweight));
                assert(false);
                return false;
            }
        }

        gt_hash_map<std::pair<group_t, group_t>, weight_t> mrs;

        for (auto e : edges_range(_g))
        {
            group_t r = _b[source(e, _g)];
            group_t s = _b[target(e, _g)];
            if (!is_directed_v<g_t> && s < r)
                std::swap(r, s);
            mrs[std::make_pair(r, s)] += _eweight[e];
            assert(_eweight[e] != 0);
            assert(int64_t(_eweight[e]) != 0);
        }

        _emat.visit_entries([&](auto me)
                            {
                                auto r = source(me, _bg);
                                auto s = target(me, _bg);
                                if (!is_directed_v<g_t> && s < r)
                                    std::swap(r, s);
                                if (!edge(r, s, _bg).second)
                                {
                                    std::println("nonexisting edge {} {} in emat!", r, s);
                                    assert(false);
                                }
                                if (mrs[std::make_pair(r, s)] == 0)
                                {
                                    std::println("empty edge {} {} present in emat!", r, s);
                                    assert(false);
                                }
                            });

        for (auto& rs_m : mrs)
        {
            auto r = rs_m.first.first;
            auto s = rs_m.first.second;
            weight_t m_rs{};
            typename graph_traits<bg_t>::edge_descriptor me;
            if (emat)
            {
                me = _emat.get_me(r, s);
                if (me != _emat.get_null_edge())
                {
                    m_rs = _mrs[me];
                    if (_mrs[me] == 0)
                    {
                        std::println("empty edge {} {} present in emat!", r, s);
                        assert(false);
                    }
                }
            }
            else
            {
                auto ret = boost::edge(size_t(r), size_t(s), _bg);
                me = ret.first;
                if (ret.second)
                    m_rs = _mrs[me];
            }

            if (m_rs != rs_m.second)
            {
                std::println("a: wrong edge count for {} {}; is: {}, should have been: {}", r, s, m_rs, rs_m.second);
                assert(false);
                return false;
            }
        }

        for (auto me : edges_range(_bg))
        {
            auto r = source(me, _bg);
            auto s = target(me, _bg);
            assert(me == _emat.get_me(r, s));
            if (!is_directed_v<g_t> && s < r)
                std::swap(r, s);
            auto m_rs = mrs[std::make_pair(r, s)];
            if (m_rs == 0)
            {
                std::println("empty edge {} {} present in graph", r, s);
                assert(false);
                return false;
            }
            if (m_rs != _mrs[me])
            {
                std::println("b: wrong edge count for {} {}; is: {}, should have been: {}", r, s, _mrs[me], m_rs);
                assert(false);
                return false;
            }

            [[maybe_unused]] auto eS = eterm_dense(r, s, m_rs, _wr[r], _wr[s], true, _bg);
            assert(!std::isinf(eS) && !std::isnan(eS));
        }

        if (propagate)
        {
            bool ret = true;
            visit_coupled_if
                ([&](auto&& cs, auto&)
                 {
                     std::println("\nnext level\n");
                     if (!cs.check_edge_counts(false))
                     {
                         assert(false);
                         ret =  false;
                     }
                 });
            return ret;
        }
        return true;
    }

    void check_node_counts()
    {
#ifndef NDEBUG
        std::vector<rw_t> wr(num_vertices(_bg));
        std::vector<typename property_traits<mrp_t>::value_type>
            mrp(num_vertices(_bg)), mrm(num_vertices(_bg));
        for (auto v : vertices_range(_g))
        {
            auto r = _b[v];
            wr[r] += _vweight[v];
            deg_t kin = in_degree(v, _g, _eweight);
            deg_t kout = out_degree(v, _g, _eweight);
            mrp[r] += kout;
            mrm[r] += kin;
        }

        size_t B = 0;
        for (auto r : vertices_range(_bg))
        {
            assert(_wr[r] == wr[r]);
            assert(_mrp[r] == mrp[r]);
            assert(!is_directed_v<g_t> || _mrm[r] == mrm[r]);
            if (_wr[r] != 0)
                B++;
        }

        assert(B == _actual_B);

        if (has_coupled())
        {
            visit_coupled
                ([&](auto& cs, auto&&)
                 {
                     for (auto r : vertices_range(_bg))
                     {
                         assert((_wr[r] != 0) == (cs._vweight[r] != 0));
                         auto nr = _wr[r];
                         auto vw = cs._vweight[r];
                         if (vw == 0)
                             assert(nr == 0);
                         else if (nr == 0)
                             assert(vw == 0);
                         else
                         {
                             nr = 1;
                             vw = 1;
                             assert(nr == vw);
                         }
                     }

                     cs.check_node_counts();
                 });
        }
#endif
    }

    void relax_update(bool relax)
    {
        // if (_egroups)
        //     _egroups->check(_bg, _mrs);
        _egroups_update = !relax;
        if (has_coupled())
        {
            visit_coupled
                ([&](auto&& cs, auto&)
                 {
                     cs.relax_update(relax);
                 });
        }
    }

    auto& get_occupied_groups(vertex_t v)
    {
        return _occupied_groups[_bclabel[_b[v]]];
    }

    constexpr static bool _has_merge = spec_t::has_merge;
    constexpr static bool _has_parallel_move = spec_t::has_parallel_move;

//private:

    idx_map<group_t, idx_set<group_t, {.shared_pos=true}>,
            {.shared_tpos=true}> _occupied_groups;
    idx_set<group_t> _empty_groups;

    typename spec_t::groups_t _groups;

    idx_map<size_t, size_t> _N;
    weight_t _E{};

    typedef spec_t::vprop_t<std::tuple<deg_t, deg_t>> degs_map_t;
    typename degs_map_t::unchecked_t _degs;

    typedef EHash<bg_t, flat_map_t::value> emat_t;
    emat_t _emat;

    std::unique_ptr<EGroups<flat_map_t::value>> _egroups;
    bool _egroups_update = true;

    size_t _actual_B = 0;

    using partition_stats_t = spec_t::partition_stats_t<BlockState>::type;
    CPartitionStats<partition_stats_t> _ps;

    STATE_SPEC::cmap_t::unchecked_t _bclabel;
    STATE_SPEC::cmap_t::unchecked_t _bpclabel;

    using deg_stats_t = spec_t::deg_stats_t<BlockState>::type;
    deg_stats_t _deg_stats;

    using m_entries_t = spec_t::m_entries_t<BlockState>::type;
    m_entries_t _m_entries;

    coupled_state_t _coupled_state{};
    coupled_entropy_args_t _coupled_entropy_args{};

    std::vector<std::shared_ptr<shared_canary_mutex>> _group_mutex;

    // for move_vertices_parallel()
    spec_t::vprop_t<group_t>::unchecked_t _vset;
    idx_set<group_t> _rset;
    vprop_map_t<rw_t> _group_delta;
    std::vector<std::vector<vertex_t>> _groups_in, _groups_out;
    std::vector<std::conditional_t<flat_map_t::value,
                                   std::flat_map<group_t, weight_t>,
                                   gt_hash_map<group_t, weight_t>>>
        _deltas;
    std::vector<m_entries_t> _m_entries_pool;

    spec_t::Imp<BlockState> _spec;
};

GEN_DISPATCH(STATE_NAME, BlockState, BLOCK_MODEL_params)

} // STATE_NAME_SPACE
} // graph_tool namespace

#endif //BLOCKMODEL_HH
