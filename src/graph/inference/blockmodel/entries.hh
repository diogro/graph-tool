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

#ifndef GRAPH_BLOCKMODEL_ENTRIES_HH
#define GRAPH_BLOCKMODEL_ENTRIES_HH

#include "graph_tool.hh"
#include "parallel_util.hh"

#include <cstddef>
#include <limits>
#include <vector>
#include <boost/property_map/property_map.hpp>
#include <boost/container/small_vector.hpp>

namespace graph_tool
{

// Manage a set of block pairs and corresponding edge counts that will be
// updated

template <bool directed, class Group, class Count, class Edge, class EW>
class EntrySet
{
public:
    typedef Edge bedge_t;

    typedef Group group_t;
    typedef Count count_t;
    static constexpr group_t null_group = std::numeric_limits<group_t>::max();

    EntrySet() : _zero{} {};

    void resize(size_t B)
    {
        _r_out_field.resize(B, _null);
        _nr_out_field.resize(B, _null);
        if constexpr (directed)
        {
            _r_in_field.resize(B, _null);
            _nr_in_field.resize(B, _null);
        }
    }

    void set_move(group_t r, group_t nr, size_t B)
    {
        clear();
        _rnr = {r, nr};
        if (B > _r_out_field.size())
            resize(B);
    }

    const std::pair<group_t, group_t>& get_move() { return _rnr; }

    template <bool First, bool Source>
    auto& get_field_rnr(group_t s, group_t t)
    {
        auto get_out_field =
            [&]() -> auto&
            {
                if constexpr (First)
                    return _r_out_field;
                else
                    return _nr_out_field;
            };

        auto get_in_field =
            [&]() -> auto&
            {
                if constexpr (First)
                    return _r_in_field;
                else
                    return _nr_in_field;
            };

        auto& out_field = get_out_field();

        if constexpr (directed)
        {
            if constexpr (Source)
                return out_field[t];
            auto& in_field = get_in_field();
            return (s == t) ? out_field[t] : in_field[s];
        }
        else
        {
            if constexpr (Source)
                return out_field[t];
            else
                return out_field[s];
        }
    }

    auto& get_field(group_t s, group_t t)
    {
        if (s == _rnr.first)
            return get_field_rnr<true, true>(s, t);
        if (t == _rnr.first)
            return get_field_rnr<true, false>(s, t);
        if (s == _rnr.second)
            return get_field_rnr<false, true>(s, t);
        if (t == _rnr.second)
            return get_field_rnr<false, false>(s, t);
        return _dummy;
    }

    auto cget_field(group_t s, group_t t) const
    {
        return const_cast<EntrySet*>(this)->get_field(s, t);
    }

    template <bool Add, class Delta>
    void insert_delta_dispatch(group_t s, group_t t, size_t& f, const Delta& d)
    {
        if (f == _null)
        {
            f = _entries.size();
            _entries.emplace_back(s, t);
            _delta.emplace_back();
        }

        assert(f < _delta.size() && f < _entries.size());

        if constexpr (Add)
            _delta[f] += d;
        else
            _delta[f] -= d;
    }

    template <bool First, bool Source, bool Add, class Delta>
    void insert_delta_rnr(group_t s, group_t t, const Delta& d)
    {
        auto& f = get_field_rnr<First, Source>(s, t);
        insert_delta_dispatch<Add>(s, t, f, d);
    }

    template <bool Add, class Delta>
    void insert_delta(group_t s, group_t t, const Delta& d)
    {
        auto& f = get_field(s, t);
        insert_delta_dispatch<Add>(s, t, f, d);
    }

    const auto& get_delta(group_t r, group_t s) const
    {
        auto f = cget_field(r, s);
        if (f == _null)
            return _zero;
        return _delta[f];
    }

    bool empty()
    {
        return _entries.empty();
    }

    void clear()
    {
        for (const auto& [r, s] : _entries)
            get_field(r, s) = _null;
        _entries.clear();
        _delta.clear();
        _mes.clear();
    }

    const auto& get_entries() const { return _entries; }
    const auto& get_deltas() const { return _delta; }

    template <class Emat>
    auto& get_mes(Emat& emat)
    {
        for (size_t i = _mes.size(); i < _entries.size(); ++i)
        {
            auto& rs = _entries[i];
            _mes.push_back(emat.get_me(rs.first, rs.second));
            assert(_mes.back() != emat.get_null_edge() || count_t(_delta[i]) >= 0);
        }
        return _mes;
    }

    template <class Emat>
    const bedge_t& get_me(group_t r, group_t s, Emat& emat) const
    {
        size_t field = cget_field(r, s);
        if (field >= _mes.size())
            return emat.get_me(r, s);
        return _mes[field];
    }

    void set_next(EntrySet& next)
    {
        _next = &next;
    }

    EntrySet* _next = nullptr;

private:
    static constexpr size_t _null = std::numeric_limits<size_t>::max();

    std::pair<group_t, group_t> _rnr;
    std::vector<size_t> _r_out_field;
    std::vector<size_t> _r_in_field;
    std::vector<size_t> _nr_out_field;
    std::vector<size_t> _nr_in_field;

    std::vector<std::pair<group_t, group_t>> _entries;
    std::vector<EW> _delta;

    std::vector<bedge_t> _mes;

    size_t _dummy = _null;
    EW _zero;
};

template <bool single>
void modify_entries_dispatch(auto v, auto r, auto nr, auto& b, auto& nb,
                             auto& g, auto& eweights, auto B, auto& m_entries,
                             auto&& efilt)
{
    m_entries.set_move(r, single ? m_entries.null_group : nr, B);

    typedef std::decay_t<decltype(g)> g_t;
    typedef std::decay_t<decltype(v)> v_t;

    typedef typename boost::property_traits<std::decay_t<decltype(eweights)>>::value_type
        weight_t;

    weight_t self_weight{};

    for (auto e : out_edges_range(v, g))
    {
        if (efilt(e, true))
            continue;

        v_t u = target(e, g);
        auto s = b[u];
        auto&& ew = eweights[e];

        m_entries.template insert_delta_rnr<true, true, false>(r, s, ew);

        if (u == v)
            s = nr;
        else
            s = nb[u];

        if (s != r)
            m_entries.template insert_delta_rnr<single, true, true>(nr, s, ew);
        else
            m_entries.template insert_delta_rnr<true, false, true>(nr, s, ew);

        if constexpr (!is_directed_v<g_t>)
        {
            if (u == v)
                self_weight += ew;
        }
    }

    if constexpr (!is_directed_v<g_t>)
    {
        if (int64_t(self_weight) != 0)
        {
            assert(int64_t(self_weight) % 2 == 0);
            self_weight /= 2;
            m_entries.template insert_delta_rnr<true, true, true>
                (r, r, self_weight);
            if constexpr (!single)
                m_entries.template insert_delta_rnr<single, true, false>
                    (nr, nr, self_weight);
        }
    }

    if constexpr (is_directed_v<g_t>)
    {
        for (auto e : in_edges_range(v, g))
        {
            if (efilt(e, false))
                continue;

            v_t u = source(e, g);
            if (u == v)
                continue;
            auto s = b[u];
            auto ew = eweights[e];

            m_entries.template insert_delta_rnr<true, false, false>(s, r, ew);

            s = nb[u];
            if (s != r)
                m_entries.template insert_delta_rnr<single, false, true>(s, nr, ew);
            else
                m_entries.template insert_delta_rnr<true, true, true>(s, nr, ew);
        }
    }
}

void modify_entries(auto v, auto nr, auto& b, auto& nb, auto& g, auto& eweights,
                    auto B, auto& m_entries, auto&& efilt)
{
    auto r = b[v];

    if (std::cmp_equal(r, nr))
        modify_entries_dispatch<true>(v, r, nr, b, nb, g, eweights, B,
                                      m_entries,  efilt);
    else
        modify_entries_dispatch<false>(v, r, nr, b, nb, g, eweights, B,
                                       m_entries, efilt);
}


// operation on a set of entries
template <class MEntries, class OP>
void entries_op(MEntries& m_entries, OP&& op)
{
    const auto& entries = m_entries.get_entries();
    const auto& delta = m_entries.get_deltas();

    for (size_t i = 0; i < entries.size(); ++i)
    {
        auto& entry = entries[i];
        auto er = entry.first;
        auto es = entry.second;
        op(er, es, delta[i]);
    }
}

template <bool lock, class State>
void update_rs(State& state, auto r, auto s, const auto& delta,
               auto&& update_egroups, auto&&... locks)
{
    typedef std::decay_t<decltype(state)> state_t;

    if (delta == 0)
        return;

    auto me = state._emat.get_me(r, s);

    if (me == state._emat.get_null_edge())
    {
        if (int64_t(delta) == 0)
            return;
        upgrade_canary_lock<{.ula={.la={.enable=lock}}}>
            ([&](bool succ)
             {
                 if constexpr (lock)
                 {
                     if (!succ)  [[unlikely]] // re-check to avoid preemption race
                     {
                         me = state._emat.get_me(r, s);
                         if (me != state._emat.get_null_edge())
                         {
                             state._spec.template update_rs<false>(me, delta, locks...);
                             return;
                         }
                     }
                 }
                 me = boost::add_edge(size_t(r), size_t(s), state._bg).first;
                 state._mrs.get_checked()[me] = typename State::weight_t{};
                 state._emat.put_me(r, s, me);
                 state._spec.template update_rs<false>(me, delta, locks...);
             }, locks...);
    }
    else
    {
        state._spec.template update_rs<lock>(me, delta, locks...);
    }

    auto&& mrs = state._mrs[me];

    atomic_inc<lock>(mrs, delta);
    atomic_inc<lock>(state._mrp[r], delta);
    if constexpr (is_directed_v<typename state_t::g_t>)
        atomic_inc<lock>(state._mrm[s], delta);
    else
        atomic_inc<lock>(state._mrp[s], delta);

    update_egroups(r, s, delta, locks...);

    assert(int64_t(state._mrs[me]) >= 0);
    assert(int64_t(state._mrp[r]) >= 0);
    assert(!is_directed_v<typename state_t::g_t> || int64_t(state._mrm[s]) >= 0);

    if (int64_t(mrs) == 0)
    {
        upgrade_canary_lock<{.ula={.la={.enable=lock}, .relock=false}}>
            ([&](bool succ)
             {
                 if constexpr (lock)
                 {
                     if (!succ)  [[unlikely]] // re-check to avoid preemption race
                     {
                         me = state._emat.get_me(r, s);
                         if (me == state._emat.get_null_edge() || state._mrs[me] != 0)
                             return;
                     }
                 }

                 state._emat.remove_me(me, state._bg);
                 boost::remove_edge(me, state._bg);
                 me = state._emat.get_null_edge();
             }, locks...);
    }
};


template <bool lock=false>
void apply_delta(auto& state, auto& m_entries, bool propagate=true)
{
    typedef std::decay_t<decltype(state)> state_t;

    if (m_entries.empty())
        return;

    auto update_egroups_dispatch =
        [&](auto&& dispatch)
        {
            if (state._egroups && state._egroups_update)
            {
                auto& egroups = *state._egroups;
                dispatch([&](auto r, auto s, const auto& delta, auto&&... locks)
                         {
                             upgrade_canary_lock<{.ula={.la={.enable=lock}}}>
                                 ([&](bool)
                                  {
                                      if (r != s)
                                      {
                                          egroups.insert_edge(r, s, delta);
                                          egroups.insert_edge(s, r, delta);
                                      }
                                      else
                                      {
                                          egroups.insert_edge(r, s, 2 * delta);
                                      }
                                  }, locks...);
                         });
            }
            else
            {
                dispatch([](auto&&...){});
            }
        };

    update_egroups_dispatch
        ([&](auto&& eup)
         {
             entries_op
                 (m_entries,
                  [&](auto r, auto s, auto&& delta)
                  {
                      state.template do_lock<state_t::lock_t::shared,
                                             {.enable=lock, .pass_lock=true}>
                          (r, s,
                           [&](auto&&... locks)
                           {
                               update_rs<lock>(state, r, s, delta, eup, locks...);
                           });
                  });
         });

    if (state._coupled_state != nullptr && propagate)
        apply_delta<lock>(*state._coupled_state, *m_entries._next);
}


} // namespace graph_tool

#endif // GRAPH_BLOCKMODEL_ENTRIES_HH
