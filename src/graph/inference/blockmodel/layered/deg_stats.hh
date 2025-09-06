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

#ifndef DEG_STATS_HH
#define DEG_STATS_HH

#include <boost/unordered/concurrent_flat_map.hpp>

#include "../deg_partition.hh"

namespace graph_tool
{

template <class Base>
class LDegStats
{
public:
    typedef typename Base::group_t group_t;
    typedef int32_t lval_t;

    LDegStats() = default;

    template <class F>
    void init(F&& iter_items)
    {
        iter_items
            ([&](auto&& deg, group_t r, auto&& w)
             {
                 add_vertex(deg, r, w);
             });
    }

    void set_N(size_t N)
    {
        _N = N;
        for (auto& kv : _deg_stats)
            get<1>(kv).set_N(N);
    }

    auto&& get_dstats(lval_t l)
    {
        auto iter = _deg_stats.find(l);
        if (iter == _deg_stats.end())
        {
            iter = _deg_stats.try_emplace(l).first;
            if constexpr (requires {iter->second.set_N(_N);})
                iter->second.set_N(_N);
        }
        return iter->second;
    }

    auto& get_lmutex(lval_t l)
    {
        auto iter = _lmutex.find(l);
        if (iter == _lmutex.end())
            iter = _lmutex.insert({l, std::make_shared<shared_canary_mutex>()}).first;
        return *iter->second;
    }

    auto visit_layer(lval_t l, auto&& f)
    {
        do_slock<{.pass_lock=true}>([&](auto& lock){ f(lock); }, get_lmutex(l));
    }

    group_t get_r(lval_t l, group_t r, auto&& lock)
    {
        group_t rl = r;
        auto& rmap = _lmap[l];
        visit_or_emplace(rmap, r,
                         [&](auto& s){  rl = s; },
                         [&](auto& s)
                         {
                             rl = s = rmap.size() - 1;
                             get_dstats(l).add_group(1);
                         },
                         lock);
        return rl;
    }
    void remove_vertex(auto&& deg, group_t r, const auto& vw)
    {
        vw.template cdispatch<true>
            ([&](auto&& h)
             {
                 iter_counts([&](auto l, auto&& c)
                             {
                                 visit_layer(l,
                                             [&](auto& lock)
                                             {
                                                 auto rl = get_r(l, r, lock);
                                                 get_dstats(l).remove_vertex(std::make_tuple(get<0>(c), get<1>(c)), rl, get<2>(c));
                                             });
                             },
                             get<0>(deg)._h, get<1>(deg)._h, h);
             });
    }

    void add_vertex(auto&& deg, group_t nr, const auto& vw)
    {
        vw.template cdispatch<true>
            ([&](auto&& h)
             {
                 iter_counts([&](auto l, auto&& c)
                             {
                                 visit_layer(l,
                                             [&](auto& lock)
                                             {
                                                 auto rl = get_r(l, nr, lock);
                                                 get_dstats(l).add_vertex(std::make_tuple(get<0>(c), get<1>(c)), rl, get<2>(c));
                                             });
                             },
                             get<0>(deg)._h, get<1>(deg)._h, h);
             });
    }

    void merge_groups(group_t s, group_t t)
    {
        if (s == t)
            return;
        for (auto& ld : _deg_stats)
        {
            auto& [l, dstats]  = ld;
            visit_layer(l,
                        [&](auto& lock)
                        {
                            dstats.merge_groups(get_r(l, s, lock), get_r(l, t, lock));
                        });
        }
    }

    void add_group(auto) {}

    double get_merge_dl(deg_dl_kind kind, group_t s, group_t t)
    {
        double S = 0;
        for (auto& ld : _deg_stats)
        {
            auto& [l, dstats]  = ld;
            visit_layer(l,
                        [&](auto& lock)
                        {
                            S += dstats.get_merge_dl(kind, get_r(l, s, lock), get_r(l, t, lock));
                        });
        }

        return S;
    }

    double get_delta_dl(auto&& deg, group_t r, group_t nr, deg_dl_kind kind, auto&& vw)
    {
        double dS = 0;
        vw.template dispatch<true>
            ([&](auto&& h)
             {
                 iter_counts
                     ([&](auto l, auto&& c)
                      {
                          auto& [kin, kout, vw] = c;
                          if (vw == 0)
                              return;

                          visit_layer
                              (l,
                               [&](auto& lock)
                               {
                                   auto lr = get_r(l, r, lock);
                                   auto lnr = get_r(l, nr, lock);
                                   dS += get_dstats(l).get_delta_dl(std::make_tuple(kin, kout),
                                                                    lr, lnr, kind, vw);
                               });

                      }, get<0>(deg)._h, get<1>(deg)._h, h);
             });
        return dS;
    }

    double get_dl(deg_dl_kind kind)
    {
        double S = 0;
        for (auto& ld : _deg_stats)
        {
            auto& [l, dstats]  = ld;
            S += dstats.get_dl(kind);
        }
        return S;
    }

    template <class T, size_t N1, size_t N2>
    double get_dl(deg_dl_kind kind, auto&& rs,
                  const std::array<std::tuple<T, int64_t>, N1>& kins,
                  const std::array<std::tuple<T, int64_t>, N2>& kouts)
    {
        std::array<const T*, N1> kis;
        std::array<const T*, N2> kos;
        for (size_t i = 0; i < N1; ++i)
            kis[i] = &get<0>(kins[i]);
        for (size_t i = 0; i < N2; ++i)
            kos[i] = &get<0>(kouts[i]);

        double S = 0;

        typedef std::decay_t<decltype(std::get<0>(*T()._h.begin()))> val_t;

        std::array<std::tuple<val_t, int64_t>, N1> lkins;
        std::array<std::tuple<val_t, int64_t>, N2> lkouts;

        iter_counts_arrays
            ([&](auto l, auto&& lkis, auto&& lkos)
             {
                 for (size_t i = 0; std::cmp_less(i, N1); ++i)
                     lkins[i] = {lkis[i], get<1>(kins[i])};
                 for (size_t i = 0; std::cmp_less(i, N2); ++i)
                     lkouts[i] = {lkos[i], get<1>(kouts[i])};

                 visit_layer
                     (l,
                      [&](auto& lock)
                      {
                          S += get_dstats(l).get_dl(kind,
                                                    rs | std::views::transform([&](auto r){ return get_r(l, r, lock); }),
                                                    lkins, lkouts);
                      });
             },
             kis, kos);

        return S;
    }

private:
    boost::unordered_flat_map<lval_t, Base> _deg_stats;
    boost::unordered_flat_map<lval_t, boost::unordered_flat_map<group_t, group_t>> _lmap;
    boost::unordered_flat_map<lval_t, std::shared_ptr<shared_canary_mutex>> _lmutex;
    size_t _N = 0;
};

} //namespace graph_tool

#endif // DEG_STATS_HH
