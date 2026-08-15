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

#ifndef DEG_PARTITION_HH
#define DEG_PARTITION_HH

#include "../support/util.hh"
#include "../support/int_part.hh"

#include "hash_map_wrap.hh"

#include <ranges>

#include <mutex>
#include <vector>
#include <ranges>
#include <flat_map>

namespace graph_tool
{

enum class deg_dl_kind
{
    UNIFORM,
    DIST
};

template <bool directed, class Deg, class Group, bool flat=false>
class DegStats
{
public:
    typedef Group group_t;
    typedef Deg deg_t;

    DegStats() = default;

    template <class F>
    void init(F&& iter_items)
    {
        group_t max_r = 0;
        iter_items
            ([&](auto&&, group_t r, auto...)
             {
                 max_r = std::max(max_r, r);
             });

        get_r(max_r);

        iter_items
            ([&](auto&& deg, group_t r, size_t w = 1)
             {
                 add_vertex(deg, r, w);
             });
    }

    template <bool out>
    auto& get_hist(group_t r)
    {
        return (out) ? _hist_out[r] : _hist_in[r];
    };

    auto get_count(const auto& h, size_t k)
    {
        auto iter = h.find(k);
        return (iter != h.end()) ? iter->second : 0;
    }

    group_t get_r(group_t r)
    {
        if (size_t(r) >= _hist_out.size())
        {
            if constexpr (directed)
            {
                soft_resize(_hist_in, r + 1);
                soft_resize(_em, r + 1);
            }
            soft_resize(_hist_out, r + 1);
            soft_resize(_total, r + 1);
            soft_resize(_ep, r + 1);
        }
        return r;
    }

    template <class Rs, class Kins, class Kouts>
    double get_dl_uniform(Rs&& rs, Kins&& kins, Kouts&& kouts, bool parallel=false)
    {
        int64_t dp = 0;
        int64_t dm = 0;
        int64_t dn = 0;

        if (!(kins.empty() && kouts.empty()))
        {
            if constexpr (directed)
            {
                for (const auto& [kin, delta] : kins)
                    dm += delta * kin;
            }

            for (const auto& [kout, delta] : kouts)
            {
                dp += delta * kout;
                dn += delta;
            }
        }

        double S = 0;

        #pragma omp parallel for schedule(runtime) reduction(+:S) if (parallel)
        for (auto r : rs)
        {
            if (_total[r] + dn == 0)
                continue;
            S += lbinom_fast(_total[r] + dn + _ep[r] + dp - 1, _ep[r] + dp);
            if constexpr (directed)
                S += lbinom_fast(_total[r] + dn + _em[r] + dm - 1, _em[r] + dm);

            assert(!std::isnan(S) && !std::isinf(S));
        }
        assert(!std::isnan(S) && !std::isinf(S));
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_dl_dist(Rs&& rs, Kins&& kins, Kouts&& kouts, bool parallel=false)
    {
        double S = 0;
        #pragma omp parallel for schedule(runtime) reduction(+:S) if (parallel)
        for (auto r : rs)
        {
            int64_t dm = 0;
            int64_t dp = 0;
            int64_t dn = 0;
            int64_t N = _total[r];
            if (kins.empty() && kouts.empty())
            {
                if (N == 0)
                    continue;
                if constexpr (directed)
                {
                    for (auto&& k_c : get_hist<false>(r))
                        S -= lgamma_fast(k_c.second + 1);
                }

                for (auto&& k_c : get_hist<true>(r))
                    S -= lgamma_fast(k_c.second + 1);
            }
            else
            {
                if constexpr (directed)
                {
                    auto& h_in = get_hist<false>(r);
                    for (const auto& [kin, delta] : kins)
                    {
                        if (kin == null_deg)
                            continue;
                        auto k_c = get_count(h_in, kin);
                        S -= lgamma_fast(k_c + delta + 1);
                        dm += delta * kin;
                    }
                }

                auto& h_out = get_hist<true>(r);
                for (const auto& [kout, delta] : kouts)
                {
                    if (kout == null_deg)
                        continue;
                    auto k_c = get_count(h_out, kout);
                    S -= lgamma_fast(k_c + delta + 1);
                    dp += delta * kout;
                    dn += delta;
                }
            }

            S += log_q(_ep[r] + dp, N + dn);
            if constexpr (directed)
                S += log_q(_em[r] + dm, N + dn);

            if constexpr (directed)
                S += 2 * lgamma_fast(N + dn + 1);
            else
                S += lgamma_fast(N + dn + 1);
        }
        return S;
    }

    template <class Rs, class Kins, class Kouts>
    double get_dl(deg_dl_kind kind, Rs&& rs, Kins&& kins, Kouts&& kouts, bool parallel=false)
    {
        switch (kind)
        {
        case deg_dl_kind::UNIFORM:
            return get_dl_uniform(rs, kins, kouts, parallel);
        case deg_dl_kind::DIST:
            return get_dl_dist(rs, kins, kouts, parallel);
        default:
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double get_dl(deg_dl_kind kind)
    {
        return get_dl(kind,
                      std::ranges::iota_view{size_t(0),  _total.size()},
                      std::array<std::tuple<size_t,int64_t>,0>(),
                      std::array<std::tuple<size_t,int64_t>,0>(), true);
    }

    double get_delta_dl(auto&& deg, group_t r, group_t nr, deg_dl_kind kind, int vw = 1)
    {
        if (r == nr)
            return 0;

        if (r != null_group)
            r = get_r(r);

        if (nr != null_group)
            nr = get_r(nr);

        auto& [kin, kout] = deg;

        std::array<std::pair<deg_t,int64_t>,1> d_in;
        std::array<std::pair<deg_t,int64_t>,1> d_out;

        d_in[0] = {kin, 0};
        d_out[0] = {kout, 0};

        double S_b = get_dl(kind, std::array<group_t, 2>{r, nr}, d_in, d_out);

        double S_a = 0;

        d_in[0] = {kin, -vw};
        d_out[0] = {kout, -vw};

        if (r != null_group)
            S_a += get_dl(kind, std::array<group_t, 1>{r}, d_in, d_out);

        d_in[0] = {kin, vw};
        d_out[0] = {kout, vw};

        if (nr != null_group)
            S_a += get_dl(kind, std::array<group_t, 1>{nr}, d_in, d_out);

        return S_a - S_b;
    }

    double get_merge_dl(deg_dl_kind kind, group_t s, group_t t)
    {
        if (s == t || _total[s] == 0 || _total[t] == 0)
            return 0;

        auto zcount = [](auto&& kc) { return std::make_tuple(get<0>(kc), 0); };

        double S_b = 0;
        if constexpr (directed)
            S_b = get_dl(kind, std::array<group_t,2>({s, t}),
                         _hist_in[s] | std::views::transform(zcount),
                         _hist_out[s] | std::views::transform(zcount));
        else
            S_b = get_dl(kind, std::array<group_t,2>({s, t}),
                         _hist_out[s] | std::views::transform(zcount),
                         _hist_out[s] | std::views::transform(zcount));

        double S_a = 0;
        if constexpr (directed)
            S_a = get_dl(kind, std::array<group_t,1>({t}),
                         _hist_in[s], _hist_out[s]);
        else
            S_a = get_dl(kind, std::array<group_t,1>({t}),
                         _hist_out[s], _hist_out[s]);

        return S_a - S_b;
    }

    void change_deg(size_t r, size_t kin, size_t kout, int64_t delta)
    {
        auto change_hist =
            [&](auto& h, size_t k)
            {
                auto iter = h.insert({k, 0}).first;
                iter->second += delta;
                if (iter->second == 0)
                {
                    h.erase(iter);
                    if (h.empty())
                    {
                        map_t empty;
                        h.swap(empty);
                    }
                }
            };

        if constexpr (directed)
            change_hist(get_hist<false>(r), kin);
        change_hist(get_hist<true>(r), kout);

        if constexpr (directed)
            _em[r] += delta * kin;
        _ep[r] += delta * kout;
        _total[r] += delta;
    }

    // concurrent
    void change_vertex(auto&& deg, group_t r, int64_t diff)
    {
        if (r == null_group || diff == 0)
            return;
        auto& [kin, kout] = deg;
        change_deg(r, kin, kout, diff);
    }

    void remove_vertex(auto&& deg, group_t r, int64_t vw = 1)
    {
        change_vertex(deg, r, -vw);
        #pragma omp atomic
        _N -= vw;
    }

    void add_vertex(auto&& deg, group_t nr, int64_t vw = 1)
    {
        if (vw == 0)
            return;
        nr = get_r(nr);
        change_vertex(deg, nr, vw);
        #pragma omp atomic
        _N += vw;
    }

    void merge_groups(group_t s, group_t t)
    {
        if (s == t)
            return;

        if constexpr (directed)
        {
            for (auto&& [k, c] : _hist_in[s])
                _hist_in[t][k] += c;
            map_t empty;
            _hist_in[s].swap(empty);
        }

        for (auto&& [k, c] : _hist_out[s])
            _hist_out[t][k] += c;

        map_t empty;
        _hist_out[s].swap(empty);

        _total[t] += _total[s];
        _ep[t] += _ep[s];
        _total[s] = _ep[s] = 0;

        if constexpr (directed)
        {
            _em[t] += _em[s];
            _em[s] = 0;
        }
    }

    void add_group(size_t n = 1)
    {
        if (n == 1)
        {
            if constexpr (directed)
            {
                _hist_in.emplace_back();
                _em.emplace_back();
            }
            _hist_out.emplace_back();
            _total.emplace_back();
            _ep.emplace_back();
        }
        else
        {
            size_t nB = _total.size() + n;
            if constexpr (directed)
            {
                _hist_in.resize(nB);
                _em.resize(nB);
            }
            _hist_out.resize(nB);
            _total.resize(nB);
            _ep.resize(nB);
        }
    }

    static constexpr group_t null_group = std::numeric_limits<group_t>::max();
    static constexpr deg_t null_deg = std::numeric_limits<deg_t>::max();

private:
    typedef std::conditional_t<flat,
                               std::flat_map<deg_t, int32_t>,
                               gt_hash_map<deg_t, int32_t>> map_t;
    std::vector<map_t> _hist_in, _hist_out;
    std::vector<deg_t> _total;
    std::vector<deg_t> _ep;
    std::vector<deg_t> _em;
    size_t _N = 0;
};

} //namespace graph_tool

#endif // DEG_PARTITION_HH
