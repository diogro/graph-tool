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

#pragma once

#include "types.hh"

#include <boost/unordered/concurrent_flat_map.hpp>

namespace graph_tool
{

template <bool directed>
struct SimpleOverlapDegStats
{
    typedef int64_t vertex_t;
    typedef int64_t deg_t;
    typedef int64_t group_t;

    SimpleOverlapDegStats() = default;

    void set_N(size_t N)
    {
        _N = N;
    }

    size_t get_out_deg(size_t v, group_t r)
    {
        auto ret = 0;
        _out_deg[r].visit(v,
                         [&](auto& vd)
                         {
                             ret = std::get<1>(vd);
                         });
        return ret;
    }

    size_t get_in_deg(size_t v, group_t r)
    {
        if constexpr (directed)
        {
            auto ret = 0;
            _in_deg[r].visit(v,
                             [&](auto& vd)
                             {
                                 ret = std::get<1>(vd);
                             });
            return ret;
        }
        else
        {
            return 0;
        }
    }

    double dl_uniform_r(auto m)
    {
        double S = 0;
        S += lbinom_fast(_N + m - 1, m);
        return S;
    }

    template <class Hist>
    double dl_dist_r(auto m, Hist& hist)
    {
        double S = 0;

        S += log_q(m, _N);

        if constexpr (is_specialization_of_v<Hist, boost::concurrent_flat_map>)
        {
            hist.visit_all
                ([&](auto& kc)
                 {
                     S -= lgamma_fast(get<1>(kc) + 1);
                 });
        }
        else
        {
            for (auto& kc : hist)
                S -= lgamma_fast(get<1>(kc) + 1);
        }

        S += lgamma_fast(_N + 1);

        return S;
    }

    double get_deg_dl_uniform()
    {
        double S = 0;
        for (size_t r = 0; r < _out_deg.size(); ++r)
        {
            S += dl_uniform_r(_ep[r]);
            if constexpr (directed)
                S += dl_uniform_r(_em[r]);
        }
        return S;
    }

    double get_deg_dl_dist()
    {
        double S = 0;
        for (size_t r = 0; r < _out_deg.size(); ++r)
        {
            S += dl_dist_r(_ep[r], _out_hist[r]);
            if constexpr (directed)
                S += dl_dist_r(_em[r], _in_hist[r]);
        }
        return S;
    }

    double get_dl(deg_dl_kind kind)
    {
        switch (kind)
        {
        case deg_dl_kind::UNIFORM:
            return get_deg_dl_uniform();
        case deg_dl_kind::DIST:
            return get_deg_dl_dist();
        default:
            return numeric_limits<double>::quiet_NaN();
        }
    }

    double get_dl(deg_dl_kind kind, auto&&, auto&&, auto&&) // FIXME: optimize
    {
        return get_dl(kind);
    }

    double get_delta_dl(auto&&, group_t r, group_t nr, deg_dl_kind kind,
                        const ovweight_t& wr)
    {
        if (r == nr)
            return 0;

        double dS = 0;
        for (auto& [v, deg] : wr)
        {
            int64_t delta_in = std::get<0>(deg);
            int64_t delta_out = std::get<1>(deg);

            switch (kind)
            {
            case deg_dl_kind::UNIFORM:
                dS += get_delta_deg_dl_uniform_change(r, -delta_out, -delta_in);
                dS += get_delta_deg_dl_uniform_change(nr, delta_out, delta_in);
                break;
            case deg_dl_kind::DIST:
                dS += get_delta_deg_dl_dist_change(r, get_out_deg(v, r), get_in_deg(v, r), -delta_out, -delta_in);
                dS += get_delta_deg_dl_dist_change(nr, get_out_deg(v, nr), get_in_deg(v, nr), delta_out, delta_in);
                break;
            default:
                dS = numeric_limits<double>::quiet_NaN();
            }
        }
        return dS;
    }

    double get_delta_deg_dl_uniform_change(size_t r, auto delta_out, auto delta_in)
    {
        double S_b = 0;
        double S_a = 0;

        auto epr = _ep[r];
        S_b += dl_uniform_r(epr);
        S_a += dl_uniform_r(epr + delta_out);

        if constexpr (directed)
        {
            auto emr = _em[r];
            S_b += dl_uniform_r(emr);
            S_a += dl_uniform_r(emr + delta_in);
        }

        return S_a - S_b;
    }

    double get_delta_deg_dl_dist_change(size_t r, size_t out_deg, size_t in_deg,
                                        auto delta_out, auto delta_in)
    {
        double S_b = 0;
        double S_a = 0;

        auto get_Sk =
            [&](auto deg, auto delta, auto& hist)
            {
                size_t nd = 0;
                auto& h = hist[r];
                h.visit(deg,
                        [&](auto& kc)
                        {
                            nd = std::get<1>(kc);
                        });
                assert(int64_t(nd) + delta >= 0);
                S_b += -lgamma_fast(nd + 1);
                S_a += -lgamma_fast(nd + delta + 1);
            };

        if (delta_out != 0)
        {
            auto epr = _ep[r];

            S_b += log_q(epr, _N);
            S_a += log_q(epr + delta_out, _N);
            get_Sk(out_deg,            -1, _out_hist);
            get_Sk(out_deg + delta_out, 1, _out_hist);
        }

        if constexpr (directed)
        {
            if (delta_in != 0)
            {
                auto emr = _em[r];
                S_b += log_q(emr, _N);
                S_a += log_q(emr + delta_in, _N);
                get_Sk(in_deg,           -1, _in_hist);
                get_Sk(in_deg + delta_in, 1, _in_hist);
            }
        }

        return S_a - S_b;
    }

    void move_vitem(vertex_t v, auto& deg, group_t r, group_t nr)
    {
        if (r != _null_group)
        {
            auto remove_item =
                [&](auto& degs, auto& hist, auto d)
                {
                    degs[r].erase_if
                        (v,
                         [&](auto& vd)
                         {
                             auto& k = get<1>(vd);

                             auto& h = hist[r];
                             h.erase_if
                                 (k,
                                  [&](auto& kc)
                                  {
                                      std::get<1>(kc)--;
                                      return std::get<1>(kc) == 0;
                                  });

                             k -= d;

                             h.emplace_or_visit(k, 1,
                                                [&](auto& kc)
                                                {
                                                    std::get<1>(kc)++;
                                                });

                             return k == 0;
                         });
                };

            remove_item(_out_deg, _out_hist, get<1>(deg));

            #pragma omp atomic
            _ep[r] -= get<1>(deg);

            if constexpr (directed)
            {
                remove_item(_in_deg, _in_hist, get<0>(deg));

                #pragma omp atomic
                _em[r] -= get<0>(deg);
            }
        }

        if (nr != _null_group)
        {
            auto add_item =
                [&](auto& degs, auto& hist, auto d)
                {
                    if (d == 0)
                        return;

                    auto f =
                        [&](auto& vd)
                        {
                            auto& k = get<1>(vd);

                            auto& h = hist[nr];

                            h.erase_if
                                (k,
                                 [&](auto& kc)
                                 {
                                     std::get<1>(kc)--;
                                     return std::get<1>(kc) == 0;
                                 });

                            k += d;

                            h.emplace_or_visit(k, 1,
                                               [&](auto& kc)
                                               {
                                                   std::get<1>(kc)++;
                                               });
                        };

                    degs[nr].emplace_and_visit(v, 0, f, f);
                };

            add_item(_out_deg, _out_hist, get<1>(deg));

            #pragma omp atomic
            _ep[nr] += get<1>(deg);

            if constexpr (directed)
            {
                add_item(_in_deg, _in_hist, get<0>(deg));

                #pragma omp atomic
                _em[nr] += get<0>(deg);
            }
        }
    }

    void move_vertex(group_t r, group_t nr, const ovweight_t& w)
    {
        for (const auto& [v, deg] : w)
            move_vitem(v, deg, r, nr);
    }

    void add_vertex(auto&&, group_t r, const ovweight_t& w)
    {
        if (size_t(r) >= _out_deg.size())
            add_group(r - _out_deg.size() + 1);
        move_vertex(_null_group, r, w);
    }

    void remove_vertex(auto&&, group_t r, const ovweight_t& w)
    {
        move_vertex(r, _null_group, w);
    }

    void add_group(size_t n)
    {
        boost::concurrent_flat_map<deg_t, size_t> empty;
        empty.emplace(0, _N);

        soft_resize(_out_deg, _out_deg.size() + n);
        soft_resize(_out_hist, _out_hist.size() + n, empty);
        soft_resize(_ep, _ep.size() + n);

        if constexpr (directed)
        {
            soft_resize(_in_deg, _in_deg.size() + n);
            soft_resize(_in_hist, _in_hist.size() + n, empty);
            soft_resize(_em, _em.size() + n);
        }
    }

    void merge_groups(group_t r, group_t s)
    {
        if (r == s)
            return;

        auto merge_degs =
            [&](auto& degs, auto& hist)
            {
                degs[r].visit_all
                    ([&](auto& vd)
                     {
                         auto& [v, d] = vd;
                         degs[s].insert_or_visit
                             (vd,
                              [&](auto& vd_)
                              {
                                  get<1>(vd_) += d;
                              });
                     });

                degs[r].clear();
                hist[r].clear();
                hist[r].emplace(0, _N);

                hist[s].clear();

                degs[s].visit_all
                    ([&](auto&& vd)
                     {
                         auto&& [v, d] = vd;
                         hist[s].insert_or_visit({d, 1},
                                                 [&](auto& kc)
                                                 {
                                                     std::get<1>(kc)++;
                                                 });
                     });

                hist[s].emplace(0, _N - degs[s].size());
            };

        merge_degs(_out_deg, _out_hist);
        _ep[s] += _ep[r];
        _ep[r] = 0;

        if constexpr (directed)
        {
            merge_degs(_in_deg, _in_hist);
            _em[s] += _em[r];
            _em[r] = 0;
        }
    }

    double get_merge_dl(deg_dl_kind kind, group_t r, group_t s)
    {
        if (r == s)
            return 0;

        double S_b = 0, S_a = 0;

        switch (kind)
        {
        case deg_dl_kind::UNIFORM:
            S_b += dl_uniform_r(_ep[r]) + dl_uniform_r(_ep[s]);
            S_a += dl_uniform_r(_ep[r] + _ep[s]);
            if constexpr (directed)
            {
                S_b += dl_uniform_r(_em[r]) + dl_uniform_r(_em[s]);
                S_a += dl_uniform_r(_em[r] + _em[s]);
            }
            break;
        case deg_dl_kind::DIST:
            {
                auto delta_dl =
                    [&](auto& degs, auto& hist, auto& m)
                    {
                        gt_hash_map<vertex_t, deg_t> ndegs;
                        gt_hash_map<deg_t,  size_t> nhist;

                        degs[s].visit_all
                            ([&](auto&& vd)
                             {
                                 auto&& [v, d] = vd;
                                 ndegs[v] = d;
                             });

                        degs[r].visit_all
                            ([&](auto& vd)
                             {
                                 auto& [v, d] = vd;
                                 auto& nd = ndegs[v];
                                 nd += d;
                             });

                        for (auto& [v, d] : ndegs)
                            nhist[d]++;
                        nhist.emplace(0, _N - ndegs.size());

                        S_b += dl_dist_r(m[r], hist[r]) + dl_dist_r(m[s], hist[s]);
                        S_a += dl_dist_r(m[r] + m[s], nhist);
                    };

                delta_dl(_out_deg, _out_hist, _ep);
                if constexpr (directed)
                    delta_dl(_in_deg, _in_hist, _em);
            }
            break;
        default:
            return numeric_limits<double>::quiet_NaN();
        }
        return S_a - S_b;
    }

private:
    size_t _N = 0;
    std::vector<boost::concurrent_flat_map<vertex_t, deg_t>> _out_deg, _in_deg;
    std::vector<boost::concurrent_flat_map<deg_t, size_t>> _out_hist, _in_hist;
    std::vector<size_t> _ep;
    std::vector<size_t> _em;

    constexpr static group_t _null_group = std::numeric_limits<group_t>::max();
};

struct DummyPartitionStats
{
    DummyPartitionStats() {}

    template <class F>
    DummyPartitionStats(F&&...) {}

    template <class F>
    void init(F&&...) {}

    constexpr double get_dl() { return 0; }

    constexpr double get_delta_dl(auto, auto, auto&&)
    {
        return 0;
    }

    constexpr double get_delta_dl(auto, auto)
    {
        return 0;
    }

    constexpr void move_item(auto, auto, auto&&)
    {
    }

    template <bool check=false>
    constexpr void add_item(auto, auto&&)
    {
    }

    constexpr void remove_item(auto, auto&&)
    {
    }

    constexpr void add_group()
    {
    }

    size_t get_B()
    {
        return 0;
    }

    typedef int64_t group_t;
    constexpr static group_t _null_group = std::numeric_limits<group_t>::max();
};


} //namespace graph_tool
