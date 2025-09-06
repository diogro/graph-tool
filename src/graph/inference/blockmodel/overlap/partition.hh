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

#ifndef GRAPH_OVERLAP_BLOCKMODEL_PARTITION_HH
#define GRAPH_OVERLAP_BLOCKMODEL_PARTITION_HH

#include "half_edge_adaptor.hh"
#include "types.hh"
#include "boost/container/small_vector.hpp"

#include "scoped_hash_map.hh"

namespace std
{
template <class Value, size_t N>
struct hash<boost::container::small_vector<Value, N>>
{
    size_t operator()(const boost::container::small_vector<Value, N>& v) const
    {
        std::size_t seed = 0;
        for (auto& x : v)
            _hash_combine(seed, x);
        return seed;
    }
};

template <>
struct hash<dcount_t>
{
    size_t operator()(const dcount_t& v) const
    {
        std::size_t seed = 0;
        _hash_combine(seed, get<0>(v));
        _hash_combine(seed, get<1>(v));
        return seed;
    }
};

}

namespace graph_tool
{

//=============================
// Partition description length
//=============================

template <bool directed, bool scoped=false>
struct OverlapPartitionStats
{
    typedef int64_t group_t;
    typedef dcount_t deg_t;
    typedef boost::container::small_vector<deg_t, 64> cdeg_t;

    typedef boost::container::small_vector<int64_t, 64> bv_t;

    typedef gt_hash_map<bv_t, size_t> bhist_t;
    typedef gt_hash_map<cdeg_t, size_t, std::hash<cdeg_t>> cdeg_hist_t;

    typedef gt_hash_map<bv_t, cdeg_hist_t> deg_hist_t;

    typedef gt_hash_map<bv_t, vector<size_t>> ebhist_t;

    typedef gt_hash_map<int64_t, int64_t> dmap_t;

    OverlapPartitionStats() = default;

    template <bool s>
    using OPS = OverlapPartitionStats<directed, s>;

    template <class F>
    requires (!std::is_base_of_v<OPS<true>, std::decay_t<F>> &&
              !std::is_base_of_v<OPS<false>, std::decay_t<F>>)
    OverlapPartitionStats(F&& iter_items)
    {
        init(std::forward<F>(iter_items));
    }

    template <bool s>
    OverlapPartitionStats(const OverlapPartitionStats<directed, s>& o)
        :_N(o._N), _actual_B(o._actual_B), _D(o._D),
         _dhist(o._dhist), _r_count(o._r_count), _bhist(o._bhist),
         _emhist(o._emhist), _ephist(o._ephist), _embhist(o._embhist),
         _epbhist(o._epbhist), _deg_hist(o._deg_hist), _bvs(o._bvs),
         _degs(o._degs), _wr(o._wr)
    {
    }

    template <class F>
    void init(F&& iter_items)
    {
        iter_items([&](group_t r, const ovweight_t& w)
                   {
                       add_item(r, w);
                   });

        _N = _bvs.size();
    }

    bool add_bv_deg(group_t r, const deg_t& deg,  bv_t& bv, cdeg_t& cdeg) const
    {
        auto iter = std::lower_bound(bv.begin(), bv.end(), r);
        if (iter == bv.end() || *iter != r)
        {
            cdeg.insert(cdeg.begin() + (iter - bv.begin()), deg);
            bv.insert(iter, r);
            return false;
        }
        else
        {
            cdeg[iter - bv.begin()] += deg;
            return true;
        }
    }

    bool remove_bv_deg(group_t r, const deg_t& deg, bv_t& bv, cdeg_t& cdeg) const
    {
        auto iter = std::lower_bound(bv.begin(), bv.end(), r);
        auto pos = iter - bv.begin();
        cdeg[pos] -= deg;
        if (cdeg[pos] == 0)
        {
            bv.erase(iter);
            cdeg.erase(cdeg.begin() + pos);
            return false;
        }
        return true;
    }

    size_t get_count(auto&& k, const auto& m) const
    {
        auto iter = m.find(k);
        if (iter != m.end())
            return iter->second;
        return 0;
    }

    double get_S_d_term(size_t d, size_t nd, size_t B) const
    {
        if (nd == 0)
            return 0;
        double x = lbinom_fast(B, d);
        double S;
        if (x > 15)
            S = nd * x - lgamma_fast(nd + 1); // Stirling's approximation
        else
            S = lbinom_careful((exp(x) + nd) - 1, nd); // not fast
        return S;
    }

    double get_dl() const
    {
        double S = 0;

        for (auto& [d, nd] : _dhist)
            S += get_S_d_term(d, nd, _actual_B);

        S += lbinom_fast(_D + _N - 1, _N) + lgamma_fast(_N + 1);

        for (auto& bh : _bhist)
            S -= lgamma_fast(bh.second + 1);

        return S;
    }

    // get deg counts without increasing the container
    size_t get_deg_count(const bv_t& bv, const cdeg_t& deg) const
    {
        auto iter = _deg_hist.find(bv);
        if (iter == _deg_hist.end())
            return 0;
        auto& hist = iter->second;
        if (hist.empty())
            return 0;
        auto diter = hist.find(deg);
        if (diter == hist.end())
            return 0;
        return diter->second;
    }

    double get_delta_dl(group_t r, group_t nr, const ovweight_t& w = ovweight_t{})
    {
        if (r == nr)
            return 0;

        if constexpr (!scoped)
        {
            if (w.size() > 1)
            {
                OverlapPartitionStats<directed, true> pstats(*this);
                return pstats.get_delta_dl(r, nr, w);
            }
        }

        double dS = 0;

        for (const auto& [v, deg] : w)
        {
            auto& bv = _bvs[v];
            auto& cdeg = _degs[v];

            size_t d = bv.size();

            bv_t n_bv = bv;
            cdeg_t n_deg = cdeg;

            bool is_same_bv = true;
            if (r != _null_group)
                is_same_bv &= remove_bv_deg(r, deg, n_bv, n_deg);
            if (nr != _null_group)
                is_same_bv &= add_bv_deg(nr, deg, n_bv, n_deg);

            assert(n_bv.size() > 0);
            if (is_same_bv)
                continue;

            size_t n_d = n_bv.size();
            size_t n_D = _D;

            if (d == _D && n_d < d && get_count(d, _dhist) == 1 && _D > 1)
            {
                n_D = _D - 1;
                while (get_count(n_D, _dhist) == 0)
                {
                    if (n_D == 0)
                        break;
                    n_D--;
                }
            }

            n_D = std::max(n_D, n_d);

            double S_a = 0, S_b = 0;

            if (n_D != _D)
            {
                S_b += lbinom_fast(_D  + _N - 1, _N);
                S_a += lbinom_fast(n_D + _N - 1, _N);
            }

            ovweight_t vw;
            vw.insert({v, deg});

            int64_t dB = 0;
            if (r != _null_group && _wr[r] == vw)
                dB--;
            if (nr != _null_group && (_wr.find(nr) == _wr.end() || _wr[nr] == 0))
                dB++;

            auto get_S_d = [&] (size_t d_i, int64_t delta, int64_t dB) -> double
                {
                    int64_t nd = int64_t(get_count(d_i, _dhist)) + delta;
                    return get_S_d_term(d_i, nd, _actual_B + dB);
                };

            if (dB == 0)
            {
                if (n_d != d)
                {
                    S_b += get_S_d(d,  0, 0) + get_S_d(n_d, 0, 0);
                    S_a += get_S_d(d, -1, 0) + get_S_d(n_d, 1, 0);
                }
            }
            else
            {
                auto get_dS_di = [&](auto di)
                {
                    S_b += get_S_d(di, 0, 0);
                    if (d != n_d)
                    {
                        if (di == d)
                            S_a += get_S_d(d, -1, dB);
                        else if (di == n_d)
                            S_a += get_S_d(n_d, 1, dB);
                        else
                            S_a += get_S_d(di, 0, dB);
                    }
                    else
                    {
                        S_a += get_S_d(di, 0, dB);
                    }
                };

                for (auto [di, ndi] : _dhist)
                    get_dS_di(di);

                if (_dhist.find(n_d) == _dhist.end())
                    get_dS_di(n_d);
            }

            size_t bv_count = get_count(bv, _bhist);
            assert(bv_count > 0);
            size_t n_bv_count = get_count(n_bv, _bhist);

            auto get_S_b = [&] (bool is_bv, int64_t delta) -> double
                {
                    assert(int64_t(bv_count) + delta >= 0);
                    if (is_bv)
                        return -lgamma_fast(bv_count + delta + 1);
                    return -lgamma_fast(n_bv_count + delta + 1);
                };

            S_b += get_S_b(true,  0) + get_S_b(false, 0);
            S_a += get_S_b(true, -1) + get_S_b(false, 1);

            dS += S_a - S_b;

            if constexpr (scoped)
            {
                move_vitem(r, nr, v, deg);
                if (r != _null_group)
                    _wr[r] -= vw;
                if (nr != _null_group)
                    _wr[nr] += vw;
            }
        }

        return dS;
    }

    void move_vitem(group_t r, group_t nr, auto v, const auto& deg)
    {
        if (r == nr)
            return;

        const auto& bv = _bvs[v];
        const auto& cdeg = _degs[v];

        size_t d = bv.size();

        bv_t n_bv = bv;
        cdeg_t n_deg = cdeg;

        bool is_same_bv = true;
        if (r != _null_group)
            is_same_bv &= remove_bv_deg(r, deg, n_bv, n_deg);

        if (nr != _null_group)
            is_same_bv &= add_bv_deg(nr, deg, n_bv, n_deg);

        size_t n_d = n_bv.size();

        if (!is_same_bv)
        {
            if (d > 0)
            {
                if (n_d != d)
                {
                    _dhist[d] -= 1;

                    if (_dhist[d] == 0)
                        _dhist.erase(d);
                }

                auto& bv_count = _bhist[bv];
                bv_count -= 1;

                if (bv_count == 0)
                {
                    _bhist.erase(bv);
                    for (auto s : bv)
                    {
                        _r_count[s]--;
                        if (_r_count[s] == 0)
                        {
                            _actual_B--;
                            _r_count.erase(s);
                        }
                    }
                }

                if (d == _D && get_count(d, _dhist) == 0 && _D > 1)
                {
                    _D--;
                    while (get_count(_D, _dhist) == 0)
                    {
                        if (_D == 0)
                            break;
                        _D--;
                    }
                }
            }

            assert(!n_bv.empty() || nr == _null_group);

            if (n_d > 0)
            {
                if (n_d != d)
                    _dhist[n_d] += 1;

                auto& n_bv_count = _bhist[n_bv];
                n_bv_count += 1;

                if (n_bv_count == 1)
                {
                    for (auto s : n_bv)
                    {
                        if (_r_count[s] == 0)
                            _actual_B++;
                        _r_count[s]++;
                    }
                }

                if (n_d > _D)
                    _D = n_d;
            }
        }

        auto kin = get<0>(deg);
        auto kout = get<1>(deg);

        if (d > 0)
        {
            auto& deg_h = _deg_hist[bv];
            auto& deg_count = deg_h[cdeg];

            assert(deg_count > 0);

            auto& bmh = _embhist[bv];
            auto& bph = _epbhist[bv];
            assert(bmh.size() == bv.size());
            assert(bph.size() == bv.size());

            deg_count -= 1;
            if (deg_count == 0)
                deg_h.erase(cdeg);

            for (size_t i = 0; i < bv.size(); ++i)
            {
                bmh[i] -= get<0>(cdeg[i]);
                bph[i] -= get<1>(cdeg[i]);
            }

            if (deg_h.empty())
            {
                _deg_hist.erase(bv);
                _embhist.erase(bv);
                _epbhist.erase(bv);
            }
        }

        if (n_d > 0)
        {
            auto& ndeg_h = _deg_hist[n_bv];
            auto& ndeg_count = ndeg_h[n_deg];

            ndeg_count += 1;

            assert(ndeg_count > 0);

            auto& nbmh = _embhist[n_bv];
            auto& nbph = _epbhist[n_bv];

            nbmh.resize(n_bv.size());
            nbph.resize(n_bv.size());

            for (size_t i = 0; i < n_bv.size(); ++i)
            {
                nbmh[i] += get<0>(n_deg[i]);
                nbph[i] += get<1>(n_deg[i]);
            }
        }

        if (r != _null_group)
        {
            _emhist[r] -= kin;
            _ephist[r] -= kout;
        }

        if (nr != _null_group)
        {
            _emhist[nr] += kin;
            _ephist[nr] += kout;
        }

        _bvs[v] = n_bv;
        _degs[v] = n_deg;
        assert(_bvs[v].size() > 0 || nr == _null_group);
    }

    void move_item(group_t r, group_t nr, const ovweight_t& w)
    {
        for (const auto& [v, deg] : w)
            move_vitem(r, nr, v, deg);
        if (r != _null_group)
            _wr[r] -= w;
        if (nr != _null_group)
            _wr[nr] += w;
    }

    void add_item(group_t r, const ovweight_t& w)
    {
        #pragma omp critical (overlap_partition) // FIXME: implement concurrency
        move_item(_null_group, r, w);
    }

    void remove_item(group_t r, const ovweight_t& w)
    {
        #pragma omp critical (overlap_partition) // FIXME: implement concurrency
        move_item(r, _null_group, w);
    }

    size_t get_actual_B()
    {
        return _actual_B;
    }

    void add_group()
    {
    }

protected:
    friend OPS<!scoped>;

    size_t _N = 0;
    size_t _actual_B = 0;
    size_t _D = 0;

    template <class M>
    using sm = std::conditional_t<scoped,
                                  scoped_hash_map<M>,
                                  M>;

    sm<gt_hash_map<size_t, int64_t>> _dhist;    // d-histogram
    sm<gt_hash_map<size_t, int64_t>> _r_count;  // m_r
    sm<bhist_t> _bhist;                         // b-histogram
    sm<gt_hash_map<size_t, size_t>> _emhist;    // e-_r histogram
    sm<gt_hash_map<size_t, size_t>> _ephist;    // e+_r histogram
    sm<ebhist_t> _embhist;                      // e+^r_b histogram
    sm<ebhist_t> _epbhist;                      // e+^r_b histogram
    sm<deg_hist_t> _deg_hist;                   // n_k^b histogram
    sm<gt_hash_map<size_t, bv_t>> _bvs;         // bv node map
    sm<gt_hash_map<size_t, cdeg_t>> _degs;      // deg node map
    sm<gt_hash_map<size_t, ovweight_t>> _wr;    // group compositions

    constexpr static group_t _null_group = std::numeric_limits<group_t>::max();
};


template <bool directed, bool scoped=false>
struct OverlapDegStats
    : public OverlapPartitionStats<directed, scoped>
{
    typedef OverlapPartitionStats<directed, scoped> base_t;

    typedef base_t::group_t group_t;
    typedef base_t::deg_t deg_t;
    typedef base_t::cdeg_t cdeg_t;
    typedef base_t::bv_t bv_t;

    using base_t::get_count;

    OverlapDegStats() = default;

    template <bool s>
    OverlapDegStats(const OverlapDegStats<directed, s>& o)
        : base_t(o)
    {}

    template <class F>
    void init(F&& iter_items)
    {
        base_t::init([&](auto&& f)
                     {
                         iter_items
                             ([&](auto&&, group_t r, auto w)
                              {
                                  f(r, w);
                              });
                     });
    }

    double get_deg_dl_uniform() const
    {
        double S = 0;

        for (auto& [bv, n_bv] : _bhist)
        {
            if (n_bv == 0)
                continue;

            const auto& bmh = _embhist.find(bv)->second;
            const auto& bph = _epbhist.find(bv)->second;

            for (size_t i = 0; i < bv.size(); ++i)
            {
                S += lbinom_fast(n_bv + bmh[i] - 1, bmh[i]);
                S += lbinom_fast(n_bv + bph[i] - 1, bph[i]);
            }
        }

        for (auto& [r, wr] : _r_count)
        {
            if (wr == 0)    // FIXME: erase zeros
                continue;
            auto emh = get_count(r, _emhist);
            auto eph = get_count(r, _ephist);
            S += lbinom_fast(wr + emh - 1,  emh);
            S += lbinom_fast(wr + eph - 1,  eph);
        }

        return S;
    }

    double get_deg_dl_dist() const
    {
        double S = 0;
        for (auto& ch : _deg_hist)
        {
            auto& bv = ch.first;
            auto& cdeg_hist = ch.second;

            size_t n_bv = _bhist.find(bv)->second;

            if (n_bv == 0)
                continue;

            const auto& bmh = _embhist.find(bv)->second;
            const auto& bph = _epbhist.find(bv)->second;

            for (size_t i = 0; i < bv.size(); ++i)
            {
                if constexpr (directed)
                {
                    S += log_q(bmh[i], n_bv);
                    S += log_q(bph[i], n_bv);
                }
                else
                {
                    S += log_q(bph[i] - n_bv, n_bv);
                }
            }

            S += lgamma_fast(n_bv + 1);

            for (auto& dh : cdeg_hist)
                S -= lgamma_fast(dh.second + 1);
        }

        // FIXME: deduplicate!
        for (auto& [r, wr] : _r_count)
        {
            if (wr == 0)    // FIXME: erase zeros
                continue;
            auto emh = base_t::get_count(r, _emhist);
            auto eph = base_t::get_count(r, _ephist);
            S += lbinom_fast(wr + emh - 1,  emh);
            S += lbinom_fast(wr + eph - 1,  eph);
        }

        return S;
    }

    double get_dl(deg_dl_kind kind) const
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

    void add_vertex(auto&&, group_t r, const ovweight_t& w)
    {
        base_t::add_item(r, w);
    }

    void remove_vertex(auto&&, group_t r, const ovweight_t& w)
    {
        base_t::remove_item(r, w);
    }

    double get_delta_dl_ecount(const bv_t& bv, const bv_t& n_bv,
                               size_t bv_count, size_t n_bv_count,
                               group_t r, group_t nr, auto&& deg)
    {
        double S_a = 0, S_b = 0;

        auto is_in = [&] (const bv_t& bv, group_t r) -> bool
            {
                auto iter = lower_bound(bv.begin(), bv.end(), r);
                if (iter == bv.end())
                    return false;
                if (*iter != r)
                    return false;
                return true;
            };

        for (group_t s : bv)
        {
            auto ws = get_count(s, _r_count);
            auto ems = get_count(s, _emhist);
            auto eps = get_count(s, _ephist);
            S_b += lbinom_fast(ws + ems - 1, ems);
            S_b += lbinom_fast(ws + eps - 1, eps);
        }

        for (group_t s : n_bv)
        {
            if (is_in(bv, s))
                continue;
            auto ws = get_count(s, _r_count);
            auto ems = get_count(s, _emhist);
            auto eps = get_count(s, _ephist);
            S_b += lbinom_fast(ws + ems - 1, ems);
            S_b += lbinom_fast(ws + eps - 1, eps);
        }

        gt_hash_map<size_t, pair<int64_t, int64_t>> deg_delta;
        gt_hash_map<size_t, int64_t> r_count_delta;

        if (bv != n_bv)
        {
            if (n_bv_count == 0)
            {
                for (auto s : n_bv)
                    r_count_delta[s] += 1;
            }

            if (bv_count == 1)
            {
                for (auto s : bv)
                   r_count_delta[s] -= 1;
            }
        }

        if (r != nr)
        {
            auto kin = get<0>(deg);
            auto kout = get<1>(deg);

            auto& d_r = deg_delta[r];
            d_r.first -= kin;
            d_r.second -= kout;

            auto& d_nr = deg_delta[nr];
            d_nr.first += kin;
            d_nr.second += kout;
        }

        for (group_t s : bv)
        {
            auto ws = get_count(s, _r_count);
            auto ems = get_count(s, _emhist);
            auto eps = get_count(s, _ephist);

            S_a += lbinom_fast(ws + r_count_delta[s] + ems + deg_delta[s].first - 1,
                               ems+ deg_delta[s].first);
            S_a += lbinom_fast(ws + r_count_delta[s] + eps + deg_delta[s].second - 1,
                               eps + deg_delta[s].second);
        }

        for (group_t s : n_bv)
        {
            if (!is_in(bv, s))
            {
                auto ws = get_count(s, _r_count);
                auto ems = get_count(s, _emhist);
                auto eps = get_count(s, _ephist);

                S_a += lbinom_fast(ws + r_count_delta[s] + ems + deg_delta[s].first - 1,
                                   ems + deg_delta[s].first);
                S_a += lbinom_fast(ws + r_count_delta[s] + eps + deg_delta[s].second - 1,
                                   eps + deg_delta[s].second);
            }
        }

        return S_a - S_b;
    }

    double get_delta_dl_uniform(auto&& degs, group_t r, group_t nr, const ovweight_t& w)
    {
        if (r == nr)
            return 0;

        if constexpr (!scoped)
        {
            if (w.size() > 1)
            {
                OverlapDegStats<directed, true> dstats(*this);
                return dstats.get_delta_dl_uniform(degs, r, nr, w);
            }
        }

        double dS = 0;

        for (const auto& [v, deg] : w)
        {
            double S_b = 0, S_a = 0;

            const bv_t& bv = _bvs.find(v)->second;
            bv_t n_bv = bv;

            const cdeg_t& cdeg = _degs.find(v)->second;
            cdeg_t n_deg = cdeg;

            bool is_same_bv = true;
            is_same_bv &= base_t::remove_bv_deg(r, deg, n_bv, n_deg);
            is_same_bv &= base_t::add_bv_deg(nr, deg, n_bv, n_deg);

            size_t bv_count = base_t::get_count(bv, _bhist);
            size_t n_bv_count = bv_count;

            if (!is_same_bv)
                n_bv_count = base_t::get_count(n_bv, _bhist);

            auto get_S_bv =
                [&](const auto& bv, auto bv_count, auto& deg, auto delta, auto deg_delta)
                {
                    double S = 0;
                    if (bv_count == 0)
                    {
                        if (delta > 0)
                        {
                            for (size_t i = 0; i < bv.size(); ++i)
                            {
                                S += lbinom_fast(delta + get<0>(deg[i]) * deg_delta - 1, get<0>(deg[i]) * deg_delta);
                                S += lbinom_fast(delta + get<1>(deg[i]) * deg_delta - 1, get<1>(deg[i]) * deg_delta);
                            }
                        }
                    }
                    else
                    {
                        const auto& bmh = _embhist.find(bv)->second;
                        const auto& bph = _epbhist.find(bv)->second;
                        for (size_t i = 0; i < bv.size(); ++i)
                        {
                            S += lbinom_fast(bv_count + delta + bmh[i] + get<0>(deg[i]) * deg_delta - 1, bmh[i] + get<0>(deg[i]) * deg_delta);
                            S += lbinom_fast(bv_count + delta + bph[i] + get<1>(deg[i]) * deg_delta - 1, bph[i] + get<1>(deg[i]) * deg_delta);
                        }
                    }
                    return S;
                };

            if (!is_same_bv)
            {
                S_b += get_S_bv(bv,   bv_count,    cdeg,  0,  0) +
                       get_S_bv(n_bv, n_bv_count, n_deg,  0,  0);
                S_a += get_S_bv(bv,   bv_count,    cdeg, -1, -1) +
                       get_S_bv(n_bv, n_bv_count, n_deg , 1,  1);
            }
            else
            {
                S_b += get_S_bv(bv, bv_count, cdeg, 0, 0);
                auto ddeg = n_deg;
                for (size_t i = 0; i < bv.size(); ++i)
                {
                    get<0>(ddeg[i]) -= get<0>(cdeg[i]);
                    get<1>(ddeg[i]) -= get<1>(cdeg[i]);
                }
                S_a += get_S_bv(bv, bv_count, ddeg, 0, 1);
            }

            dS += S_a - S_b;

            dS += get_delta_dl_ecount(bv, n_bv, bv_count, n_bv_count, r, nr, deg);

            if constexpr (scoped)
                base_t::move_vitem(r, nr, v, deg);
        }

        return dS;
    }

    double get_delta_dl_dist(auto&& degs, group_t r, group_t nr, const ovweight_t& w)
    {
        if (r == nr)
            return 0;

        if constexpr (!scoped)
        {
            if (w.size() > 1)
            {
                OverlapDegStats<directed, true> dstats(*this);
                return dstats.get_delta_dl_dist(degs, r, nr, w);
            }
        }

        double dS = 0;

        for (const auto& [v, deg] : w)
        {
            double S_b = 0, S_a = 0;

            const bv_t& bv = _bvs.find(v)->second;
            bv_t n_bv = bv;

            const cdeg_t& cdeg = _degs.find(v)->second;
            cdeg_t n_deg = cdeg;

            bool is_same_bv = true;
            is_same_bv &= base_t::remove_bv_deg(r, deg, n_bv, n_deg);
            is_same_bv &= base_t::add_bv_deg(nr, deg, n_bv, n_deg);
            size_t bv_count = base_t::get_count(bv, _bhist);
            size_t n_bv_count = bv_count;

            auto get_S_bv = [&] (bool is_bv, int64_t delta) -> double
                {
                    if (is_bv)
                        return lgamma_fast(bv_count + delta + 1);
                    return lgamma_fast(n_bv_count + delta + 1);
                };

            auto get_S_e = [&] (bool is_bv, int64_t bdelta, int64_t deg_delta) -> double
                {
                    size_t bv_c = ((is_bv) ? bv_count : n_bv_count) + bdelta;
                    if (bv_c == 0)
                        return 0.;

                    const cdeg_t& deg_i = (is_bv) ? cdeg : n_deg;
                    const auto& bv_i = (is_bv) ? bv : n_bv;

                    double S = 0;
                    if (((is_bv) ? bv_count : n_bv_count) > 0)
                    {
                        const auto& bmh = _embhist.find(bv_i)->second;
                        const auto& bph = _epbhist.find(bv_i)->second;

                        assert(bmh.size() == bv_i.size());
                        assert(bph.size() == bv_i.size());

                        for (size_t i = 0; i < bv_i.size(); ++i)
                        {
                            if constexpr (directed)
                            {
                                S += log_q(size_t(bmh[i] + deg_delta * int64_t(get<0>(deg_i[i]))), bv_c);
                                S += log_q(size_t(bph[i] + deg_delta * int64_t(get<1>(deg_i[i]))), bv_c);
                            }
                            else
                            {
                                S += log_q(size_t(bph[i] + deg_delta * int64_t(get<1>(deg_i[i]))) - bv_c, bv_c);
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bv_i.size(); ++i)
                        {
                            if constexpr (directed)
                            {
                                S += log_q(size_t(deg_delta * int64_t(get<0>(deg_i[i]))), bv_c);
                                S += log_q(size_t(deg_delta * int64_t(get<1>(deg_i[i]))), bv_c);
                            }
                            else
                            {
                                S += log_q(size_t(deg_delta * int64_t(get<1>(deg_i[i]))) - bv_c, bv_c);
                            }
                        }
                    }

                    return S;
                };

            auto get_S_e2 = [&] (int64_t deg_delta, int64_t ndeg_delta) -> double
                {
                    double S = 0;

                    const auto& bmh = _embhist.find(bv)->second;
                    const auto& bph = _epbhist.find(bv)->second;

                    for (size_t i = 0; i < bv.size(); ++i)
                    {
                        if constexpr (directed)
                        {
                            S += log_q(size_t(bmh[i] +
                                              deg_delta * int64_t(get<0>(cdeg[i])) +
                                              ndeg_delta * int64_t(get<0>(n_deg[i]))),
                                       bv_count);
                            S += log_q(size_t(bph[i] +
                                              deg_delta * int64_t(get<1>(cdeg[i])) +
                                              ndeg_delta * int64_t(get<1>(n_deg[i]))),
                                       bv_count);
                        }
                        else
                        {
                            S += log_q(size_t(bph[i] +
                                              deg_delta * int64_t(get<1>(cdeg[i])) +
                                              ndeg_delta * int64_t(get<1>(n_deg[i])))
                                       - bv_count,
                                       bv_count);
                        }
                    }
                    return S;
                };

            if (!is_same_bv)
            {
                n_bv_count = base_t::get_count(n_bv, _bhist);

                S_b += get_S_bv(true,  0) + get_S_bv(false, 0);
                S_a += get_S_bv(true, -1) + get_S_bv(false, 1);

                S_b += get_S_e(true,  0,  0) + get_S_e(false, 0, 0);
                S_a += get_S_e(true, -1, -1) + get_S_e(false, 1, 1);
            }
            else
            {
                S_b += get_S_e2( 0, 0);
                S_a += get_S_e2(-1, 1);
            }

            size_t deg_count = base_t::get_deg_count(bv, cdeg);
            size_t n_deg_count = base_t::get_deg_count(n_bv, n_deg);

            auto get_S_deg = [&] (bool is_deg, int64_t delta) -> double
                {
                    if (is_deg)
                        return -lgamma_fast(deg_count + delta + 1);
                    return -lgamma_fast(n_deg_count + delta + 1);
                };

            S_b += get_S_deg(true,  0) + get_S_deg(false, 0);
            S_a += get_S_deg(true, -1) + get_S_deg(false, 1);

            dS += S_a - S_b;

            dS += get_delta_dl_ecount(bv, n_bv, bv_count, n_bv_count, r, nr, deg);

            if constexpr (scoped)
                base_t::move_vitem(r, nr, v, deg);
        }
        return dS;
    }

    double get_delta_dl(auto&& d, group_t r, group_t nr, deg_dl_kind kind,
                        const ovweight_t& w)
    {
        switch (kind)
        {
        case deg_dl_kind::UNIFORM:
            return get_delta_dl_uniform(d, r, nr, w);
        case deg_dl_kind::DIST:
            return get_delta_dl_dist(d, r, nr, w);
        default:
            return numeric_limits<double>::quiet_NaN();
        }
    }

    void add_group(size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            base_t::add_group();
    }

    void merge_groups(group_t s, group_t t)
    {
        for (auto& [v, deg] : _wr[s])
            base_t::move_vitem(s, t, v, deg);

        _wr[t] += _wr[s];
        _wr[s] = 0;
    }

    double get_merge_dl(deg_dl_kind kind, group_t r, group_t nr)
    {
        return get_delta_dl(std::monostate(), r, nr, kind, _wr[r]);
    }

private:
    using base_t::_deg_hist;
    using base_t::_bhist;
    using base_t::_embhist;
    using base_t::_epbhist;
    using base_t::_emhist;
    using base_t::_ephist;
    using base_t::_r_count;
    using base_t::_bvs;
    using base_t::_degs;
    using base_t::_wr;
};


} // namespace graph_tool

#endif // GRAPH_OVERLAP_BLOCKMODEL_PARTITION_HH



