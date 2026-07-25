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

#ifndef PARTITION_HH
#define PARTITION_HH

#include "../../graph_util.hh"
#include "../support/util.hh"

#include <vector>

namespace graph_tool
{

// ===============
// Partition stats
// ===============

template <class Group = size_t>
class PartitionStats
{
public:
    typedef Group group_t;

    PartitionStats() = default;

    size_t get_count(group_t r)
    {
        if (std::cmp_less(r, _count.size()))
            return _count[r];
        return 0;
    }

    size_t get_actual_B()
    {
        return _actual_B;
    }

    size_t get_B()
    {
        return _count.size();
    }

    size_t get_N()
    {
        return _N;
    }

    void add_group(size_t n = 1)
    {
        if (n == 1)
            _count.push_back(0);
        else
            soft_resize(_count, _count.size() + n);
    }

    // concurrent
    template <bool check = false>
    void add_item(group_t r, size_t w = 1)
    {
        if constexpr (check)
        {
            if (std::cmp_greater_equal(r, _count.size()))
                add_group(r - _count.size() + 1);
        }

        if (_count[r] == 0 && w > 0)
        {
            #pragma omp atomic
            _actual_B++;
        }
        _count[r] += w;
        #pragma omp atomic
        _N += w;
        assert(_count[r] >= 0);
    }

    // concurrent
    void remove_item(group_t r, size_t w = 1)
    {
        assert(_count[r] >= w);
        if (_count[r] == w && w > 0)
        {
            #pragma omp atomic
            _actual_B--;
        }
        _count[r] -= w;
        #pragma omp atomic
        _N -= w;
    }

    double get_dl()
    {
        if (_N == 0)
            return 0;

        double S = 0;
        S += lbinom_fast(_N - 1, _actual_B - 1);
        S += lgamma_fast(_N + 1);

        for (auto nr : _count)
            S -= lgamma_fast(nr + 1);

        S += safelog_fast(_N);

        return S;
    }

    double get_delta_dl(group_t r, group_t nr, size_t w = 1)
    {
        if (r == nr || w == 0)
            return 0;

        double S_b = 0, S_a = 0;

        if (r != _null_group)
        {
            auto wr = get_count(r);
            assert(wr >= w);
            S_b += -lgamma_fast(wr + 1);
            S_a += -lgamma_fast(wr - w + 1);
        }

        if (nr != _null_group)
        {
            auto wnr = get_count(nr);
            S_b += -lgamma_fast(wnr + 1);
            S_a += -lgamma_fast(wnr + w + 1);
        }

        int64_t dN = 0;
        if (r == _null_group)
            dN += w;
        if (nr == _null_group)
            dN -= w;

        int64_t dB = 0;
        if (r != _null_group && get_count(r) == w)
            dB--;
        if (nr != _null_group && get_count(nr) == 0)
            dB++;

        if (dN != 0 || dB != 0)
        {
            S_b += lbinom_fast(_N - 1, _actual_B - 1);
            S_a += lbinom_fast(_N - 1 + dN, _actual_B + dB - 1);
        }

        if (dN != 0)
        {
            S_b += lgamma_fast(_N + 1);
            S_a += lgamma_fast(_N + dN + 1);
            S_b += safelog_fast(_N);
            S_a += safelog_fast(_N + dN);
        }

        return S_a - S_b;
    }

    static constexpr group_t _null_group = std::numeric_limits<group_t>::max();

private:
    size_t _N = 0;
    size_t _actual_B = 0;
    size_t _B = 0;
    std::vector<size_t> _count;
};

template <class PStats>
class CPartitionStats
{
public:
    typedef PStats::group_t group_t;

    void add_group(int64_t c, size_t n = 1)
    {
        _ps[c].add_group(n);
    }

    template <bool check = false>
    void add_item(group_t r, int64_t c, auto&&... w)
    {
        if constexpr (check)
        {
            if (std::cmp_greater_equal(c, _ps.size()))
            {
                soft_resize(_ps, c + 1);
                soft_resize(_empty_labels, c + 1);
            }
            mod_r<true>(r, c);
        }

        _ps[c].template add_item<check>(get_r(r, c), w...);
    }

    template <bool check = false>
    void remove_item(group_t r, int64_t c, auto&&... w)
    {
        auto s = get_r(r, c);
        _ps[c].remove_item(s, w...);
        if constexpr (check && requires {_ps[c].get_count(s);})
        {
            if (_ps[c].get_count(s) == 0)
                mod_r<false>(r, c);
        }
    }

    double get_dl()
    {
        double S = 0;
        for (auto& ps : _ps)
            S += ps.get_dl();
        return S;
    }

    double get_delta_dl(group_t r, group_t nr, int64_t c, auto&&... w)
    {
        return _ps[c].get_delta_dl(get_r(r, c), get_r(nr, c), w...);
    }

    size_t get_actual_B()
    {
        size_t B = 0;
        for (auto& ps : _ps)
            B += ps.get_actual_B();
        return B;
    }

    static constexpr group_t _null_group = PStats::_null_group;

private:

    group_t get_r(group_t r, int64_t c)
    {
        if (r == _null_group)
            return r;
        auto& rlabel = _r_label[c];
        auto iter = rlabel.find(r);
        if (iter != rlabel.end())
            return iter->second;
        return get_empty(c);
    }

    template <bool pop=false>
    group_t get_empty(int64_t c)
    {
        auto& el = _empty_labels[c];
        if (el.empty())
            return _r_label[c].size();
        auto s = el.back();
        if constexpr (pop)
            el.pop_back();
        return s;
    }

    template <bool add>
    auto mod_r(group_t r, int64_t c)
    {
        auto& ls = _r_label[c];
        auto iter = ls.find(r);
        auto& el = _empty_labels[c];
        if constexpr (add)
        {
            if (iter == ls.end())
                return ls[r] = get_empty<true>(c);
            else
                return iter->second;
        }
        else
        {
            if (iter != ls.end())
            {
                el.push_back(iter->second);
                ls.erase(iter);
            }
        }
    }

    std::vector<PStats> _ps;
    idx_map<group_t, idx_map<group_t, group_t, {.shared_pos=true}>,
            {.shared_tpos=true}> _r_label;
    std::vector<std::vector<group_t>> _empty_labels;
};


} //namespace graph_tool

#endif // PARTITION_HH
