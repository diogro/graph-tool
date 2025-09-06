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

#ifndef PARTITION_HH
#define PARTITION_HH

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

    template <class IterItems>
    PartitionStats(IterItems&& iter_items)
    {
        init(std::forward<IterItems>(iter_items));
    }

    template <class IterItems>
    void init(IterItems&& iter_items)
    {
        size_t B = 0;
        iter_items([&](size_t r, size_t w = 1)
                   {
                       if (r >= _count.size())
                           _count.resize(std::max(r + 1, 2 * _count.size()));
                       B = std::max(r + 1, B);
                       _count[r] += w;
                       _N += w;
                   });

        _count.resize(B);

        for (auto n : _count)
        {
            if (n > 0)
                _actual_B++;
        }
    }

    size_t get_count(group_t r)
    {
        if (size_t(r) < _count.size())
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
            _count.resize(_count.size() + n, 0);
    }

    // concurrent
    void add_item(group_t r, size_t w = 1)
    {
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

    double get_delta_dl(size_t r, size_t nr, size_t w = 1)
    {
        if (r == nr || w == 0)
            return 0;

        double S_b = 0, S_a = 0;

        if (r != null_group)
        {
            assert(_count[r] >= w);
            S_b += -lgamma_fast(_count[r] + 1);
            S_a += -lgamma_fast(_count[r] - w + 1);
        }

        if (nr != null_group)
        {
            S_b += -lgamma_fast(_count[nr] + 1);
            S_a += -lgamma_fast(_count[nr] + w + 1);
        }

        int64_t dN = 0;
        if (r == null_group)
            dN += w;
        if (nr == null_group)
            dN -= w;

        int64_t dB = 0;
        if (r != null_group && _count[r] == w)
            dB--;
        if (nr != null_group && _count[nr] == 0)
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

    static constexpr group_t null_group = std::numeric_limits<group_t>::max();

private:
    size_t _N = 0;
    size_t _actual_B = 0;
    size_t _B = 0;
    std::vector<size_t> _count;
};

} //namespace graph_tool

#endif // PARTITION_HH
