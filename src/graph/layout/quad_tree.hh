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

#include <array>
#include <vector>
#include <tuple>
#include <cmath>
#include <bit>
#include <cstdint>
#include <ranges>
#include <span>

#include "parallel_util.hh"

namespace graph_tool
{

template <Indexable X, Indexable Y>
auto& operator+=(X& x, const Y& y)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] += y[i];
    return x;
}

template <Indexable X, Indexable Y>
auto& operator-=(X& x, const Y& y)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] -= y[i];
    return x;
}

template <Indexable X, Indexable Y>
auto& operator*=(X& x, const Y& y)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] *= y[i];
    return x;
}

template <Indexable X, Indexable Y>
auto& operator/=(X& x, const Y& y)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] /= y[i];
    return x;
}

template <Indexable X, Scalar Y>
auto& operator+=(X& x, Y c)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] += c;
    return x;
}

template <Indexable X, Scalar Y>
auto& operator-=(X& x, Y c)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] -= c;
    return x;
}

template <Indexable X, Scalar Y>
auto& operator*=(X& x, Y c)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] *= c;
    return x;
}

template <Indexable X, Scalar Y>
auto& operator/=(X& x, Y c)
{
    for (size_t i = 0; i < 2; ++i)
        x[i] /= c;
    return x;
}

template <Indexable X, class Y>
auto operator+(const X& x, const Y& y)
{
    X z(x);
    z += y;
    return z;
}

template <Indexable X, class Y>
auto operator-(const X& x, const Y& y)
{
    X z(x);
    z -= y;
    return z;
}

template <Indexable X, class Y>
auto operator*(const X& x, const Y& y)
{
    X z(x);
    z *= y;
    return z;
}

template <Indexable X, class Y>
auto operator/(const X& x, const Y& y)
{
    X z(x);
    z /= y;
    return z;
}


template <Indexable X, Indexable Y>
bool operator==(const X& x, const Y& y)
{
    for (size_t i = 0; i < 2; ++i)
    {
        if (x[i] != y[i])
            return false;
    }
    return true;
}

template <bool enable=true, class T, size_t N, Indexable D>
decltype(auto) atomic_inc(std::array<T, N>& x, D&& d)
{
    for (size_t i = 0; i < N; ++i)
        atomic_inc<enable>(x[i], d[i]);
    return x;
}

template <class Val, class Weight>
class QuadTree
{
public:

    typedef typename std::array<Val, 2> pos_t;

    QuadTree() = default;

    template <class... As>
    QuadTree(As&&... args)
    {
        init(std::forward<As>(args)...);
    }

    uint32_t init(auto&& pos_range,
                  uint32_t depth = 0,
                  uint32_t max_depth = 32,
                  bool square = true)
    {
        bool parallel = std::ranges::size(pos_range) > get_openmp_min_thresh();

        Val xmin = std::numeric_limits<Val>::max();
        Val xmax = std::numeric_limits<Val>::lowest();
        Val ymin = std::numeric_limits<Val>::max();
        Val ymax = std::numeric_limits<Val>::lowest();
        size_t N = 0;

        #pragma omp parallel if (parallel) \
            reduction(min: xmin, ymin) reduction(max: xmax, ymax) reduction(+:N)
        parallel_loop_no_spawn
            (pos_range,
             [&](auto, auto&& xw)
             {
                 auto&& [x, w] = xw;
                 xmin = std::min(x[0], xmin);
                 xmax = std::max(x[0], xmax);
                 ymin = std::min(x[1], ymin);
                 ymax = std::max(x[1], ymax);
                 N++;
             });

        _pmin = {xmin, ymin};
        _L = {xmax - xmin, ymax - ymin};

        if (square)
            _L = {std::max(_L[0], _L[1]),
                  std::max(_L[0], _L[1])};

        // a zero extent (e.g. a single point) maps every point to cell 0
        _inv_L = {_L[0] > 0 ? 1. / _L[0] : 0., _L[1] > 0 ? 1. / _L[1] : 0.};

        if (depth == 0 && N > 1)
            depth = uint32_t(std::ceil(std::ceil(std::log2(N)) / 2));
        // the cells of deeper levels would overflow the 64-bit indices
        _depth = std::min({depth, max_depth, uint32_t(30)});

        _tree.resize(level_offset(_depth + 1));

        // Only the cells of the bottom level are accumulated atomically, and
        // those of the levels above are obtained by summing their children,
        // since otherwise all points would contend for the top cells.
        size_t thresh = parallel ? get_openmp_min_thresh() :
            std::numeric_limits<size_t>::max();
        parallel_loop(level(_depth),
                      [&](auto, auto idx) { _tree[idx] = tree_node{}; },
                      thresh);

        parallel_loop
            (pos_range,
             [&](auto, auto&& xw)
             {
                 auto&& [x, w] = xw;
                 auto& t = _tree[index_of(x, _depth)];
                 atomic_inc(t.cm, x * w);
                 atomic_inc(t.w, w);
                 #pragma omp atomic
                 t.count++;
             }, thresh);

        for (int l = int(_depth) - 1; l >= 0; --l)
        {
            parallel_loop
                (level(l),
                 [&](auto, auto idx)
                 {
                     tree_node t{};
                     for (uint64_t k = 1; k <= 4; ++k)
                     {
                         auto& c = _tree[4 * idx + k];
                         t.cm += c.cm;
                         t.w += c.w;
                         t.count += c.count;
                     }
                     _tree[idx] = t;
                 }, thresh);
        }

        // the points of every cell are contiguous in _points, from its
        // offset on, in the order of the cells
        _tree[0].offset = 0;
        for (uint32_t l = 0; l < _depth; ++l)
        {
            parallel_loop
                (level(l),
                 [&](auto, auto idx)
                 {
                     uint32_t offset = _tree[idx].offset;
                     for (uint64_t k = 1; k <= 4; ++k)
                     {
                         auto& c = _tree[4 * idx + k];
                         c.offset = offset;
                         offset += c.count;
                     }
                 }, thresh);
        }

        uint32_t max_count = 0;
        #pragma omp parallel if (parallel) reduction(max: max_count)
        parallel_loop_no_spawn
            (_tree,
             [&](auto idx, auto& t)
             {
                 if (t.w != Weight{})
                     t.cm /= t.w;
                 if (idx >= level_offset(_depth))
                     max_count = std::max(t.count, max_count);
             });

        auto leaves = level(_depth);
        _fill.resize(leaves.size());
        parallel_loop(leaves,
                      [&](auto i, auto) { _fill[i] = 0; },
                      thresh);

        _points.resize(N);
        _order.resize(N);
        parallel_loop
            (pos_range,
             [&](auto i, auto&& xw)
             {
                 auto&& [x, w] = xw;
                 auto idx = index_of(x, _depth);
                 uint32_t p;
                 #pragma omp atomic capture
                 p = _fill[idx - leaves.front()]++;
                 auto k = _tree[idx].offset + p;
                 auto& [lx, lw] = _points[k];
                 lx[0] = x[0];
                 lx[1] = x[1];
                 lw = w;
                 _order[k] = i;
             }, thresh);

        return max_count;
    }

    void barnes_hut_init(double theta)
    {
        double s0 = std::max(_L[0], _L[1]);
        for (uint32_t l = 0; l <= _depth; ++l)
        {
            double s = s0 / (1ULL << l);
            _s2[l] = (s * s) / (theta * theta);
        }
    }

    void barnes_hut(auto&& x, auto&& visit)
    {
        uint64_t p = 0;
        uint32_t l = 0;

        while (true)
        {
            auto& t = _tree[p];
            if (t.w != 0)
            {
                double dx = t.cm[0] - x[0];
                double dy = t.cm[1] - x[1];
                double d2 = dx*dx + dy*dy;

                // s/d < theta, squared; a cell of zero extent is always
                // opened, since otherwise a point would interact with its own
                // (rounded) center of mass when all points coincide
                if (_s2[l] > 0 && _s2[l] < d2)
                {
                    visit(t.cm, t.w, p, l);
                }
                else if (l == _depth)
                {
                    for (auto&& [cm, w] : leaf_points(t))
                        visit(cm, w, p, l);
                }
                else                     // open subtree
                {
                    p = 4 * p + 1;
                    ++l;
                    continue;
                }
            }

            // move along

            while (p != 0 && (p & 3) == 0)
            {                                // ascend past last children
                p = (p - 1) >> 2;
                --l;
            }

            if (p == 0)
                break;                       // back at root: done

            ++p;                             // next sibling
        }
    }

    // indices of the points (in the range given to init()) in the order of
    // the cells, so that consecutive points are close
    const std::vector<size_t>& order()
    {
        return _order;
    }

    // idx = (4^l - 1) / 3 + morton(i, j);
    // i = floor(x 2^l / W); j = floor(y 2^l / H)
    uint64_t index_of(auto&& x, uint32_t l)
    {
        int64_t n = 1LL << l;
        uint64_t i = std::min(n - 1,
                              int64_t((x[0] - _pmin[0]) * (n * _inv_L[0])));
        uint64_t j = std::min(n - 1,
                              int64_t((x[1] - _pmin[1]) * (n * _inv_L[1])));
        return level_offset(l) + interleave_bits(i, j);
    }

    // l = floor(log_4(3p + 1)), m = p - (4^l-1)/3
    auto pos_of(uint64_t idx)
    {
        uint32_t l = uint32_t(std::bit_width(3 * idx + 1) - 1) >> 1;
        uint64_t m = idx - level_offset(l);
        uint64_t i = compact_bits(m);
        uint64_t j = compact_bits(m >> 1);
        uint64_t n = 1ULL << l;

        return pos_t{_pmin[0] + (i + .5) * _L[0] / n,
                     _pmin[1] + (j + .5) * _L[1] / n};
    }

    uint32_t get_depth()
    {
        return _depth;
    }

    const pos_t& get_L()
    {
        return _L;
    }

    const pos_t& get_cm()
    {
        return _tree[0].cm;
    }

private:

    pos_t _pmin;
    pos_t _L;
    std::array<Val, 2> _inv_L;
    uint32_t _depth;

    struct tree_node
    {
        pos_t cm{};
        Weight w{};
        uint32_t count{};
        uint32_t offset{};
    };

    std::vector<tree_node> _tree;
    std::vector<std::tuple<pos_t, Weight>> _points;
    std::vector<size_t> _order;
    std::vector<uint32_t> _fill;
    std::array<double, 32> _s2;

    std::span<std::tuple<pos_t, Weight>> leaf_points(const tree_node& t)
    {
        return {_points.data() + t.offset, t.count};
    }

    auto level(uint32_t l)
    {
        return std::views::iota(level_offset(l), level_offset(l + 1));
    }

    uint64_t interleave_bits(uint64_t x, uint64_t y)
    {
        return spread_bits(x) | (spread_bits(y) << 1);
    }

    // spread the low 32 bits into the even bit positions
    uint64_t spread_bits(uint64_t v)
    {
        v &= 0x00000000FFFFFFFFULL;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v <<  8)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v <<  2)) & 0x3333333333333333ULL;
        v = (v | (v <<  1)) & 0x5555555555555555ULL;
        return v;
    }

    // inverse
    uint64_t compact_bits(uint64_t v)
    {
        v &= 0x5555555555555555ULL;
        v = (v | (v >>  1)) & 0x3333333333333333ULL;
        v = (v | (v >>  2)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v >>  4)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v >>  8)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v >> 16)) & 0x00000000FFFFFFFFULL;
        return v;
    }

    // (4^l - 1) / 3, valid l <= 31
    uint64_t level_offset(uint32_t l)
    {
        return ((1ULL << (2*l)) - 1) / 3;
    }
};

} // namespace graph_tool
