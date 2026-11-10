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

#include "config.h"
#include "../../openmp.hh"

#include <vector>
#include <cmath>
#include <utility>
#include <mutex>
#include <type_traits>

#include <boost/math/special_functions/gamma.hpp>

namespace graph_tool
{

// thread-safe lgamma; doubles are not promoted to long double internally,
// which is much slower and makes no difference at double precision

auto safe_lgamma(auto x)
{
    using boost::math::policies::domain_error;
    using boost::math::policies::pole_error;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::make_policy;
    using boost::math::policies::ignore_error;
    using boost::math::policies::promote_double;
    return boost::math::lgamma(x, make_policy(promote_double<false>(),
                                              domain_error<ignore_error>(),
                                              pole_error<ignore_error>(),
                                              overflow_error<ignore_error>(),
                                              evaluation_error<ignore_error>()));
}

// Repeated computation of x*log(x), log(x), and lgamma(x) actually adds up to a
// lot of time. A significant speedup can be made by caching pre-computed
// values.

extern std::vector<double> __safelog_cache;
extern std::vector<double> __xlogx_cache;
extern std::vector<double> __lgamma_cache;

constexpr size_t __max_size = std::numeric_limits<size_t>::max();

template <class T>
size_t get_next_size(T n)
{
    size_t k = 1;
    while (std::cmp_less(k, n))
        k <<= 1;
    return k;
}

template <class T, class F, class Cache>
[[gnu::const]] [[gnu::always_inline]]
inline double get_cached(T x, F&& f, Cache& cache)
{
    if (std::cmp_greater_equal(x, cache.size())) [[unlikely]]
        return f(x);
    else                                         [[likely]]
        return cache[x];
}

template <class T, class F, class Cache>
void init_cache(T x, F&& f, Cache& cache)
{
    if (std::cmp_greater_equal(x, cache.size()) &&
        std::cmp_less(x, __max_size))
    {
        size_t old_size = cache.size();
        cache.resize(get_next_size(x + 1));
        #pragma omp parallel for
        for (size_t y = old_size; y < cache.size(); ++y)
            cache[y] = f(y);
    }
}

template <Scalar T>
[[gnu::const]] [[gnu::always_inline]]
inline double safelog(T x)
{
    if (x == 0)
        return 0;
    return std::log(x);
}

template <Scalar T>
[[gnu::const]] [[gnu::always_inline]]
inline double safelog_fast(T x)
{
    return get_cached(x, [](T y) { return safelog(y); }, __safelog_cache);
}

template <class T>
inline void init_safelog_fast(T x)
{
    return init_cache(x, [](T y) { return safelog(y); }, __safelog_cache);
}

template <Scalar T>
[[gnu::const]] [[gnu::always_inline]]
inline double xlogx(T x)
{
    return x * safelog(x);
}

template <Scalar T>
[[gnu::const]] [[gnu::always_inline]]
inline double xlogx_fast(T x)
{
    return get_cached(x, [](T y) { return xlogx(y); }, __xlogx_cache);
}

template <Scalar T>
void init_xlogx_fast(T x)
{
    return init_cache(x, [](T y) { return xlogx(y); }, __xlogx_cache);
}

template <Scalar T>
[[gnu::const]] [[gnu::always_inline]]
inline double lgamma_fast(T x)
{
    return get_cached(x, [](T y) { return safe_lgamma(y); }, __lgamma_cache);
}

template <Scalar T>
void init_lgamma_fast(T x)
{
    return init_cache(x, [](T y) { return safe_lgamma(y); }, __lgamma_cache);
}

} // graph_tool namespace
