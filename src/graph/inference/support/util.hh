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

#ifndef UTIL_HH
#define UTIL_HH

#include "config.h"

#include <cmath>

#include "cache.hh"

namespace graph_tool
{
using namespace boost;

template <Scalar T1, Scalar T2>
//[[gnu::const]]
double lbinom(T1 N, T2 k)
{
    if (N == 0 || k == 0 || T1(k) >= N)
        return 0;
    return ((boost::math::lgamma(N + 1) - boost::math::lgamma(k + 1)) - boost::math::lgamma((N - k) + 1));
}

template <Scalar T1, Scalar T2>
//[[gnu::const]] [[gnu::always_inline]]
inline double lbinom_fast(T1 N, T2 k)
{
    if (N == 0 || k == 0 || std::cmp_greater_equal(k, N))
        return 0;
    return ((lgamma_fast(N + 1) - lgamma_fast(k + 1)) - lgamma_fast((N - k) + 1));
}

template <Scalar T1, Scalar T2>
//[[gnu::const]]
double lbinom_careful(T1 N, T2 k)
{
    if (N == 0 || k == 0 || T1(k) >= N)
        return 0;
    double lgN = boost::math::lgamma(N + 1);
    double lgk = boost::math::lgamma(k + 1);
    if (lgN - lgk > 1e8)
    {
        // We have N >> k. Use Stirling's approximation: ln N! ~ N ln N - N
        // and reorder
        double N_ = N;
        return - N * std::log1p(-k / N_) - k * std::log1p(-k / N_) - k - lgk + k * std::log(N_);
    }
    else
    {
        return lgN - boost::math::lgamma(N - k + 1) - lgk;
    }
}

template <Scalar T>
//[[gnu::const]]
auto lbeta(T x, T y)
{
    return (boost::math::lgamma(x) + boost::math::lgamma(y)) - boost::math::lgamma(x + y);
}

template <Scalar T, bool handle_inf=true>
//[[gnu::const]]
auto log_sum_exp(T a, T b)
{
    if constexpr (handle_inf)
    {
        if (a == b)  // handles infinity
            return a + std::log(T(2));
    }
    if (a > b)
        return a + std::log1p(std::exp(b-a));
    else
        return b + std::log1p(std::exp(a-b));
}

template <Scalar T, Scalar... Ts>
//[[gnu::const]]
auto log_sum_exp(T a, T b, Ts... cs)
{
    if constexpr (sizeof...(Ts) == 0)
        return log_sum_exp(a, b);
    else
        return log_sum_exp(log_sum_exp(a, b), cs...);
}

template <class V, bool handle_inf=true>
//[[gnu::const]]
auto log_sum_exp(const V& v)
{
    typedef std::remove_reference_t<decltype(*v.begin())> val_t;
    auto ret = -std::numeric_limits<val_t>::infinity();
    for (const auto& x : v)
        ret = log_sum_exp<val_t, handle_inf>(ret, x);
    return ret;
}

} // namespace graph_tool

#endif // UTIL_HH
