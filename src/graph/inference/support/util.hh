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

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "cache.hh"

namespace graph_tool
{
using namespace boost;

template <Scalar T1, Scalar T2>
[[gnu::const]]
double lbinom(T1 N, T2 k)
{
    if (N == 0 || k == 0 || T1(k) >= N)  [[unlikely]]
        return 0;
    return ((boost::math::lgamma(N + 1) - boost::math::lgamma(k + 1)) - boost::math::lgamma((N - k) + 1));
}

template <Scalar T1, Scalar T2>
[[gnu::const]] [[gnu::always_inline]]
inline double lbinom_fast(T1 N, T2 k)
{
    if (N == 0 || k == 0 || std::cmp_greater_equal(k, N)) [[unlikely]]
        return 0;
    return ((lgamma_fast(N + 1) - lgamma_fast(k + 1)) - lgamma_fast((N - k) + 1));
}

// Coefficients B_{2r} / (2r (2r - 1)), r = 1, ..., 8, of the Stirling series
// log Gamma(x) = (x - 1/2) log x - x + log(2 pi) / 2 + sum_r c_r x^(1-2r), which
// is accurate to double precision for x >= 10
constexpr std::array<double, 8> __stirling_c =
    {1./12, -1./360, 1./1260, -1./1680, 1./1188, -691./360360, 1./156,
     -3617./122400};

// sum_r c_r x^(1-2r)
[[gnu::const]]
inline double stirling_delta(double x)
{
    double ix = 1 / x, ix2 = ix * ix;
    double S = 0, p = ix;
    for (auto c : __stirling_c)
    {
        S += c * p;
        p *= ix2;
    }
    return S;
}

// log Gamma(z + h) - log Gamma(z), for z >= 10 and h >= 0, from the
// difference of the Stirling series, i.e. without the cancellation in the
// difference of the log-gamma functions if z >> h
[[gnu::const]]
inline double lgamma_ratio(double z, double h)
{
    double zh = z + h;
    return ((zh - 0.5) * std::log1p(h / z) + h * std::log(z) - h) +
        (stirling_delta(zh) - stirling_delta(z));
}

// log binom(N, k), for real N and k, accurate to double precision also if N
// is large: with k <= N - k (by symmetry), if k is small the ratio
// Gamma(N + 1) / Gamma(N - k + 1) is obtained from the difference of the
// Stirling series, and otherwise all three log-gamma functions are expanded,
// so that the terms -n cancel exactly, giving -(k + 1/2) log(k / N) - (N - k +
// 1/2) log(1 - k / N) - log(2 pi N) / 2 plus the Stirling corrections, without
// cancellations.
template <Scalar T1, Scalar T2>
[[gnu::const]]
double lbinom_careful(T1 N_, T2 k_)
{
    if (N_ == 0 || k_ == 0 || T1(k_) >= N_)
        return 0;
    double N = N_;
    double k = std::min(double(k_), N - k_);
    if (N - k + 1 < 10)
        return (safe_lgamma(N + 1) - safe_lgamma(k + 1)) -
            safe_lgamma(N - k + 1);
    if (k < 10)
        return lgamma_ratio(N - k + 1, k) - safe_lgamma(k + 1);
    double p = k / N;
    return ((-(k + 0.5) * std::log(p) - ((N - k) + 0.5) * std::log1p(-p)) -
            0.5 * std::log(2 * std::numbers::pi * N)) +
        (stirling_delta(N) - stirling_delta(k) - stirling_delta(N - k));
}

template <Scalar T>
[[gnu::const]]
auto lbeta(T x, T y)
{
    return (boost::math::lgamma(x) + boost::math::lgamma(y)) - boost::math::lgamma(x + y);
}

template <Scalar T, bool handle_inf=true>
[[gnu::const]]
auto log_sum_exp(T a, T b)
{
    if constexpr (handle_inf)
    {
        if (a == b) [[unlikely]]  // handles infinity
            return a + std::numbers::ln2_v<T>;
    }
    const T hi = std::max(a, b);
    const T lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

template <Scalar T, Scalar... Ts>
[[gnu::const]]
auto log_sum_exp(T a, T b, Ts... cs)
{
    if constexpr (sizeof...(Ts) == 0)
        return log_sum_exp(a, b);
    else
        return log_sum_exp(log_sum_exp(a, b), cs...);
}

template <class V, bool handle_inf=true>
[[gnu::const]]
auto log_sum_exp(const V& v)
{
    typedef std::remove_reference_t<decltype(*v.begin())> val_t;
    auto ret = -std::numeric_limits<val_t>::infinity();
    for (const auto& x : v)
        ret = log_sum_exp<val_t, handle_inf>(ret, x);
    return ret;
}

#pragma omp declare reduction(log_sum_exp : double : omp_out = log_sum_exp(omp_out, omp_in)) \
    initializer(omp_priv = -std::numeric_limits<double>::infinity())

} // namespace graph_tool
