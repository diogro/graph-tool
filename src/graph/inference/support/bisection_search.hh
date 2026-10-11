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

#ifndef BISECTION_SEARCH_HH
#define BISECTION_SEARCH_HH

#include <algorithm>
#include <array>
#include <random>
#include <cmath>

namespace graph_tool
{

template <class Value>
class BisectionSearch
{
public:

    constexpr static bool _discrete = std::is_integral_v<Value>;
    constexpr static double _epsilon = 1e-8;

    BisectionSearch() {}

    template <class F, class... RNG>
    std::tuple<Value, double>
    search(Value x_min, Value x_max, F&& f, size_t maxiter = 0,
           Value tol = 0, RNG&... rng)
    {
        Value x_mid;
        return search(x_min, x_mid, x_max, std::forward<F>(f), maxiter, tol,
                      rng...);
    }

    template <class F, class... RNG>
    std::tuple<Value, double>
    search(Value& x_min, Value& x_mid, Value& x_max, F&& f,
           size_t maxiter = 0, Value tol = 0, RNG&... rng)
    {
        // initial bracketing
        x_mid = get_mid(x_min, x_max, rng...);
        double f_max = f(x_max);
        double f_mid = std::numeric_limits<double>::infinity();
        double f_min = f(x_min);

        auto ret_min =
            [&]()
            {
                std::array<Value,3> xs = {x_min, x_mid, x_max};
                std::array<double,3> fs = {f_min, f_mid, f_max};
                size_t pos = std::min_element(fs.begin(), fs.end()) - fs.begin();
                return std::make_tuple(xs[pos], fs[pos]);
            };

        size_t niter = 2;

        if (maxiter > 0 && niter >= maxiter)
            return ret_min();

        f_mid = f(x_mid);
        niter++;

        if (maxiter > 0 && niter == maxiter)
            return ret_min();

        Value md = _discrete ? 1 : 0;

        while (f_mid > f_min || f_mid > f_max)
        {
            if (f_min < f_max)
            {
                x_max = x_mid;
                f_max = f_mid;
                x_mid = get_mid(x_min, x_mid, rng...);
            }
            else
            {
                x_min = x_mid;
                f_min = f_mid;
                x_mid = get_mid(x_mid, x_max, rng...);
            }

            f_mid = f(x_mid);
            niter++;

            if (maxiter > 0 && niter == maxiter)
                return ret_min();

            if (x_min == x_mid && (x_max - x_mid) <= md)
                break;
        }

        // Bisection search
        while (x_max - x_mid > md)
        {
            Value x;
            if (x_max - x_mid > x_mid - x_min)
                x = get_mid(x_mid, x_max, rng...);
            else
                x = get_mid(x_min, x_mid, rng...);

            double f_x = f(x);
            niter++;

            if (maxiter > 0 && niter == maxiter)
                return ret_min();

            if (f_x < f_mid)
            {
                if (x_max - x_mid > x_mid - x_min)
                {
                    x_min = x_mid;
                    f_min = f_mid;
                }
                else
                {
                    x_max = x_mid;
                    f_max = f_mid;
                }
                x_mid = x;
                f_mid = f_x;
            }
            else
            {
                if (x_max - x_mid > x_mid - x_min)
                {
                    x_max = x;
                    f_max = f_x;
                }
                else
                {
                    x_min = x;
                    f_min = f_x;
                }
            }

            if constexpr (!_discrete)
            {
                if (x_max - x_min < tol)
                    break;
            }
        }

        return ret_min();
    }

    Value fibo(size_t n)
    {
        return Value(std::round(std::pow(_phi, n) / std::sqrt(5)));
    }

    Value fibo_n_floor(Value x)
    {
        return std::floor(std::log(x * std::sqrt(5) + .5) / std::log(_phi));
    }

    Value get_mid(Value a, Value b)
    {
        if (a == b)
            return a;
        if constexpr (_discrete)
        {
            auto n = fibo_n_floor(b - a);
            return b - fibo(n - 1);
        }
        else
        {
            return (_phi * a + b) / (_phi + 1);
        }
    }

    template <class RNG>
    Value get_mid(Value a, Value b, RNG& rng)
    {
        if (a == b)
            return a;
        if constexpr (_discrete)
        {
            if (b - a < 2)
                return a;
            std::uniform_int_distribution<Value> sample(a + 1, b - 1);
            return sample(rng);
        }
        else
        {
            std::uniform_real_distribution<Value> sample(a, b);
            return sample(rng);
        }
    }

private:
#ifndef __clang__
    constexpr static
#endif
    double _phi = (1 + std::sqrt(5)) / 2;
};

}
#endif // BISECTION_SEARCH_HH
