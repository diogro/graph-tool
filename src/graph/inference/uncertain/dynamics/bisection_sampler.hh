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

#ifndef BISECTION_SAMPLER_HH
#define BISECTION_SAMPLER_HH

#include <map>

#include <boost/math/tools/minima.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include "segment_sampler.hh"

#include "../../support/util.hh"
#include "../../support/fibonacci_search.hh"

namespace graph_tool
{

struct bisect_args_t
{
    double min_bound = -numeric_limits<double>::infinity();
    double max_bound = numeric_limits<double>::infinity();
    double min_init = -numeric_limits<double>::infinity();
    double max_init = numeric_limits<double>::infinity();
    std::uintmax_t maxiter = 0;
    double tol = 2e-3;
    double ftol = 100;
    bool reversible = true;
    size_t nmax_extend = 8;
};

template <class T>
T quantize(T x, T delta)
{
    if (delta == 0)
        return x;
    return std::floor(x/delta) * delta;
}

class BisectionSampler
{
public:
    BisectionSampler()
    {}

    template <class F>
    BisectionSampler(F&& f, const bisect_args_t& args)
        : _f(f), _args(args)
    {}

    bool update_bounds(double nx, double& xmin, double& xmax)
    {
        auto xmin_ = xmin;
        auto xmax_ = xmax;

        auto close =
            [&] (auto a, auto b)
            {
                return abs(a - b) < std::max(2 * _args.tol, (xmax_ - xmin_) / 10);
            };

        if (nx < xmin)
            xmin = nx;

        if (nx > xmax)
            xmax = nx;

        double dS_min = std::min_element(_fcache.begin(), _fcache.end(),
                                         [&](auto& a, auto& b)
                                         { return a.second < b.second; })->second;

        if (close(nx, xmin) || _fcache.begin()->second - dS_min < _args.ftol)
        {
            if (xmin < 0)
                xmin *= 10;
            else
                xmin /= 10;
            xmin = std::max(xmin, _args.min_bound);
        }

        if (close(nx, xmax) || _fcache.rbegin()->second - dS_min < _args.ftol)
        {
            if (xmax > 0)
                xmax *= 10;
            else
                xmax /= 10;
            xmax = std::min(xmax, _args.max_bound);
        }

        if (xmin != xmin_ || xmax != xmax_)
            return true;

        return false;
    }

    double bisect(double x, double delta)
    {
        _seg_sampler = SegmentSampler();

        double nx = numeric_limits<double>::quiet_NaN();

        auto maxiter = _args.maxiter;
        auto xmin = std::max(_args.min_init, _args.min_bound);
        auto xmax = std::min(_args.max_init, _args.max_bound);

        if (maxiter == 0)
            maxiter = std::numeric_limits<std::uintmax_t>::max();

        if (std::isinf(xmin) || std::isinf(xmax))
        {
            if (_args.reversible)
            {
                xmin = std::isinf(_args.min_bound) ? -1. : _args.min_bound;
                xmax = std::isinf(_args.max_bound) ?  1. : _args.max_bound;
            }
            else
            {
                xmin = (x < 0) ? x * 2 : x / 2;
                xmax = (x > 0) ? x * 2 : x / 2;
                xmin = std::max(xmin - 1, _args.min_bound);
                xmax = std::min(xmax + 1, _args.max_bound);
            }
        }

        size_t it = 0;
        do
        {
            double dS;
            std::tie(nx, dS) =
                boost::math::tools::brent_find_minima([&](auto x){ return f(x);},
                                                      xmin, xmax,
                                                      ceil(1-log2(_args.tol)),
                                                      maxiter);
            if (f(xmin) < dS)
                std::tie(nx, dS) = std::make_tuple(xmin, f(xmin));
            if (f(xmax) < dS)
                std::tie(nx, dS) = std::make_tuple(xmax, f(xmax));
        }
        while ((boost::math::epsilon_difference(f(xmin), f(xmax)) > 3) &&
               update_bounds(nx, xmin, xmax) &&
               (++it < _args.nmax_extend));

        nx = std::min_element(_fcache.begin(), _fcache.end(),
                              [&](auto& a, auto& b)
                              { return a.second < b.second; })->first;

        return quantize(nx, delta);
    }

    template <class... RNG>
    double bisect_fb(const std::vector<double>& vals, RNG&... rng)
    {
        _seg_sampler = SegmentSampler();
        FibonacciSearch<size_t> fb;
        auto [i, dS] = fb.search(0, vals.size() - 1,
                                 [&](size_t i){ return f(vals[i]); },
                                 0, 0, rng...);
        return vals[i];
    }

    template <class RNG>
    double sample(double beta, double delta, RNG& rng)
    {
        double val;
        if (std::isinf(beta))
        {
            val = std::min_element(_fcache.begin(), _fcache.end(),
                                   [&](auto& a, auto& b)
                                   { return a.second < b.second; })->first;
        }
        else
        {
            auto& seg = get_seg_sampler(beta);
            val = seg.sample(rng);
        }
        return quantize(val, delta);
    }

    double lprob(double x, double beta, double delta)
    {
        auto& seg = get_seg_sampler(beta);
        if (delta == 0)
            return seg.lprob(x);
        return seg.lprob_int(x, delta);
    }

    double f(double x, bool add = true)
    {
        auto iter = _fcache.find(x);
        if (iter != _fcache.end())
            return iter->second;
        auto f_x = _f(x);
        if (add)
            _fcache[x] = f_x;
        return f_x;
    }

    SegmentSampler& get_seg_sampler(double beta)
    {
        if (!_seg_sampler.get_xs().empty() && beta == _beta)
            return _seg_sampler;

        std::vector<double> xs;
        std::vector<double> ws;

        for (auto& [x, w] : _fcache)
        {
            xs.push_back(x);
            ws.push_back(-w * beta);
        }

        _seg_sampler = SegmentSampler(xs, ws);
        _beta = beta;
        return _seg_sampler;
    }

    const std::map<double, double>& get_fcache()
    {
        return _fcache;
    }

private:

    std::function<double(double)> _f;
    bisect_args_t _args;
    std::vector<double> _vals;
    std::map<double, double> _fcache;
    double _beta = 1;
    SegmentSampler _seg_sampler;
};

class SetBisectionSampler
{
public:
    SetBisectionSampler(const std::vector<double>& vals, double pu,
                        BisectionSampler& sampler)
        : _vals(vals), _pu(pu), _sampler(sampler)
    {
    }


    std::tuple<double, double, double>
    bracket_closest(double x,
                    double skip = numeric_limits<double>::quiet_NaN(),
                    double add = numeric_limits<double>::quiet_NaN())
    {
        auto iter = std::lower_bound(_vals.begin(), _vals.end(), x);
        auto begin = iter - std::min<size_t>(3, iter - _vals.begin());
        auto end = iter + std::min<size_t>(4, _vals.end() - iter);

        std::vector<double> pts;
        for (auto iter = begin; iter != end; ++iter)
        {
            if (*iter == skip)
                continue;
            pts.push_back(*iter);
        }

        if (!std::isnan(add) && (pts.empty() || (pts.front() < add && pts.back() > add)))
            pts.insert(std::lower_bound(pts.begin(), pts.end(), add), add);

        auto pos = std::lower_bound(pts.begin(), pts.end(), x); // *pos >= x
        if (pos == pts.end() || ((pos != pts.begin()) &&
                                 (x - *(pos-1)) < (*pos - x)))
            --pos;

        auto y = *pos;

        double a = (pos != pts.begin()) ? *(pos - 1) + (y - *(pos - 1))/2 :
            -numeric_limits<double>::infinity();
        double b = ((pos + 1) != pts.end()) ? y + (*(pos + 1) - y)/2 :
            numeric_limits<double>::infinity();
        return {a, y, b};
    }

    double find_closest(double x,
                        double skip = numeric_limits<double>::quiet_NaN(),
                        double add = numeric_limits<double>::quiet_NaN())
    {
        return get<1>(bracket_closest(x, skip, add));
    }

    template <class RNG>
    double sample(double beta, RNG& rng)
    {
        std::bernoulli_distribution unif(_pu);
        if (unif(rng))
        {
            return uniform_sample(_vals, rng);
        }
        else
        {
            auto nx = _sampler.sample(beta, 0., rng);
            return find_closest(nx);
        }
    }

    double lprob(double nx, double beta,
                 double skip = numeric_limits<double>::quiet_NaN(),
                 double add = numeric_limits<double>::quiet_NaN())
    {
        size_t N = _vals.size() - int(!std::isnan(skip)) + int(!std::isnan(add));
        if (N == 0)
            return -numeric_limits<double>::infinity();
        if (_pu == 1)
            return -log(N);
        auto [a, y, b] = bracket_closest(nx, skip, add);
        auto& seg = _sampler.get_seg_sampler(beta);
        a = std::max(a, seg.get_xs().front());
        b = std::min(b, seg.get_xs().back());
        double l = seg.lprob_int(a, b - a);
        return log_sum_exp(log1p(-_pu) + l, log(_pu) - log(N));
    }

private:
    const std::vector<double>& _vals;
    double _pu;
    BisectionSampler& _sampler;
};

}
#endif // BISECTION_SAMPLER_HH
