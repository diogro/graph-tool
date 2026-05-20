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

#ifndef DYNAMICS_DISCRETE_POTTS_HH
#define DYNAMICS_DISCRETE_POTTS_HH

#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

#include "dynamics_discrete.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class Spec, bool keep_k, bool tshift, int xmin>
class PottsStateBase
    : public NSumStateBase<Spec, std::vector<double>, true, keep_k, tshift>
{
public:
    typedef NSumStateBase<Spec, std::vector<double>, true, keep_k, tshift> base_t;

    template <class S, class... Args>
    PottsStateBase(python::dict params, S& state, Args&&... s)
        : NSumStateBase<Spec, std::vector<double>, true, keep_k, tshift>(state, s..., false),
          _f(get_array<double,2>(params["f"])),
          _f_sym(python::extract<bool>(params["f_sym"])),
          _q(_f.shape()[0]),
          _state(state)
    {
        for (auto& sn : base_t::_s)
        {
            for (auto v : vertices_range(state._u))
            {
                for (auto s : sn[v])
                {
                    if (s < 0 || s > int(_q) - 1)
                    {
                        throw ValueException("Expected state values in the range {0, ..., "
                                             + std::to_string(_q - 1)
                                             + "}, got: " + std::to_string(s));
                    }
                }
            }
        }

        if (!norm_f())
            throw ValueException("Invalid Potts interaction matrix f: "
                                 "normalization sum is zero or non-finite");

        base_t::reset_m(state);
    }

    template <class Val>
    constexpr Val transform_input(size_t, size_t, Val x) { return x; }

    template <class Val>
    void transform_theta(std::vector<Val>&, Val) {}

    typedef std::vector<double> m_t;
    m_t get_m_zero() { return m_t(_q); }

    template <class Val>
    void update_dm(m_t& dm, Val s, double dx)
    {
        dm[s] += dx;
    }

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P_disp(const std::vector<double>& theta, const m_t& m, int r)
    {
        double x = theta[r];
        auto&& f = _f[r];
        for (size_t s = 0; s < _q; ++s)
            x += f[s] * m[s];

        double Z = std::numeric_limits<double>::lowest();

        for (size_t r = 0; r < _q; ++r)
        {
            double val = theta[r];
            auto&& f = _f[r];
            for (size_t s = 0; s < _q; ++s)
                val += f[s] * m[s];
            Z = log_sum_exp<double, false>(Z, val);
        }
        return x - Z;
    }

    template <size_t q>
    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P_disp_q(const std::vector<double>& theta, const m_t& m, int r)
    {
        double x = theta[r];
        auto&& f = _f[r];
#if GCC_VERSION >= 140000
        #pragma GCC unroll q
#endif
        for (size_t s = 0; s < q; ++s)
            x += f[s] * m[s];

        double vmax = std::numeric_limits<double>::lowest();
        std::array<double, q> vals;

#if GCC_VERSION >= 140000
        #pragma GCC unroll q
#endif
        for (size_t r = 0; r < q; ++r)
        {
            auto& val = vals[r];
            val = theta[r];
            auto&& f = _f[r];
#if GCC_VERSION >= 140000
            #pragma GCC unroll q
#endif
            for (size_t s = 0; s < q; ++s)
                val += f[s] * m[s];
            if (val > vmax)
                vmax = val;
        }

        double tmp = 0;
#if GCC_VERSION >= 140000
        #pragma GCC unroll q
#endif
        for (size_t r = 0; r < q; ++r)
            tmp += exp(vals[r] - vmax);

        return x - (vmax + log(tmp));
    }

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(const std::vector<double>& theta, const m_t& m, int r)
    {
        switch (_q)
        {
        case 1:
            return log_P_disp_q<1>(theta, m, r);
            break;
        case 2:
            return log_P_disp_q<2>(theta, m, r);
            break;
        case 3:
            return log_P_disp_q<3>(theta, m, r);
            break;
        case 4:
            return log_P_disp_q<4>(theta, m, r);
            break;
        case 5:
            return log_P_disp_q<5>(theta, m, r);
            break;
        case 6:
            return log_P_disp_q<6>(theta, m, r);
            break;
        case 7:
            return log_P_disp_q<7>(theta, m, r);
            break;
        case 8:
            return log_P_disp_q<8>(theta, m, r);
            break;
        case 9:
            return log_P_disp_q<9>(theta, m, r);
            break;
        case 10:
            return log_P_disp_q<10>(theta, m, r);
            break;
        default:
            return log_P_disp(theta, m, r);
        }
    }

    size_t get_q() { return _q; }

    bool norm_f()
    {
        double Z = 0;
        for (size_t r = 0; r < _q; ++r)
        {
            for (size_t s = _f_sym ? r : 0; s < _q; ++s)
            {
                _f[r][s] = std::max(_f[r][s], 0.);
                if (_f_sym)
                    _f[s][r] = _f[r][s];
                Z += _f[r][s];
            }
        }

        if (Z <= 0 || !std::isfinite(Z))
            return false;

        for (size_t r = 0; r < _q; ++r)
            for (size_t s = 0; s < _q; ++s)
                _f[r][s] /= Z;

        return true;
    }

    double sample_f(size_t r, size_t s, double beta, dentropy_args_t& ea,
                    const bisect_args_t& ba, bool verbose, rng_t& rng)
    {
        boost::multi_array<double,2> f_temp = _f;
        auto x = _f[r][s];

        auto set_x =
            [=,this](auto x)
            {
                _f = f_temp;
                if (_f_sym)
                    _f[r][s] = _f[s][r] = x;
                else
                    _f[r][s] = x;
                return norm_f();
            };

        auto f =
            [=,this](auto x)
            {
                if (!set_x(x))
                {
                    _f = f_temp;
                    if (verbose)
                        std::println("reject invalid normalization for f matrix: {} {} {}", r, s, x);
                    return std::numeric_limits<double>::infinity();
                }
                return _state.entropy(ea);
            };

        BisectionSampler<> sampler(f, ba);

        double nx;
        if (ba.min_bound == ba.max_bound)
            nx = ba.min_bound;
        else
            nx = sampler.bisect(x);

        sampler.f(nx, true);

        nx = sampler.sample(beta, 0, rng);
        double dS = sampler.f(nx) - sampler.f(x);

        if ((std::isinf(beta) && dS > 0) || !std::isfinite(dS))
        {
            _f = f_temp;
            return 0;
        }

        double lf = sampler.lprob(nx, beta);
        double lb = sampler.lprob(x, beta);

        std::uniform_real_distribution<> u(0, 1);

        double a = -beta * dS + lb - lf;

        if (std::isinf(lb) || (a <= 0 && exp(a) <= u(rng)))
        {
            _f = f_temp;
            return 0;
        }

        set_x(nx);

        return dS;
    }

private:
    boost::multi_array_ref<double,2> _f;
    bool _f_sym;
    size_t _q = 0;
    DynBase& _state;
};

class PseudoPottsState
    : public PottsStateBase<PseudoPottsState, false, false, 0>
{
public:
    template <class... Args>
    PseudoPottsState(python::dict params, Args&&... s)
        : PottsStateBase<PseudoPottsState, false, false, 0>(params, s...)
    {}
};

class GlauberPottsState
    : public PottsStateBase<PseudoPottsState, false, false, 0>
{
public:
    template <class... Args>
    GlauberPottsState(python::dict params, Args&&... s)
        : PottsStateBase<PseudoPottsState, false, false, 0>(params, s...)
    {}

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(const std::vector<double>& theta, std::vector<double>& m, int, int ns)
    {
        return PottsStateBase<PseudoPottsState, false, false, 0>::log_P(theta, m, ns);
    }
};


} // graph_tool namespace

#endif //DYNAMICS_DISCRETE_POTTS_HH
