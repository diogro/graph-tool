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

#ifndef SPEC_HH
#define SPEC_HH

#define STATE_NAME partition_centroid

#define SPEC_BASE_params

typedef decltype(hana::tuple_t<std::true_type, std::false_type>) bool_tr;

#define SPEC_params                                                            \
    ((bs, bs_t, 0))                                                            \
    ((RMI, bool_tr, 1))

#define EARGS_params

#include "../base/spec_base.hh"

#include "../support/cache.hh"

#include <boost/python.hpp>

namespace graph_tool
{

namespace STATE_NAME
{

typedef multi_array_ref<int64_t,2> bs_t;

struct StateSpec
    : public SpecBase
{
    typedef int64_t group_t;
    typedef multi_array_ref<group_t,1> vmap_t;

    using SpecBase::entropy_args_t;

    template <class State>
    struct Imp
        : public SpecBase::Imp<State>
    {
        GET_PARAMS_SPEC_TYPEDEF(SPEC_params)
        GET_PARAMS_SPEC_DEFINE(SPEC_params)

        Imp(State& state)
            : SpecBase::Imp<State>(state),
              GET_PARAMS_SPEC_INIT(SPEC_params)
              _state(state),
              _mrs(_bs.shape()[0]),
              _nr(_bs.shape()[0])
        {
            GILRelease gil_release;

            _state._N = _bs.shape()[1];
            for (size_t i = 0; i < _mrs.size(); ++i)
            {
                for (size_t v = 0; v < _state._N; ++v)
                {
                    auto r = _state._b[v];
                    auto s = _bs[i][v];
                    _mrs[i][{r,s}]++;
                    _nr[i][s]++;
                }
            }
        }

        static constexpr bool _is_RMI = std::decay_t<decltype(_RMI)>::value;

        State& _state;
        std::vector<gt_hash_map<std::tuple<group_t, group_t>, size_t>> _mrs;
        std::vector<gt_hash_map<group_t, size_t>> _nr;
        void move_vertex(size_t v, group_t r, group_t nr)
        {
            #pragma omp parallel for schedule(runtime)                         \
                if (_mrs.size() > get_openmp_min_thresh())
            for (size_t i = 0; i < _mrs.size(); ++i)
            {
                auto& mrsi = _mrs[i];
                size_t s = _bs[i][v];
                auto iter = mrsi.find({r, s});
                assert(iter != mrsi.end());
                iter->second--;
                if (iter->second == 0)
                    mrsi.erase(iter);
                mrsi[{nr,s}]++;
            }
        }

        double virtual_move(size_t v, group_t r, group_t nr, const entropy_args_t& ea)
        {
            double Sb = 0;
            double Sa = 0;

            if constexpr (_is_RMI)
            {
                Sb = entropy(ea);
                _state.move_vertex(v, nr);
                Sa = entropy(ea);
                _state.move_vertex(v, r);
                return (Sa - Sb);
            }
            else
            {
                auto wr = _state._groups[r].size();
                auto wnr = _state._groups[nr].size();

                Sb += (xlogx_fast(wr) + xlogx_fast(wnr)) * _mrs.size();
                Sa += (xlogx_fast(wr-1) + xlogx_fast(wnr+1)) * _mrs.size();

                #pragma omp parallel for schedule(runtime) reduction(+:Sa, Sb)  \
                    if (_mrs.size() > get_openmp_min_thresh())
                for (size_t i = 0; i < _mrs.size(); ++i)
                {
                    auto& mrsi = _mrs[i];
                    group_t s = _bs[i][v];

                    auto iter = mrsi.find({r,s});
                    size_t mrs = (iter != mrsi.end()) ? iter->second : 0;
                    assert(mrs > 0);
                    iter = mrsi.find({nr, s});
                    size_t mnrs = (iter != mrsi.end()) ? iter->second : 0;

                    Sb += -2 * (xlogx_fast(mrs) + xlogx_fast(mnrs));
                    Sa += -2 * (xlogx_fast(mrs-1) + xlogx_fast(mnrs+1));
                }
                return (Sa - Sb);
            }
        }

        double entropy(const entropy_args_t&)
        {
            if constexpr (_is_RMI)
            {
                double S = 0, S_n = 0;
                gt_hash_map<int64_t, int64_t> wr;
                for (auto& [r, vs] : _state._groups)
                {
                    if (vs.empty())
                        continue;
                    S_n -= lgamma_fast(vs.size() + 1);
                    wr[r] = vs.size();
                }

                S += (lgamma_fast(_state._N + 1) + S_n) * _mrs.size();

                #pragma omp parallel for schedule(runtime) reduction(+:S)
                for (size_t i = 0; i < _mrs.size(); ++i)
                {
                    for (auto& c : _mrs[i])
                        S += lgamma_fast(c.second + 1);
                    for (auto& rn : _nr[i])
                        S -= lgamma_fast(rn.second + 1);
                    S -= log_omega(wr, _nr[i], [](auto& x) {return x.second;});
                }
                return -S;
            }
            else
            {
                double S = 0, S_n = 0;
                for (auto& [r,vs] : _state._groups)
                    S_n += xlogx_fast(vs.size());

                #pragma omp parallel for schedule(runtime) reduction(+:S)
                for (size_t i = 0; i < _mrs.size(); ++i)
                {
                    for (auto& c : _mrs[i])
                        S -= 2 * xlogx_fast(c.second);
                    for (auto& rn : _nr[i])
                        S += xlogx_fast(rn.second);
                    S += S_n;
                }
                return S;
            }
        }

    };

    template <class Vr, class Vs, class T>
    static double log_omega(Vr&& nr, Vs&& ns, T&& get_n)
    {
        double S = 0;

        size_t N = 0;
        size_t Bx = 0;
        size_t By = 0;
        for (auto& n : nr)
        {
            Bx += (get_n(n) > 0);
            N += get_n(n);
        }
        for (auto& n : ns)
            By += (get_n(n) > 0);

        S += (Bx - 1) * (By - 1) * log(N + (Bx * By) / 2.);

        double w = N / (N + (Bx * By) / 2.);

        double y2 = 0, lny = 0;
        double x2 = 0, lnx = 0;

        for (auto& n : nr)
        {
            double x_r = (1-w)/Bx + (w * get_n(n))/N;
            x2 += x_r * x_r;
            lnx += log(x_r);
        }

        for (auto& n : ns)
        {
            double y_s = (1-w)/By + (w * get_n(n))/N;
            y2 += y_s * y_s;
            lny += log(y_s);
        }

        double mu = (Bx + 1) / (Bx * y2) - 1./Bx;
        double nu = (By + 1) / (By * x2) - 1./By;

        S += (Bx + nu - 2) * lny / 2;
        S += (By + mu - 2) * lnx / 2;

        S += safe_lgamma(mu * Bx)/2 + safe_lgamma(nu * By)/2;
        S -= (By * (safe_lgamma(nu) + safe_lgamma(Bx))/2. +
              Bx * (safe_lgamma(mu) + safe_lgamma(By))/2.);

        return S;
    }

    static void python_export()
    {
        using namespace boost::python;
        def("log_omega",
            +[](python::object onr, python::object ons) -> double
             {
                 auto nr = get_array<int64_t, 1>(onr);
                 auto ns = get_array<int64_t, 1>(ons);
                 return log_omega(nr, ns, [](auto& x) {return x;});
             });

        def("expected_MI",
            +[](python::object onr, python::object ons) -> double
             {
                 auto nr = get_array<int64_t, 1>(onr);
                 auto ns = get_array<int64_t, 1>(ons);

                 int64_t N = 0;
                 for (auto c : nr)
                     N += c;

                 double EMI = 0;

                 #pragma omp parallel for schedule(runtime) reduction(+:EMI) collapse(2)
                 for (size_t r = 0; r < nr.size(); ++r)
                 {
                     for (size_t s = 0; s < ns.size(); ++s)
                     {
                         auto a = nr[r];
                         auto b = ns[s];
                         for (int64_t m = max(int64_t(1), a + b - N); m <= min(a, b); ++m)
                         {
                             double T = (m * (safelog_fast(m) + safelog_fast(N) - safelog_fast(a) - safelog_fast(b))) / N;
                             double lT = lgamma_fast(a + 1) + lgamma_fast(b + 1);
                             lT += lgamma_fast(N - a + 1) + lgamma_fast(N - b + 1);
                             lT -= lgamma_fast(N + 1) + lgamma_fast(m + 1);
                             lT -= lgamma_fast(a - m + 1) + lgamma_fast(b - m + 1);
                             lT -= lgamma_fast(N - a - b + m + 1);
                             EMI += T * exp(lT);
                         }
                     }
                 }

                 return EMI;
             });
    }
};

} // STATE_NAME

} // graph_tool namespace

#endif //SPEC_HH
