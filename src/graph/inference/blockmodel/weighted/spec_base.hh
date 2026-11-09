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

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <print>

#include "../spec.hh"

#include "../../support/graph_state.hh"

#include "../../../parallel_util.hh"

#define WEARGS_params                                                          \
    ((rec, bool))                                                              \
    ((rec_dl, bool))

#define WEIGHTED_SPEC_params                                                   \
    ((rec_types, std::vector<int32_t>, 0))                                     \
    ((wparams, std::vector<std::vector<double>>&, 0))                          \
    ((recdx, std::vector<STATE_SPEC::rcount_t>&, 0))                           \
    ((Lrecdx, std::vector<STATE_SPEC::rcount_t>&, 0))                          \
    ((epsilon, std::vector<double>&, 0))                                       \
    ((is_base, bool, 0))

#include "types.hh"
#include "entropy.hh"

namespace graph_tool
{

using namespace std;

template <bool directed, class Group, class Count, class Edge, class EW, class DRcount, class Rcount>
struct WEntrySet
    : public EntrySet<directed, Group, Count, Edge, EW>
{
    typedef EntrySet<directed, Group, Count, Edge, EW> base_t;

    WEntrySet()
        : _dL(std::make_shared<DRcount>()),
          _LdBdx(std::make_shared<std::vector<Rcount>>())
    {}

    std::shared_ptr<DRcount> _dL;
    std::shared_ptr<std::vector<Rcount>> _LdBdx;

    void set_next(WEntrySet& next)
    {
        next._dL = _dL;
        next._LdBdx = _LdBdx;
        _next = &next;
    }

    void set_move(auto r, auto nr, size_t B)
    {
        base_t::set_move(r, nr, B);
        *_dL = 0;
        _LdBdx->clear();
    }

    WEntrySet* _next = nullptr;
};

template <class T>
constexpr void enable_concurrent(T&, bool)
{
}

template <class Base = BlockStateSpec, size_t WN = WVEC_N>
struct WeightedBlockStateSpec
    : public Base
{
    typedef decltype(hana::tuple_t<std::false_type>) flat_map_tr;

    typedef typename Base::count_t count_t;
    typedef typename Base::group_t group_t;
    static constexpr group_t null_group = std::numeric_limits<group_t>::max();

    typedef double rcount_t;
    typedef eprop_map_t<rcount_t> ewmap_t;
    typedef ewmap_t bewmap_t;
    typedef int64_t drcount_t;

    typedef WeightsPropMap<typename eprop_map_t<count_t>::unchecked_t,
                           std::vector<ewmap_t::unchecked_t>, WN> wmap_t;

    typedef decltype(hana::tuple_t<wmap_t>) wmap_tr;

    template <class State>
    struct m_entries_t
    {
        typedef WEntrySet<is_directed_v<typename State::g_t>,
                          group_t,
                          count_t,
                          GraphInterface::edge_t,
                          typename State::weight_t, drcount_t, rcount_t> type;
    };

    DEFINE_STRUCT_BASED(entropy_args_t, Base::entropy_args_t, WEARGS_params);

    static void python_export()
    {
        using namespace boost::python;

        Base::python_export();

        EXPORT_STRUCT_BASED(entropy_args_t, typename Base::entropy_args_t, WEARGS_params);

        __MOD__::def_obj<enum_<weight_type>>
            ("rec_type")
             ([](auto& c)
              {
                  c.value("none", weight_type::NONE)
                      .value("count", weight_type::COUNT)
                      .value("real_exponential", weight_type::REAL_EXPONENTIAL)
                      .value("real_normal", weight_type::REAL_NORMAL)
                      .value("discrete_geometric", weight_type::DISCRETE_GEOMETRIC)
                      .value("discrete_poisson", weight_type::DISCRETE_POISSON)
                      .value("discrete_binomial", weight_type::DISCRETE_BINOMIAL);
              });
    }

    template <class State>
    static void python_export_state(auto& c)
    {
        Base::template python_export_state<State>(c);

        c.def("get_B_E",
              +[](State& state)
               {
                   return state._spec._B_E;
               });

        c.def("get_B_E_D",
              +[](State& state)
               {
                   return state._spec._B_E_D;
               });

        c.def("print_B_E",
              +[](State& state)
               {
                   std::println("{}", state._spec._B_E);
               });

        c.def("print_B_E_D",
              +[](State& state)
               {
                   std::println("{}", state._spec._B_E_D);
               });

        c.def("print_eweight",
              +[](State& state)
               {
                   for (auto e : edges_range(state._g))
                       std::println("({}, {}) = ({}) | {}",
                                    source(e, state._g), target(e, state._g),
                                    int64_t(state._eweight[e]),
                                    state._eweight[e]);
               });

        c.def("get_eweight",
              +[](State& state, python::object oew)
               {
                   gt_dispatch<>()
                       ([&](auto& ew)
                        {
                            for (auto e : edges_range(state._g))
                                ew[e] = state._eweight[e];
                        }, writable_scalar_edge_properties)(oew);
               });

        c.def("print_mrs",
              +[](State& state)
               {
                   for (auto e : edges_range(state._bg))
                       std::println("({}, {}) = ({}) | {}", source(e, state._bg), target(e, state._bg),
                                    int64_t(state._mrs[e]),
                                    state._mrs[e]);
               });

        c.def("print_recsum",
              +[](State& state)
               {
                   for (auto& x : state._spec._recsum)
                       std::println("{}", x);
               });

        c.def("print_Lrecdx",
              +[](State& state)
               {
                   for (auto& x : state._Lrecdx)
                       std::println("{}", x);
               });
    }

    template <class State>
    struct Imp
        : public Base::template Imp<State>
    {
        using ibase_t = Base::template Imp<State>;
        using spec_t = State::spec_t;
        using rcount_t = spec_t::rcount_t;
        using drcount_t = spec_t::drcount_t;

        typedef spec_t sp;

        GET_PARAMS_SPEC_TYPEDEF(WEIGHTED_SPEC_params)
        GET_PARAMS_SPEC_DEFINE(WEIGHTED_SPEC_params)

        Imp(State& state)
            : ibase_t(state),
              GET_PARAMS_SPEC_INIT(WEIGHTED_SPEC_params)
              _state(state),
              _B_E{}, _B_E_D{}, _wc(_rec_types)
        {
            _recx2.resize(_rec_types.size());
            _recdx.resize(_rec_types.size());

            parallel_edge_loop
                (_state._g,
                 [&](const auto& e)
                 {
                     set_weight(_state._eweight[e], 0,
                                get_count(_state._eweight[e]) > 0);

                     for (size_t i : _wc._normals)
                         set_dev(_state._eweight[e], i,
                                 std::pow(get_weight(_state._eweight[e], i), 2));
                 });

            if (_epsilon.size() != _rec_types.size())
            {
                _epsilon.resize(_rec_types.size(),
                                std::numeric_limits<double>::infinity());
                for (auto e : edges_range(_state._g))
                {
                    if (_state._eweight[e] == 0)
                        continue;
                    for (size_t i = 0; i < _rec_types.size(); ++i)
                    {
                        double w = get_weight(_state._eweight[e], i);
                        if (w != 0)
                            _epsilon[i] = std::min(_epsilon[i], abs(w));
                    }
                }

                for (auto& x : _epsilon)
                    x /= 10;
            }
        }

        template <class T>
        static T unity(T x)
        {
            if constexpr (std::is_scalar_v<T>)
            {
                return 1;
            }
            else
            {
                x /= x;
                return x;
            }
        }

        void post_init()
        {
            ibase_t::post_init();

            _state._eweight.soft_reserve(int64_t(_state._E) + get_num_threads());

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                rcount_t x{};
                enable_concurrent(x, true);

                parallel_edge_loop
                    (_state._bg,
                     [&](const auto& me)
                     {
                         atomic_inc(x, get_weight(_state._mrs[me], i));
                     });

                _recsum.push_back(x);
            }

            if (!_rec_types.empty())
            {
                set_concurrent(true);

                parallel_edge_loop
                    (_state._bg,
                     [&](const auto& me)
                     {
                        if (get_weight(_state._mrs[me], 0) > 0)
                        {
                            atomic_inc(_B_E, thresh_diff(rcount_t(), _state._mrs[me]));
                            for (size_t i : _wc._normals)
                            {
                                atomic_inc(_recx2[i], std::pow(get_weight(_state._mrs[me], i), 2));

                                if (get_weight(_state._mrs[me], 0) > 1)
                                {
                                    auto dx = (get_dev(_state._mrs[me], i) -
                                               std::pow(get_weight(_state._mrs[me], i), 2) / get_weight(_state._mrs[me], 0));
                                    atomic_inc(_recdx[i], dx);
                                }
                            }
                        }

                        if (get_weight(_state._mrs[me], 0) - unity(get_weight(_state._mrs[me], 0)) > 0)
                            atomic_inc(_B_E_D, unity(get_weight(_state._mrs[me], 0) - unity(get_weight(_state._mrs[me], 0))));
                     });

                set_concurrent(false);
            }
        }

        double entropy(const entropy_args_t& ea) const
        {
            double S = ibase_t::entropy(ea);

            double S_dl = 0;
            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                auto& wp = _wparams[i];
                switch (_rec_types[i])
                {
                case weight_type::COUNT:
                    break;
                case weight_type::REAL_EXPONENTIAL:
                    if (ea.rec)
                    {
                       #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._bg,
                             [&](auto&& me)
                             {
                                 auto ers = get_weight(_state._mrs[me], 0);
                                 auto xrs = get_weight(_state._mrs[me], i);
                                 S += -positive_w_log_P(ers, xrs, wp[0], wp[1],
                                                        _epsilon[i]);
                             });
                    }
                    if (ea.rec_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                        S_dl += -positive_w_log_P(_B_E, _recsum[i], wp[0], wp[1],
                                                  _epsilon[i]);
                    break;
                case weight_type::DISCRETE_GEOMETRIC:
                    if (ea.rec)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._bg,
                             [&](auto&& me)
                             {
                                 auto ers = get_weight(_state._mrs[me], 0);
                                 auto xrs = get_weight(_state._mrs[me], i);
                                 S += -geometric_w_log_P(ers, xrs, wp[0], wp[1]);
                             });
                    }
                    if (ea.rec_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                        S_dl += -geometric_w_log_P(_B_E, _recsum[i], wp[0], wp[1]);
                    break;
                case weight_type::DISCRETE_POISSON:
                    if (ea.rec)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._bg,
                             [&](auto&& me)
                             {
                                 auto ers = get_weight(_state._mrs[me], 0);
                                 auto xrs = get_weight(_state._mrs[me], i);
                                 S += -poisson_w_log_P(ers, xrs, wp[0], wp[1]);
                             });
                    }
                    #pragma omp parallel reduction(+:S)
                    parallel_edge_loop_no_spawn
                        (_state._g,
                         [&](auto&& e)
                         {
                             S += safe_lgamma(get_weight(_state._eweight[e], i) + 1);
                         });
                    if (ea.rec_dl && std::isnan(wp[0]) && std::isnan(wp[1]))
                        S_dl += -geometric_w_log_P(_B_E, _recsum[i], wp[0],
                                                   wp[1]);
                    break;
                case weight_type::DISCRETE_BINOMIAL:
                    if (ea.rec)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._bg,
                             [&](auto&& me)
                             {
                                 auto ers = get_weight(_state._mrs[me], 0);
                                 auto xrs = get_weight(_state._mrs[me], i);
                                 S += -binomial_w_log_P(ers, xrs, wp[0], wp[1], wp[2]);
                             });
                        #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._g,
                             [&](auto&& e)
                             {
                                 S -= lbinom(wp[0], get_weight(_state._eweight[e], i));
                             });
                    }
                    if (ea.rec_dl && std::isnan(wp[1]) && std::isnan(wp[2]))
                        S_dl += -geometric_w_log_P(_B_E, _recsum[i], wp[1],
                                                   wp[2]);
                    break;
                case weight_type::REAL_NORMAL:
                    if (ea.rec)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_edge_loop_no_spawn
                            (_state._bg,
                             [&](auto&& me)
                             {
                                 auto ers = get_weight(_state._mrs[me], 0);
                                 auto xrs = get_weight(_state._mrs[me], i);
                                 auto x2rs = get_dev(_state._mrs[me], i);
                                 S += -signed_w_log_P(ers, xrs, x2rs, wp[0], wp[1],
                                                      wp[2], wp[3], _epsilon[i]);
                             });
                    }
                    if (std::isnan(wp[0]) && std::isnan(wp[1]))
                    {
                        if (ea.rec_dl)
                            S_dl += -signed_w_log_P(_B_E, _recsum[i],
                                                    _recx2[i], wp[0], wp[1],
                                                    wp[2], wp[3], _epsilon[i]);
                        S += -positive_w_log_P(_B_E_D, _recdx[i], wp[2],
                                               wp[3], _epsilon[i]);
                    }

                    if (_Lrecdx[0] >= 0)
                    {
                        S_dl += -safelog_fast(_B_E_D);
                        // the prior over all levels is counted once, at the
                        // top of the hierarchy, so that it is included in the
                        // (propagated) entropy of every level
                        if (!_state.has_coupled())
                        {
                            S_dl += -positive_w_log_P(_Lrecdx[0],
                                                      _Lrecdx[i+1],
                                                      wp[2], wp[3],
                                                      _epsilon[i]);
                        }
                    }

                    break;
                }
            }
            return S + ea.beta_dl * S_dl;
        }

        double entries_dS(const entropy_args_t& ea, const auto& m_entries) const
        {
            drcount_t ddL{};
            double dS = 0, dS_dl = 0;

            auto& LdBdx = *m_entries._LdBdx;
            auto& dL = *m_entries._dL;

            if (LdBdx.empty())
                LdBdx.resize(_rec_types.size());

            dS += ibase_t::entries_dS(ea, m_entries);

            auto positive_entries_op =
                [&](size_t i, auto&& w_log_P, auto&& w_log_prior, bool micro_dl)
                {
                    drcount_t dB_E{};

                    entries_op
                        (m_entries,
                         [&](auto t, auto u, const auto& delta)
                         {
                             auto me = _state._emat.get_me(t, u);

                             std::decay_t<decltype(get_weight(_state._mrs[me], 0))> ers{};
                             std::decay_t<decltype(get_weight(_state._mrs[me], i))> xrs{};

                             if (me != _state._emat.get_null_edge())
                             {
                                 ers = get_weight(_state._mrs[me], 0);
                                 xrs = get_weight(_state._mrs[me], i);
                             }

                             auto d = get_weight(delta, 0);
                             auto dx = get_weight(delta, i);

                             if (ea.rec)
                             {
                                 dS -= -w_log_P(ers, xrs);
                                 dS += -w_log_P(ers + d, xrs + dx);
                             }

                             if (ea.rec_dl)
                             {
                                 dB_E += thresh_diff(ers, ers + d);
                                 dB_E -= thresh_diff(ers + d, ers);
                             }
                         });

                    if (dB_E != drcount_t{} && ea.rec_dl && micro_dl)
                    {
                        dS_dl -= -w_log_prior(_B_E);
                        dS_dl += -w_log_prior(_B_E + dB_E);
                    }
                };

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                auto& wp = _wparams[i];
                switch (_rec_types[i])
                {
                case weight_type::COUNT:
                    break;
                case weight_type::REAL_EXPONENTIAL:
                    positive_entries_op(i,
                                        [&](auto N, auto x)
                                        { return positive_w_log_P(N, x, wp[0],
                                                                  wp[1],
                                                                  _epsilon[i]);
                                        },
                                        [&](auto B_E)
                                        { return positive_w_log_P(B_E,
                                                                  _recsum[i],
                                                                  wp[0], wp[1],
                                                                  _epsilon[i]);
                                        }, std::isnan(wp[0]) && std::isnan(wp[1]));
                break;
                case weight_type::DISCRETE_GEOMETRIC:
                    positive_entries_op(i,
                                        [&](auto N, auto x)
                                        { return geometric_w_log_P(N, x, wp[0],
                                                                   wp[1]);
                                        },
                                        [&](auto B_E)
                                        { return geometric_w_log_P(B_E,
                                                                   _recsum[i],
                                                                   wp[0],
                                                                   wp[1]);
                                        }, std::isnan(wp[0]) && std::isnan(wp[1]));
                    break;
                case weight_type::DISCRETE_POISSON:
                    positive_entries_op(i,
                                        [&](auto N, auto x)
                                        { return poisson_w_log_P(N, x, wp[0],
                                                                 wp[1]);
                                        },
                                        [&](auto B_E)
                                        { return geometric_w_log_P(B_E,
                                                                   _recsum[i],
                                                                   wp[0],
                                                                   wp[1]);
                                        }, std::isnan(wp[0]) && std::isnan(wp[1]));
                break;
                case weight_type::DISCRETE_BINOMIAL:
                    positive_entries_op(i,
                                        [&](auto N, auto x)
                                        { return binomial_w_log_P(N, x, wp[0],
                                                                  wp[1], wp[2]);
                                        },
                                        [&](auto B_E)
                                        { return geometric_w_log_P(B_E,
                                                                   _recsum[i],
                                                                   wp[1],
                                                                   wp[2]);
                                        }, std::isnan(wp[1]) && std::isnan(wp[2]));
                    break;
                case weight_type::REAL_NORMAL:
                    {
                        decltype(_B_E) dB_E{};
                        decltype(_B_E_D) dB_E_D{};
                        rcount_t dBx2{};
                        rcount_t dBdx{};

                        entries_op
                            (m_entries,
                             [&](auto t, auto u, auto&& delta)
                             {
                                 auto me = _state._emat.get_me(t, u);

                                 rcount_t ers{};
                                 rcount_t xrs{}, x2rs{};

                                 if (me != _state._emat.get_null_edge())
                                 {
                                     ers = get_weight(_state._mrs[me], 0);
                                     xrs = get_weight(_state._mrs[me], i);
                                     x2rs = get_dev(_state._mrs[me], i);
                                 }

                                 auto d = get_weight(delta, 0);
                                 auto dx = get_weight(delta, i);
                                 auto dx2 = get_dev(delta, i);

                                 auto n_ers = ers + d;

                                 if (ea.rec)
                                 {
                                     dS -= -signed_w_log_P(ers, xrs, x2rs,
                                                           wp[0], wp[1],
                                                           wp[2], wp[3],
                                                           _epsilon[i]);
                                     dS += -signed_w_log_P(n_ers,
                                                           xrs + dx,
                                                           x2rs + dx2,
                                                           wp[0], wp[1],
                                                           wp[2], wp[3],
                                                           _epsilon[i]);
                                     assert(std::isfinite(dS));
                                 }

                                 if (std::isnan(wp[0]) && std::isnan(wp[1]))
                                 {

                                     dB_E += thresh_diff(ers, n_ers);
                                     dB_E -= thresh_diff(n_ers, ers);

                                     dB_E_D += thresh_diff(ers, n_ers, 1);
                                     dB_E_D -= thresh_diff(n_ers, ers, 1);

                                     if (n_ers != 0)
                                         dBdx += (x2rs + dx2 -
                                                  std::pow(xrs + dx, 2) / n_ers) * (n_ers > 1);

                                     if (ers != 0)
                                         dBdx -= (x2rs - std::pow(xrs, 2) / ers) * (ers > 1);

                                     dBx2 += (std::pow(xrs + dx, 2) -
                                              std::pow(xrs, 2));
                                 }
                             });

                        if (std::isnan(wp[0]) && std::isnan(wp[1]))
                        {
                            if (ea.rec_dl && (dB_E != 0 || dBx2 != 0))
                            {
                                dS_dl -= -signed_w_log_P(_B_E, _recsum[i],
                                                         _recx2[i], wp[0], wp[1],
                                                         wp[2], wp[3],
                                                         _epsilon[i]);
                                dS_dl += -signed_w_log_P(_B_E + dB_E,
                                                         _recsum[i],
                                                         _recx2[i] + dBx2, wp[0],
                                                         wp[1], wp[2], wp[3],
                                                         _epsilon[i]);
                                assert(std::isfinite(dS_dl));
                            }

                            if ((dB_E_D != drcount_t{} || dBdx != drcount_t{}))
                            {
                                dS -= -positive_w_log_P(_B_E_D, _recdx[i],
                                                        wp[2], wp[3],
                                                        _epsilon[i]);
                                dS += -positive_w_log_P(_B_E_D + dB_E_D,
                                                        _recdx[i] + dBdx,
                                                        wp[2], wp[3],
                                                        _epsilon[i]);
                                assert(std::isfinite(dS_dl));
                            }

                            auto dL_p = dL;
                            if (ddL == 0)   // FIXME: suspect...
                            {
                                atomic_inc(ddL, thresh_diff(_B_E_D, _B_E_D + dB_E_D));
                                atomic_dec(ddL, thresh_diff(_B_E_D + dB_E_D, _B_E_D));
                                dL += ddL;
                            }

                            if (_Lrecdx[0] >= 0)
                            {
                                auto N_B_E_D = _B_E_D + dB_E_D;

                                dS_dl -= -safelog_fast(_B_E_D);
                                dS_dl += -safelog_fast(N_B_E_D);

                                if constexpr (std::is_scalar_v<std::decay_t<decltype(LdBdx[i])>>)
                                    assert(!std::isnan(LdBdx[i]));

                                auto LdBdx_p = LdBdx[i];
                                LdBdx[i] -= _recdx[i] * _B_E_D;
                                LdBdx[i] += (_recdx[i] + dBdx) * N_B_E_D;

                                if constexpr (std::is_scalar_v<std::decay_t<decltype(LdBdx[i])>>)
                                    assert(!std::isnan(LdBdx[i]));

                                auto& L = _Lrecdx[0];
                                dS_dl -= -positive_w_log_P(L + dL_p,
                                                           _Lrecdx[i+1] + LdBdx_p,
                                                           wp[2], wp[3],
                                                           _epsilon[i]);
                                dS_dl += -positive_w_log_P(L + dL,
                                                           _Lrecdx[i+1] + LdBdx[i],
                                                           wp[2], wp[3],
                                                           _epsilon[i]);

                                assert(std::isfinite(dS_dl));
                            }
                            assert(std::isfinite(dS));
                            assert(std::isfinite(dS_dl));
                        }
                    }
                    break;
                }
            }
            assert(std::isfinite(dS));
            assert(std::isfinite(dS_dl));
            return dS + ea.beta_dl * dS_dl;
        }

        template <bool lock>
        void update_rs(auto& me, const auto& delta, auto&&... locks)
        {
            ibase_t::template update_rs<lock>(me, delta);

            upgrade_canary_lock<{.ula={.la={.enable=lock}}}>
                ([&](bool)
                 {
                     auto mrs = get_weight(_state._mrs[me], 0);
                     auto n_mrs = mrs + get_weight(delta, 0);

                     auto dB_E = thresh_diff(mrs, n_mrs) - thresh_diff(n_mrs, mrs);

                     atomic_inc(_B_E, dB_E);

                     if (dB_E != 0)
                     {
                         _state.visit_coupled_if
                             ([&](auto&& cs, auto&)
                              {
                                  auto&& ew = cs._eweight.get_checked()[me];
                                  set_weight(ew, 0, get_weight(ew, 0) + dB_E);
                                  cs._eweight.sync_size();
                              });
                     }

                     if (!_wc._normals.empty())
                     {
                         if (_Lrecdx[0] >= 0)
                         {
                             for (auto i : _wc._normals)
                                 _Lrecdx[i+1] -= _recdx[i] * _B_E_D;
                         }

                         auto dB_E_D = thresh_diff(mrs, n_mrs, 1) - thresh_diff(n_mrs, mrs, 1);

                         if (_Lrecdx[0] >= 0)
                             _Lrecdx[0] += (thresh_diff(_B_E_D, _B_E_D + dB_E_D) -
                                            thresh_diff(_B_E_D + dB_E_D, _B_E_D));

                         atomic_inc(_B_E_D, dB_E_D);

                         for (auto i : _wc._normals)
                         {
                             auto x = get_weight(_state._mrs[me], i);
                             atomic_dec(_recx2[i], std::pow(x, 2));
                             atomic_inc(_recx2[i], std::pow(x + get_weight(delta, i), 2));
                         }

                         if (n_mrs != 0)
                         {
                             for (auto i : _wc._normals)
                             {
                                 auto dx = (get_dev(_state._mrs[me], i) + get_dev(delta, i)
                                            - std::pow((get_weight(_state._mrs[me], i) +
                                                        get_weight(delta, i)), 2) / n_mrs);
                                 atomic_inc(_recdx[i], dx * (n_mrs > 1));
                             }
                         }

                         if (mrs != 0)
                         {
                             for (auto i : _wc._normals)
                             {
                                 auto dx = (get_dev(_state._mrs[me], i) -
                                            std::pow(get_weight(_state._mrs[me], i), 2) / mrs);
                                 atomic_dec(_recdx[i], dx * (mrs > 1));
                             }
                         }

                         _state.visit_coupled_if
                             ([&](auto&& s, auto&)
                              {
                                  for (auto i : _wc._normals)
                                  {
                                      auto&& ew = s._eweight.get_checked()[me];
                                      set_dev(ew, i, std::pow(get_weight(ew, i) + get_weight(delta, i), 2));
                                  }
                              });

                         if (_Lrecdx[0] >= 0)
                         {
                             for (auto i : _wc._normals)
                                 _Lrecdx[i+1] += _recdx[i] * _B_E_D;
                         }
                     }
                 }, locks...);
        }

        void propagate_delta(auto& bottom, auto u, auto v, auto& delta) const
        {
            ibase_t::propagate_delta(bottom, u, v, delta);

            auto e = bottom._emat.get_me(u, v);

            if (e == bottom._emat.get_null_edge() || get_weight(bottom._mrs[e], 0) == 0)
            {
                if (get_weight(delta, 0) != 0)
                    set_weight(delta, 0, unity(get_weight(delta, 0)));
                else
                    set_weight(delta, 0, 0);

                for (auto i : _wc._normals)
                    set_dev(delta, i, std::pow(get_weight(delta, i), 2));
            }
            else
            {
                auto ers = get_weight(bottom._mrs[e], 0);
                auto n_ers = ers + get_weight(delta, 0);

                auto dB_E = thresh_diff(ers, n_ers) - thresh_diff(n_ers, ers);

                set_weight(delta, 0, dB_E);

                for (auto i : _wc._normals)
                {
                    auto x = get_weight(_state._eweight[e], i);
                    auto nx = x + get_weight(delta, i);
                    set_dev(delta, i, (std::pow(nx, 2) - std::pow(x, 2)));
                }
            }
        }

        void couple_state(auto& state)
        {
            ibase_t::couple_state(state);

            _state.visit_coupled
                ([&](auto&& cs, auto&)
                 {
                    cs._spec._recsum = _recsum;

                    if (cs._Lrecdx[0] < 0)
                        cs._Lrecdx[0] = 0;

                    if (cs._spec._B_E_D != 0)
                        cs._spec._Lrecdx[0] += unity(cs._spec._B_E_D);
                    for (auto i : _wc._normals)
                        cs._spec._Lrecdx[i+1] += cs._spec._recdx[i] * cs._spec._B_E_D;
                 });

            if (_is_base)
            {
                if (_B_E_D != 0)
                    _Lrecdx[0] += unity(_B_E_D);
                for (auto i : _wc._normals)
                    _Lrecdx[i+1] += _recdx[i] * _B_E_D;
            }
        }

        void set_concurrent(bool enable, size_t M=0)
        {
            ibase_t::set_concurrent(enable, M);

            enable_concurrent(_B_E, enable);
            enable_concurrent(_B_E_D, enable);
            for (size_t i : _wc._normals)
            {
                enable_concurrent(_recx2[i], enable);
                enable_concurrent(_recdx[i], enable);
            }

            if (enable)
            {
                _state._E.resize(_wc);
                _state._eweight.sync_size();
                _state._mrs.sync_size();
            }
        }

        void set_concurrent_delta(auto& delta, bool enable) const
        {
            ibase_t::set_concurrent_delta(delta, enable);
            if (enable)
                delta.resize(_wc);
        }

        State& _state;

        std::vector<typename spec_t::rcount_t> _recsum;
        std::vector<typename spec_t::rcount_t> _recx2;
        drcount_t _B_E;
        drcount_t _B_E_D;
        WeightContext _wc;
    };
};

template <class CMap, class WMaps, size_t N>
struct extract_param<WeightsPropMap<CMap, WMaps, N>>
{
    auto operator()(boost::python::object obj)
    {
        python::object acount = obj[0];
        python::object aweights = obj[1];
        python::object adevs = obj[2];
        python::object atypes = obj[3];

        auto count = extract_param<typename CMap::checked_t>()(acount).get_unchecked();

        typedef typename WMaps::value_type::checked_t ewmap_t;

        std::vector<typename ewmap_t::unchecked_t> weights;
        for (int i = 0; i < python::len(aweights); ++i)
            weights.push_back(extract_param<ewmap_t>()(aweights[i]).get_unchecked());

        std::vector<typename ewmap_t::unchecked_t> devs;
        for (int i = 0; i < python::len(adevs); ++i)
            devs.push_back(extract_param<ewmap_t>()(adevs[i]).get_unchecked());

        std::vector<int32_t> types;
        for (int i = 0; i < python::len(atypes); ++i)
            types.push_back(extract_param<int32_t>()(atypes[i]));

        return make_weights_prop<N>(count, weights, devs, types);
    };
};

template <class Value, class Index>
struct extract_param<std::vector<boost::checked_vector_property_map<Value,Index>>>
{
    auto operator()(boost::python::object obj)
    {
        typedef boost::checked_vector_property_map<Value,Index> pmap_t;
        std::vector<pmap_t> ret;
        for (int i = 0; i < boost::python::len(obj); ++i)
            ret.push_back(extract_param<pmap_t>()(obj[i]));
        return ret;
    }
};

} //namespace graph_tool

namespace boost
{
using namespace graph_tool;
template <class Type, class Index>
struct uncheck<std::vector<boost::checked_vector_property_map<Type,Index>>>
{
    typedef std::vector<boost::unchecked_vector_property_map<Type,Index>> type;

    auto operator()(auto&& p)
    {
        std::vector<boost::unchecked_vector_property_map<Type,Index>> np;
        for (auto& x : p)
            np.push_back(x.get_unchecked());
        return np;
    }
};
} // namespace boost
