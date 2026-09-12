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

#ifndef GRAPH_LAYERED_BLOCKMODEL_BASE_HH
#define GRAPH_LAYERED_BLOCKMODEL_BASE_HH

#include "config.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <print>

#include "demangle.hh"

#include "../spec.hh"

#include "../../support/graph_state.hh"

#define LAYERED_SPEC_params                                                    \
    ((cvweight, STATE_SPEC::cvweight_t, 0))

#include "types.hh"
#include "entropy.hh"
#include "deg_stats.hh"

// for layer membership
#include "../overlap/partition.hh"

namespace graph_tool
{
using namespace std;

template <class Base = BlockStateSpec>
struct LayeredBlockStateSpec
    : public Base
{
    using count_t = Base::count_t;
    using group_t = Base::group_t;
    static constexpr group_t null_group = std::numeric_limits<group_t>::max();
    using vcount_t = Base::vcount_t;

    using cmap_t = Base::cmap_t;
    using bmap_t = Base::bmap_t;

    template <class Val>
    using vprop_t = Base::template vprop_t<Val>;

    typedef vprop_t<int64_t> cvweight_t;

    template <class CVal>
    using lwmap_t = LayersPropMap<eprop_map_t<std::vector<lval_t>>::unchecked_t,
                                  typename eprop_map_t<std::vector<CVal>>::unchecked_t, true>;

    template <class CVal>
    using cwmap_t = LayersPropMap<eprop_map_t<std::vector<lval_t>>::unchecked_t,
                                  typename eprop_map_t<std::vector<CVal>>::unchecked_t, false>;

    static inline auto wmaps =
        hana::flatten(hana::transform(typename Base::wmap_tr(),
                                      [](auto&& x)
                                      {
                                          using W = typename decltype(+x)::type::value_type;
                                          return hana::tuple<hana::type<lwmap_t<W>>,
                                                             hana::type<cwmap_t<W>>>();
                                      }));

    typedef decltype(wmaps) wmap_tr;

    struct get_mrp_t
        : public last_type_func_t
    {
        template <class T>
        constexpr static auto get(T)
        {
            return hana::type_c
                <std::conditional_t<T::type::value_type::_layered,
                                    vprop_map_t<LayerCount<count_t, false, true>>,
                                    vprop_map_t<count_t>>>;
        }
    };
    constexpr inline static auto get_mrp_type = hana::tuple_t<get_mrp_t>;

    struct get_wr_t
        : public all_types_func_t
    {
        constexpr static auto get(auto s)
        {
            constexpr auto t = s[BOOST_HANA_STRING("eweight")];
            if constexpr (decltype(t)::type::value_type::_layered)
                return hana::type_c<vprop_map_t<VLayerCount<vcount_t>>>;
            else
                return hana::type_c<vprop_map_t<vcount_t>>;
        }
    };

    constexpr inline static auto get_wr_type = hana::tuple_t<get_wr_t>;

    struct get_vweight_t
        : public last_type_func_t
    {
        template <class T>
        constexpr static auto get(T)
        {
            return hana::type_c
                <std::conditional_t<!std::is_scalar_v<typename T::type::value_type>,
                                    vprop_t<VLayerCount<vcount_t>>,
                                    vprop_t<vcount_t>>>;
        }
    };
    constexpr inline static auto get_vweight_type = hana::tuple_t<get_vweight_t>;

    typedef decltype(get_mrp_type) mrp_tr;
    typedef decltype(same_arg_type) mrm_tr;
    typedef decltype(get_wr_type) wr_tr;
    typedef decltype(get_vweight_type) vweight_tr;

    template <class State>
    struct deg_t
    {
        using base_deg_t = Base::template deg_t<State>::type;
        typedef std::conditional_t<State::weight_t::_layered,
                                   LayerCount<base_deg_t, false, true>,
                                   base_deg_t> type;
    };

    template <class State>
    struct deg_stats_t
    {
        using base_deg_stats_t = Base::template deg_stats_t<State>::type;
        typedef std::conditional_t<State::weight_t::_layered,
                                   LDegStats<base_deg_stats_t>,
                                   base_deg_stats_t> type;
    };

    struct entropy_args_t
        : public Base::entropy_args_t
    {
        bool lmemb = true;
    };

    static void python_export()
    {
        using namespace boost::python;

        Base::python_export();

        __MOD__::def_obj<class_<entropy_args_t, bases<typename Base::entropy_args_t>>>
            (get_type_name<entropy_args_t>().c_str())
            ([](auto& c){ c.def_readwrite("lmemb", &entropy_args_t::lmemb);});
    }

    template <class State>
    static void python_export_state(auto& c)
    {
        Base:: template python_export_state<State>(c);

        c.def("get_C",
              +[](State& state)
               {
                   return state._spec.get_C();
               });
        c.def("get_B",
              +[](State& state)
               {
                   return state._spec.get_B();
               });
        c.def("get_E",
              +[](State& state)
               {
                   boost::python::dict E;
                   for (const auto& [l, c_] : state._spec.get_E())
                       E[l] = c_;
                   return E;
               });
        c.def("get_N",
              +[](State& state)
               {
                   size_t N = 0;
                   for (auto v : vertices_range(state._g))
                       N += state._vweight[v];
                   return N;
               });
        c.def("get_wr",
              +[](State& state, boost::python::object owr)
               {
                 auto wr = extract_param<vprop_map_t<typename State::count_t>>()(owr).get_unchecked();
                 parallel_vertex_loop
                     (state._bg,
                      [&](auto r)
                      {
                          wr[r] = state._wr[r];
                      });
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
        c.def("get_vweight",
              +[](State& state, boost::python::object ovw)
               {
                 auto vw = extract_param<typename State::spec_t::template vprop_t<typename State::count_t>>()(ovw);
                 parallel_vertex_loop
                     (state._g,
                      [&](auto v)
                      {
                          vw[v] = state._vweight[v];
                      });
               });
        c.def("get_lwr",
              +[](State& state, boost::python::object ols, boost::python::object ocs)
               {
                 auto ls = extract_param<vprop_map_t<std::vector<lval_t>>>()(ols).get_unchecked();
                 auto cs = extract_param<vprop_map_t<std::vector<typename State::count_t>>>()(ocs).get_unchecked();
                 parallel_vertex_loop
                     (state._bg,
                      [&](auto r)
                      {
                          VLayerCount<typename State::vcount_t> w;
                          if constexpr (std::is_scalar_v<typename State::vw_t>)
                          {
                              for (auto v : state._groups[r])
                              {
                                  auto kin = in_degree(v, state._g, state._eweight);
                                  auto kout = out_degree(v, state._g, state._eweight);
                                  auto vw = kin + kout;
                                  vw = 1;
                                  w += VLayerCount<typename State::vcount_t>(vw);
                              }
                          }
                          else
                          {
                              w = state._wr[r];
                          }

                          for (auto&& [l, c_] : w._h)
                          {
                              ls[r].push_back(l);
                              cs[r].push_back(c_);
                          }
                      });
               });
        c.def("get_lvweight",
              +[](State& state, boost::python::object als, boost::python::object acs)
               {
                 auto ls = extract_param<vprop_t<std::vector<lval_t>>>()(als).get_unchecked();
                 auto cs = extract_param<vprop_t<std::vector<typename State::count_t>>>()(acs).get_unchecked();
                 parallel_vertex_loop
                     (state._g,
                      [&](auto v)
                      {
                          VLayerCount<typename State::vcount_t> w;
                          if constexpr (std::is_scalar_v<typename State::vw_t>)
                          {
                              auto kin = in_degree(v, state._g, state._eweight);
                              auto kout = out_degree(v, state._g, state._eweight);
                              w = VLayerCount<typename State::vcount_t>(kin + kout);
                              w = 1;
                          }
                          else
                          {
                              w = state._vweight[v];
                          }

                          for (auto&& [l, c_] : w._h)
                          {
                              ls[v].push_back(l);
                              cs[v].push_back(c_);
                          }
                      });
               });
    }

    template <class State>
    static auto get_evweight(auto, auto& e, State& state)
    {
        using vw_t = State::vw_t;
        return vw_t(state._eweight[e]);
    }

    template <class State>
    struct Imp
        : public Base::template Imp<State>
    {
        using ibase_t = Base::template Imp<State>;
        using vw_t = State::vw_t;

        GET_PARAMS_SPEC_TYPEDEF(LAYERED_SPEC_params)
        GET_PARAMS_SPEC_DEFINE(LAYERED_SPEC_params)

        Imp(State& state)
            : ibase_t(state),
              GET_PARAMS_SPEC_INIT(LAYERED_SPEC_params)
              _state(state)
        {
            _state._vweight.soft_reserve(num_vertices(_state._g));

            if constexpr (State::weight_t::_layered)
            {
                size_t M = base_graph(_state._g).get_edge_index_range();
                _state._eweight.soft_reserve(M);

                _state.set_concurrent(true);

                parallel_vertex_loop
                    (_state._g,
                     [&](const auto& v)
                     {
                         auto& vw = _state._vweight[v];
                         vw.set_concurrent(true);
                     });

                parallel_edge_loop
                    (_state._g,
                     [&](const auto& e)
                     {
                         auto u = source(e, _state._g);
                         auto v = target(e, _state._g);
                         _state._vweight[u] += State::spec_t::get_evweight(u, e, _state);
                         _state._vweight[v] += State::spec_t::get_evweight(v, e, _state);
                     });

                parallel_vertex_loop
                    (_state._g,
                     [&](const auto& v)
                     {
                         auto& vw = _state._vweight[v];
                         vw.set_concurrent(false);
                         for (auto&& lc : vw._h)
                             get<1>(lc) = _cvweight[v];
                         vw._total = _cvweight[v];
                     });

                _state.set_concurrent(false);

                for (auto v : vertices_range(_state._g))
                {
                    for (auto&& [l, c] : _state._vweight[v]._h)
                    {
                        ovweight_t w;
                        w[l] = {0, c};
                        _lstats.add_item(l, w);
                    }
                }
            }
        }

        size_t get_C()
        {
            return _state._E._h.size();
        }

        size_t get_B()
        {
            size_t B = 0;
            parallel_vertex_loop
                (_state._bg,
                 [&](const auto& r)
                 {
                     if (int64_t(_state._wr[r]) == 0)
                         return;
                     #pragma omp atomic
                     B++;
                 });
            return B;
        }

        const auto& get_E()
        {
            return _state._E._h;
        }

        double entropy(const entropy_args_t& ea) const
        {
            double S = ibase_t::entropy(ea);
            if (_state._mrs._layered)
            {
                if (ea.lmemb)
                    S += _lstats.get_dl();
            }
            return S;
        }

        void couple_state(auto& state)
        {
            ibase_t::couple_state(state);
            _state.visit_coupled
                ([&](auto&& cs, auto&)
                 {
                     cs._eweight.set_mutex(_state._mrs.get_mutex());
                     cs._eweight.sync_size();
                 });
        }

        void update_group_weight(auto r, auto&& dw)
        {
            ibase_t::update_group_weight(r, dw);
            if constexpr (State::weight_t::_layered)
            {
                _state.visit_coupled_if
                    ([&](auto&& cs, auto&)
                     {
                         auto& wr = _state._wr[r];
                         for (auto&& [l, d] : dw._h)
                         {
                             if (d == 0)
                                 continue;
                             if (wr.get(l) == 0)
                                 cs._spec.set_vertex_weight(r, l, 1);
                             else if (wr.get(l) + d == 0)
                                 cs._spec.set_vertex_weight(r, l, 0);
                         }
                     });
            }
        }

        void set_vertex_weight(auto v, auto&& w)
        {
            ibase_t::set_vertex_weight(v, w);
            if constexpr (State::weight_t::_layered)
                _state._cvweight[v] = w;
        }

        void set_vertex_weight(auto v, auto l, auto w)
        {
            if constexpr (State::weight_t::_layered)
            {
                auto & vw = _state._vweight[v];
                if (w == 1)
                {
                    if (vw._total == 0)
                        vw._total = 1;
                    vw._h[l] = 1;
                }
                else
                {
                    vw._h.erase(l);
                    if (vw._h.empty())
                        vw._total = 0;
                }
                auto r = _state._b[v];
                typename State::vw_t d;
                d._h[l] = (w == 0) ? -1 : 1;
                update_group_weight(r, d);
                _state._wr[r] += d;
            }
        }

        void coupled_resize_vertices(size_t N)
        {
            ibase_t::coupled_resize_vertices(N);

            _state._vweight.soft_reserve(N);

            if constexpr (State::weight_t::_layered)
                _state._cvweight.soft_reserve(N);
        }

        void set_concurrent(bool enable, size_t M)
        {
            ibase_t::set_concurrent(enable, M);

            _state._eweight.sync_size();
            _state._mrs.sync_size();

            _state._E.set_concurrent(enable);
            _state._mrs.set_concurrent(enable);

            if constexpr (State::weight_t::_layered)
            {
                parallel_vertex_loop
                    (_state._bg,
                     [&](auto r)
                     {
                         _state._mrp[r].set_concurrent(enable);
                         if constexpr (is_directed_v<typename State::g_t>)
                             _state._mrm[r].set_concurrent(enable);
                     });
            }
        }

        void set_concurrent_delta(auto&& delta, bool enable = true)
        {
            ibase_t::set_concurrent_delta(delta, enable);
            delta.set_concurrent(enable);
        }

        State& _state;
        size_t _L;

        OverlapPartitionStats<false> _lstats;
    };
};

template <LayerCounts T>
constexpr void enable_concurrent(T& x, bool enable = true)
{
    x.set_concurrent(enable);
}

// parameter extraction

template <class EC, class W, bool layered>
struct extract_param<LayersPropMap<EC, W, layered>>
{
    auto operator()(boost::python::object obj)
    {
        typedef LayersPropMap<EC, W, layered> lpmap_t;
        boost::python::extract<std::any> oecx(obj);
        if (oecx.check())
        {
            auto a = oecx();
            lpmap_t* ret = std::any_cast<lpmap_t>(&a);
            if (ret == nullptr)
                throw bad_extract("invalid lmap_t");
            return *ret;
        }

        using w_val_t = boost::property_traits<W>::value_type::value_type;
        auto& gi = extract_param<GraphInterface&>()(obj[0]);
        auto ec = extract_param<eprop_map_t<std::vector<lval_t>>>()(obj[1]).get_unchecked();
        auto w = extract_param<eprop_map_t<std::vector<w_val_t>>>()(obj[2]).get_unchecked();
        bool l = extract_param<bool>()(obj[3]);

        if (l != layered)
            throw bad_extract("inconsistent independent layered state");

        lpmap_t ret;
        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g)
             {
                 ret = make_layers_prop<layered>(ec, w, g);
             }, all_graph_views)
            (gi.get_graph_view());

        return ret;
    }
};

template <>
struct extract_param<vprop_map_t<VLayerCount<int64_t>>>
{
    auto operator()(boost::python::object)
    {
        return vprop_map_t<VLayerCount<int64_t>>();
    }
};

template <>
struct extract_param<vprop_map_t<LayerCount<int64_t, false, true>>>
{
    auto operator()(boost::python::object)
    {
        return vprop_map_t<LayerCount<int64_t, false, true>>();
    }
};

} //namespace graph_tool

#endif // GRAPH_LAYERED_BLOCKMODEL_BASE_HH
