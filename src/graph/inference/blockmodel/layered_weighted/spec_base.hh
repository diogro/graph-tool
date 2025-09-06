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

#ifndef GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_BASE_HH
#define GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_BASE_HH


// layered spec
#include "../layered/spec_base.hh"

// weighted spec
#include "../weighted/spec_base.hh"

#define LAYERED_WEIGHTED_SPEC_params LAYERED_SPEC_params WEIGHTED_SPEC_params

#include "types.hh"
#include "entropy.hh"

#define LWVEC_N 2

namespace graph_tool
{
using namespace std;

template <class Base = WeightedBlockStateSpec<BlockStateSpec, LWVEC_N>>
struct LayeredWeightedBlockStateSpec
    : public LayeredBlockStateSpec<Base>
{
    typedef LayeredBlockStateSpec<Base> base_t;
    using count_t = base_t::count_t;
    using group_t = base_t::group_t;

    typedef LayerCount<typename base_t::rcount_t, false, true> rcount_t;
    typedef LayerCount<count_t, false, true> drcount_t;
    typedef eprop_map_t<rcount_t> ewmap_t;
    typedef ewmap_t bewmap_t;

    template <class State>
    struct m_entries_t
    {
        typedef WEntrySet<is_directed_v<typename State::g_t>,
                          group_t,
                          count_t,
                          GraphInterface::edge_t,
                          typename State::weight_t, drcount_t, rcount_t> type;
    };

    template <class State>
    struct deg_t
    {
        typedef std::conditional_t<State::weight_t::_layered,
                                   LayerCount<count_t, false, true>,
                                   count_t> type;
    };

    struct entropy_args_t
        : public base_t::entropy_args_t
    {
    };

    static void python_export()
    {
        using namespace boost::python;

        base_t::python_export();

        __MOD__::def_obj<class_<entropy_args_t, bases<typename base_t::entropy_args_t>>>
            (get_type_name<entropy_args_t>().c_str());
        __MOD__::def_obj<class_<ewmap_t>>
            (get_type_name<ewmap_t>().c_str(), no_init);
        __MOD__::def_obj<class_<std::vector<rcount_t>>>
            ("rcount_vec")
            ([](auto&& o)
             {
                 o.def("resize", +[](std::vector<rcount_t>& v, size_t n) { v.resize(n); });
             });
        __MOD__::def_obj<class_<std::vector<drcount_t>>>
            ("drcount_vec");
    }

    template <class State>
    static void python_export_state(auto& c)
    {
        base_t::template python_export_state<State>(c);
        c.def("get_mrs",
              +[](State& state)
               {
                   auto mrs = state._mrs.copy();
                   auto& wc = state._spec._coupled_wc;
                   wc = state._spec._wc;
                   wc._normals.clear();

                   for (size_t i = 0; i < wc._types.size(); ++i)
                   {
                       switch (wc._types[i])
                       {
                       case weight_type::REAL_NORMAL:
                           wc._normals.push_back(i);
                           break;
                       case weight_type::DISCRETE_GEOMETRIC:
                       case weight_type::DISCRETE_BINOMIAL:
                       case weight_type::DISCRETE_POISSON:
                           wc._types[i] = weight_type::DISCRETE_GEOMETRIC;
                           break;
                       default:
                           break;
                       }
                   }

                   parallel_edge_loop
                     (state._bg,
                      [&](auto e)
                      {
                          auto&& w = mrs[e];
                          for (auto&& [l, c] : w._h)
                          {
                              c.resize(wc);
                              for (size_t i = 0; i < c.get_types().size(); ++ i)
                              {
                                  switch (c.get_types()[i])
                                  {
                                  case weight_type::COUNT:
                                      c._weights[i] = 1;
                                      break;
                                  case weight_type::REAL_NORMAL:
                                      c._devs[i] = std::pow(c._weights[i], 2);
                                      break;
                                  default:
                                      c._devs[i] = 0;
                                      break;
                                  }
                              }
                          }
                      });

                   return std::any(mrs);
               })
            .def("print_eweight",
              +[](State& state)
               {
                   for (auto e : edges_range(state._g))
                   {
                       auto&& w = state._eweight[e];
                       std::println("{} {} : {}", source(e, state._g),
                                    target(e, state._g), state._eweight[e]);
                   }
               });
    }

    template <class State>
    struct Imp
        : public base_t::template Imp<State>
    {
        using ibase_t = base_t::template Imp<State>;

        GET_PARAMS_SPEC_TYPEDEF(WEIGHTED_SPEC_params)
        GET_PARAMS_SPEC_DEFINE(WEIGHTED_SPEC_params)

        auto& setup_eweight(auto& state)
        {
            parallel_edge_loop
                (state._g,
                 [&](auto e)
                 {
                     for (auto&& [l, c] : state._eweight[e]._h)
                         c._wc = &state._spec._wc;
                 });
            return state;
        }

        Imp(State& state)
            : ibase_t(setup_eweight(state)),
              GET_PARAMS_SPEC_INIT(WEIGHTED_SPEC_params)
              _state(state)
        {
            if (_state._Lrecdx.empty())
            {
                _state._Lrecdx.resize(ibase_t::_rec_types.size() + 1);
                _state._Lrecdx[0] = -1;
            }
        }

        template <bool lock>
        void update_rs(auto& me, const auto& delta, auto&&... locks)
        {
            ibase_t::template update_rs<lock>(me, delta, locks...);

            if (_state._coupled_state != nullptr)
            {
                upgrade_canary_lock<{.ula={.la={.enable=lock}}}> // TODO: is lock needed?
                    ([&](bool)
                     {
                         auto&& w = _state._coupled_state->_eweight[me];
                         w += delta;
                         for (auto&& [l, c] : w._h)
                         {
                             if (get_count(c) != 0)
                                 set_weight(c, 0, 1);
                             else
                                 set_weight(c, 0, 0);
                         }
                         w.clean();

                         for (auto i : ibase_t::_wc._normals)
                         {
                             auto x = get_weight(w, i);
                             set_dev(w, i, std::pow(x, 2));;
                         }
                     }, locks...);
            }
        }

        bool check_edge_counts() const
        {
            if (_state._coupled_state != nullptr)
            {
                for (auto e : edges_range(_state._bg))
                {
                    typename State::weight_t m = _state._mrs[e];
                    for (auto&& [l, c] : m._h)
                    {
                        if (get_count(c) != 0)
                            set_weight(c, 0, 1);
                        else
                            set_weight(c, 0, 0);
                    }
                    m.clean();
                    assert(m != 0);

                    for (auto i : ibase_t::_wc._normals)
                    {
                        auto x = get_weight(m, i);
                        set_dev(m, i, std::pow(x, 2));;
                    }

                    auto&& w = _state._coupled_state->_eweight[e];
                    if (m != w)
                    {
                        std::println("wrong edge weight for {} {}, should be: {}, is: {}",
                                     source(e, _state._bg),
                                     target(e, _state._bg), m, w);
                        return false;
                    }
                    assert(m == w);
                    return true;
                }
            }
            return true;
        }

        State& _state;
        WeightContext _coupled_wc;
    };

};

template <>
struct extract_param<eprop_map_t<std::vector<WeightCount<double, int64_t, LWVEC_N>>>>
{
    auto operator()(boost::python::object obj)
    {
        typedef eprop_map_t<std::vector<WeightCount<double, int64_t, LWVEC_N>>> ecx_t;

        auto& gi = extract_param<GraphInterface&>()(obj[0]);
        auto ec = extract_param<eprop_map_t<std::vector<lval_t>>>()(obj[1]).get_unchecked();
        auto w = extract_param<eprop_map_t<std::vector<int64_t>>>()(obj[2]).get_unchecked();
        auto x = extract_param<WeightsPropMap<eprop_map_t<std::vector<int64_t>>::unchecked_t,
                                              std::vector<eprop_map_t<std::vector<double>>::unchecked_t>,
                                              LWVEC_N>>()(obj[3]);

        ecx_t ecx;

        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g)
             {
                 ecx.resize(base_graph(g).get_edge_index_range());
                 parallel_edge_loop
                     (g,
                      [&](auto e)
                      {
                          auto& we = ecx[e];
                          auto&& xe = x[e];
                          size_t idx = 0;
                          for ([[maybe_unused]] auto&& [l, c] : std::views::zip(ec[e], w[e]))
                          {
                              we.emplace_back();
                              we.back().resize(x._wc);
                              we.back()._count = get_count(xe)[idx];
                              for (size_t i = 0; i < xe.get_types().size(); ++i)
                              {
                                  if (std::min(get_count(xe).size(),
                                               get_weight(xe, i).size()) <= idx)
                                      throw ValueException(std::format("edge covariate shape not compatible with number of layers: {} {} {} {}",
                                                                       idx,
                                                                       get_count(xe).size(),
                                                                       get_weight(xe, i).size(),
                                                                       get_dev(xe, i).size()));
                                  set_weight(we.back(), i, get_weight(xe, i)[idx]);
                                  //set_dev(we.back(), i, get_dev(xe, i)[idx]);
                              }
                              ++idx;
                          }
                      }, get_openmp_min_thresh(), (ValueException*) nullptr);
             }, all_graph_views)
            (gi.get_graph_view());

        return ecx;
    }
};

} // namespace boost

#endif // GRAPH_LAYERED_WEIGHTED_BLOCKMODEL_BASE_HH
