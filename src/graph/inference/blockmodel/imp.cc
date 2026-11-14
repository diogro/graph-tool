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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "state.hh"
#include "../uncertain/sample_edge_sbm.hh"

#include <array>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>

#define MAKE_STATE BOOST_PP_CAT(make_,BOOST_PP_CAT(STATE_NAME,_state))

// The state types can be split among GT_DISPATCH_SLICES translation units by
// the type of their first parameter (the graph), which define
// GT_DISPATCH_SLICE = 0, ..., GT_DISPATCH_SLICES - 1 before including this
// file, to reduce the memory needed to compile them. Each one creates and
// exports the states of its slice, and the first one also defines the Python
// function that creates the state, which tries the slices in turn (see
// StateWrap::make_dispatch()).
#ifndef GT_DISPATCH_SLICES
#define GT_DISPATCH_SLICES 1
#define GT_DISPATCH_SLICE 0
#endif

#define MAKE_STATE_SLICE(i) BOOST_PP_CAT(MAKE_STATE, BOOST_PP_CAT(_slice_, i))

namespace graph_tool {
namespace STATE_NAME {

#define _MAKE_STATE_SLICE_DECL(z, i, data)                                     \
    python::object MAKE_STATE_SLICE(i)(boost::python::object ostate);
BOOST_PP_REPEAT(GT_DISPATCH_SLICES, _MAKE_STATE_SLICE_DECL, _)

python::object MAKE_STATE_SLICE(GT_DISPATCH_SLICE)(boost::python::object ostate)
{
    python::object state;
    STATE_NAME::make_dispatch<GT_DISPATCH_SLICES, GT_DISPATCH_SLICE>
        (ostate, [&](auto& s){state = python::object(s);});
    return state;
}

#if GT_DISPATCH_SLICE == 0
python::object MAKE_STATE(boost::python::object ostate)
{
#if GT_DISPATCH_SLICES == 1
    return MAKE_STATE_SLICE(0)(ostate);
#else
#define _MAKE_STATE_SLICE_PTR(z, i, data) &MAKE_STATE_SLICE(i)
    std::array<python::object (*)(boost::python::object), GT_DISPATCH_SLICES>
        slices = {BOOST_PP_ENUM(GT_DISPATCH_SLICES, _MAKE_STATE_SLICE_PTR, _)};
    std::vector<std::string> trs;
    std::string val;
    for (auto make : slices)
    {
        try
        {
            return make(ostate);
        }
        catch (const DispatchNotFound& e)
        {
            // the graph has the type of another slice
            trs.insert(trs.end(), e.get_trs().begin(), e.get_trs().end());
            val = e.get_val();
        }
    }
    STATE_NAME::first_param_not_found(trs, val);
#endif
}
#endif

} // namespace graph_tool
} //namespace STATE_NAME


#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    using namespace graph_tool;
    using namespace __MOD__;
    using namespace STATE_NAME;

#if GT_DISPATCH_SLICE == 0
    STATE_SPEC::python_export();

    def(BOOST_PP_STRINGIZE(MAKE_STATE), &MAKE_STATE);
#endif

    graph_tool::STATE_NAME::STATE_NAME::template dispatch<GT_DISPATCH_SLICES,
                                                          GT_DISPATCH_SLICE,
                                                          true>
        ([&](auto* s)
         {
             typedef std::decay_t<decltype(*s)> state_t;
             typedef typename state_t::group_t group_t;
             typedef typename state_t::vertex_t vertex_t;

             void (state_t::*move_vertex)(vertex_t, group_t) =
                 &state_t::move_vertex;
             void (state_t::*move_vertices)(python::object, python::object, bool) =
                 &state_t::move_vertices;
             double (state_t::*virtual_move)(vertex_t, group_t,
                                             const typename state_t::entropy_args_t&) =
                 &state_t::virtual_move;
             double (state_t::*virtual_merge)(group_t, group_t,
                                              const typename state_t::entropy_args_t&) =
                 &state_t::virtual_merge;
             void (state_t::*merge_groups)(group_t, group_t) =
                 &state_t::merge_groups;
             vertex_t (state_t::*random_neighbor)(vertex_t, rng_t&) =
                 &state_t::random_neighbor;
             double (state_t::*get_move_prob)(vertex_t, group_t, group_t, double,
                                              double, bool) =
                 &state_t::get_move_prob;
             void (state_t::*set_partition)(std::any&, bool) =
                 &state_t::set_partition;

             auto& c = get_class<state_t, std::shared_ptr<state_t>, boost::noncopyable>
                 (get_type_name<state_t>().c_str(), no_init);

             c.def("move_vertex", move_vertex)
                 .def("move_vertices", move_vertices)
                 .def("set_partition", set_partition)
                 .def("virtual_move", virtual_move)
                 .def("virtual_merge", virtual_merge)
                 .def("merge_groups", merge_groups)
                 .def("sample_group",
                      +[](state_t& state, vertex_t v, double c_, double d, rng_t& rng)
                      {
                          return state.sample_group(v, c_, d, rng);
                      })
                 .def("sample_neighbor", random_neighbor)
                 .def("entropy", &state_t::entropy)
                 .def("get_move_prob", get_move_prob)
                 .def("modify_edge",
                      +[](state_t& state, vertex_t u, vertex_t v,
                          typename state_t::weight_t dm)
                      {
                          auto e = edge(u, v, state._g).first;
                          if (typename state_t::count_t(dm) > 0)
                          {
                              state.template modify_edge<true> (u, v, e, dm);
                          }
                          else
                          {
                              dm *= -1;
                              state.template modify_edge<false>(u, v, e, dm);
                          }
                      })
                 .def("modify_edge_dS",
                      +[](state_t& state,vertex_t u, vertex_t v,
                          typename state_t::weight_t dm,
                          const typename state_t::entropy_args_t& ea)
                      {
                          auto e = edge(u, v, state._g).first;
                          return state.modify_edge_dS(u, v, e, dm, ea);
                      })
                 .def("gen_eargs",
                      +[](state_t&)
                      {
                          return typename state_t::entropy_args_t();
                      })
                 .def("couple_state",
                      +[](state_t& state, python::object ostate, python::object oea)
                      {
                          if constexpr (std::is_pointer_v<typename state_t::coupled_state_t>)
                          {
                              typedef std::remove_pointer_t<typename state_t::coupled_state_t> cstate_t;
                              auto& cs = extract_param<cstate_t&>()(ostate);
                              auto ea = extract_param<typename cstate_t::entropy_args_t>()(oea);
                              state.couple_state(&cs, ea);
                          }
                          else
                          {
                              constexpr auto types_of =
                                  []<class T, class... Ts>(hana::basic_type<std::variant<T, Ts...>>)
                                  {
                                      return hana::tuple_t<std::remove_pointer_t<Ts>&...>;
                                  };

                              constexpr auto ts = types_of(hana::type_c<typename state_t::coupled_state_t>);

                              gt_dispatch<>()
                                  ([&](auto& cs)
                                   {
                                       typedef std::decay_t<decltype(cs)> cstate_t;
                                       auto ea = extract_param<typename cstate_t::entropy_args_t>()(oea);
                                       state.couple_state(&cs, ea);
                                   }, ts)(ostate);
                          }
                      })
                 .def("decouple_state",
                      &state_t::decouple_state)
                 .def("get_coupled_state",
                      +[](state_t& state)
                      {
                          return state.template visit_coupled<true>
                              ([&](auto&& cs, auto&)
                               {
                                   if constexpr (std::is_same_v<std::decay_t<decltype(cs)>,
                                                                std::monostate>)
                                       return size_t(0);
                                   else
                                       return size_t(&cs);
                               });
                      })
                 .def("get_ptr",
                      +[](state_t& state)
                      {
                          return size_t(&state);
                      })
                 .def("clear_egroups",
                      &state_t::clear_egroups)
                 .def("sync_emat",
                      &state_t::sync_emat)
                 .def("get_edge_sampler",
                      +[](state_t& state, bool edges_only)
                      {
                          return std::make_shared<SBMEdgeSampler<state_t>>(state, edges_only ? 1. : 0.25);
                      })
                 .def("get_actual_B", &state_t::get_actual_B)
                 .def("check_edge_counts", &state_t::check_edge_counts)
                 .def("check_node_counts", &state_t::check_node_counts)
                 .def("get_any",
                      +[](state_t& state)
                      {
                          return std::any(&state);
                      });

             STATE_SPEC::python_export_state<state_t>(c);

             typedef SBMEdgeSampler<state_t> edge_sampler_t;
             class_<edge_sampler_t, std::shared_ptr<edge_sampler_t>, boost::noncopyable>
                 esamp(get_type_name<edge_sampler_t>().c_str(),
                    no_init);
             esamp.def("sample",
                    +[](edge_sampler_t& es, rng_t& rng)
                    {
                        auto e = es.sample(rng);
                        return python::make_tuple(get<0>(e), get<1>(e));
                    })
                 .def("log_prob",
                      +[](edge_sampler_t& es, vertex_t u, vertex_t v, size_t m)
                      {
                          return es.log_prob(u, v, m, 0);
                      })
                 .def("sample_out_neighbor",
                      +[](edge_sampler_t& es, vertex_t v, rng_t& rng)
                      {
                          return es.sample_out_neighbor(v, rng);
                      })
                 .def("log_prob_out",
                      +[](edge_sampler_t& es, vertex_t v, vertex_t u)
                      {
                          return es.log_prob_out(v, u);
                      })
                 .def("sample_in_neighbor",
                      +[](edge_sampler_t& es, vertex_t v, rng_t& rng)
                      {
                          return es.sample_in_neighbor(v, rng);
                      })
                 .def("log_prob_in",
                      +[](edge_sampler_t& es, vertex_t v, vertex_t u)
                      {
                          return es.log_prob_in(v, u);
                      });
         });
}, MODULE_PRIO);
