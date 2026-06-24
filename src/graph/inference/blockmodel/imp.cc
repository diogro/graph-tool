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

#define MAKE_STATE BOOST_PP_CAT(make_,BOOST_PP_CAT(STATE_NAME,_state))

namespace graph_tool {
namespace STATE_NAME {
python::object MAKE_STATE(boost::python::object ostate)
{
    python::object state;
    STATE_NAME::make_dispatch(ostate,
                              [&](auto& s){state = python::object(s);});
    return state;
}
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

    STATE_SPEC::python_export();

    def(BOOST_PP_STRINGIZE(MAKE_STATE), &MAKE_STATE);

    graph_tool::STATE_NAME::STATE_NAME::dispatch
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
             group_t (state_t::*sample_group)(vertex_t, double, double, rng_t&) =
                 &state_t::sample_group;
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
                 .def("sample_group", sample_group)
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
                      &state_t::couple_state)
                 .def("decouple_state",
                      &state_t::decouple_state)
                 .def("get_coupled_state",
                       +[](state_t& state)
                      {
                          return size_t(state.get_coupled_state());
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
                 es(get_type_name<edge_sampler_t>().c_str(),
                    no_init);
             es.def("sample",
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
