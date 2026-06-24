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

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "numpy_bind.hh"

#include "graph_selectors.hh"
#include "graph_properties.hh"
#include "graph_python_interface.hh"

#include "graph_normal_bp.hh"

#include "random.hh"


using namespace std;
using namespace boost;
using namespace graph_tool;

python::object make_normal_bp_state(GraphInterface& gi, std::any ax,
                                    std::any amu, std::any atheta,
                                    std::any aem_m, std::any aem_s,
                                    std::any avm_m, std::any avm_s,
                                    bool marginal_init,
                                    std::any afrozen, rng_t& rng)
{
    NormalBPState::emap_t x =
        std::any_cast<NormalBPState::emap_t::checked_t>(ax).get_unchecked();
    NormalBPState::vmap_t mu =
        std::any_cast<NormalBPState::vmap_t::checked_t>(amu).get_unchecked();
    NormalBPState::vmap_t theta =
        std::any_cast<NormalBPState::vmap_t::checked_t>(atheta).get_unchecked();
    NormalBPState::emmap_t em_m =
        std::any_cast<NormalBPState::emmap_t::checked_t>(aem_m).get_unchecked();
    NormalBPState::emmap_t em_s =
        std::any_cast<NormalBPState::emmap_t::checked_t>(aem_s).get_unchecked();
    NormalBPState::vmap_t vm_m =
        std::any_cast<NormalBPState::vmap_t::checked_t>(avm_m).get_unchecked();
    NormalBPState::vmap_t vm_s =
        std::any_cast<NormalBPState::vmap_t::checked_t>(avm_s).get_unchecked();
    NormalBPState::vfmap_t frozen =
        std::any_cast<NormalBPState::vfmap_t::checked_t>(afrozen).get_unchecked();

    python::object state;
    run_action<{.dargs={.gil_release=false}}>()
        (gi,
         [&](auto& g)
         {
             GILRelease gil_release;
             NormalBPState pstate(g, x, mu, theta, em_m, em_s, vm_m, vm_s,
                                  marginal_init, frozen, rng);
             gil_release.restore();
             state = python::object(pstate);
         })();
    return state;
}

#define __MOD__ dynamics
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     python::class_<NormalBPState>("NormalBPState", python::no_init)
         .def("iterate",
              +[](NormalBPState& state, GraphInterface& gi, size_t niter)
              {
                  double delta = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           delta = state.iterate(g, niter);
                       })();
                  return delta;
              })
         .def("iterate_parallel",
              +[](NormalBPState& state, GraphInterface& gi, size_t niter)
              {
                  double delta = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           delta = state.iterate_parallel(g, niter);
                       })();
                  return delta;
              })
         .def("update_marginals",
              +[](NormalBPState& state, GraphInterface& gi)
              {
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           state.update_marginals(g);
                       })();
              })
         .def("log_Z",
              +[](NormalBPState& state, GraphInterface& gi)
              {
                  double lZ = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           lZ = state.log_Z(g);
                       })();
                  return lZ;
              })
         .def("energy",
              +[](NormalBPState& state, GraphInterface& gi, std::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.energy(g, s);
                       }, scalar_vertex_properties)(as);
                  return H;
              })
         .def("energies",
              +[](NormalBPState& state, GraphInterface& gi, std::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.energies(g, s);
                       }, scalar_vector_vertex_properties)(as);
                  return H;
              })
         .def("marginal_lprob",
              +[](NormalBPState& state, GraphInterface& gi, std::any as)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           L = state.marginal_lprob(g, s);
                       }, scalar_vertex_properties)(as);
                  return L;
              })
         .def("marginal_lprobs",
              +[](NormalBPState& state, GraphInterface& gi, std::any as)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           L = state.marginal_lprobs(g, s);
                       }, scalar_vector_vertex_properties)(as);
                  return L;
              })
         .def("marginal_sample",
              +[](NormalBPState& state, GraphInterface& gi, std::any as, rng_t& rng)
              {
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           state.marginal_sample(g, s, rng);
                       }, writable_scalar_vertex_properties)(as);
              });

     def("make_normal_bp_state", &make_normal_bp_state);
});
