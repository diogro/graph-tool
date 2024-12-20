// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_potts_bp.hh"

#include "random.hh"


using namespace std;
using namespace boost;
using namespace graph_tool;

python::object make_potts_bp_state(GraphInterface& gi, boost::python::object of,
                                   boost::any ax, boost::any atheta, boost::any aem,
                                   boost::any avm, bool marginal_init,
                                   boost::any afrozen, rng_t& rng)
{
    PottsBPState::emap_t x =
        boost::any_cast<PottsBPState::emap_t::checked_t>(ax).get_unchecked();
    PottsBPState::vmap_t theta =
        boost::any_cast<PottsBPState::vmap_t::checked_t>(atheta).get_unchecked();
    PottsBPState::emmap_t em =
        boost::any_cast<PottsBPState::emmap_t::checked_t>(aem).get_unchecked();
    PottsBPState::vmmap_t vm =
        boost::any_cast<PottsBPState::vmmap_t::checked_t>(avm).get_unchecked();
    PottsBPState::vfmap_t frozen =
        boost::any_cast<PottsBPState::vfmap_t::checked_t>(afrozen).get_unchecked();

    auto f = get_array<double, 2>(of);

    python::object state;
    run_action<>(false)
        (gi,
         [&](auto& g)
         {
             GILRelease gil_release;
             PottsBPState pstate(g, f, x, theta, em, vm, marginal_init, frozen, rng);
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
     python::class_<PottsBPState>("PottsBPState", python::no_init)
         .def("iterate",
              +[](PottsBPState& state, GraphInterface& gi, size_t niter)
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
              +[](PottsBPState& state, GraphInterface& gi, size_t niter)
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
              +[](PottsBPState& state, GraphInterface& gi)
              {
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           state.update_marginals(g);
                       })();
              })
         .def("log_Z",
              +[](PottsBPState& state, GraphInterface& gi)
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
              +[](PottsBPState& state, GraphInterface& gi, boost::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.energy(g, s);
                       }, vertex_scalar_properties)(as);
                  return H;
              })
         .def("energies",
              +[](PottsBPState& state, GraphInterface& gi, boost::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.energies(g, s);
                       }, vertex_scalar_vector_properties)(as);
                  return H;
              })
         .def("marginal_lprob",
              +[](PottsBPState& state, GraphInterface& gi, boost::any as)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           L = state.marginal_lprob(g, s);
                       }, vertex_scalar_properties)(as);
                  return L;
              })
         .def("marginal_lprobs",
              +[](PottsBPState& state, GraphInterface& gi, boost::any as)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           L = state.marginal_lprobs(g, s);
                       }, vertex_scalar_vector_properties)(as);
                  return L;
              })
         .def("sample",
              +[](PottsBPState& state, GraphInterface& gi, boost::any as, rng_t& rng)
              {
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           state.sample(g, s, rng);
                       }, writable_vertex_scalar_properties)(as);
              });

     def("make_potts_bp_state", &make_potts_bp_state);
});
