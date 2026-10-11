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

#include "selectors.hh"
#include "properties.hh"
#include "python_interface.hh"

#include "potts_bp.hh"

#include "random.hh"


using namespace std;
using namespace boost;
using namespace graph_tool;

python::object make_potts_bp_state(GraphInterface& gi, boost::python::object of,
                                   std::any ax, std::any atheta, std::any am,
                                   std::any aem, std::any avm, bool marginal_init,
                                   std::any afrozen, bool init_vm, bool init_em,
                                   bool init_m, rng_t& rng)
{
    PottsBPState::emap_t x =
        std::any_cast<PottsBPState::emap_t::checked_t>(ax).get_unchecked();
    PottsBPState::vmap_t theta =
        std::any_cast<PottsBPState::vmap_t::checked_t>(atheta).get_unchecked();
    PottsBPState::emmap_t m =
        std::any_cast<PottsBPState::emmap_t::checked_t>(am).get_unchecked();
    PottsBPState::emmap_t em =
        std::any_cast<PottsBPState::emmap_t::checked_t>(aem).get_unchecked();
    PottsBPState::vmmap_t vm =
        std::any_cast<PottsBPState::vmmap_t::checked_t>(avm).get_unchecked();
    PottsBPState::vfmap_t frozen =
        std::any_cast<PottsBPState::vfmap_t::checked_t>(afrozen).get_unchecked();

    auto f = get_array<double, 2>(of);

    python::object state;
    run_action<{.dargs={.gil_release=false}}>()
        (gi,
         [&](auto& g)
         {
             GILRelease gil_release;
             PottsBPState pstate(g, f, x, theta, m, em, vm, marginal_init,
                                 frozen, init_vm, init_em, init_m, rng);
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
              +[](PottsBPState& state, GraphInterface& gi)
              {
                  double U = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           U = state.energy(g);
                       })();
                  return U;
              })
         .def("entropy",
              +[](PottsBPState& state, GraphInterface& gi)
              {
                  double S = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           S = state.entropy(g);
                       })();
                  return S;
              })
         .def("marginal_entropy",
              +[](PottsBPState& state, GraphInterface& gi)
              {
                  double S = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           S = state.marginal_entropy(g);
                       })();
                  return S;
              })
         .def("marginal_mean",
              +[](PottsBPState& state, GraphInterface& gi)
              {
                  double M = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           M = state.marginal_mean(g);
                       })();
                  return M;
              })
         .def("sample_energy",
              +[](PottsBPState& state, GraphInterface& gi, std::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.sample_energy(g, s);
                       }, scalar_vertex_properties)(as);
                  return H;
              })
         .def("sample_energies",
              +[](PottsBPState& state, GraphInterface& gi, std::any as)
              {
                  double H = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           H = state.sample_energies(g, s);
                       }, scalar_vector_vertex_properties)(as);
                  return H;
              })
         .def("bethe_lprob",
              +[](PottsBPState& state, GraphInterface& gi, std::any as)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           L = state.bethe_lprob(g, s);
                       }, scalar_vertex_properties)(as);
                  return L;
              })
         .def("marginal_lprob",
              +[](PottsBPState& state, GraphInterface& gi, std::any as)
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
              +[](PottsBPState& state, GraphInterface& gi, std::any as)
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
         .def("marginal_pair_lprob",
              +[](PottsBPState& state, GraphInterface& gi, size_t u, size_t v,
                  size_t r, size_t s)
              {
                  double L = 0;
                  run_action<>()
                      (gi,
                       [&](auto& g)
                       {
                           L = state.marginal_pair_lprob(g, u, v, r, s);
                       })();
                  return L;
              })
         .def("sample",
              +[](PottsBPState& state, GraphInterface& gi, python::object ovorder,
                  std::any as, bool parallel, size_t maxint, double epsilon,
                  rng_t& rng)
              {
                  auto vorder = get_array<uint64_t,1>(ovorder);
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           state.sample(g, vorder, s, parallel, maxint, epsilon,
                                        rng);
                       }, writable_scalar_vertex_properties)(as);
              })
         .def("marginal_sample",
              +[](PottsBPState& state, GraphInterface& gi, std::any as, rng_t& rng)
              {
                  run_action<>()
                      (gi,
                       [&](auto& g, auto s)
                       {
                           state.marginal_sample(g, s, rng);
                       }, writable_scalar_vertex_properties)(as);
              });

     def("make_potts_bp_state", &make_potts_bp_state);
});
