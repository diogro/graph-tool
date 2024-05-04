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

#define BOOST_PYTHON_MAX_ARITY 30
#include <boost/python.hpp>

#include "graph_tool.hh"
#include "random.hh"

#define GRAPH_VIEWS never_filtered_never_reversed
#include "../../blockmodel/graph_blockmodel.hh"

#include "dynamics.hh"
#include "segment_sampler.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
struct Dyn : Dynamics<BaseState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

python::object make_dynamics_state(boost::python::object oblock_state,
                                   boost::python::object odynamics_state)
{
    python::object state;
    auto dispatch = [&](auto& block_state)
        {
            typedef typename std::remove_reference<decltype(block_state)>::type
            state_t;

            dynamics_state<state_t>::make_dispatch
                (odynamics_state,
                 [&](auto& s)
                 {
                     state = python::object(s);
                 },
                 block_state);
        };
    block_state::dispatch(oblock_state, dispatch);
    return state;
}

template <class Map>
auto to_dict(Map& m)
{
    boost::python::dict d;
    for (auto [k, v] : m)
        d[k] = v;
    return d;
};

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;

    class_<dentropy_args_t, bases<uentropy_args_t>>
        ("dentropy_args", init<uentropy_args_t>())
        .def_readwrite("alpha", &dentropy_args_t::alpha)
        .def_readwrite("delta", &dentropy_args_t::delta)
        .def_readwrite("xdist", &dentropy_args_t::xdist)
        .def_readwrite("tdist", &dentropy_args_t::tdist)
        .def_readwrite("xdist_uniform", &dentropy_args_t::xdist_uniform)
        .def_readwrite("tdist_uniform", &dentropy_args_t::tdist_uniform)
        .def_readwrite("xl1", &dentropy_args_t::xl1)
        .def_readwrite("tl1", &dentropy_args_t::tl1)
        .def_readwrite("normal", &dentropy_args_t::normal)
        .def_readwrite("mu", &dentropy_args_t::mu)
        .def_readwrite("sigma", &dentropy_args_t::sigma);

    class_<DStateBase, bases<>, std::shared_ptr<DStateBase>,
           boost::noncopyable>("DStateBase", no_init);

    class_<SegmentSampler>("SegmentSampler", init<std::vector<double>&,
                                                  std::vector<double>&>())
        .def("sample",
             +[](SegmentSampler& s, rng_t& rng)
              {
                  return s.sample(rng);
              })
        .def("lprob", &SegmentSampler::lprob)
        .def("lprob_int", &SegmentSampler::lprob_int);

    def("qlaplace_lprob", &qlaplace_lprob<double>);

    def("make_dynamics_state", &make_dynamics_state);

    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             dynamics_state<block_state_t>::dispatch
                 ([&](auto* s)
                  {
                      typedef typename std::remove_reference<decltype(*s)>::type state_t;
                      auto name = name_demangle(typeid(state_t).name());
                      auto& c = __MOD__::get_class<state_t, bases<>, std::shared_ptr<state_t>,
                                                   boost::noncopyable>(name.c_str(), no_init);
                      c.def("remove_edge",
                            +[](state_t& state, size_t u, size_t v, int dm)
                            {
                                state.remove_edge(u, v, dm);
                            })
                          .def("add_edge",
                               +[](state_t& state, size_t u, size_t v, int dm, double x)
                                {
                                    state.add_edge(u, v, dm, x);
                                })
                          .def("update_edge",
                               +[](state_t& state, size_t u, size_t v, double x)
                                {
                                    state.update_edge(u, v, x);
                                })
                          .def("update_node",
                               +[](state_t& state, size_t v, double x)
                                {
                                    state.update_node(v, x);
                                })
                          .def("remove_edge_dS",
                               +[](state_t& state, size_t u, size_t v, int dm,
                                   const dentropy_args_t& ea)
                                {
                                    return state.remove_edge_dS(u, v, dm, ea, true);
                                })
                          .def("add_edge_dS",
                               +[](state_t& state, size_t u, size_t v, int dm, double x,
                                   const dentropy_args_t& ea)
                                {
                                    return state.add_edge_dS(u, v, dm, x, ea, true, false);
                                })
                          .def("update_edge_dS",
                               +[](state_t& state, size_t u, size_t v, double nx,
                                   const dentropy_args_t& ea)
                                {
                                    return state.update_edge_dS(u, v, nx, ea, true);
                                })
                          .def("update_node_dS",
                               +[](state_t& state, size_t v, double nt,
                                   const dentropy_args_t& ea)
                                {
                                    return state.update_node_dS(v, nt, ea, true);
                                })
                          .def("edge_diff",
                               +[](state_t& state, size_t u, size_t v, double h,
                                   const dentropy_args_t& ea)
                                {
                                    return state.edge_diff(u, v, h, ea);
                                })
                          .def("node_diff",
                               +[](state_t& state, size_t v, double h,
                                   const dentropy_args_t& ea)
                                {
                                    return state.node_diff(v, h, ea);
                                })
                          .def("xvals_sweep",
                               +[](state_t& state, double beta, double r, size_t maxiter, double tol,
                                   size_t min_size, const dentropy_args_t& ea, rng_t& rng)
                                {
                                    auto [dS, nmoves] =
                                        state.xvals_sweep(beta, r, maxiter, tol,
                                                          min_size, ea, rng);
                                    return python::make_tuple(dS, nmoves);
                                })
                          .def("tvals_sweep",
                               +[](state_t& state, double beta, double r, size_t maxiter, double tol,
                                   size_t min_size, const dentropy_args_t& ea, rng_t& rng)
                                {
                                    auto [dS, nmoves] =
                                        state.tvals_sweep(beta, r, maxiter, tol,
                                                          min_size, ea, rng);
                                    return python::make_tuple(dS, nmoves);
                                })
                          .def("entropy", &state_t::entropy)
                          .def("get_node_prob", &state_t::get_node_prob)
                          .def("get_edge_prob",
                               +[](state_t& state, size_t u, size_t v, double x,
                                   const dentropy_args_t& ea, double epsilon)
                                {
                                    return get_edge_prob(state, u, v, ea, epsilon, x);
                                })
                          .def("get_edges_prob",
                               +[](state_t& state, python::object edges,
                                   python::object probs, const dentropy_args_t& ea,
                                   double epsilon)
                                {
                                    get_xedges_prob(state, edges, probs, ea, epsilon);
                                })
                          .def("bisect_x",
                               +[](state_t& state, size_t u, size_t v,
                                   size_t maxiter, double tol, const dentropy_args_t& ea, bool reversible, bool fp, rng_t& rng)
                                {
                                    auto [x, xcache] = state.bisect_x(u, v, maxiter, tol, ea, reversible, fp, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("bisect_t",
                               +[](state_t& state, size_t v,
                                   size_t maxiter, double tol, const dentropy_args_t& ea, bool reversible, bool fp, rng_t& rng)
                                {
                                    auto [x, xcache] = state.bisect_t(v, maxiter, tol, ea, reversible, fp, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_x",
                               +[](state_t& state, size_t u,  size_t v,
                                   double beta, size_t maxiter,
                                   double tol, const dentropy_args_t& ea, bool fb, rng_t& rng)
                                {
                                    auto [x, xcache] = state.sample_x(u, v, beta, maxiter, tol, ea, fb, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_t",
                               +[](state_t& state,  size_t v, double beta, size_t maxiter, double tol,
                                   const dentropy_args_t& ea, bool fb, rng_t& rng)
                                {
                                    auto [x, xcache] = state.sample_t(v, beta, maxiter, tol, ea, fb, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_xl1",
                               +[](state_t& state, double minval, double maxval, double beta, size_t maxiter, double tol,
                                   const dentropy_args_t& ea, rng_t& rng)
                                {
                                    auto [x, xcache] = state.sample_xl1(beta, minval, maxval, maxiter, tol, ea, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_tl1",
                               +[](state_t& state, double minval, double maxval, double beta, size_t maxiter, double tol,
                                   const dentropy_args_t& ea, rng_t& rng)
                                {
                                    auto [x, xcache] = state.sample_tl1(beta, minval, maxval, maxiter, tol, ea, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_delta",
                               +[](state_t& state, double minval, double maxval, double beta, size_t maxiter, double tol,
                                   const dentropy_args_t& ea, rng_t& rng)
                                {
                                    auto [x, xcache] = state.sample_delta(beta, minval, maxval, maxiter, tol, ea, rng);
                                    return python::make_tuple(x, to_dict(xcache));
                                })
                          .def("sample_val_lprob",
                               +[](state_t& state, double x, python::object xc, double beta)
                                {
                                    std::map<double, double> xcache;
                                    auto from_dict = [](python::dict d, auto& m)
                                        {
                                            auto items = python::list(d.items());
                                            for (int i = 0; i < len(items); ++i)
                                            {
                                                m[python::extract<double>(items[i][0])] =
                                                    python::extract<double>(items[i][1]);
                                            }
                                        };
                                    from_dict(python::dict(xc), xcache);
                                    return state.sample_val_lprob(x, xcache, beta);
                                })
                          .def("quantize_x",
                               +[](state_t& state, double delta)
                                {
                                    state.quantize_x(delta);
                                })
                          .def("get_xvals", +[](state_t& state)
                                             {
                                                 return wrap_vector_not_owned<double>(state.get_xvals());
                                             })
                          .def("get_tvals", +[](state_t& state)
                                             {
                                                 return wrap_vector_not_owned<double>(state.get_tvals());
                                             })
                          .def("node_TE", &state_t::node_TE)
                          .def("node_MI", &state_t::node_MI)
                          .def("node_cov", &state_t::node_cov)
                          .def("get_candidate_edges",
                               +[](state_t& state, GraphInterface& gi, size_t k,
                                   double r, size_t max_rk, double epsilon,
                                   bool c_stop, size_t max_iter, boost::any aw,
                                   boost::any aei, bool keep_iter,
                                   dentropy_args_t ea, bool exact,
                                   bool knn, bool keep_all, bool gradient, double h,
                                   size_t f_max_iter, double tol, bool allow_edges,
                                   bool include_edges, bool use_hint, size_t nrandom,
                                   bool verbose, rng_t& rng)
                                {
                                    typedef eprop_map_t<double>::type wmap_t;
                                    wmap_t w = boost::any_cast<wmap_t>(aw);
                                    typedef eprop_map_t<int32_t>::type eimap_t;
                                    eimap_t ei = boost::any_cast<eimap_t>(aei);
                                    std::tuple<size_t, size_t, size_t> ret;
                                    run_action<always_directed_never_filtered_never_reversed>()
                                        (gi,
                                         [&](auto& g)
                                         {
                                             if (keep_iter)
                                                 ret = state.template get_candidate_edges<true>
                                                          (g, k, r, max_rk, epsilon, c_stop, max_iter,
                                                           w, ei, ea, exact, knn, keep_all,
                                                           gradient, h, f_max_iter, tol,
                                                           allow_edges, include_edges,
                                                           use_hint, nrandom, verbose, rng);
                                             else
                                                 ret = state.template get_candidate_edges<false>
                                                          (g, k, r, max_rk, epsilon, c_stop, max_iter,
                                                           w, ei, ea, exact, knn, keep_all,
                                                           gradient, h, f_max_iter, tol,
                                                           allow_edges, include_edges,
                                                           use_hint, nrandom, verbose, rng);
                                         })();
                                    return python::make_tuple(get<0>(ret), get<1>(ret), get<2>(ret));
                                })
                          .def("set_dstate", &state_t::set_dstate)
                          .def("set_params", &state_t::set_params)
                          .def("reset_m",
                               +[](state_t&)// state)
                                {
                                    //state._dstate->reset_m(state);
                                });
                  });
         });
}, 5);
