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

#include "bisection_sampler.hh"

template <class Map>
auto to_dict(const Map& m)
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
    using namespace graph_tool;

    class_<bisect_args_t>
        ("bisect_args")
        .def_readwrite("min_bound", &bisect_args_t::min_bound)
        .def_readwrite("max_bound", &bisect_args_t::max_bound)
        .def_readwrite("min_init", &bisect_args_t::min_init)
        .def_readwrite("max_init", &bisect_args_t::max_init)
        .def_readwrite("maxiter", &bisect_args_t::maxiter)
        .def_readwrite("tol", &bisect_args_t::tol)
        .def_readwrite("ftol", &bisect_args_t::ftol)
        .def_readwrite("reversible", &bisect_args_t::reversible)
        .def_readwrite("nmax_extend", &bisect_args_t::nmax_extend);

    class_<BisectionSampler,
           std::shared_ptr<BisectionSampler>>("BisectionSampler",
                                              no_init)
        .def("__init__",
             make_constructor
             (+[](python::object f_, const bisect_args_t& args)
               {
                   auto f =
                       [f_](double x)
                       {
                           return python::extract<double>(f_(x))();
                       };
                   return std::make_shared<BisectionSampler>(f, args);
               }))
        .def("bisect",
             +[](BisectionSampler& s, double x, double delta)
              {
                  return s.bisect(x, delta);
              })
        .def("bisect_fb",
             +[](BisectionSampler& s, const std::vector<double>& vals,
                 bool random, rng_t& rng)
              {
                  if (random)
                      return s.bisect_fb(vals, rng);
                  return s.bisect_fb(vals);
              })
        .def("sample",
             +[](BisectionSampler& s, double beta, double delta, rng_t& rng)
              {
                  return s.sample(beta, delta, rng);
              })
        .def("lprob",
             +[](BisectionSampler& s, double x, double beta, double delta)
              {
                  return s.lprob(x, beta, delta);
              })
        .def("f",
             +[](BisectionSampler& s, double x, bool add)
              {
                  return s.f(x, add);
              })
        .def("get_seg_sampler",
             +[](BisectionSampler& s, double beta)
              {
                  return s.get_seg_sampler(beta);
              })
        .def("get_fcache",
             +[](BisectionSampler& s)
              {
                  return to_dict(s.get_fcache());
              });

    class_<SetBisectionSampler,
           std::shared_ptr<SetBisectionSampler>>("SetBisectionSampler",
                                                 init<const std::vector<double>&,
                                                      double, BisectionSampler&>())
        .def("sample",
             +[](SetBisectionSampler& s, double beta, rng_t& rng)
              {
                  return s.sample(beta, rng);
              })
        .def("lprob",
             +[](SetBisectionSampler& s, double nx, double beta, double skip,
                 double add)
              {
                  return s.lprob(nx, beta, skip, add);
              })
        .def("bracket_closest",
             +[](SetBisectionSampler& s, double x, double skip, double add)
              {
                  auto [a, y, b] = s.bracket_closest(x, skip, add);
                  return python::make_tuple(a, y, b);
              });
        // .def("get_close_int",
        //      +[](SetBisectionSampler& s, double x, double skip, double add)
        //       {
        //           auto [a, c] = s.get_close_int(x, skip, add);
        //           return python::make_tuple(a, c);
        //       });
}, 1);
