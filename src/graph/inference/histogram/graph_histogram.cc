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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "graph_histogram.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

template <template <class T> class VT>
GEN_DISPATCH(hist_state, HistD<VT>::template HistState, HIST_STATE_params)

python::object make_hist_state(boost::python::object ostate, size_t D)
{
    python::object state;
    switch (D)
    {
    case 1:
        {
            hist_state<HVa<1>::type>::make_dispatch(ostate,
                                                    [&](auto& s){state = python::object(s);});
        }
        break;
    case 2:
        {
            hist_state<HVa<2>::type>::make_dispatch(ostate,
                                                    [&](auto& s){state = python::object(s);});
        }
        break;
    case 3:
        {
            hist_state<HVa<3>::type>::make_dispatch(ostate,
                                                    [&](auto& s){state = python::object(s);});
        }
        break;
    case 4:
        {
            hist_state<HVa<4>::type>::make_dispatch(ostate,
                                                    [&](auto& s){state = python::object(s);});
        }
        break;
    case 5:
        {
            hist_state<HVa<5>::type>::make_dispatch(ostate,
                                                    [&](auto& s){state = python::object(s);});
        }
        break;
    default:
        {
            hist_state<HVec>::make_dispatch(ostate,
                                            [&](auto& s){state = python::object(s);});
        }
    }
    return state;
}

template <class State>
void dispatch_state_def(State*)
{
    using namespace boost::python;

    class_<State, bases<>, std::shared_ptr<State>>
        c(name_demangle(typeid(State).name()).c_str(),
          no_init);
    c.def("entropy", &State::entropy)
        .def("get_x",
             +[](State& state)
             {
                 auto& x = state.get_x();
                 return wrap_multi_array_not_owned(x);
             })
        .def("get_w",
             +[](State& state)
             {
                 auto& w = state.get_w();
                 return wrap_vector_not_owned(w);
             })
        .def("get_lpdf",
             +[](State& state, python::object xo, bool mle)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 return state.get_lpdf(x, mle);
             })
        .def("get_cond_mean",
             +[](State& state, python::object xo, size_t mean_j, bool mle)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 return state.get_cond_mean(x, mean_j, mle);
             })
        .def("sample",
             +[](State& state, size_t n, python::object cxo, rng_t& rng)
             {
                 auto cx = get_array<typename State::value_t, 1>(cxo);
                 auto x = state.sample(n, cx, rng);
                 return wrap_multi_array_owned(x);
             })
        .def("replace_point_dS",
             +[](State& state, size_t pos, python::object xo)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 return state.virtual_replace_point_dS(pos, x);
             })
        .def("replace_point",
             +[](State& state, size_t pos, size_t w, python::object xo)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 state.replace_point(pos, x, w);
             })
        .def("add_point",
             +[](State& state, size_t pos, size_t w, python::object xo)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 state.template modify_point<true, true>(pos, x, w);
             })
        .def("remove_point",
             +[](State& state, size_t pos)
             {
                 state.template modify_point<false, false>(pos);
             })
        .def("trim_points", &State::trim_points);
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_hist_state", &make_hist_state);

    {
        hist_state<HVa<1>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<HVa<2>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<HVa<3>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<HVa<4>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<HVa<5>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<HVec>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
});
