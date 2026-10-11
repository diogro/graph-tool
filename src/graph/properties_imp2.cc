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
#include "properties.hh"
#include "graph_filtering.hh"
#include "selectors.hh"
#include "graph_util.hh"
#include "python_interface.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class Val1, class Val2>
static void operator+=(std::vector<Val1>& v1, const std::vector<Val2>& v2)
{
    if (v2.size() > v1.size())
        v1.resize(v2.size());
    for (size_t i = 0; i < v2.size(); ++i)
        v1[i] += v2[i];
}

static void operator*=(std::string&, const std::string&)
{
    throw GraphException("Cannot multiply strings.");
}

template <class Val1, class Val2>
static void operator*=(std::vector<Val1>& v1, const std::vector<Val2>& v2)
{
    if (v2.size() > v1.size())
        v1.resize(v2.size());
    for (size_t i = 0; i < v2.size(); ++i)
        v1[i] *= v2[i];
}

struct SumOp
{
    template <class Graph, class Vertex, class EProp, class VProp>
    void operator()(Vertex v, EProp& eprop, VProp& vprop, Graph& g) const
    {
        typedef typename property_traits<VProp>::value_type vval_t;

        size_t count = 0;
        for (const auto& e : out_edges_range(v, g))
        {
            if (count == 0)
                vprop[v] = convert<vval_t>(eprop[e]);
            else
                vprop[v] += convert<vval_t>(eprop[e]);
            ++count;
        }
    }
};

struct ProdOp
{
    template <class Graph, class Vertex, class EProp, class VProp>
    void operator()(Vertex v, EProp& eprop, VProp& vprop, Graph& g) const
    {
        typedef typename property_traits<VProp>::value_type vval_t;

        size_t count = 0;
        for (const auto& e : out_edges_range(v, g))
        {
            if (count == 0)
                vprop[v] = convert<vval_t>(eprop[e]);
            else
                vprop[v] *= convert<vval_t>(eprop[e]);
            ++count;
        }
    }
};


struct MinOp
{
    template <class Graph, class Vertex, class EProp, class VProp>
    void operator()(Vertex v, EProp& eprop, VProp& vprop, Graph& g) const
    {
        typedef typename property_traits<VProp>::value_type vval_t;

        for (const auto& e : out_edges_range(v, g))
        {
            vprop[v] = convert<vval_t>(eprop[e]);
            break;
        }

        for (const auto& e : out_edges_range(v, g))
            vprop[v] = std::min(vprop[v], convert<vval_t>(eprop[e]));
    }
};

struct MaxOp
{
    template <class Graph, class Vertex, class EProp, class VProp>
    void operator()(Vertex v, EProp& eprop, VProp& vprop, Graph& g) const
    {
        typedef typename property_traits<VProp>::value_type vval_t;

        for (const auto& e : out_edges_range(v, g))
        {
            vprop[v] = convert<vval_t>(eprop[e]);
            break;
        }

        for (const auto& e : out_edges_range(v, g))
            vprop[v] = std::max(vprop[v], convert<vval_t>(eprop[e]));
    }
};

struct do_out_edges_op
{
    template <class Graph, class EProp, class OP>
    void operator()(Graph& g, EProp eprop, std::any avprop, OP op) const
    {
        typedef typename property_traits<EProp>::value_type eval_t;
        typedef property_map_t<
            std::conditional_t<std::is_same_v<eval_t, size_t>,
                               int64_t, eval_t>,
            GraphInterface::vertex_index_map_t> VProp;

        auto vprop = std::any_cast<VProp>(avprop).get_unchecked(num_vertices(g));

        #ifdef _OPENMP
        size_t __attribute__ ((unused)) nt = omp_get_num_threads();
        if (std::is_same<eval_t, python::object>::value)
            nt = 1; // python is not thread-safe
        #endif

        #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
            num_threads(nt)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 op(v, eprop, vprop, g);
             });
    }
};

void out_edges_op(GraphInterface& gi, std::any eprop, std::any vprop,
                  std::string op)
{
    if (op == "sum")
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_out_edges_op()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), vprop, SumOp());
             },
             edge_properties)(eprop);
    }
    else if (op == "prod")
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_out_edges_op()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), vprop, ProdOp());
             },
             edge_properties)(eprop);
    }
    else if (op == "min")
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_out_edges_op()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), vprop, MinOp());
             },
             edge_properties)(eprop);
    }
    else if (op == "max")
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_out_edges_op()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2), vprop, MaxOp());
             },
             edge_properties)(eprop);
    }
}
