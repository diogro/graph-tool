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

#include "graph_filtering.hh"
#include "graph_python_interface.hh"

#include <boost/python.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_util.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

inline constexpr auto graph_view_ptrs =
    hana::transform(all_graph_views,
                    [](auto t)
                    {
                        return hana::type<std::shared_ptr<typename decltype(+t)::type>>();
                    });

template <class GP>
class BFVisitorWrapper
{
public:
    BFVisitorWrapper(GP& gp, python::object vis)
        : _gp(gp), _vis(vis) {}

    template <class Edge, class Graph>
    void examine_edge(Edge e, Graph& g)
    {
        _vis.attr("examine_edge")
            (PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void edge_relaxed(Edge e, Graph& g)
    {
        _vis.attr("edge_relaxed")
            (PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void edge_not_relaxed(Edge e, Graph& g)
    {
        _vis.attr("edge_not_relaxed")
            (PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void edge_minimized(Edge e, Graph& g)
    {
        _vis.attr("edge_minimized")
            (PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void edge_not_minimized(Edge e, Graph& g)
    {
        _vis.attr("edge_not_minimized")
            (PythonEdge<Graph>(_gp, e));
    }

private:
    GP _gp;
    python::object _vis;
};


class BFCmp
{
public:
    BFCmp() {}
    BFCmp(python::object cmp): _cmp(cmp) {}

    template <class Value1, class Value2>
    bool operator()(const Value1& v1, const Value2& v2) const
    {
        return python::extract<bool>(_cmp(v1, v2));
    }

private:
    python::object _cmp;
};

class BFCmb
{
public:
    BFCmb() {}
    BFCmb(python::object cmb): _cmb(cmb) {}

    template <class Value1, class Value2 >
    Value1 operator()(const Value1& v1, const Value2& v2) const
    {
        return python::extract<Value1>(_cmb(v1, v2));
    }

private:
    python::object _cmb;
};

struct do_bf_search
{
    template <class Graph, class DistanceMap, class Visitor>
    void operator()(Graph& g, size_t s, DistanceMap dist, boost::any pred_map,
                    boost::any aweight, Visitor vis, pair<BFCmp,
                    BFCmb> cm, pair<python::object, python::object> range,
                    bool& ret) const
    {
        typedef typename property_traits<DistanceMap>::value_type dtype_t;
        dtype_t z = python::extract<dtype_t>(range.first);
        dtype_t i = python::extract<dtype_t>(range.second);

        typedef vprop_map_t<int64_t> pred_t;
        pred_t pred = any_cast<pred_t>(pred_map);
        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        DynamicPropertyMapWrap<dtype_t, edge_t> weight(aweight,
                                                       edge_properties);
        ret = bellman_ford_shortest_paths
            (g, hard_num_vertices(g),
             root_vertex(vertex(s, g)).visitor(vis).weight_map(weight).
             distance_map(dist).
             predecessor_map(pred).
             distance_compare(cm.first).
             distance_combine(cm.second).distance_inf(i).
             distance_zero(z));
    }
};


bool bellman_ford_search(GraphInterface& g, size_t source, boost::any dist_map,
                         boost::any pred_map, boost::any weight,
                         python::object vis, python::object cmp,
                         python::object cmb, python::object zero,
                         python::object inf)
{
    bool ret = false;
    run_action<decltype(graph_view_ptrs), false>(false)
        (g,
         [&](auto&& gp, auto&& a2)
         {
             return do_bf_search()
                 (*gp, source,
                  std::forward<decltype(a2)>(a2), pred_map, weight,
                  BFVisitorWrapper<decltype(gp)>(gp, vis), make_pair(BFCmp(cmp), BFCmb(cmb)),
                  make_pair(zero, inf), ret);
         },
         writable_vertex_properties)(dist_map);
    return ret;
}

#define __MOD__ search
#include "module_registry.hh"
REGISTER_MOD
([]
 {
    using namespace boost::python;
    def("bellman_ford_search", &bellman_ford_search);
 });
