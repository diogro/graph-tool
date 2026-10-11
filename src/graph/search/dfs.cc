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

#include "graph_filtering.hh"
#include "python_interface.hh"

#include <boost/python.hpp>
#include <boost/graph/depth_first_search.hpp>

#include "graph.hh"
#include "selectors.hh"
#include "graph_util.hh"

#include "coroutine.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

inline constexpr auto graph_view_ptrs =
    hana::transform(all_graph_views,
                    [](auto t)
                    {
                        return hana::type<std::shared_ptr<std::decay_t<typename decltype(+t)::type>>>();
                    });

template <class GP>
class DFSVisitorWrapper
{
public:
    DFSVisitorWrapper(GP& gp, python::object vis)
        : _gp(gp), _vis(vis) {}


    template <class Vertex, class Graph>
    void initialize_vertex(Vertex u, Graph&)
    {
        _vis.attr("initialize_vertex")(PythonVertex<Graph>(_gp, u));
    }
    template <class Vertex, class Graph>
    void start_vertex(Vertex u, Graph&)
    {
        _vis.attr("start_vertex")(PythonVertex<Graph>(_gp, u));
    }
    template <class Vertex, class Graph>
    void discover_vertex(Vertex u, Graph&)
    {
        _vis.attr("discover_vertex")(PythonVertex<Graph>(_gp, u));
    }

    template <class Edge, class Graph>
    void examine_edge(Edge e, Graph&)
    {
        _vis.attr("examine_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void tree_edge(Edge e, Graph&)
    {
        _vis.attr("tree_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void back_edge(Edge e, Graph&)
    {
        _vis.attr("back_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void forward_or_cross_edge(Edge e, Graph&)
    {
        _vis.attr("forward_or_cross_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Vertex, class Graph>
    void finish_vertex(Vertex u, Graph&)
    {
        _vis.attr("finish_vertex")(PythonVertex<Graph>(_gp, u));
    }

private:
    GP _gp;
    python::object _vis;
};

template <class Graph, class Visitor>
void do_dfs(Graph& g, size_t s, Visitor&& vis)
{
    vprop_map_t<default_color_type>
        color(get(vertex_index_t(), g));
    auto v = vertex(s, g);
    if (v == graph_traits<Graph>::null_vertex())
        depth_first_search(g, vis, color);
    else
        depth_first_visit(g, v, vis, color);
}

void dfs_search(GraphInterface& gi, size_t s, python::object vis)
{
    run_action<{.tr=graph_tool::type<decltype(graph_view_ptrs)>, .dargs={.gil_release=false}}>()
        (gi, [&](auto& gp) { do_dfs(*gp, s, DFSVisitorWrapper<decltype(gp)>(gp, vis));})();
}

#ifdef HAVE_BOOST_COROUTINE

template <class GP>
class DFSGeneratorVisitor : public dfs_visitor<>
{
public:
    DFSGeneratorVisitor(GP& gp, coro_t::push_type& yield)
        : _gp(gp), _yield(yield) {}

    template <class Edge, class Graph>
    void tree_edge(const Edge& e, Graph&)
    {
        _yield(boost::python::object(PythonEdge<Graph>(_gp, e)));
    }

private:
    GP _gp;
    coro_t::push_type& _yield;
};

#endif // HAVE_BOOST_COROUTINE


boost::python::object dfs_search_generator(GraphInterface& g, size_t s)
{
#ifdef HAVE_BOOST_COROUTINE
    auto dispatch = [&](auto& yield)
        {
            run_action<{.tr=graph_tool::type<decltype(graph_view_ptrs)>, .dargs={.gil_release=false}}>()
                (g, [&](auto& gp) { do_dfs(*gp, s, DFSGeneratorVisitor<decltype(gp)>(gp, yield)); })();
        };
    return boost::python::object(CoroGenerator(dispatch));
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif
}

class DFSArrayVisitor: public dfs_visitor<>
{
public:
    DFSArrayVisitor(std::vector<std::array<size_t, 2>>& edges)
        : _edges(edges) {}

    template <class Edge, class Graph>
    void tree_edge(const Edge& e, Graph& g)
    {
        _edges.push_back({{source(e, g), target(e,g)}});
    }

private:
    std::vector<std::array<size_t, 2>>& _edges;
};

boost::python::object dfs_search_array(GraphInterface& gi, size_t s)
{
    std::vector<std::array<size_t, 2>> edges;
    DFSArrayVisitor vis(edges);
    run_action<{.tr=graph_tool::type<decltype(graph_tool::all_graph_views)>,
                .dargs={.gil_release=false}}>()
        (gi, [&](auto &g){ do_dfs(g, s, vis); })();
    return wrap_vector_owned<size_t,2>(edges);
}

#define __MOD__ search
#include "module_registry.hh"
REGISTER_MOD
([]
 {
    using namespace boost::python;
    def("dfs_search", &dfs_search);
    def("dfs_search_generator", &dfs_search_generator);
    def("dfs_search_array", &dfs_search_array);
 });
