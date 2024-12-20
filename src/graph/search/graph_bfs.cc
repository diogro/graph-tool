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
#include <boost/graph/breadth_first_search.hpp>

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_util.hh"

#include "coroutine.hh"

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
class BFSVisitorWrapper
{
public:
    BFSVisitorWrapper(GP& gp, python::object vis)
        : _gp(gp), _vis(vis) {}

    template <class Vertex, class Graph>
    void initialize_vertex(Vertex u, Graph& g)
    {
        _vis.attr("initialize_vertex")(PythonVertex<Graph>(_gp, u));
    }

    template <class Vertex, class Graph>
    void discover_vertex(Vertex u, Graph& g)
    {
        _vis.attr("discover_vertex")(PythonVertex<Graph>(_gp, u));
    }

    template <class Vertex, class Graph>
    void examine_vertex(Vertex u, Graph& g)
    {
        _vis.attr("examine_vertex")(PythonVertex<Graph>(_gp, u));
    }

    template <class Edge, class Graph>
    void examine_edge(Edge e, Graph& g)
    {
        _vis.attr("examine_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void tree_edge(Edge e, Graph& g)
    {
        _vis.attr("tree_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void non_tree_edge(Edge e, Graph& g)
    {
        _vis.attr("non_tree_edge")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void gray_target(Edge e, Graph& g)
    {
        _vis.attr("gray_target")(PythonEdge<Graph>(_gp, e));
    }

    template <class Edge, class Graph>
    void black_target(Edge e, Graph& g)
    {
        _vis.attr("black_target")(PythonEdge<Graph>(_gp, e));
    }

    template <class Vertex, class Graph>
    void finish_vertex(Vertex u, Graph& g)
    {
        _vis.attr("finish_vertex")(PythonVertex<Graph>(_gp, u));
    }

private:
    GP _gp;
    boost::python::object _vis;
};

template <class Graph, class Visitor>
void do_bfs(Graph& g, size_t s, Visitor&& vis)
{
    vprop_map_t<default_color_type>
        color(get(vertex_index_t(), g));

    auto v = vertex(s, g);
    if (v == graph_traits<Graph>::null_vertex())
    {
        for (auto u : vertices_range(g))
            vis.initialize_vertex(u, g);

        for (auto u : vertices_range(g))
        {
            if (color[u] == color_traits<default_color_type>::black())
                continue;
            breadth_first_visit(g, u, visitor(vis).color_map(color));
        }
    }
    else
    {
        breadth_first_search(g, v, visitor(vis).color_map(color));
    }
}

void bfs_search(GraphInterface& gi, size_t s, python::object vis)
{
    run_action<decltype(graph_view_ptrs), false>(false)
        (gi, [&](auto& gp){ do_bfs(*gp, s, BFSVisitorWrapper<decltype(gp)>(gp, vis)); })();
}

#ifdef HAVE_BOOST_COROUTINE

template <class GP>
class BFSGeneratorVisitor : public bfs_visitor<>
{
public:
    BFSGeneratorVisitor(GP& gp, coro_t::push_type& yield)
        : _gp(gp), _yield(yield) {}

    template <class Edge, class Graph>
    void tree_edge(const Edge& e, Graph& g)
    {
        _yield(boost::python::object(PythonEdge<Graph>(_gp, e)));
    }

private:
    GP _gp;
    coro_t::push_type& _yield;
};

#endif // HAVE_BOOST_COROUTINE

boost::python::object bfs_search_generator(GraphInterface& g, size_t s)
{
#ifdef HAVE_BOOST_COROUTINE
    auto dispatch = [&](auto& yield)
        {
            run_action<decltype(graph_view_ptrs), false>(false)
                (g, [&](auto& gp){ do_bfs(*gp, s, BFSGeneratorVisitor<decltype(gp)>(gp, yield)); })();
        };
    return boost::python::object(CoroGenerator(dispatch));
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif
}

class BFSArrayVisitor : public bfs_visitor<>
{
public:
    BFSArrayVisitor(std::vector<std::array<size_t, 2>>& edges)
        : _edges(edges) {}

    template <class Edge, class Graph>
    void tree_edge(const Edge& e, Graph& g)
    {
        _edges.push_back({{source(e, g), target(e,g)}});
    }

private:
    std::vector<std::array<size_t, 2>>& _edges;
};

boost::python::object bfs_search_array(GraphInterface& g, size_t s)
{
    std::vector<std::array<size_t, 2>> edges;
    BFSArrayVisitor vis(edges);
    run_action<decltype(all_graph_views), false>()
        (g, [&](auto &g){ do_bfs(g, s, vis); })();
    return wrap_vector_owned<size_t,2>(edges);
}

#define __MOD__ search
#include "module_registry.hh"
REGISTER_MOD
([]
 {
    using namespace boost::python;
    def("bfs_search", &bfs_search);
    def("bfs_search_generator", &bfs_search_generator);
    def("bfs_search_array", &bfs_search_array);
 });
