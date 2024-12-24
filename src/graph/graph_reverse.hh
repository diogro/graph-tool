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

//  (C) Copyright David Abrahams 2000.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef GRAPH_REVERSE
# define GRAPH_REVERSE

#include <boost/property_map/property_map.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include <utility>

namespace boost {

template <class Graph>
class reversed_graph
    : private Graph
{
public:
    template <class... Ts>
    reversed_graph(Ts&&... args) : Graph(args...) {}

    operator Graph() const = delete;

    typedef Graph original_graph_t;
    typedef typename Graph::all_edge_iterator all_edge_iterator;
    typedef typename graph_traits<Graph>::adjacency_iterator in_adjacency_iterator;

    Graph& original_graph() { return reinterpret_cast<Graph&>(*this); }
    const Graph& original_graph() const { return reinterpret_cast<const Graph&>(*this); }

    typedef typename graph_traits<Graph>::edge_descriptor edge_descriptor;
    void reverse_edge(edge_descriptor& e) const { original_graph().reverse_edge(e); }
};

template <class Graph>
auto& get_reversed_graph(Graph& g)
{
    if constexpr (std::is_convertible_v<typename boost::graph_traits<Graph>::directed_category,
                                        boost::directed_tag>)
        return reinterpret_cast<reversed_graph<Graph>&>(g);
    else
        return g;
}

template <class Graph>
auto& get_reversed_graph(const Graph& g)
{
    if constexpr (std::is_convertible_v<typename boost::graph_traits<Graph>::directed_category,
                                        boost::directed_tag>)
        return reinterpret_cast<const reversed_graph<Graph>&>(g);
    else
        return g;
}

template <class Graph>
struct graph_traits<reversed_graph<Graph>>
    : public graph_traits<Graph>
{
    typedef typename graph_traits<Graph>::in_edge_iterator out_edge_iterator;
    typedef typename graph_traits<Graph>::out_edge_iterator in_edge_iterator;

    typedef typename Graph::in_adjacency_iterator adjacency_iterator;
    typedef typename graph_traits<Graph>::adjacency_iterator in_adjacency_iterator;
};

template <class Graph>
struct graph_traits<const reversed_graph<Graph>>
    : public graph_traits<Graph>
{
    typedef typename graph_traits<Graph>::in_edge_iterator out_edge_iterator;
    typedef typename graph_traits<Graph>::out_edge_iterator in_edge_iterator;

    typedef typename Graph::in_adjacency_iterator adjacency_iterator;
    typedef typename graph_traits<Graph>::adjacency_iterator in_adjacency_iterator;
};

// These are separate so they are not instantiated unless used (see bug 1021)
template <class Graph>
struct vertex_property_type<reversed_graph<Graph> >
{
    typedef typename boost::vertex_property_type<Graph>::type type;
};

template <class Graph>
struct edge_property_type<reversed_graph<Graph> >
{
    typedef typename boost::edge_property_type<Graph>::type type;
};

template <class Graph>
struct graph_property_type<reversed_graph<Graph> >
{
    typedef typename boost::graph_property_type<Graph>::type type;
};

template <class Graph>
auto vertices(const reversed_graph<Graph>& g)
{
    return vertices(g.original_graph());
}

template <class Graph>
auto edges(const reversed_graph<Graph>& g)
{
    return edges(g.original_graph());
}

template <class Graph>
auto out_edges(const typename graph_traits<Graph>::vertex_descriptor u,
               const reversed_graph<Graph>& g)
{
    return in_edges(u, g.original_graph());
}

template <class Graph>
auto
num_vertices(const reversed_graph<Graph>& g)
{
    return num_vertices(g.original_graph());
}

template <class Graph>
auto
num_edges(const reversed_graph<Graph>& g)
{
    return num_edges(g.original_graph());
}

template <class Graph>
auto
out_degree(const typename graph_traits<Graph>::vertex_descriptor u,
           const reversed_graph<Graph>& g)
{
    return in_degree(u, g.original_graph());
}

template <class Graph>
auto
vertex(const typename graph_traits<Graph>::vertices_size_type v,
       const reversed_graph<Graph>& g)
{
    return vertex(v, g.original_graph());
}

template <class Graph>
auto
edge(const typename graph_traits<Graph>::vertex_descriptor u,
     const typename graph_traits<Graph>::vertex_descriptor v,
     const reversed_graph<Graph>& g)
{
    return edge(v, u, g.original_graph());
}

template <class Graph>
auto
in_edges(const typename graph_traits<Graph>::vertex_descriptor u,
         const reversed_graph<Graph>& g)
{
    return out_edges(u, g.original_graph());
}

template <class Graph>
auto
all_edges(const typename graph_traits<Graph>::vertex_descriptor u,
          const reversed_graph<Graph>& g)
{
    return all_edges(u, g.original_graph());
}

template <class Graph>
auto
out_neighbors(typename graph_traits<Graph>::vertex_descriptor u,
               const reversed_graph<Graph>& g)
{
    return in_neighbors(u, g.original_graph());
}

template <class Graph>
auto
in_neighbors(typename graph_traits<Graph>::vertex_descriptor u,
              const reversed_graph<Graph>& g)
{
    return out_neighbors(u, g.original_graph());
}

template <class Graph>
auto
all_neighbors(typename graph_traits<Graph>::vertex_descriptor u,
               const reversed_graph<Graph>& g)
{
    return all_neighbors(u, g.original_graph());
}

template <class Graph>
auto
adjacent_vertices(typename graph_traits<Graph>::vertex_descriptor u,
                  const reversed_graph<Graph>& g)
{
    return out_neighbors(u, g);
}


template <class Graph>
auto
in_degree(const typename graph_traits<Graph>::vertex_descriptor u,
          const reversed_graph<Graph>& g)
{
    return out_degree(u, g.original_graph());
}

template <class Graph>
auto
degree(const typename graph_traits<Graph>::vertex_descriptor u,
       const reversed_graph<Graph>& g)
{
    return degree(u, g.original_graph());
}


template <class Graph>
auto
source(const typename reversed_graph<Graph>::edge_descriptor& e,
       const reversed_graph<Graph>& g)
{
    return target(e, g.original_graph());
}

template <class Graph>
auto
target(const typename reversed_graph<Graph>::edge_descriptor& e,
       const reversed_graph<Graph>& g)
{
    return source(e, g.original_graph());
}

template <class Graph>
auto
_all_edges_out(const typename graph_traits<Graph>::vertex_descriptor u,
               const reversed_graph<Graph>& g)
{
    return _all_edges_in(u, g.original_graph());
}

template <class Graph>
auto
_all_edges_in(const typename graph_traits<Graph>::vertex_descriptor u,
               const reversed_graph<Graph>& g)
{
    return _all_edges_out(u, g.original_graph());
}

template <class Graph, class Pred>
void
clear_vertex(typename boost::graph_traits<reversed_graph<Graph>>
                ::vertex_descriptor v,
             reversed_graph<Graph>& g, Pred&& pred)
{
    clear_vertex(v, const_cast<Graph&>(g.original_graph()), pred);
}

template <class Graph>
auto
add_edge(typename boost::graph_traits<reversed_graph<Graph>>
             ::vertex_descriptor u,
         typename boost::graph_traits<reversed_graph<Graph>>
             ::vertex_descriptor v,
         reversed_graph<Graph>& g)
{
    typedef typename boost::graph_traits<reversed_graph<Graph>>
           ::edge_descriptor e_t;
    std::pair<typename boost::graph_traits<Graph>::edge_descriptor,
              bool> ret =
        add_edge(v, u, const_cast<Graph&>(g.original_graph())); // insert reversed
    return std::make_pair(e_t(ret.first), ret.second);
}

template <class Graph>
void remove_edge(typename boost::graph_traits<reversed_graph<Graph>>
                    ::edge_descriptor e,
                 reversed_graph<Graph>& g)
{
    return remove_edge(e,const_cast<Graph&>(g.original_graph()));
}

template <class Graph>
void remove_vertex(typename boost::graph_traits<reversed_graph<Graph>>
                      ::vertex_descriptor v,
                   reversed_graph<Graph>& g)
{
    return remove_vertex(v,const_cast<Graph&>(g.original_graph()));
}

template <class Graph>
void remove_vertex_fast(typename boost::graph_traits<reversed_graph<Graph>>
                            ::vertex_descriptor v,
                        reversed_graph<Graph>& g)
{
    return remove_vertex_fast(v,const_cast<Graph&>(g.original_graph()));
}

template <class Graph>
void clear_vertex(typename boost::graph_traits<reversed_graph<Graph>>
                     ::vertex_descriptor v,
                  reversed_graph<Graph>& g)
{
    return clear_vertex(v,const_cast<Graph&>(g.original_graph()));
}

template <class Graph>
auto
add_vertex(reversed_graph<Graph>& g)
{
    return add_vertex(const_cast<Graph&>(g.original_graph()));
}

//========================================================================
// Vertex and edge index property maps
//========================================================================

template <class Graph, class Tag>
struct property_map<reversed_graph<Graph>, Tag>
{
    typedef typename property_map<Graph,Tag>::type type;
    typedef typename property_map<Graph,Tag>::const_type const_type;
};

template <class Graph, class Tag>
struct property_map<const reversed_graph<Graph>, Tag>
{
    typedef typename property_map<const Graph,Tag>::type type;
    typedef typename property_map<const Graph,Tag>::const_type
        const_type;
};

template <class Graph, class Tag>
inline auto
get(Tag t, reversed_graph<Graph>& g)
{
    return get(t, g.original_graph());
}

template <class Graph, class Tag>
inline auto
get(Tag t, const reversed_graph<Graph>& g)
{
    return get(t, g.original_graph());
}

} // namespace boost

#endif
