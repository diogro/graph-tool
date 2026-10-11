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

#ifndef SELECTORS_HH
#define SELECTORS_HH

#pragma GCC visibility push(default)

#include <utility>
#include <ranges>
#include <type_traits>
#include <any>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/graph/graph_traits.hpp>

#include "properties.hh"
#include "graph.hh"

namespace graph_tool
{

// This file contain selector for degree types for different types of
// graphs. Namely, they will return the in degree, out degree, and total degree
// for both directed and undirected graphs. There is also an scalar selector,
// which will return a specific scalar property

// This file also contains selectors for in_edge iterators of graphs, which
// return an empty range for undirected graphs

namespace detail
{
    struct no_weightS {};

    template <class Weight>
    struct get_weight_type
    {
        typedef typename boost::property_traits<typename std::remove_reference<Weight>::type>::value_type type;
    };

    template <>
    struct get_weight_type<no_weightS&>
    {
        typedef size_t type;
    };

    template <>
    struct get_weight_type<no_weightS>
    {
        typedef size_t type;
    };
}

struct in_degreeS
{
    typedef size_t value_type;

    in_degreeS() {}

    template <class Graph>
    auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                    const Graph& g) const
    {
        return in_degreeS::operator()(v, g, detail::no_weightS());
    }

    template <class Graph, class Weight>
    auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                    const Graph& g, Weight&& weight) const
    {
        typedef typename std::is_convertible
            <typename boost::graph_traits<Graph>::directed_category,
             boost::directed_tag>::type is_directed;
        return get_in_degree(v, g, is_directed(), std::forward<Weight>(weight));
    }

    template <class Graph>
    auto get_in_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                       const Graph& g, std::true_type,
                       detail::no_weightS) const
    {
        return in_degree(v, g);
    }

    template <class Graph, class Key, class Value>
    auto get_in_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                       const Graph& g, std::true_type,
                       const ConstantPropertyMap<Value, Key>& weight) const
    {
        return in_degree(v, g) * weight.c;
    }

    template <class Graph, class Key, class Value>
    auto get_in_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                       const Graph& g, std::true_type,
                       const UnityPropertyMap<Value, Key>&) const
    {
        return in_degree(v, g);
    }

    template <class Graph, class Weight>
    auto get_in_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                       const Graph& g, std::true_type, Weight& weight) const
    {
        typedef typename boost::property_traits<Weight>::value_type val_t;
        val_t d = val_t();
        typename boost::graph_traits<Graph>::in_edge_iterator e, e_end;
        for (std::tie(e, e_end) = in_edges(v, g); e != e_end; ++e)
            d += get(weight, *e);
        return d;
    }

    template <class Graph, class Weight>
    auto get_in_degree(const typename boost::graph_traits<Graph>::vertex_descriptor&,
                       const Graph&, std::false_type, Weight&&)  const
    {
        if constexpr (std::is_same_v<std::decay_t<Weight>, detail::no_weightS>)
        {
            return 0;
        }
        else
        {
            typedef typename boost::property_traits<std::decay_t<Weight>>::value_type val_t;
            return val_t();
        }
    }
};

auto in_degree(auto v, const auto& g, const auto& ew)
{
    return in_degreeS()(v, g, ew);
}

struct out_degreeS
{
    typedef size_t value_type;

    out_degreeS() {}

    template <class Graph>
    auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                    const Graph& g) const
    {
        return out_degreeS::operator()(v, g, detail::no_weightS());
    }

    template <class Graph, class Weight>
    auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                    const Graph& g, Weight&& weight) const
    {
        return get_out_degree(v, g, std::forward<Weight>(weight));
    }

    template <class Graph, class Key, class Value>
    auto get_out_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g,
                        const ConstantPropertyMap<Value, Key>& weight) const
    {
        return out_degree(v, g) * weight.c;
    }

    template <class Graph, class Key, class Value>
    auto get_out_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g,
                        const UnityPropertyMap<Value, Key>&) const
    {
        return out_degree(v, g);
    }

    template <class Graph, class Weight>
    auto get_out_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g, const Weight& weight) const
    {
        typedef typename boost::property_traits<Weight>::value_type val_t;
        val_t d = val_t();
        typename boost::graph_traits<Graph>::out_edge_iterator e, e_end;
        for (std::tie(e, e_end) = out_edges(v, g); e != e_end; ++e)
            d += get(weight, *e);
        return d;
    }

    template <class Graph>
    auto get_out_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g, detail::no_weightS) const
    {
        return out_degree(v, g);
    }
};

auto out_degree(auto v, const auto& g, const auto& ew)
{
    return out_degreeS()(v, g, ew);
}

struct total_degreeS
{
    typedef size_t value_type;

    total_degreeS() {}
    template <class Graph>
        auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g) const
    {
        return total_degreeS::operator()(v, g, detail::no_weightS());
    }

    template <class Graph, class Weight>
        auto operator()(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g, Weight&& weight) const
    {
        typedef typename std::is_convertible
            <typename boost::graph_traits<Graph>::directed_category,
             boost::directed_tag>::type is_directed;
        return get_total_degree(v, g, is_directed(),
                                std::forward<Weight>(weight));
    }

    template <class Graph, class Weight>
        auto get_total_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                              const Graph& g, std::true_type,
                              Weight&& weight) const
    {
        return in_degreeS()(v, g, std::forward<Weight>(weight)) +
            out_degreeS()(v, g, std::forward<Weight>(weight));
    }

    template <class Graph, class Weight>
        auto get_total_degree(typename boost::graph_traits<Graph>::vertex_descriptor v,
                              const Graph& g, std::false_type, Weight&& weight) const
    {
        return out_degreeS()(v, g, std::forward<Weight>(weight));
    }
};

auto total_degree(auto v, const auto& g, const auto& ew)
{
    return total_degreeS()(v, g, ew);
}

auto total_degree(auto v, const auto& g)
{
    return total_degreeS()(v, g);
}

auto in_or_out_degree(auto v, const auto& g, auto&&... w)
{
    if constexpr (std::is_convertible_v
                  <typename boost::graph_traits<std::decay_t<decltype(g)>>::directed_category,
                  boost::directed_tag>)
        return in_degree(v, g, w...);
    else
        return out_degree(v, g, w...);
}

template <class PropertyMap>
struct scalarS
{
    typedef typename boost::property_traits<PropertyMap>::value_type value_type;

    scalarS() {}
    scalarS(PropertyMap pmap): _pmap(pmap) {}

    template <class Descriptor, class Graph>
    auto operator()(const Descriptor& d, const Graph&) const
    {
        return get(_pmap, d);
    }

    PropertyMap _pmap;
};

struct get_degree_selector
{
    typedef boost::mpl::map
        <boost::mpl::pair<in_degreeS,
                          boost::mpl::int_<GraphInterface::IN_DEGREE> >,
         boost::mpl::pair<out_degreeS,
                          boost::mpl::int_<GraphInterface::OUT_DEGREE> >,
         boost::mpl::pair<total_degreeS,
                          boost::mpl::int_<GraphInterface::TOTAL_DEGREE> > >
        degree_selector_index;

    template <class Selector>
    void operator()(Selector, int deg_index, std::any& deg) const
    {
        if (boost::mpl::at<degree_selector_index, Selector>::type::value == deg_index)
            deg = Selector();
    }
};

struct get_scalar_selector
{
    template <class PropertyMap>
    void operator()(PropertyMap, std::any prop, std::any& sec, bool& found) const
    {
        try
        {
            PropertyMap map = std::any_cast<PropertyMap>(prop);
            sec = scalarS<PropertyMap>(map);
            found = true;
        }
        catch (std::bad_any_cast&) {}
    }
};

inline constexpr auto degree_selectors =
    hana::tuple_t<in_degreeS, out_degreeS, total_degreeS>;


inline constexpr auto scalar_selectors =
    hana::transform(scalar_vertex_properties,
                    [](auto t){ return hana::type<scalarS<typename decltype(t)::type>>();});

inline constexpr auto prop_selectors =
    hana::transform(vertex_properties,
                    [](auto t){ return hana::type<scalarS<typename decltype(t)::type>>();});

inline constexpr auto all_selectors =
    hana::concat(degree_selectors, scalar_selectors);

inline constexpr auto all_prop_selectors =
    hana::concat(degree_selectors, prop_selectors);


// retrieves the appropriate degree selector
std::any degree_selector(GraphInterface::deg_t deg);

// helper types for in_edge_iteratorS

template <class Graph, class IsDirected>
struct get_in_edges
{
    BOOST_MPL_ASSERT((std::is_same<IsDirected,std::true_type>));
    BOOST_MPL_ASSERT((std::is_convertible
                      <typename boost::graph_traits<Graph>::directed_category,
                      boost::directed_tag>));
    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    typedef typename boost::graph_traits<Graph>::in_edge_iterator type;
    static std::pair<type,type> get_edges(vertex_descriptor v,
                                          const Graph& g)
    {
        return in_edges(v, g);
    }
};

template <class Graph>
struct get_in_edges<Graph,std::false_type>
{
    BOOST_MPL_ASSERT((std::is_convertible
                      <typename boost::graph_traits<Graph>::directed_category,
                      boost::undirected_tag>));
    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    typedef typename boost::graph_traits<Graph>::out_edge_iterator type;
    static std::pair<type,type> get_edges(vertex_descriptor,
                                          const Graph&)
    {
        return std::make_pair(type(), type());
    }
};

// this in-edge iterator selector returns the in-edge range for directed graphs
// and an empty out-edge range for undirected graphs. The iterator type is given
// by in_edge_iteratorS<Graph>::type.
template <class Graph>
struct in_edge_iteratorS
{
    typedef typename boost::graph_traits<Graph>::directed_category
        directed_category;
    typedef typename std::is_convertible<directed_category,
                                         boost::directed_tag>::type is_directed;
    typedef typename get_in_edges<Graph,is_directed>::type type;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type,type> get_edges(vertex_descriptor v,
                                          const Graph& g)
    {
        return get_in_edges<Graph,is_directed>::get_edges(v, g);
    }
};

template <class Graph>
struct _all_edges_in_iteratorS
{
    typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef typename boost::graph_traits<Graph>::in_edge_iterator type;

    static std::pair<type,type> get_edges(vertex_descriptor v,
                                          const Graph& g)
    {
        return _all_edges_in(v, g);
    }
};

// out edges selector for completeness
template <class Graph>
struct out_edge_iteratorS
{
    typedef typename boost::graph_traits<Graph>::out_edge_iterator type;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type,type> get_edges(vertex_descriptor v,
                                          const Graph& g)
    {
        return out_edges(v, g);
    }
};

// this "all edges" iterator selector returns the in-edge + out-edge ranges for
// directed graphs and the out-edge range for undirected graphs. The
// iterator type is given by all_edges_iteratorS<Graph>::type.
template <class Graph>
struct all_edges_iteratorS
{
    typedef typename Graph::all_edge_iterator type;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type, type> get_edges(vertex_descriptor v, const Graph& g)
    {
        return all_edges(v, g);
    }
};

// this "in or out" iterator selector returns the in-edge range for directed
// graphs and the out-edge range for undirected graphs. The iterator type is
// given by in_or_out_edges_iteratorS<Graph>::type
template <class Graph>
struct in_or_out_edge_iteratorS
    : public std::conditional
          <std::is_convertible
               <typename boost::graph_traits<Graph>::directed_category,
                boost::directed_tag>::value,
           in_edge_iteratorS<Graph>,
           _all_edges_in_iteratorS<Graph>>::type
{};

// helper types for in_neighbor_iteratorS

template <class Graph, class IsDirected>
struct get_in_neighbors
{
    BOOST_MPL_ASSERT((std::is_same<IsDirected,std::true_type>));
    BOOST_MPL_ASSERT((std::is_convertible
                      <typename boost::graph_traits<Graph>::directed_category,
                      boost::directed_tag>));
    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;
    typedef typename Graph::in_adjacency_iterator type;

    static std::pair<type,type> get_neighbors(vertex_descriptor v,
                                              const Graph& g)
    {
        return in_neighbors(v, g);
    }
};

template <class Graph>
struct get_in_neighbors<Graph,std::false_type>
{
    BOOST_MPL_ASSERT((std::is_convertible
                      <typename boost::graph_traits<Graph>::directed_category,
                      boost::undirected_tag>));
    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;
    typedef typename boost::graph_traits<Graph>::adjacency_iterator type;

    static std::pair<type,type> get_neighbors(vertex_descriptor,
                                              const Graph&)
    {
        return make_pair(type(), type());
    }
};

// this in-neighbor iterator selector returns the in-neighbor range for
// directed graphs and an empty out-neighbor range for undirected graphs. The
// iterator type is given by in_neighbor_iteratorS<Graph>::type.
template <class Graph>
struct in_neighbor_iteratorS
{
    typedef typename boost::graph_traits<Graph>::directed_category
        directed_category;
    typedef typename std::is_convertible<directed_category,
                                         boost::directed_tag>::type is_directed;
    typedef typename get_in_neighbors<Graph,is_directed>::type type;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type,type> get_neighbors(vertex_descriptor v,
                                              const Graph& g)
    {
        return get_in_neighbors<Graph,is_directed>::get_neighbors(v, g);
    }
};

// out edges selector for completeness
template <class Graph>
struct out_neighbor_iteratorS
{
    typedef typename boost::graph_traits<Graph>::adjacency_iterator type;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type,type> get_neighbors(vertex_descriptor v,
                                              const Graph& g)
    {
        return out_neighbors(v, g);
    }
};

// this "all neighbors" iterator selector returns the in-neighbors +
// out-neighbors ranges for directed graphs and the out-neighbors range for
// undirected graphs. The iterator type is given by
// all_neighbors_iteratorS<Graph>::type.
template <class Graph>
struct all_neighbors_iteratorS
{
    typedef typename boost::graph_traits<Graph>::adjacency_iterator type;
    typedef typename boost::graph_traits<Graph>::vertex_descriptor
        vertex_descriptor;

    static std::pair<type,type> get_neighbors(vertex_descriptor v,
                                              const Graph& g)
    {
        return all_neighbors(v, g);
    }
};

// this "in or out" iterator selector returns the in-neighbor range for
// directed graphs and the out-neighbor range for undirected graphs. The
// iterator type is given by in_or_adjacency_iteratorS<Graph>::type
template <class Graph>
struct in_or_out_neighbors_iteratorS
    : public std::conditional
          <std::is_convertible
               <typename boost::graph_traits<Graph>::directed_category,
                boost::directed_tag>::value,
           in_neighbor_iteratorS<Graph>,
           out_neighbor_iteratorS<Graph>>::type
{};

// range adaptors

template <class Iter>
class IterRange
    : public std::ranges::view_interface<IterRange<Iter>>
{
public:
    explicit IterRange(std::pair<Iter, Iter>&& range)
        : _range(std::move(range)) {}
    const Iter& begin() const { return _range.first; }
    const Iter& end() const { return _range.second; }
private:
    std::pair<Iter, Iter> _range;
};

template <class Iter>
auto mk_range(std::pair<Iter, Iter>&& range)
{
    return IterRange<Iter>(std::forward<std::pair<Iter, Iter>>(range));
}

template <class Graph>
auto vertices_range(const Graph& g)
{
    return mk_range(vertices(g));
}

template <class Graph>
auto edges_range(const Graph& g)
{
    return mk_range(edges(g));
}

template <class Graph>
auto adjacent_vertices_range(typename boost::graph_traits<Graph>::vertex_descriptor v,
                             const Graph& g)
{
    return mk_range(adjacent_vertices(v, g));
}

template <class Graph>
auto out_edges_range(typename out_edge_iteratorS<Graph>::vertex_descriptor v,
                     const Graph& g)
{
    return mk_range(out_edge_iteratorS<Graph>::get_edges(v, g));
}

template <class Graph>
auto out_neighbors_range(typename boost::graph_traits<Graph>::vertex_descriptor v,
                         const Graph& g)
{
    return mk_range(out_neighbors(v, g));
}


template <class Graph>
auto in_edges_range(typename in_edge_iteratorS<Graph>::vertex_descriptor v,
                    const Graph& g)
{
    return mk_range(in_edge_iteratorS<Graph>::get_edges(v, g));
}

template <class Graph>
auto in_neighbors_range(typename boost::graph_traits<Graph>::vertex_descriptor v,
                        const Graph& g)
{
    return mk_range(in_neighbors(v, g));
}


template <class Graph>
auto all_edges_range(typename all_edges_iteratorS<Graph>::vertex_descriptor v,
                     const Graph& g)
{
    return mk_range(all_edges(v, g));
}

template <class Graph>
auto all_neighbors_range(typename all_neighbors_iteratorS<Graph>::vertex_descriptor v,
                         const Graph& g)
{
    return mk_range(all_neighbors(v, g));
}

template <class Graph>
auto in_or_out_edges_range(typename in_or_out_edge_iteratorS<Graph>::vertex_descriptor v,
                           const Graph& g)
{
    return mk_range(in_or_out_edge_iteratorS<Graph>::get_edges(v, g));
}

template <class Graph>
auto in_or_out_neighbors_range(typename in_or_out_neighbors_iteratorS<Graph>::vertex_descriptor v,
                               const Graph& g)
{
    return mk_range(in_or_out_neighbors_iteratorS<Graph>::get_neighbors(v, g));
}

template <class Graph>
auto _all_edges_in_range(typename all_edges_iteratorS<Graph>::vertex_descriptor v,
                         const Graph& g)
{
    return mk_range(_all_edges_in(v, g));
}

template <class Graph>
auto _all_edges_out_range(typename all_edges_iteratorS<Graph>::vertex_descriptor v,
                          const Graph& g)
{
    return mk_range(_all_edges_out(v, g));
}

template <class Range, class Graph>
decltype(auto) self_loop_filter_idx(Range&& range, const Graph& g)
{
    auto eidx = boost::get(boost::edge_index_t(), g);
    gt_hash_set<size_t> count;
    return range |
        std::views::filter([eidx,count,&g](auto&& e) mutable
                           {
                               if (source(e, g) != target(e, g)) [[likely]]
                                   return true;
                               auto iter = count.find(eidx[e]);
                               if (iter == count.end())
                               {
                                   count.insert(eidx[e]);
                                   return true;
                               }
                               return false;
                           });
}

template <class Range, class Graph>
decltype(auto) self_loop_filter_single(Range&& range, const Graph& g)
{
    bool self = false;
    return range |
        std::views::filter([self,&g](auto&& e) mutable
                           {
                               if (source(e, g) != target(e, g)) [[likely]]
                                   return true;
                               if (self)
                                   return false;
                               self = true;
                               return true;
                           });
}

} //namespace graph_tool

#pragma GCC visibility pop

#endif
