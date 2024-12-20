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

#ifndef FILTERING_HH
#define FILTERING_HH

#include <boost/graph/graph_traits.hpp>
#include <boost/iterator/iterator_traits.hpp>

#include <boost/hana.hpp>
namespace hana = boost::hana;
using namespace hana::literals;

#include "graph_adjacency.hh"
#include "graph_adaptor.hh"
#include "graph_filtered.hh"
#include "graph_reverse.hh"
#include "graph_properties.hh"
#include "graph_util.hh"
#include "gil_release.hh"

#include <type_traits>
#include <iostream>

namespace graph_tool
{

// Graph views
// -----------
//
// We want to generate versions of a template algorithm for every possible type
// of graph views. The types of graph views are the following:
//
//    - The original directed multigraph
//
//    - Filtered graphs, based on MaskFilter below
//
//    - A reversed view of each directed graph (original + filtered)
//
//    - An undirected view of each directed (unreversed) graph (original +
//      filtered)
//
// The total number of graph views is then: 1 + 1 + 2 + 2 = 6
//
// The specific specialization can be called at run time (and generated at
// compile time) with the gt_dispatch() function, which takes as arguments the
// function to be called, the type ranges of each argument (as hana::tuple), and
// the actual parameters as boost::any values.
//
// Example:
//
// template <class Graph, class ValueType>
// void my_algorithm(Graph& g, ValueType val)
// {
//      ... do something ...
// };
//
// ...
//
// GraphInterface gi;
// auto value_types = hana::tuple_c<int, double, string>;
// double foo = 42.0;
// gt_dispatch<>()([](auto& g, auto val){ my _algorithm(g, val); },
//                 all_graph_views, value_tyoes)
//                 (g.get_graph_views(), boost::any(foo));
//
// The above line will run my_algorithm with Graph being the appropriate graph
// view type and ValueType being 'double' and val = 42.0.

class DispatchNotFound: public GraphException
{
public:
    DispatchNotFound(const std::type_info& dispatch,
                     const std::vector<const std::type_info*>& args);
    virtual ~DispatchNotFound() throw () {}
private:
    const std::type_info& _dispatch;
    std::vector<const std::type_info*> _args;
};

template <bool uncheck=true>
struct gt_dispatch
{
    gt_dispatch(bool gil_release=true)
        : _gil_release(gil_release) {};

    template <class Type, class IndexMap>
    static auto pmap(boost::checked_vector_property_map<Type,IndexMap>& a)
    {
        if constexpr (uncheck)
            return a.get_unchecked();
        else
            return a;
    }

    template <class Type>
    static Type&& pmap(Type&& a)
    {
        return std::forward<Type>(a);
    }

    template <class Dispatch, class... TRS>
    auto operator()(Dispatch&& a, TRS... trs)
    {
        return [=](auto&&... as)
               {
                   GILRelease gil(_gil_release);

                   constexpr auto cp = hana::cartesian_product(hana::make_tuple(trs...));

                   bool found = false;
                   struct stop_dispatch {};
                   try
                   {
                       hana::for_each
                           (cp,
                            [&](auto ats)
                            {
                                struct stop_transform {};
                                try
                                {
                                    auto args =
                                        hana::transform
                                        (hana::zip(ats, hana::make_tuple(std::ref(as)...)),
                                         [](auto&& tv)
                                         {
                                             using T = typename decltype(+tv[0_c])::type;
                                             auto& x = tv[1_c].get();
                                             T* val = boost::any_cast<T>(&x);
                                             if (val == nullptr)
                                             {
                                                 auto rval = boost::any_cast<std::reference_wrapper<T>>(&x);
                                                 if (rval == nullptr)
                                                 {
                                                     auto sval = boost::any_cast<std::shared_ptr<T>>(&x);
                                                     if (sval == nullptr)
                                                         throw stop_transform();
                                                     else
                                                         val = sval->get();
                                                 }
                                                 else
                                                 {
                                                     val = &rval->get();
                                                 }
                                             }
                                             return val;
                                         });
                                    hana::unpack(args, [&](auto&... vals){ a(pmap(*vals)...); });
                                    found = true;
                                    throw stop_dispatch();
                                }
                                catch (stop_transform) {}
                            });
                   }
                   catch (stop_dispatch) {}

                   if (!found)
                   {
                       std::vector<const std::type_info*> args_t = {(&(as).type())...};
                       throw DispatchNotFound(typeid(Dispatch), args_t);
                   }
               };
    }

    bool _gil_release;
};


template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(DescriptorProperty filtered_property)
        : _filtered_property(filtered_property) {}
    MaskFilter() {}

    template <class Descriptor>
    inline bool operator() (Descriptor&& d) const
    {
        return get(_filtered_property, d);
    }

    DescriptorProperty& get_filter() { return _filtered_property; }

private:
    DescriptorProperty _filtered_property;
};

template <bool directed, bool reversed, bool filtered>
struct graph_view
{
    typedef boost::adj_list<size_t> base_t;
    typedef std::conditional_t<directed,
                               base_t,
                               boost::undirected_adaptor<base_t>> directed_t;
    typedef std::conditional_t<reversed && is_directed_v<directed_t>,
                               boost::reversed_graph<directed_t>,
                               directed_t> reversed_t;

    typedef eprop_map_t<uint8_t>::unchecked_t eprop;
    typedef vprop_map_t<uint8_t>::unchecked_t vprop;

    typedef boost::filt_graph<reversed_t,
                              MaskFilter<eprop>,
                              MaskFilter<vprop>> filtered_graph_t;

    typedef std::conditional_t<filtered,
                               filtered_graph_t,
                               reversed_t> type;
};

template <bool directed, bool reversed, bool filtered>
using graph_view_t = typename graph_view<directed, reversed, filtered>::type;


template <class Directed, class Reversed, class Filtered>
constexpr auto get_graph_views(Directed dir = hana::tuple_t<std::true_type, std::false_type>,
                               Reversed rev = hana::tuple_t<std::true_type, std::false_type>,
                               Filtered filt = hana::tuple_t<std::true_type, std::false_type>)
{
    auto prod = hana::cartesian_product(hana::make_tuple(dir, rev, filt));
    auto cp = hana::filter(prod,
                           [](auto const& view)
                           {
                               return hana::integral_constant<bool, !(!decltype(+view[0_c])::type::value &&
                                                                      decltype(+view[1_c])::type::value)>();
                           });
    auto xs = hana::transform(cp,
                              [](auto const& view)
                              {
                                  return hana::type<graph_view_t<decltype(+view[0_c])::type::value,
                                                                 decltype(+view[1_c])::type::value,
                                                                 decltype(+view[2_c])::type::value>>();
                              });
    return  hana::to<hana::tuple_tag>(hana::to<hana::set_tag>(xs));
}

inline constexpr auto all_graph_views =
    get_graph_views(hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>);

inline constexpr auto always_directed =
    get_graph_views(hana::tuple_t<std::true_type>,
                    hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>);

inline constexpr auto never_directed =
    get_graph_views(hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>);

inline constexpr auto never_reversed =
    get_graph_views(hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>);

inline constexpr auto always_directed_never_reversed =
    get_graph_views(hana::tuple_t<std::true_type>,
                    hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>);

inline constexpr auto never_filtered =
    get_graph_views(hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::false_type>);

inline constexpr auto never_filtered_never_reversed =
    get_graph_views(hana::tuple_t<std::true_type, std::false_type>,
                    hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::false_type>);

inline constexpr auto always_directed_never_filtered_never_reversed =
    get_graph_views(hana::tuple_t<std::true_type>,
                    hana::tuple_t<std::false_type>,
                    hana::tuple_t<std::false_type>);

// legacy syntax
template <class TR = decltype(all_graph_views), bool uncheck = true>
struct run_action
    : public gt_dispatch<uncheck>
{
    run_action(bool gil_release=true)
        : gt_dispatch<uncheck>(gil_release) {}

    template <class Dispatch, class... TRS>
    auto operator()(GraphInterface& gi, Dispatch&& a, TRS&&... trs)
    {
        auto dispatcher =
            gt_dispatch<uncheck>::operator()(std::forward<Dispatch>(a), TR(),
                                             std::forward<TRS>(trs)...);
        return [=,&gi](auto&&... as)
               {
                   dispatcher(gi.get_graph_view(),
                              std::forward<decltype(as)>(as)...);
               };
    }
};

} //graph_tool namespace

// Overload add_vertex() and add_edge() to filtered graphs, so that the new
// descriptors are always valid

namespace boost
{
template <class Graph, class EdgeProperty, class VertexProperty>
auto
add_vertex(boost::filt_graph<Graph,
                             graph_tool::MaskFilter<EdgeProperty>,
                             graph_tool::MaskFilter<VertexProperty>>& g)
{
    auto v = add_vertex(const_cast<Graph&>(g._g));
    auto& filt = g._vertex_pred.get_filter();
    auto cfilt = filt.get_checked();
    cfilt[v] = true;
    return v;
}

template <class Graph, class EdgeProperty, class VertexProperty, class Vertex>
auto
add_edge(Vertex s, Vertex t, filt_graph<Graph,
                                        graph_tool::MaskFilter<EdgeProperty>,
                                        graph_tool::MaskFilter<VertexProperty>>& g)
{
    auto e = add_edge(s, t, const_cast<Graph&>(g._g));
    auto& filt = g._edge_pred.get_filter();
    auto cfilt = filt.get_checked();
    cfilt[e.first] = true;
    return e;
}

// Used to skip filtered vertices

template <class Graph, class EP, class VP, class Vertex>
bool is_valid_vertex(Vertex v, const boost::filt_graph<Graph,EP,VP>& g)
{
    return (v < num_vertices(g) &&
            vertex(v, g) != graph_traits<boost::filt_graph<Graph,EP,VP>>::null_vertex());
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const boost::reversed_graph<Graph>& g)
{
    return is_valid_vertex(v, g.original_graph());
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const boost::undirected_adaptor<Graph>& g)
{
    return is_valid_vertex(v, g.original_graph());
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const Graph& g)
{
    return v < num_vertices(g);
}


} // namespace boost

#endif // FILTERING_HH
