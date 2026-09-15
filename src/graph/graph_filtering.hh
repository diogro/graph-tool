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

#ifndef GRAPH_FILTERING_HH
#define GRAPH_FILTERING_HH

#pragma GCC visibility push(default)

#include <boost/graph/graph_traits.hpp>
#include <boost/iterator/iterator_traits.hpp>

#include "graph_adjacency.hh"
#include "graph_adaptor.hh"
#include "graph_filtered.hh"
#include "graph_reverse.hh"
#include "graph_properties.hh"
#include "graph_util.hh"
#include "gil_release.hh"
#include "dispatch.hh"

#include <type_traits>
#include <iostream>
#include <print>

namespace graph_tool
{

template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(DescriptorProperty filtered_property)
        : _filtered_property(filtered_property) {}
    MaskFilter() {}

    template <class Descriptor>
    bool operator() (Descriptor&& d) const
    {
        return boost::get(_filtered_property, d);
    }

    template <class Descriptor>
    void set_val(Descriptor&& d, bool existing)
    {
        boost::put(get_checked(_filtered_property), d, existing);
    }

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

#ifndef DISABLE_GRAPH_FILTERING
    typedef std::conditional_t<reversed && is_directed_v<directed_t>,
                               boost::reversed_graph<directed_t>,
                               directed_t> reversed_t;
#else
    typedef directed_t reversed_t;
#endif

    typedef eprop_map_t<uint8_t>::unchecked_t eprop;
    typedef vprop_map_t<uint8_t>::unchecked_t vprop;

    typedef boost::filt_graph<reversed_t,
                              MaskFilter<eprop>,
                              MaskFilter<vprop>>& filtered_graph_t;

#ifndef DISABLE_GRAPH_FILTERING
    typedef std::conditional_t<filtered,
                               filtered_graph_t,
                               reversed_t>& type;
#else
    typedef reversed_t& type;
#endif
};

template <bool directed, bool reversed, bool filtered>
using graph_view_t = typename graph_view<directed, reversed, filtered>::type;

template <class Directed, class Reversed, class Filtered>
constexpr auto get_graph_views(Directed dir = hana::tuple_t<std::true_type, std::false_type>,
                               [[maybe_unused]] Reversed rev = hana::tuple_t<std::true_type, std::false_type>,
                               [[maybe_unused]] Filtered filt = hana::tuple_t<std::true_type, std::false_type>)
{
    auto prod =
#ifndef DISABLE_GRAPH_FILTERING
        hana::cartesian_product(hana::make_tuple(dir, rev, filt));
#else
        hana::cartesian_product(hana::make_tuple(dir,
                                                 hana::tuple_t<std::false_type>,
                                                 hana::tuple_t<std::false_type>));
#endif

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

template <typename T>
struct type_t {
    using type = T;
};

template <typename T>
inline constexpr type_t<T> type{};

template <class TR = decltype(all_graph_views)>
struct run_action_args
{
    type_t<TR> tr = type<TR>;
    gt_dispatch_args dargs = gt_dispatch_args();
};

template <run_action_args rargs = {}>
struct run_action
    : public gt_dispatch<rargs.dargs>
{
    template <class G, class Dispatch, class... TRS>
    auto operator()(G&& gi, Dispatch&& a, TRS&&... trs)
    {
        using TR = decltype(rargs.tr)::type;
        auto dispatcher =
            gt_dispatch<rargs.dargs>()
            (std::forward<Dispatch>(a), TR(),
             std::forward<TRS>(trs)...);
        return [=, &gi](auto&&... as)
               {
                   if constexpr (std::is_same_v<std::decay_t<G>, GraphInterface>)
                       dispatcher(gi.get_graph_view(),
                                  std::forward<decltype(as)>(as)...);
                   else
                       dispatcher(gi,
                                  std::forward<decltype(as)>(as)...);
               };
    }
};


// Access underlying base graph

template <class Graph>
decltype(auto) base_graph(Graph&& g)
{
    if constexpr (is_specialization_of_v<std::decay_t<Graph>, boost::adj_list>)
        return g;
    else
        return base_graph(g.original_graph());
}

} //graph_tool namespace

namespace boost
{
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

#pragma GCC visibility pop

#endif // GRAPH_FILTERING_HH
