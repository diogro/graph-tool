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
#include "graph.hh"
#include "graph_properties.hh"

#if (BOOST_VERSION >= 108800)
#include <boost/graph/maximum_weighted_matching.hpp>
#else
#include <boost/graph/maximum_weighted_matching_new.hpp>
#endif
#include <boost/graph/maximum_weighted_matching_brute.hpp>

#include "graph_bipartite_weighted_matching.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;


void get_max_matching(GraphInterface& gi, std::string initial_matching,
                      std::any omatching)
{
    typedef vprop_map_t<int64_t> vprop_t;

    vprop_t::unchecked_t matching = std::any_cast<vprop_t>(omatching).get_unchecked();

    run_action<{.tr=graph_tool::type<decltype(graph_tool::never_directed)>}>()
        (gi,
         [&](auto& g)
         {
             auto vindex = get(vertex_index, g);
             typedef decltype(vindex) vindex_t;
             typedef std::remove_reference_t<decltype(g)> g_t;

             if (initial_matching == "empty")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, empty_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else if (initial_matching == "greedy")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, greedy_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else if (initial_matching == "extra_greedy")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, extra_greedy_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else
                 throw ValueException("invalid initial matching: " +
                                      initial_matching);

             for (auto v : vertices_range(g))
             {
                 if (matching[v] == int64_t(graph_traits<g_t>::null_vertex()))
                     matching[v] = std::numeric_limits<int64_t>::max();
             }

         })();
}

void get_max_weighted_matching(GraphInterface& gi, std::any oweight,
                               std::any omatching, bool brute_force)
{
    typedef vprop_map_t<int64_t> vprop_t;

    vprop_t::unchecked_t matching = std::any_cast<vprop_t>(omatching).get_unchecked();

    run_action<{.tr=graph_tool::type<decltype(graph_tool::never_directed)>}>()
        (gi,
         [&](auto& g, auto w)
         {
             typedef std::remove_reference_t<decltype(g)> g_t;

             typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
             vprop_map_t<vertex_t> match(get(vertex_index, g));

             if (brute_force)
                 alt::brute_force_maximum_weighted_matching(g, w, match);
             else
                 maximum_weighted_matching(g, match, get(vertex_index, g), w);

             for (auto v : vertices_range(g))
             {
                 if (match[v] == graph_traits<g_t>::null_vertex())
                     matching[v] = std::numeric_limits<int64_t>::max();
                 else
                     matching[v] = match[v];
             }
         },
         scalar_edge_properties)(oweight);
}

void get_max_bip_weighted_matching(GraphInterface& gi, std::any opartition,
                                   std::any oweight, std::any omatching)
{
    typedef vprop_map_t<int64_t> vprop_t;

    vprop_t::unchecked_t matching = std::any_cast<vprop_t>(omatching).get_unchecked();

    typedef UnityPropertyMap<int, GraphInterface::edge_t> ecmap_t;
    auto weight_props_t = hana::append(scalar_edge_properties,
                                       hana::type<ecmap_t>());

    if (!oweight.has_value())
        oweight = ecmap_t();

    run_action<{.tr=graph_tool::type<decltype(graph_tool::never_directed)>}>()
        (gi,
         [&](auto& g, auto part, auto w)
         {
             typedef std::remove_reference_t<decltype(g)> g_t;

             typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
             vprop_map_t<vertex_t> match(get(vertex_index,g));

             maximum_bipartite_weighted_matching(g, part, w, match);

             for (auto v : vertices_range(g))
             {
                 if (match[v] == graph_traits<g_t>::null_vertex())
                     matching[v] = std::numeric_limits<int64_t>::max();
                 else
                     matching[v] = match[v];
             }
         },
         vertex_properties, weight_props_t)(opartition, oweight);
}

void match_edges(GraphInterface& gi, std::any omatching,
                 std::any oematching)
{
    typedef vprop_map_t<int64_t> vprop_t;
    typedef eprop_map_t<uint8_t> eprop_t;

    vprop_t::unchecked_t matching = std::any_cast<vprop_t>(omatching).get_unchecked();
    eprop_t::unchecked_t ematching = std::any_cast<eprop_t>(oematching).get_unchecked();

    run_action<{.tr=graph_tool::type<decltype(graph_tool::never_directed)>}>()
        (gi,
         [&](auto& g)
         {
             for (auto v : vertices_range(g))
             {
                 auto u = matching[v];
                 if (size_t(u) > num_vertices(g))
                     continue;
                 ematching[edge(v, u, g).first] = true;
             }
         })();
}

#include <boost/python.hpp>
using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("get_max_matching", &get_max_matching);
     def("get_max_weighted_matching", &get_max_weighted_matching);
     def("get_max_bip_weighted_matching", &get_max_bip_weighted_matching);
     def("match_edges", &match_edges);
 });
