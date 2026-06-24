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
#include "graph_filtering.hh"
#include "graph_properties.hh"
#include "graph_util.hh"
#include "graph_selectors.hh"

#include "graph_motifs.hh"

#include "graph_python_interface.hh"
#include <boost/python.hpp>

using namespace graph_tool;

struct append_to_list
{
    template <class Graph>
    void operator()(Graph& g, std::vector<d_graph_t>& glist) const
    {
        glist.emplace_back();
        graph_copy(g, glist.back());
    }
};

struct retrieve_from_list
{
    template <class Graph>
    void operator()(Graph& g, std::vector<d_graph_t>& glist, bool& done) const
    {
        if (glist.empty())
        {
            done = true;
            return;
        }
        auto& uback = wrap_directed(g, glist.back());
        graph_copy(uback, g);
        glist.pop_back();
    }
};

void get_motifs(GraphInterface& g, size_t k, boost::python::list subgraph_list,
                boost::python::list hist, boost::python::list pvmaps,
                bool collect_vmaps, boost::python::list p, bool comp_iso,
                bool fill_list, rng_t& rng)
{
    std::vector<d_graph_t> list;
    for (int i = 0; i <  boost::python::len(subgraph_list); ++i)
    {
        GraphInterface& sub =
            boost::python::extract<GraphInterface&>(subgraph_list[i]);
        run_action<>()
            (sub,
             [&](auto&& graph)
             {
                 return append_to_list()
                     (std::forward<decltype(graph)>(graph), list);
             })();
    }

    std::vector<size_t> phist;
    std::vector<double> plist;
    double total = 1;
    for (int i = 0; i < boost::python::len(p); ++i)
    {
        plist.push_back(boost::python::extract<double>(p[i]));
        total *= plist[i];
    }

    std::any sampler;
    if (total == 1.0)
        sampler = sample_all();
    else
        sampler = sample_some(plist, rng);

    typedef vprop_map_t<int32_t> vmap_t;
    std::vector<std::vector<vmap_t> > vmaps;

    run_action<>()
        (g,
         [&](auto& graph, auto& samp)
         {
             return get_all_motifs(collect_vmaps, plist[0], comp_iso, fill_list,
                                   rng)(graph, k, list, phist, vmaps, samp);
         },
         hana::tuple_t<sample_all&, sample_some&>)(sampler);

    for (size_t i = 0; i < phist.size(); ++i)
        hist.append(phist[i]);

    for (size_t i = 0; i < vmaps.size(); ++i)
    {
        boost::python::list vlist;
        for (size_t j = 0; j < vmaps[i].size(); ++j)
            vlist.append(PythonPropertyMap<vmap_t>(vmaps[i][j]));
        pvmaps.append(vlist);
    }

    if (fill_list)
    {
        for (int i = 0; i < boost::python::len(subgraph_list); ++i)
            subgraph_list.pop();

        bool done = false;
        while (!done)
        {
            GraphInterface sub;
            sub.set_directed(g.get_directed());
            run_action<>()
                (sub,
                 [&](auto& graph)
                 {
                     return retrieve_from_list()(graph, list, done);
                 })();
            if (!done)
                subgraph_list.append(sub);
        }
        subgraph_list.reverse();
    }
}

#define __MOD__ clustering
#include "module_registry.hh"
REGISTER_MOD
([]
 {
         def("get_motifs", &get_motifs);
 });
