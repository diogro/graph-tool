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

#include "graph_histograms.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

struct deg_check
{
    void operator()(std::any a) const
    {
        if (!belongs(scalar_vertex_properties, a))
            throw ValueException("Vertex property must be of scalar type.");
    }

    template<class T>
    void operator()(const T&) const {}
};

// this will return the vertex histogram of degrees or scalar properties
python::object
get_vertex_histogram(GraphInterface& gi, GraphInterface::deg_t deg,
                     const vector<long double>& bins)
{
    std::visit(deg_check(), deg);

    python::object hist;
    python::object ret_bins;

    run_action<{.dargs={.gil_release=false}}>()
        (gi,
         get_histogram<VertexHistogramFiller>(hist, bins, ret_bins),
         all_selectors)(degree_selector(deg));

    return python::make_tuple(hist, ret_bins);
}

// this will return the vertex histogram of degrees or scalar properties
python::object
get_edge_histogram(GraphInterface& gi, std::any prop,
                   const vector<long double>& bins)
{
    if (!belongs(scalar_edge_properties, prop))
        throw ValueException("Edge property must be of scalar type.");

    python::object hist;
    python::object ret_bins;

    bool directed = gi.get_directed();
    gi.set_directed(true);
    run_action<{.tr=graph_tool::type<decltype(graph_tool::always_directed)>, .dargs={.gil_release=false}}>()
        (gi, get_histogram<EdgeHistogramFiller>(hist, bins, ret_bins),
         scalar_edge_properties)(prop);
    gi.set_directed(directed);

    return python::make_tuple(hist, ret_bins);
}

using namespace boost::python;

#define __MOD__ stats
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("get_vertex_histogram", &get_vertex_histogram);
     def("get_edge_histogram", &get_edge_histogram);
 });
