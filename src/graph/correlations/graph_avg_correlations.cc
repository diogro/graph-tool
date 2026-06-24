// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include <boost/python.hpp>

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_avg_correlations.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

typedef UnityPropertyMap<int,GraphInterface::edge_t> cweight_map_t;

boost::python::object
get_vertex_avg_correlation(GraphInterface& gi,
                           GraphInterface::deg_t deg1,
                           GraphInterface::deg_t deg2,
                           std::any weight,
                           const vector<long double>& bins)
{
    boost::python::object avg, dev;
    boost::python::object ret_bins;

    any weight_prop;
    typedef DynamicPropertyMapWrap<long double, GraphInterface::edge_t>
        wrapped_weight_t;

    if (weight.has_value())
        weight_prop = wrapped_weight_t(weight, edge_scalar_properties);
    else
        weight_prop = cweight_map_t();

    run_action<{.dargs={.gil_release=false}}>()
        (gi, get_avg_correlation<GetNeighborsPairs>
         (avg, dev, bins, ret_bins),
         all_selectors, all_selectors,
         boost::hana::tuple_t<wrapped_weight_t, cweight_map_t>)
        (degree_selector(deg1), degree_selector(deg2), weight_prop);

    return boost::python::make_tuple(avg, dev, ret_bins);
}

using namespace boost::python;

#define __MOD__ correlations
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("vertex_avg_correlation", &get_vertex_avg_correlation);
 });
