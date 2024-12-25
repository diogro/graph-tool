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

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_properties.hh"

#include "graph_properties_copy.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void GraphInterface::copy_vertex_property(const GraphInterface& src,
                                          std::any prop_src,
                                          std::any prop_tgt)
{
    gt_dispatch<>(false)
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             std::tuple<bool, std::string> ret;
             #pragma omp parallel if (parallel)
             ret = parallel_vertex_loop_no_spawn
                 (g,
                  [&](auto v)
                  {
                      tgt[v] = get(src, v);
                  }, (ValueException*)(nullptr));

             if (get<0>(ret))
                 throw ValueException(get<1>(ret));
         },
         all_graph_views,
         writable_vertex_properties,
         hana::concat(vprop_same_type, dynamic_prop_type))
         (src.get_graph_view(), prop_tgt,
          vprop_map_as_dynamic(prop_src, prop_tgt));
}

void GraphInterface::copy_edge_property(const GraphInterface& src,
                                        std::any prop_src,
                                        std::any prop_tgt)
{
    gt_dispatch<>(false)
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             std::tuple<bool, std::string> ret;
             #pragma omp parallel if (parallel)
             ret = parallel_edge_loop_no_spawn
                 (g,
                  [&](auto e)
                  {
                      tgt[e] = get(src, e);
                  },
                  (ValueException*)(nullptr));

             if (get<0>(ret))
                 throw ValueException(get<1>(ret));
         },
         all_graph_views,
         writable_edge_properties,
         hana::concat(eprop_same_type, dynamic_prop_type))
         (src.get_graph_view(), prop_tgt,
          eprop_map_as_dynamic(prop_src, prop_tgt));
}

bool compare_vertex_properties(const GraphInterface& src,
                               std::any prop_src,
                               std::any prop_tgt)
{
    bool ret = true;
    gt_dispatch<>(false)
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             bool ret_ = true; // clang workaround

             std::tuple<bool, std::string> eret;
             #pragma omp parallel if (parallel)
             eret = parallel_vertex_loop_no_spawn
                 (g,
                  [&](auto v)
                  {
                      if (tgt[v] != get(src, v))
                          ret_ = false;
                  },
                  (ValueException*)(nullptr));

             if (get<0>(eret))
                 throw ValueException(get<1>(eret));

             ret = ret_;
         },
         all_graph_views,
         writable_vertex_properties,
         hana::concat(vprop_same_type, dynamic_prop_type))
         (src.get_graph_view(), prop_tgt,
          vprop_map_as_dynamic(prop_src, prop_tgt));
    return ret;
}

bool compare_edge_properties(const GraphInterface& src,
                             std::any prop_src,
                             std::any prop_tgt)
{
    bool ret = true;
    gt_dispatch<>(false)
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             bool ret_ = true; // clang workaround

             std::tuple<bool, std::string> eret;
             #pragma omp parallel if (parallel)
             eret = parallel_edge_loop_no_spawn
                 (g,
                  [&](auto e)
                  {
                      if (tgt[e] != get(src, e))
                          ret_ = false;
                  },
                  (ValueException*)(nullptr));

             if (get<0>(eret))
                 throw ValueException(get<1>(eret));

             ret = ret_;
         },
         all_graph_views,
         writable_edge_properties,
         hana::concat(eprop_same_type, dynamic_prop_type))
         (src.get_graph_view(), prop_tgt,
          eprop_map_as_dynamic(prop_src, prop_tgt));

    return ret;
}
