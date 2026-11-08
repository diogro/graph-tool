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
#include "properties.hh"

#include "properties_copy.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

// NaN values are considered equal, so that a property map always compares
// equal to a copy of itself
template <class T>
bool prop_value_equal(const T& a, const T& b)
{
    if constexpr (std::is_floating_point_v<T>)
        return a == b || (std::isnan(a) && std::isnan(b));
    else if constexpr (is_vector_v<T>)
        return std::ranges::equal(a, b,
                                  [](const auto& x, const auto& y)
                                  { return prop_value_equal(x, y); });
    else
        return static_cast<bool>(a == b);
}

void GraphInterface::copy_vertex_property(const GraphInterface& g_src,
                                          std::any prop_src,
                                          std::any prop_tgt)
{
    gt_dispatch<{.gil_release=false}>()
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             [[maybe_unused]] bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             std::exception_ptr eptr{};
             #pragma omp parallel if (parallel)
             eptr = parallel_vertex_loop_no_spawn<true>
                 (g,
                  [&](auto v)
                  {
                      tgt[v] = get(src, v);
                  });
             if (eptr)
                 std::rethrow_exception(eptr);
         },
         all_graph_views,
         writable_vertex_properties,
         hana::concat(vprop_same_type, dynamic_prop_type))
         (g_src.get_graph_view(), prop_tgt,
          vprop_map_as_dynamic(prop_src, prop_tgt));
}

void GraphInterface::copy_edge_property(const GraphInterface& g_src,
                                        std::any prop_src,
                                        std::any prop_tgt)
{
    gt_dispatch<{.gil_release=false}>()
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             [[maybe_unused]] bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             std::exception_ptr eptr{};
             #pragma omp parallel if (parallel)
             eptr = parallel_edge_loop_no_spawn<true>
                 (g,
                  [&](auto e)
                  {
                      tgt[e] = get(src, e);
                  });
             if (eptr)
                 std::rethrow_exception(eptr);
         },
         all_graph_views,
         writable_edge_properties,
         hana::concat(eprop_same_type, dynamic_prop_type))
         (g_src.get_graph_view(), prop_tgt,
          eprop_map_as_dynamic(prop_src, prop_tgt));
}

bool compare_vertex_properties(const GraphInterface& g_src,
                               std::any prop_src,
                               std::any prop_tgt)
{
    bool ret = true;
    gt_dispatch<{.gil_release=false}>()
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             [[maybe_unused]] bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             bool ret_ = true; // clang workaround

             std::exception_ptr eptr{};
             #pragma omp parallel if (parallel)
             eptr = parallel_vertex_loop_no_spawn<true>
                 (g,
                  [&](auto v)
                  {
                      if (!prop_value_equal(get(tgt, v), get(src, v)))
                          ret_ = false;
                  });
             if (eptr)
                 std::rethrow_exception(eptr);

             ret = ret_;
         },
         all_graph_views,
         vertex_properties,
         hana::concat(vprop_same_type, dynamic_prop_type))
         (g_src.get_graph_view(), prop_tgt,
          vprop_map_as_dynamic(prop_src, prop_tgt));
    return ret;
}

bool compare_edge_properties(const GraphInterface& g_src,
                             std::any prop_src,
                             std::any prop_tgt)
{
    bool ret = true;
    gt_dispatch<{.gil_release=false}>()
        ([&](auto& g, auto tgt, auto src)
         {
             bool is_python =
                 (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));

             GILRelease gil(!is_python);

             [[maybe_unused]] bool parallel =
                 (num_vertices(g) > get_openmp_min_thresh() && !is_python);

             bool ret_ = true; // clang workaround

             std::exception_ptr eptr{};
             #pragma omp parallel if (parallel)
             eptr = parallel_edge_loop_no_spawn<true>
                 (g,
                  [&](auto e)
                  {
                      if (!prop_value_equal(get(tgt, e), get(src, e)))
                          ret_ = false;
                  });
             if (eptr)
                 std::rethrow_exception(eptr);

             ret = ret_;
         },
         all_graph_views,
         edge_properties,
         hana::concat(eprop_same_type, dynamic_prop_type))
         (g_src.get_graph_view(), prop_tgt,
          eprop_map_as_dynamic(prop_src, prop_tgt));

    return ret;
}
