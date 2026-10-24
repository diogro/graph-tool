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

// As a special exception, you have permission to link this program
// with the CGAL library and distribute executables, as long as you
// follow the requirements of the GNU GPL in regard to all of the
// software in the executable aside from CGAL.

#include "graph.hh"
#include "graph_util.hh"
#include "graph_filtering.hh"

#if (GCC_VERSION < 40400 || defined __clang__)
#   define CGAL_CFG_NO_TR1_ARRAY
#   define CGAL_CFG_NO_TR1_TUPLE
#endif


#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

// the vertices store the index of the graph vertex
typedef CGAL::Triangulation_vertex_base_with_info_3<size_t, Kernel> Vb;
typedef CGAL::Triangulation_3<Kernel, CGAL::Triangulation_data_structure_3<Vb>>
    SimpleTriangulation;

// with TBB, the points of the Delaunay triangulation are inserted in parallel
#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag dt_concurrency_t;
#else
typedef CGAL::Sequential_tag dt_concurrency_t;
#endif
typedef CGAL::Triangulation_data_structure_3
    <Vb, CGAL::Delaunay_triangulation_cell_base_3<Kernel>, dt_concurrency_t>
    DelaunayTds;
typedef CGAL::Delaunay_triangulation_3<Kernel, DelaunayTds>
    DelaunayTriangulation;

// periodic triangulation is only available in more recent versions of CGAL
#if (CGAL_VERSION_NR >= 1040700000)
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<Kernel> GT;
typedef CGAL::Triangulation_vertex_base_with_info_3
    <size_t, GT,
     CGAL::Triangulation_vertex_base_3
         <GT, CGAL::Periodic_3_triangulation_ds_vertex_base_3<>>> PVb;
typedef CGAL::Triangulation_cell_base_3
    <GT, CGAL::Periodic_3_triangulation_ds_cell_base_3<>> PCb;
typedef CGAL::Periodic_3_Delaunay_triangulation_3
    <GT, CGAL::Triangulation_data_structure_3<PVb, PCb>>
    PeriodicDelaunayTriangulation;
#endif

#include "triangulation.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void triangulation(GraphInterface& gi, boost::python::object points,
                   std::any pos, string type, bool periodic)
{
    auto& g = get_undirected_adaptor(gi.get_graph());
    multi_array_ref<double,2> points_array = get_array<double,2>(points);
    typedef vprop_map_t<vector<double>> pos_type_t;
    pos_type_t pos_map = std::any_cast<pos_type_t>(pos);

    if (type == "simple")
    {
        get_triangulation<SimpleTriangulation, std::false_type>()
            (g, points_array, pos_map);
    }
    else if (type == "delaunay")
    {
        if (!periodic)
        {
            get_triangulation<DelaunayTriangulation, std::false_type>()
                (g, points_array, pos_map);
        }
        else
        {
#if (CGAL_VERSION_NR >= 1040700000)
            get_triangulation<PeriodicDelaunayTriangulation, std::true_type>()
                (g, points_array, pos_map);
#else
            throw ValueException("Periodic Delaunay triangulation is only "
                                 "available with versions of CGAL newer than "
                                 "4.7.0.");
#endif
        }
    }
}

using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("triangulation", &triangulation);
 });
