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

#pragma once

#include "graph_util.hh"

#include <ranges>
#include <boost/iterator/function_output_iterator.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

// The vertices of the triangulation store the index of the corresponding
// graph vertex as their info, and the points are inserted as a single range,
// which is spatially sorted for the Delaunay triangulations, and done in
// parallel if the triangulation data structure is concurrent (which requires
// TBB). The periodic triangulation does not support the insertion of a range
// with info, and its points are inserted one at a time.
template <class Triang, class IsPeriodic>
struct get_triangulation
{
    template <class Graph, class Points, class PosMap>
    void operator()(Graph& g, Points& points, PosMap pos) const
    {
        typedef typename Triang::Point point_t;
        std::vector<std::pair<point_t, size_t>> vpoints;
        vpoints.reserve(points.shape()[0]);
        for (size_t i = 0; i < points.shape()[0]; ++i)
        {
            auto v = add_vertex(g);
            pos[v].resize(3);
            for (size_t j = 0; j < 3; ++j)
                pos[v][j] = points[i][j];
            vpoints.emplace_back(point_t(points[i][0], points[i][1],
                                         points[i][2]), v);
        }

        if constexpr (IsPeriodic::value)
        {
            Triang T;
            for (auto& [p, v] : vpoints)
                T.insert(p)->info() = v;
            add_edges(g, T);
        }
#ifdef CGAL_LINKED_WITH_TBB
        else if constexpr (std::is_same_v<typename Triang::Concurrency_tag,
                                          CGAL::Parallel_tag>)
        {
            // the lock grid needs a bounding box of non-zero volume
            std::array<double, 3> lo, hi;
            lo.fill(std::numeric_limits<double>::infinity());
            hi.fill(-std::numeric_limits<double>::infinity());
            for (size_t i = 0; i < points.shape()[0]; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    lo[j] = std::min(lo[j], double(points[i][j]));
                    hi[j] = std::max(hi[j], double(points[i][j]));
                }
            }
            double w = 0;
            for (size_t j = 0; j < 3; ++j)
                w = std::max(w, hi[j] - lo[j]);
            w = (w > 0) ? w * 1e-3 : 1;
            typename Triang::Lock_data_structure
                lock(CGAL::Bbox_3(lo[0] - w, lo[1] - w, lo[2] - w,
                                  hi[0] + w, hi[1] + w, hi[2] + w), 50);
            Triang T(typename Triang::Geom_traits(), &lock);
            {
                tbb_num_threads tbb_threads;
                T.insert(vpoints.begin(), vpoints.end());
            }
            add_edges(g, T);
        }
#endif
        else
        {
            Triang T;
            T.insert(vpoints.begin(), vpoints.end());
            add_edges(g, T);
        }
    }

    template <class Graph, class T>
    void add_edges(Graph& g, T& t) const
    {
        // The periodic triangulation maps the neighbors in the covering space
        // back to the original vertices, which is not thread safe, and the
        // thread-safe traversal of the other triangulations is slower than
        // the default one (which marks the visited cells), so it is only
        // worth it with more than one thread.
        if (IsPeriodic::value || get_num_threads() == 1 ||
            t.number_of_vertices() <= get_openmp_min_thresh())
        {
            std::vector<typename T::Vertex_handle> ws;
            for (auto v = t.finite_vertices_begin();
                 v != t.finite_vertices_end(); ++v)
            {
                ws.clear();
                if constexpr (IsPeriodic::value)
                    t.incident_vertices(v, std::back_inserter(ws));
                else
                    t.finite_incident_vertices(v, std::back_inserter(ws));

                // every edge is seen from both endpoints, and the periodic
                // triangulation may list the same neighbor more than once,
                // if the point set is small
                size_t s = v->info();
                for (auto w : ws)
                {
                    size_t u = w->info();
                    if (s >= u)
                        continue;
                    if (IsPeriodic::value && is_adjacent(s, u, g))
                        continue;
                    add_edge(s, u, g);
                }
            }
            return;
        }

        if constexpr (!IsPeriodic::value)
        {
            std::vector<typename T::Vertex_handle> vs;
            vs.reserve(t.number_of_vertices());
            for (auto v = t.finite_vertices_begin();
                 v != t.finite_vertices_end(); ++v)
                vs.push_back(v);

            // the neighbors of every vertex with a larger index, so that
            // every edge is listed once
            std::vector<std::vector<size_t>> ns(vs.size());
            parallel_loop
                (std::views::iota(size_t(0), vs.size()),
                 [&](auto, size_t i)
                 {
                     size_t s = vs[i]->info();
                     auto& n = ns[i];
                     auto push = [&](auto w)
                     {
                         size_t u = w->info();
                         if (s < u)
                             n.push_back(u);
                     };
                     t.tds().adjacent_vertices_threadsafe
                         (vs[i], boost::make_function_output_iterator(push),
                          [&](auto w) { return t.is_infinite(w); });
                 });

            for (auto [i, v] : std::views::enumerate(vs))
            {
                size_t s = v->info();
                for (auto u : ns[i])
                    add_edge(s, u, g);
            }
        }
    }
};

} // namespace graph_tool
