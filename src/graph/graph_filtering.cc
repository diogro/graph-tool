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
#include "demangle.hh"

using namespace graph_tool;
using namespace graph_tool::detail;
using namespace boost;
using namespace std;

// Whenever no implementation is called, the following exception is thrown
graph_tool::DispatchNotFound::DispatchNotFound(const type_info& dispatch,
                                               const vector<const type_info*>& args)
    : GraphException(""), _dispatch(dispatch), _args(args)
{
    _error =
        "No static implementation was found for the desired routine. "
        "This is a graph_tool bug. :-( Please submit a bug report at "
        PACKAGE_BUGREPORT ". What follows is debug information.\n\n";

    _error += "Dispatch: " + name_demangle(_dispatch.name()) + "\n\n";
    for (size_t i = 0; i < _args.size(); ++i)
    {
        _error += "Arg " + lexical_cast<string>(i+1) + ": " +
            name_demangle(_args[i]->name()) + "\n\n";
    }
}


// gets the correct graph view at run time
std::any GraphInterface::get_graph_view() const
{
    size_t pos = 0;

    auto check_filt = [&](auto& u) -> std::any
        {
            typedef std::remove_const_t<std::remove_reference_t<decltype(*u)>>
                g_t;
            if (_edge_filter_active || _vertex_filter_active)
            {
#ifdef DISABLE_GRAPH_FILTERING
                throw GraphException("Graph filtering was disabled during compilation!");
#endif
                auto max_eindex = _mg->get_edge_index_range();
                if (max_eindex > 0)
                    _edge_filter_map.reserve(max_eindex);
                if (num_vertices(*u) > 0)
                    _vertex_filter_map.reserve(num_vertices(*u));

                typedef filt_graph<g_t,
                                   MaskFilter<edge_filter_t>,
                                   MaskFilter<vertex_filter_t>> fg_t;

                MaskFilter<edge_filter_t> e_filter(_edge_filter_map);
                MaskFilter<vertex_filter_t> v_filter(_vertex_filter_map);

                auto& gviews = const_cast<decltype(_gviews)&>(_gviews);
                if (pos >= gviews.size())
                    gviews.resize(pos + 1);
                auto& a = gviews[pos];
                if (!a.has_value())
                    a = std::make_shared<fg_t>(*u, e_filter, v_filter);
                return a;
            }
            else
            {
                return u;
            }
        };

    auto check_reverse = [&](auto& u) -> std::any
        {
            if (_reversed)
            {
#ifdef DISABLE_GRAPH_FILTERING
                throw GraphException("Graph filtering was disabled during compilation!");
#endif
                pos = 1;
                typedef std::remove_const_t<std::remove_reference_t<decltype(*u)>>
                    g_t;
                typedef reversed_graph<g_t> rg_t;
                std::shared_ptr<rg_t> rg = std::reinterpret_pointer_cast<rg_t>(u);
                return check_filt(rg);
            }
            return check_filt(u);
        };

    auto check_directed = [&](auto& u) -> std::any
        {
            if (_directed)
                return check_reverse(u);
            pos = 2;
            typedef std::remove_const_t<std::remove_reference_t<decltype(*u)>>
                g_t;
            typedef undirected_adaptor<g_t> ug_t;
            std::shared_ptr<ug_t> ug = std::reinterpret_pointer_cast<ug_t>(u);
            return check_filt(ug);
        };

    return check_directed(_mg);
}

// these test whether or not the vertex and edge filters are active
bool GraphInterface::is_vertex_filter_active() const
{ return _vertex_filter_active; }

bool GraphInterface::is_edge_filter_active() const
{ return _edge_filter_active; }


// this function will reindex all the edges, in the order in which they are
// found
void GraphInterface::re_index_edges()
{
    _mg->reindex_edges();
}

// this will definitively remove all the edges from the graph, which are being
// currently filtered out. This will also disable the edge filter
void GraphInterface::purge_edges()
{
    if (!is_edge_filter_active())
        return;

    MaskFilter<edge_filter_t> filter(_edge_filter_map);
    vector<graph_traits<multigraph_t>::edge_descriptor> deleted_edges;
    for (auto v : vertices_range(*_mg))
    {
        for (auto e : out_edges_range(v, *_mg))
            if (!filter(e))
                deleted_edges.push_back(e);
        for (auto& e  : deleted_edges)
            remove_edge(e, *_mg);
        deleted_edges.clear();
    }
}


// this will definitively remove all the vertices from the graph, which are
// being currently filtered out. This will also disable the vertex filter
void GraphInterface::purge_vertices(std::any aold_index)
{
    if (!is_vertex_filter_active())
        return;

    typedef vprop_map_t<int64_t> index_prop_t;
    index_prop_t old_index = std::any_cast<index_prop_t>(aold_index);

    MaskFilter<vertex_filter_t> filter(_vertex_filter_map);
    size_t N = num_vertices(*_mg);
    vector<bool> deleted(N, false);
    for (size_t i = 0; i < N; ++i)
        deleted[i] = !filter(vertex(i, *_mg));
    vector<int> old_indices;

    //remove vertices
    for (int64_t i = N-1; i >= 0; --i)
    {
        if (deleted[i])
            remove_vertex(vertex(i, *_mg), *_mg);
        else
            old_indices.push_back(i);
    }

    N = old_indices.size();
    for (int64_t i = N-1; i >= 0; --i)
        old_index[vertex((N - 1) - i, *_mg)] = old_indices[i];
}

void GraphInterface::set_vertex_filter_property(std::any property)
{
    try
    {
        _vertex_filter_map =
            std::any_cast<vertex_filter_t::checked_t>(property).get_unchecked();
        _vertex_filter_active = true;
    }
    catch (std::bad_any_cast&)
    {
        if (property.has_value())
            throw GraphException("Invalid vertex filter property!");
        _vertex_filter_active = false;
    }
}

void GraphInterface::set_edge_filter_property(std::any property)
{
    try
    {
        _edge_filter_map =
            std::any_cast<edge_filter_t::checked_t>(property).get_unchecked();
        _edge_filter_active = true;
    }
    catch(bad_any_cast&)
    {
        if (property.has_value())
            throw GraphException("Invalid edge filter property!");
        _edge_filter_active = false;
    }
}
