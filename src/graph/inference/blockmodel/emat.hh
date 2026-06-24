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

#ifndef GRAPH_BLOCKMODEL_EMAT_HH
#define GRAPH_BLOCKMODEL_EMAT_HH

#include <boost/multi_array.hpp>
#include "hash_map_wrap.hh"

namespace graph_tool
{

// this structure speeds up the access to the edges between given blocks, by
// using a simple adjacency matrix

template <class BGraph>
class EMat
{
public:
    EMat() {}

    EMat(BGraph& bg)
    {
        sync(bg);
    }

    void sync(BGraph& bg)
    {
        size_t B = num_vertices(bg);
        _mat.resize(boost::extents[B][B]);
        std::fill(_mat.data(), _mat.data() + _mat.num_elements(), _null_edge);

        for (auto e : edges_range(bg))
        {
            assert(get_me(source(e, bg),target(e, bg)) == _null_edge);
            _mat[source(e, bg)][target(e, bg)] = e;
            if constexpr (!is_directed_v<BGraph>)
                _mat[target(e, bg)][source(e, bg)] = e;
        }
    }

    void add_group(BGraph& bg)
    {
        sync(bg);
    }

    typedef typename graph_traits<BGraph>::vertex_descriptor vertex_t;
    typedef typename graph_traits<BGraph>::edge_descriptor edge_t;

    const edge_t& get_me(vertex_t r, vertex_t s) const
    {
        return _mat[r][s];
    }

    void put_me(vertex_t r, vertex_t s, const edge_t& e)
    {
        assert(e != _null_edge);
        _mat[r][s] = e;
        if (!is_directed_v<BGraph> && r != s)
            _mat[s][r] = e;
    }

    void remove_me(const edge_t& me, BGraph& bg)
    {
        auto r = source(me, bg);
        auto s = target(me, bg);
        _mat[r][s] = _null_edge;
        if constexpr (!is_directed_v<BGraph>)
            _mat[s][r] = _null_edge;
    }

    const edge_t& get_null_edge() const { return _null_edge; }

private:
    multi_array<edge_t, 2> _mat;
    static const edge_t _null_edge;
};

template <class BGraph>
const typename EMat<BGraph>::edge_t EMat<BGraph>::_null_edge;


// this structure speeds up the access to the edges between given blocks, since
// we're using an adjacency list to store the block structure (this is like
// EMat above, but takes less space and is slower)

template <class BGraph>
class EHash
{
public:
    typedef BlockStateSpec::group_t group_t;

    EHash() {}

    EHash(BGraph& bg)
    {
        sync(bg);
    }

    void sync(BGraph& bg)
    {
        _h.clear();
        add_group(bg);

        for (auto e : edges_range(bg))
        {
            assert(get_me(source(e, bg), target(e, bg)) == _null_edge);
            put_me(source(e, bg), target(e, bg), e);
        }
    }

    void add_group(BGraph& bg)
    {
        if (num_vertices(bg) > _h.size())
            _h.resize(num_vertices(bg));
    }

    typedef typename graph_traits<BGraph>::vertex_descriptor vertex_t;
    typedef typename graph_traits<BGraph>::edge_descriptor edge_t;

    const edge_t& get_me(vertex_t r, vertex_t s) const
    {
        if (!is_directed_v<BGraph> && r > s)
            std::swap(r, s);
        auto& h = _h[r];
        auto iter = h.find(s);
        if (iter == h.end())
            return _null_edge;
        return iter->second;
    }

    void put_me(vertex_t r, vertex_t s, const edge_t& e)
    {
        assert(e != _null_edge);
        if (!is_directed_v<BGraph> && r > s)
            _h[s].insert_or_assign(r, e);
        else
            _h[r].insert_or_assign(s, e);
    }

    void remove_me(const edge_t& me, BGraph& bg)
    {
        auto r = source(me, bg);
        auto s = target(me, bg);
        if (!is_directed_v<BGraph> && r > s)
            std::swap(r, s);
        auto& hr = _h[r];
        hr.erase(s);
        if (hr.empty())
        {
            gt_hash_map<group_t, edge_t> empty;
            hr.swap(empty);
        }
    }

    const edge_t& get_null_edge() const { return _null_edge; }

private:
    std::vector<gt_hash_map<group_t, edge_t>> _h;
    static const edge_t _null_edge;
};

template <class BGraph>
const typename EHash<BGraph>::edge_t EHash<BGraph>::_null_edge;

} // namespace graph_tool

#endif // GRAPH_BLOCKMODEL_EMAT_HH
