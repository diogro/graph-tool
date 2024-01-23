// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2023 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_SAMPLE_EDGE_HH
#define GRAPH_SAMPLE_EDGE_HH

#include <tuple>
#include <queue>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "../../generation/sampler.hh"

#include "random.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Graph>
class EdgeSampler
{
public:
    template <class Iter>
    EdgeSampler(Graph& g, Iter tgts_begin, Iter tgts_end)
        : _g(g), _tgts(tgts_begin, tgts_end),
          _visited(get(vertex_index, _g), num_vertices(_g)),
          _ds(get(vertex_index, _g), num_vertices(_g))
    {
    }

    EdgeSampler(Graph& g)
        : EdgeSampler(g, vertices(g).first, vertices(g).second)
    {
    }

    EdgeSampler(const EdgeSampler&) = delete;

    void get_ns(size_t v, size_t d, int delta = 0,
                size_t z = std::numeric_limits<size_t>::max())
    {
        if (delta != 0 && z == v)
            delta = 0;

        _ns.clear();
        _ds[v] = 0;
        _visited[v] = true;

        _queue.push(v);

        while (!_queue.empty())
        {
            auto u = _queue.front();
            _queue.pop();
            _ns.insert(u);

            for (auto w : out_neighbors_range(u, _g))
            {
                if (graph_tool::is_directed(_g))
                {
                    if (_visited[w] || (delta < 0 && ((u == v && w == z))))
                        continue;
                }
                else
                {
                    if (_visited[w] || (delta < 0 && ((u == v && w == z) ||
                                                      (u == z && w == v))))
                        continue;
                }

                _visited[w] = true;
                _ds[w] = _ds[u] + 1;
                if (_ds[w] < d)
                    _queue.push(w);
            }

            if (graph_tool::is_directed(_g))
            {
                for (auto w : in_neighbors_range(u, _g))
                {
                    if (_visited[w] || (delta < 0 && ((u == z && w == v))))
                        continue;

                    _visited[w] = true;
                    _ds[w] = _ds[u] + 1;
                    if (_ds[w] < d)
                        _queue.push(w);
                }
            }

            if (delta > 0 && u == v && !_visited[z])
            {
                _visited[z] = true;
                _ds[z] = _ds[u] + 1;
                if (_ds[z] < d)
                    _queue.push(z);
            }
        }

        _visited[v] = false;
        for (auto u : _ns)
            _visited[u] = false;
        if (delta != 0)
            _visited[z] = false;
    }

    template <class RNG>
    size_t sample_neighbor(size_t v, size_t d, RNG& rng)
    {
        get_ns(v, d);
        return uniform_sample(_ns, rng);
    }

    template <class RNG>
    std::tuple<size_t, size_t> sample(size_t d, RNG& rng)
    {
        size_t v = uniform_sample(_tgts, rng);
        size_t u = sample_neighbor(v, d, rng);
        return {u, v};
    }

    double log_prob(size_t v, size_t u, size_t d, size_t m, int dm,
                    bool neighbor = false)
    {
        if ((m > 0) == (m + dm > 0))
            get_ns(v, d);
        else if (m + dm > 0)
            get_ns(v, d, 1, u);
        else
            get_ns(v, d, -1, u);

        if (_ns.find(u) == _ns.end())
            return -numeric_limits<double>::infinity();

        if (neighbor)
            return -log(_ns.size());
        return -log(_ns.size()) - log(_tgts.size());
    }

    std::tuple<size_t, size_t> get_null_edge()
    {
        return _null_edge;
    }

private:

    Graph& _g;
    std::vector<size_t> _tgts;

    typename vprop_map_t<bool>::type::unchecked_t _visited;
    typename vprop_map_t<bool>::type::unchecked_t _ds;

    std::queue<size_t> _queue;
    idx_set<size_t> _ns;

    std::tuple<size_t, size_t> _null_edge =             \
            {std::numeric_limits<size_t>::max(),
             std::numeric_limits<size_t>::max()};
};


} // graph_tool namespace

#endif // GRAPH_SAMPLE_EDGE_HH
