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

#ifndef GRAPH_BLOCKMODEL_ELIST_HH
#define GRAPH_BLOCKMODEL_ELIST_HH

#include "samplers.hh"
#include "dynamic_sampler.hh"

namespace graph_tool
{

// ====================================
// Construct and manage half-edge lists
// ====================================

template <bool flat>
class EGroups
{
public:
    template <class BGraph, class Eprop>
    EGroups(BGraph& bg, Eprop& mrs)
        : _egroups(num_vertices(bg)),
          _pos(num_vertices(bg))
    {
        parallel_vertex_loop
            (bg,
             [&](auto r)
             {
                 for (auto e : out_edges_range(r, bg))
                     insert_edge(r, target(e, bg), mrs[e]);
                 for (auto e : in_edges_range(r, bg))
                     insert_edge(r, source(e, bg), mrs[e]);
             });

        //check(bg, mrs);
    }

    void add_group()
    {
        _egroups.emplace_back();
        _pos.emplace_back();
    }

    void insert_edge(size_t s, size_t t, int64_t weight)
    {
        auto& pos = _pos[s];
        auto iter = pos.find(t);
        if (iter == pos.end())
            iter = pos.insert({t, _null_pos}).first;
        insert_edge(t, weight, _egroups[s], iter->second);
        if (iter->second == _null_pos)
        {
            pos.erase(iter);
            if (pos.empty())
            {
                pmap_t empty;
                pos.swap(empty);
            }
        }
    }

    template <class EV>
    void insert_edge(size_t t, int64_t weight, EV& elist, auto& pos)
    {
        if (pos != _null_pos)
        {
            assert(elist.is_valid(pos) && elist[pos] == t);
            elist.update(pos, weight, true);
            if (elist.get_prob(pos) == 0)
            {
                elist.remove(pos);
                pos = _null_pos;
                if (elist.empty())
                    elist.clear(true);
            }
        }
        else
        {
            if (weight > 0)
                pos = elist.insert(t, weight);
            else
                pos = _null_pos;
        }
    }

    template <class RNG>
    size_t sample_edge(size_t r, RNG& rng)
    {
        return _egroups[r].sample(rng);
    }

    template <class Eprop, class BGraph>
    void check([[maybe_unused]] BGraph& bg, [[maybe_unused]] Eprop& mrs)
    {
        for (auto e : edges_range(bg))
        {
            auto r = source(e, bg);
            auto s = target(e, bg);

            assert(r < _pos.size());
            auto& pos = _pos[r];
            auto iter = pos.find(s);
            assert(iter != pos.end());

            [[maybe_unused]] auto p = _egroups[r].get_prob(iter->second);

            if (!is_directed(bg) || r == s)
            {
                assert(p == mrs[e] * (r == s ? 2 : 1));
            }
            else
            {
                auto ne = edge(s, r, bg);
                if (ne.second)
                    assert(p == mrs[e] + mrs[ne.first]);
                else
                    assert(p == mrs[e]);
            }
        }
    }

private:
    std::vector<DynamicSampler<size_t>> _egroups;
    typedef std::conditional_t<flat,
                               std::flat_map<size_t, uint32_t>,
                               gt_hash_map<size_t, uint32_t>> pmap_t;
    std::vector<pmap_t> _pos;
    static constexpr uint32_t _null_pos = numeric_limits<uint32_t>::max();
};

} // namespace graph_tool

#endif //GRAPH_BLOCKMODEL_ELIST_HH
