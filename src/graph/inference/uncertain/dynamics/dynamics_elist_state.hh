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

#ifndef DYNAMICS_ELIST_STATE
#define DYNAMICS_ELIST_STATE

#include <queue>

#include "config.h"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef multi_array_ref<int64_t,2> elist_t;

template <class Graph>
class elist_state_t
{
public:

    enum class move_t { self = 0, uniform, edge, nearby, candidates };

    elist_state_t(elist_t& elist, std::vector<std::vector<size_t>>& candidates,
                  size_t d, double pself, double puniform,
                  double pedge, double pnearby, double pcandidates, const Graph& g)
        : _moves({move_t::self, move_t::uniform, move_t::edge,
                  move_t::nearby, move_t::candidates}),
          _probs({pself, puniform, pedge, pnearby, pcandidates}),
          _move_sampler(_moves, _probs),
          _ds(num_vertices(g)),
          _d(d),
          _pself(pself),
          _puniform(puniform),
          _pedge(pedge),
          _pnearby(pnearby),
          _pcandidates(pcandidates),
          _g(g),
          _candidates(candidates)
    {
        if (_candidates.empty())
        {
            _candidates.resize(num_vertices(g));
            for (size_t i = 0; i < elist.shape()[0]; ++i)
            {
                size_t u = elist[i][0];
                size_t v = elist[i][1];
                _candidates[u].push_back(v);
                if (u != v)
                    _candidates[v].push_back(u);
            }

            idx_set<size_t> tmp;
            for (auto& cs : _candidates)
            {
                tmp.clear();
                for (auto v : cs)
                    tmp.insert(v);
                cs.clear();
                cs.insert(cs.end(), tmp.begin(), tmp.end());
                std::sort(cs.begin(), cs.end());
            }
        }
    }

    std::vector<move_t> _moves;
    std::vector<double> _probs;

    Sampler<move_t> _move_sampler;

    typedef typename graph_traits<adj_list<>>::edge_descriptor edge_t;
    gt_hash_set<edge_t> _es;


    template <class RNG>
    size_t sample_edge(size_t u, RNG& rng)
    {
        move_t move = _move_sampler(rng);

        get_ns(u, 1);

        if (_ns.size() - 1 == 0 && (move == move_t::edge ||
                                    move == move_t::nearby))
            move = move_t::uniform;

        auto& cs = _candidates[u];
        if (cs.empty() && move == move_t::candidates)
            move = move_t::uniform;

        size_t v = u;
        switch (move)
        {
        case move_t::self:
            break;
        case move_t::uniform:
            {
                std::uniform_int_distribution<size_t> rand(0, num_vertices(_g) - 1);
                v = rand(rng);
            }
            break;
        case move_t::edge:
            v = uniform_sample(_ns.begin() + 1, _ns.end(), rng);
            break;
        case move_t::nearby:
            get_ns(u, _d);
            v = uniform_sample(_ns.begin() + 1, _ns.end(), rng);
            break;
        case move_t::candidates:
            v = uniform_sample(cs, rng);
            break;
        default:
            break;
        }

        return v;
    }

    template <class RNG>
    std::tuple<size_t, size_t> sample_edge(RNG& rng)
    {
        std::uniform_int_distribution<size_t> rand(0, num_vertices(_g) - 1);
        size_t u = rand(rng);
        return {u, sample_edge(u, rng)};
    }

    template <bool conditional=false>
    double log_prob(size_t u, size_t v, int delta = 0,
                    size_t s = std::numeric_limits<size_t>::max(),
                    size_t t = std::numeric_limits<size_t>::max())
    {
        double lself = -numeric_limits<double>::infinity();
        double luniform = -numeric_limits<double>::infinity();
        double ledge = -numeric_limits<double>::infinity();
        double lnearby = -numeric_limits<double>::infinity();
        double lcandidates = -numeric_limits<double>::infinity();

        double pself = _pself;
        double puniform = _puniform;
        double pedge = _pedge;
        double pnearby = _pnearby;
        double pcandidates = _pcandidates;

        get_ns(u, 1, delta, s, t);
        size_t k = _ns.size() - 1;

        if (k == 0)
        {
            puniform += pedge + pnearby;
            pedge = pnearby = 0;
        }

        auto& cs = _candidates[u];
        if (cs.empty())
        {
            puniform += pcandidates;
            pcandidates = 0;
        }

        double ltot = log(pself + puniform + pedge + pnearby + pcandidates);

        if (pself > 0 && v == u)
            lself = log(pself) - ltot;

        if (puniform > 0)
            luniform = log(puniform) - ltot - safelog_fast(num_vertices(_g));

        if (pedge > 0 && u != v && _ns.find(v) != _ns.end())
            ledge = log(pedge) - ltot - log(k);

        if (pnearby > 0)
        {
            get_ns(u, _d, delta, s, t);
            if (u != v && _ns.find(v) != _ns.end())
                lnearby = log(pnearby) - ltot - log(_ns.size()-1);
        }

        if (pcandidates > 0)
        {
            auto iter = std::lower_bound(cs.begin(), cs.end(), v);
            if (iter != cs.end() && *iter == v)
                lcandidates = log(pcandidates) - ltot - log(cs.size());
        }

        double l = log_sum_exp(lself, luniform, ledge, lnearby, lcandidates);
        if constexpr (!conditional)
            l -= std::log(num_vertices(_g));
        return l;
    }

    bool same_edge(size_t u, size_t v, size_t s, size_t t)
    {
        if (u == s && v == t)
            return true;
        if (u == t && v == s)
            return true;
        return false;
    }

    void get_ns(size_t u, size_t d, int delta = 0,
                size_t s = std::numeric_limits<size_t>::max(),
                size_t t = std::numeric_limits<size_t>::max())
    {
        _ns.clear();
        _ds[u] = 0;
        _ns.insert(u);

        _queue.push(u);

        while (!_queue.empty())
        {
            auto v = _queue.front();
            _queue.pop();

            for (auto w : all_neighbors_range(v, _g))
            {
                if (_ns.find(w) != _ns.end() ||
                    (delta < 0 && same_edge(v, w, s, t)))
                    continue;
                _ds[w] = _ds[v] + 1;
                _ns.insert(w);
                if (_ds[w] < d)
                    _queue.push(w);
            }

            if (delta > 0 && (v == s || v == t))
            {
                size_t z = (v == s) ? t : s;
                if (_ns.find(z) != _ns.end())
                    continue;
                _ds[z] = _ds[v] + 1;
                _ns.insert(z);
                if (_ds[z] < d)
                    _queue.push(z);
            }
        }
    }

private:
    std::vector<size_t> _ds;
    std::queue<size_t> _queue;
    idx_set<size_t> _ns;

    size_t _d;
    double _pself;
    double _puniform;
    double _pedge;
    double _pnearby;
    double _pcandidates;

    const Graph& _g;

    std::vector<std::vector<size_t>>& _candidates;
};


} // graph_tool namespace

#endif //DYNAMICS_ELIST_STATE
