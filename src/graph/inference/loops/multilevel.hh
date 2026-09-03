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

#ifndef MULTILEVEL_HH
#define MULTILEVEL_HH

#include <queue>
#include <cmath>
#include <print>

#include "mcmc_loop.hh"

#include <boost/range/combine.hpp>
#include "../../topology/graph_bipartite_weighted_matching.hh"
#include "../support/bisection_search.hh"
#include "../support/contingency.hh"
#include "../support/util.hh"

namespace graph_tool
{

template <class State>
struct vlist_t
{
    typedef std::conditional_t<std::is_scalar_v<typename State::vertex_t>,
                               std::vector<typename State::vertex_t>&,
                               std::vector<typename State::vertex_t>> type;
};

#define MULTILEVEL_STATE_params(State)                                         \
    ((state, State&, 0))                                                       \
    ((beta, double, 0))                                                        \
    ((c, double, 0))                                                           \
    ((d, double, 0))                                                           \
    ((r, double, 0))                                                           \
    ((dB_epsilon, double, 0))                                                  \
    ((bisection, bool, 0))                                                     \
    ((refine, bool, 0))                                                        \
    ((random_bisection, bool, 0))                                              \
    ((merge_sweeps, size_t, 0))                                                \
    ((mh_sweeps, size_t, 0))                                                   \
    ((fast_proposal, bool, 0))                                                 \
    ((sample_group_thres, size_t, 0))                                          \
    ((parallel, bool, 0))                                                      \
    ((parallel_thres, size_t, 0))                                              \
    ((parallel_merge, bool, 0))                                                \
    ((init_min_iter, size_t, 0))                                               \
    ((init_min_iter_beta, size_t, 0))                                          \
    ((init_r, double, 0))                                                      \
    ((init_beta, double, 0))                                                   \
    ((gibbs, bool, 0))                                                         \
    ((M, size_t, 0))                                                           \
    ((global_moves, bool, 0))                                                  \
    ((cache_states, bool, 0))                                                  \
    ((B_min, size_t, 0))                                                       \
    ((B_max, size_t, 0))                                                       \
    ((b_min, typename State::bmap_t, 0))                                       \
    ((b_max, typename State::bmap_t, 0))                                       \
    ((vertices, typename vlist_t<State>::type, 0))                             \
    ((force_accept, bool, 0))                                                  \
    ((entropy_args, typename State::entropy_args_t, 0))                        \
    ((verbose, int, 0))                                                        \
    ((niter, size_t, 0))


struct MultilevelStateBase
{
    // mandatory
    constexpr static size_t _null_group = std::numeric_limits<size_t>::max();

    template <class F>
    void iter_nodes(F&& f) = delete;

    template <class F>
    void iter_groups(F&& f) = delete;

    template <class V>
    size_t get_group(size_t v) = delete;

    template <class V, class RNG>
    size_t get_new_group(V v, bool inherit, RNG& rng, bool force=false) = delete;

    template <class V, class Group>
    void move_node(V v, Group r, bool cache) = delete;

    template <class V, class Group>
    double virtual_move(V v, Group r, size_t tnum) = delete;

    template <class V, class RNG>
    auto sample_group(V v, bool allow_random, bool allow_empty,
                      bool init_heuristic, RNG& rng) = delete;

    template <class V, class Group>
    double get_move_prob(V v, Group r, Group s, bool allow_random,
                         bool allow_empty, bool reverse, size_t tnum=0) = delete;

    double entropy() = delete;

    //optional
    constexpr static bool _has_merge = false;
    constexpr static bool _has_parallel_move = false;
    constexpr static bool _has_parallel_virtual_move = true;

    constexpr static bool _has_relabel = false;
    constexpr static bool _has_parallel_relabel = false;
    constexpr static bool _has_extra_sweep = false;

    template <class VS>
    size_t get_Bmin(VS&)
    {
        return 1;
    }

    template <class V>
    size_t get_b_min(V)
    {
        return 1;
    }

    template <class V>
    V get_b_max(V v)
    {
        return v;
    }

    void relax_update(bool) {}

    template <class V>
    void push_state(V&&) {}
    void pop_state(bool) {}
    void swap_state() {}
    void clear_state() {}

    template <class Group>
    double virtual_merge(const Group& r, const Group& s, int tnum) = delete;

    template <class Group>
    void merge_groups(const Group& r, const Group& s) = delete;

    template <class Vs, class Groups>
    double move_nodes_parallel(Vs&& vs, Groups&& rs) = delete;

    int group_label(auto) { return 0; }
    int sample_group_label(auto, auto&) { return 0; }
    double get_relabel_prob(auto, auto, auto, bool) { return 0; }
    double relabel_group_dS(auto, auto) { return 0; }
    void relabel_group(auto, auto) {}
    double relabel_groups_parallel(auto&, auto&) { return 0; }

    double extra_sweep(auto&, double, auto&, bool, auto&) { return 0; }

    void put_cache(size_t) {}
    void get_cache(size_t) {}
};


template <class State, class Node, class Group,
          template <class> class VSet,
          template <class, class> class VMap,
          template <class> class GSet,
          template <class, class> class GMap,
          class GSMap>
struct Multilevel:
        public State,
        public MetropolisStateBase
{
    enum class move_t { multilevel=0, null };

    template <class... TS>
    Multilevel(TS&&... as)
        : State(as...)
        {
            State::iter_nodes
                ([&](const auto& v)
                {
                    auto r = State::get_group(v);
                    _groups[r].insert(v);
                    _N++;
                    _nodes.insert(v);
                });

            State::iter_groups
                ([&](const auto& r)
                {
                    _rlist.insert(r);
                });
        }

    VSet<Node> _nodes;
    GSMap _groups;

    using State::_bisection;
    using State::_refine;
    using State::_merge_sweeps;
    using State::_mh_sweeps;
    using State::_sample_group_thres;
    using State::_r;
    using State::_init_r;
    using State::_init_min_iter;
    using State::_init_min_iter_beta;
    using State::_gibbs;
    using State::_null_group;
    using State::_beta;
    using State::_init_beta;
    using State::_cache_states;
    using State::_verbose;
    using State::_force_accept;

    using State::_B_min;
    using State::_B_max;
    using State::_dB_epsilon;

    using State::_has_merge;
    using State::_has_parallel_move;
    using State::_has_parallel_virtual_move;
    using State::_parallel;
    using State::_parallel_merge;
    using State::_has_relabel;
    using State::_has_parallel_relabel;
    using State::_has_extra_sweep;

    size_t _nmoves = 0;

    typedef decltype(std::declval<State>().group_label(Group())) group_label_t;

    std::vector<std::vector<std::tuple<Node,Group>>> _bstack;
    std::vector<std::vector<std::tuple<Group,group_label_t>>> _lstack;

    Sampler<move_t, false> _move_sampler;

    template <class Vs>
    void push_b(const Vs& vs)
    {
        _bstack.emplace_back();
        auto& back = _bstack.back();
        back.resize(vs.size());
        #pragma omp parallel for if (vs.size() > 1000)
        for (size_t i = 0; i < vs.size(); ++i)
            back[i] = {vs[i], State::get_group(vs[i])};
        if constexpr (_has_relabel)
        {
            _lstack.emplace_back();
            auto& lback = _lstack.back();
            gt_hash_set<Group> rs;
            for (auto v : vs)
            {
                auto r = State::get_group(v);
                if (rs.find(r) == rs.end())
                {
                    rs.insert(r);
                    lback.emplace_back(r, State::group_label(r));
                }
            }
            for (auto r : _rlist)
            {
                if (rs.find(r) == rs.end())
                {
                    rs.insert(r);
                    lback.emplace_back(r, State::group_label(r));
                }
            }
        }
        State::push_state(vs);
    }

    void pop_b(bool discard = false)
    {
        if (not discard)
        {
            auto& back = _bstack.back();
            for (auto& [v, s] : back)
            {
                auto r = State::get_group(v);
                if (r == s)
                    continue;
                move_node(v, s, false, _has_parallel_move);
            }

            if constexpr (_has_parallel_move)
            {
                State::move_nodes_parallel
                    (back | std::views::transform([](auto& vr) { return get<0>(vr); }),
                     back | std::views::transform([](auto& vr) { return get<1>(vr); }));
            }

            if constexpr (_has_relabel)
            {
                auto& lback = _lstack.back();
                if constexpr (_has_parallel_relabel)
                {
                    State::relabel_groups_parallel
                        (lback | std::views::transform([](auto& ru) { return get<0>(ru); }),
                         lback | std::views::transform([](auto& ru) { return get<1>(ru); }));
                }
                else
                {
                    for (auto& [r, u] : lback)
                        State::relabel_group(r, u);
                }
            }
        }

        _bstack.pop_back();
        if constexpr (_has_relabel)
            _lstack.pop_back();
        State::pop_state(discard);
    }

    void swap_b()
    {
        std::swap(_bstack[_bstack.size() - 2], _bstack[_bstack.size() - 1]);
        if constexpr (_has_relabel)
            std::swap(_lstack[_lstack.size() - 2], _lstack[_lstack.size() - 1]);
        State::swap_state();
    }

    void clear_b()
    {
        _bstack.clear();
        if constexpr (_has_relabel)
            _lstack.clear();
        State::clear_state();
    }

    GSet<Group> _rlist;

    std::vector<Node> _vs;
    move_t _move;
    GSet<Group> _rs, _rs_prev;
    GMap<Group, pair<Group, double>> _best_merge;
    GMap<Group, Group> _root;

    VSet<Node> _visited;

    constexpr static size_t _null_move = 1;

    size_t _N = 0;

    double _dS;
    double _a;

    Group node_state(const Node&)
    {
        return Group();
    }

    constexpr bool skip_node(const Node&)
    {
        return false;
    }

    void move_node(const Node& v, const Group& r, bool cache = false,
                   bool skip_move = false)
    {
        Group s = State::get_group(v);
        if (s == r)
            return;
        if (!skip_move)
            State::move_node(v, r, cache);
        auto& vs = _groups[s];
        vs.erase(v);
        if (vs.empty())
            _groups.erase(s);
        _groups[r].insert(v);
        _nmoves++;
    }

    template <bool clear=true>
    void get_group_vs(const Group& r, std::vector<Node>& vs)
    {
        if constexpr (clear)
            vs.clear();
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            vs.insert(vs.end(), iter->second.begin(), iter->second.end());
    }

    size_t get_wr(const Group& r)
    {
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            return iter->second.size();
        return 0;
    }

    std::vector<size_t> _vis;

    template <bool smart, class RNG>
    double mh_sweep(std::vector<Node>& vs, GSet<Group>& rs, double beta,
                    RNG& rng, size_t B_min = 0,
                    size_t B_max = std::numeric_limits<size_t>::max(),
                    bool init_heuristic = false, bool allow_new = false)
    {
        if (rs.size() == 1 || (rs.size() == vs.size() && B_min == rs.size()))
            return 0;

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);

        double S = 0;
        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);
            Group s;

            if constexpr (smart)
            {
                s = State::sample_group(v, false, false, init_heuristic, rng); // c == 0!
                if (rs.find(s) == rs.end() && !allow_new)
                    continue;
            }
            else
            {
                rs.erase(r);
                s = uniform_sample(rs, rng);
                rs.insert(r);
            }

            int dB = 0;
            if (get_wr(s) == 0)
                dB++;
            if (get_wr(r) == 1)
                dB--;

            double dS;
            if ((dB < 0 && rs.size() + dB < B_min) ||
                (dB > 0 && rs.size() + dB > B_max))
                dS = std::numeric_limits<double>::infinity();
            else
                dS = State::virtual_move(v, s);

            double pf = 0, pb = 0;
            if constexpr (smart)
            {
                if (!std::isinf(beta) && s != r && !init_heuristic)
                {
                    pf = State::get_move_prob(v, r, s, false, rs.size() > B_min, false);
                    pb = State::get_move_prob(v, s, r, false, rs.size() > B_min, true);
                }
            }

            double ap = 0, rp = 0;
            bool accept;
            if constexpr (smart)
            {
                accept = metropolis_accept(dS, pb - pf, beta, rng);
            }
            else
            {
                if (!std::isinf(beta))
                {
                    double logZ = log_sum_exp(-beta * dS, 0.);
                    ap = -beta * dS - logZ;
                    rp = -logZ;

                    std::bernoulli_distribution u(exp(ap));
                    accept = u(rng);
                }
                else
                {
                    accept = dS < 0;
                    ap = 0;
                    rp = -std::numeric_limits<double>::infinity();
                    if (!accept)
                        std::swap(ap, rp);
                }
            }

            if (accept)
            {
                if (get_wr(s) == 0)
                    rs.insert(s);

                move_node(v, s, true);
                S += dS;

                if (get_wr(r) == 0)
                    rs.erase(r);

                assert(r != s || dS == 0);
            }
        }

        return S;
    }

    std::vector<Group> _mh_moves;

    template <bool smart, class RNG>
    double pseudo_mh_sweep(std::vector<Node>& vs, GSet<Group>& rs, double beta,
                           RNG& rng_, size_t B_min = 0,
                           size_t B_max = std::numeric_limits<size_t>::max(),
                           bool init_heuristic = false, bool allow_new=false)
    {
        if (rs.size() == 1 || (rs.size() == vs.size() && B_min == rs.size()))
            return 0;

        auto& moves = _mh_moves;
        moves.resize(vs.size());

        if (std::isinf(beta))
            push_b(vs);

        parallel_rng<rng_t> prng(rng_);

        #pragma omp parallel
        {
            auto tnum = get_thread_num();
            auto& rng = prng.get(rng_, tnum);

            #pragma omp for schedule(runtime)
            for (size_t vi = 0; vi < std::ranges::size(vs); ++vi) // clang doesn't allow a range loop here
            {
                auto& v = vs[vi];
                auto r = State::get_group(v);
                Group s;

                if constexpr (smart)
                {
                    s = State::sample_group(v, false, false, init_heuristic, rng); // c == 0!
                    if ((rs.find(s) == rs.end() && !allow_new && rs.size() >= B_max) || s == r)
                    {
                        moves[vi] = r;
                        continue;
                    }
                }
                else
                {
                    s = r;
                    while (s == r)
                        s = uniform_sample(rs, rng);
                }

                double dS = State::virtual_move(v, s, tnum);

                double pf = 0, pb = 0;
                if constexpr (smart)
                {
                    if (!std::isinf(beta) && s != r && !init_heuristic)
                    {
                        pf = State::get_move_prob(v, r, s, false, rs.size() > B_min, false, tnum);
                        pb = State::get_move_prob(v, s, r, false, rs.size() > B_min, true, tnum);
                    }
                }

                if (metropolis_accept(dS, pb - pf, beta, rng))
                    moves[vi] = s;
                else
                    moves[vi] = r;
            }
        }

        double Sb = 0;

        if constexpr (!_has_parallel_move)
            Sb = State::entropy();

        std::vector<Node> nvs;
        std::vector<Group> nrs;

        for (const auto& [vi, v] : enumerate(vs))
        {
            auto r = State::get_group(v);
            auto s = moves[vi];
            if (r == s || (get_wr(s) == 0 && !allow_new))
                continue;
            int dB = 0;
            if (get_wr(s) == 0)
                dB++;
            if (get_wr(r) == 1)
                dB--;
            if ((dB < 0 && rs.size() + dB < B_min) ||
                (dB > 0 && rs.size() + dB > B_max))
                continue;
            if (get_wr(s) == 0)
                rs.insert(s);
            move_node(v, s, false, _has_parallel_move);
            if (get_wr(r) == 0)
                rs.erase(r);
            if constexpr (_has_parallel_move)
            {
                nvs.push_back(v);
                nrs.push_back(s);
            }
        }

        double dS;

        if constexpr (_has_parallel_move)
            dS = State::move_nodes_parallel(nvs, nrs);
        else
            dS = State::entropy() - Sb;

        if (std::isinf(beta))
        {
            pop_b(dS <= 0);
            if (dS > 0)
            {
                dS = 0;
                rs.clear();
                for (auto v : vs)
                    rs.insert(State::get_group(v));
            }
        }

        return dS;
    }

    template <class RNG>
    double gibbs_sweep(std::vector<Node>& vs, GSet<Group>& rs, RNG& rng)
    {
        if (rs.size() == 1 || rs.size() == vs.size())
            return 0;

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        double S = 0;

        std::vector<double> dS(rs.size());
        std::vector<double> probs(rs.size());
        std::vector<double> ps(rs.size());
        std::vector<size_t> ris(rs.size());
        std::iota(ris.begin(), ris.end(), 0);

        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);

            for (size_t j = 0; j < rs.size(); ++j)
            {
                auto iter = rs.begin();
                std::advance(iter, j);
                Group s = *iter;
                if (s != r && get_wr(State::get_group(v)) == 1)
                    dS[j] = std::numeric_limits<double>::infinity();
                else
                    dS[j] = State::virtual_move(v, s);
            }

            double Z = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < rs.size(); ++j)
            {
                if (!std::isinf(_beta) && !std::isinf(dS[j]))
                {
                    ps[j] = -dS[j] * _beta;
                }
                else
                {
                    if (dS[j] < 0)
                        ps[j] = 0;
                    else
                        ps[j] = -std::numeric_limits<double>::infinity();
                }
                Z = log_sum_exp(Z, ps[j]);
            }

            size_t si = rs.size();
            for (size_t j = 0; j < rs.size(); ++j)
                probs[j] = exp(ps[j] - Z);

            Sampler<size_t> sampler(ris, probs);
            si = sampler.sample(rng);

            if (si >= rs.size() || std::isinf(dS[si]))
                break;

            auto iter = rs.begin();
            std::advance(iter, si);
            move_node(v, *iter);
            S += dS[si];
        }

        return S;
    }

    template <class RNG>
    double relabel_sweep(GSet<Group>& rs, double beta, RNG& rng_)
    {
        double S = 0;

        if constexpr (_has_parallel_relabel)
        {
            std::vector<decltype(State::group_label(*rs.begin()))> moves(rs.size());
            parallel_rng<rng_t> prng(rng_);
            #pragma omp parallel
            {
                auto tnum = get_thread_num();
                auto& rng = prng.get(rng_, tnum);

                #pragma omp for schedule(runtime)
                for (size_t ri = 0; ri < std::ranges::size(rs); ++ri) // clang doesn't allow a range loop here
                {
                    auto&& r = *(std::ranges::begin(rs) + ri);
                    auto v = State::group_label(r);
                    auto u = State::sample_group_label(r, rng);
                    double dS = State::relabel_group_dS(r, u);
                    double pf = 0, pb = 0;
                    if (!std::isinf(beta))
                    {
                        pf = State::get_relabel_prob(r, v, u, false);
                        pb = State::get_relabel_prob(r, u, v, true);
                    }
                    if (metropolis_accept(dS, pb - pf, beta, rng))
                        moves[ri] = u;
                    else
                        moves[ri] = v;
                }
            }

            S -= State::entropy();
            State::relabel_groups_parallel(rs, moves);
            S += State::entropy();
        }
        else
        {
            for (auto r : rs)
            {
                auto v = State::group_label(r);
                auto u = State::sample_group_label(r, rng_);
                double dS = State::relabel_group_dS(r, u);
                double pf = 0, pb = 0;
                if (!std::isinf(beta))
                {
                    pf = State::get_relabel_prob(r, v, u, false);
                    pb = State::get_relabel_prob(r, u, v, true);
                }
                if (metropolis_accept(dS, pb - pf, beta, rng_))
                {
                    State::relabel_group(r, u);
                    S += dS;
                }
            }
        }
        return S;
    }

    double virtual_merge_dS(const Group& r, const Group& s)
    {
        std::vector<Node> mvs;
        State::relax_update(true);

        double dS = 0;
        for (auto& v : _groups[r])
        {
            assert(State::get_group(v) == r);
            double ddS = State::virtual_move(v, s);
            dS += ddS;
            if (std::isinf(ddS))
                break;
            State::move_node(v, s, true);
            mvs.push_back(v);
        }

        for (auto& v : mvs)
            State::move_node(v, r, false);

        State::relax_update(false);

        return dS;
    }

    void merge(const Group& r, const Group& s)
    {
        assert(r != s);

        std::vector<Node> mvs;
        get_group_vs(r, mvs);
        for (auto& v : mvs)
            move_node(v, s, false, _has_merge);

        if constexpr (_has_merge)
            State::merge_groups(r, s);
    }

    template <class RNG>
    double
    merge_sweep(GSet<Group>& rs, size_t B, size_t niter, RNG& rng_,
                bool strict = false,
                size_t B_max = numeric_limits<size_t>::infinity())
    {
        assert(rs.size() >= B);

        std::vector<Group> rlist;
        _best_merge.clear();
        for (const auto& r : rs)
        {
            _best_merge[r] = std::make_pair(r, numeric_limits<double>::infinity());
            rlist.push_back(r);
        }

        if (_has_merge)
            std::shuffle(rlist.begin(), rlist.end(), rng_);


        parallel_rng<rng_t> prng(rng_);
        #pragma omp parallel if (_has_merge)
        {
            int tnum = get_thread_num();
            #pragma omp for schedule(runtime)
            for (size_t j = 0; j < rlist.size(); ++j)
            {
                auto& rng = prng.get(rng_, tnum);

                gt_hash_set<Group> past_merges;
                auto& r = rlist[j];
                auto find_candidates =
                    [&](bool allow_random)
                    {
                        for (size_t i = 0; i < niter; ++i)
                        {
                            auto v = uniform_sample(_groups[r], rng);
                            auto s = State::sample_group(v, allow_random, false, false, rng);
                            if (s != r &&
                                rs.find(s) != rs.end() &&
                                past_merges.find(s) == past_merges.end())
                            {
                                double dS;
                                if constexpr (_has_merge)
                                    dS = State::virtual_merge(r, s, tnum);
                                else
                                    dS = virtual_merge_dS(r, s);
                                if (!std::isinf(dS) && dS < _best_merge[r].second)
                                    _best_merge[r] = {s, dS};
                                past_merges.insert(s);
                            }
                        }
                    };

                // Prefer smart constrained moves. If no candidates were found, the
                // group is likely to be "stuck" (e.g. isolated or constrained by
                // clabel); attempt random movements instead.

                find_candidates(false);

                if (_best_merge[r].first == r)
                    find_candidates(true);
            }
        }

        std::vector<std::pair<Group, Group>> pairs;
        std::vector<double> dS;
        for (const auto& r : rs)
        {
            auto& m = _best_merge[r];
            if (m.first == r)
                continue;
            pairs.emplace_back(r, m.first);
            dS.emplace_back(m.second);
        }

        std::vector<size_t> pis(pairs.size());
        std::iota(pis.begin(), pis.end(), 0);
        std::shuffle(pis.begin(), pis.end(), rng_);
        std::priority_queue queue(pis.begin(), pis.end(),
                                  [&](size_t i, size_t j) { return dS[i] > dS[j]; });

        _root.clear();
        auto get_root = [&](Group r)
        {
            Group s = r;
            if (_root.find(r) == _root.end())
            {
                _root[r] = r;
            }
            else
            {
                while (_root[r] != r)
                    r = _root[r];
            }
            _root[s] = r;
            return r;
        };

        double S = 0;

        if (_parallel_merge)
        {
            std::vector<Node> vs;

            while (rs.size() > B && !queue.empty())
            {
                auto i = queue.top();
                queue.pop();
                auto& [s, t] = pairs[i];
                t = get_root(t);
                if (s == t)
                    continue;
                get_group_vs<false>(s, vs);
                _root[s] = t;
                rs.erase(s);
            }

            if (strict)
                push_b(vs);

            if constexpr (_has_parallel_move)
            {
                S = -State::entropy();

                for (auto v : vs)
                    move_node(v, get_root(State::get_group(v)), false, true);

                std::shuffle(vs.begin(), vs.end(), rng_);

                State::move_nodes_parallel
                    (vs,
                     vs | std::views::transform([&](auto& v) { return get_root(State::get_group(v)); }));

                S += State::entropy();
            }
            else
            {
                for (auto v : vs)
                {
                    auto s = get_root(State::get_group(v));
                    S += State::virtual_move(v, s);
                    move_node(v, s);
                }
            }

            if (strict)
            {
                if (S > 0 && rs.size() < B_max)
                {
                    pop_b();
                    S = 0;
                    for (auto i : pis)
                        rs.insert(pairs[i].first);
                }
                else
                {
                    pop_b(true);
                }
            }
        }
        else
        {
            while (rs.size() > B && !queue.empty())
            {
                auto i = queue.top();
                queue.pop();

                auto& [s, t] = pairs[i];

                t = get_root(t);

                if (s == t)
                    continue;

                double ndS = dS[i];

                if constexpr (_has_merge)
                    ndS = State::virtual_merge(s, t);
                else
                    ndS = virtual_merge_dS(s, t);

                if (!queue.empty() && ndS > dS[queue.top()])
                {
                    dS[i] = ndS;
                    queue.push(i);
                    continue;
                }

                if (ndS > 0 && strict && rs.size() < B_max)
                    break;

                _root[s] = t;
                merge(s, t);
                S += ndS;
                rs.erase(s);
                assert(get_wr(s) == 0);
            }
        }

        assert(rs.size() >= B);
        return S;
    }

    template <class RNG>
    double
    single_move_init(GSet<Group>& rs, std::vector<Node>& vs, double init_beta,
                     size_t init_min_iter, size_t B_min, size_t B_max,
                     bool init_heuristic, RNG& rng)
    {
        size_t Bprev;
        size_t i = 0;
        double S = 0;

        do
        {
            double beta = _beta;
            if (i < _init_min_iter_beta)
                beta = init_beta;

            Bprev = rs.size();
            double dS = 0;

            if constexpr (_has_relabel)
                dS += relabel_sweep(rs, beta, rng);

            if (_parallel && _has_parallel_virtual_move)
                dS += pseudo_mh_sweep<true>(vs, rs, beta, rng, B_min, B_max,
                                            init_heuristic);
            else
                dS += mh_sweep<true>(vs, rs, beta, rng, B_min, B_max,
                                     init_heuristic);

            if constexpr (_has_extra_sweep)
                dS += State::extra_sweep(rs, beta, _parallel, rng);

            S += dS;

            if (_verbose)
                println("init, beta: {:3.1f}, B: {}, ratio: {:.3f}, dS: {:10.1f}",
                        beta, rs.size(), rs.size()/double(Bprev), _dS + S);
            ++i;
        }
        while ((i < init_min_iter || rs.size()/double(Bprev) < _init_r) && rs.size() > B_min);

        return S;
    }

    template <class RNG>
    double spread_groups(std::vector<Node>& vs, GSet<Group>& rs, RNG& rng)
    {
        double S = 0;
        rs.clear();
        if constexpr (_has_parallel_move)
        {
            S -= State::entropy();
            std::vector<Group> vrs;
            for (auto& v : vs)
            {
                auto s = State::get_group(v);
                auto t = s;
                if (State::_has_b_max)
                {
                    t = State::get_b_max(v);
                }
                else if (get_wr(t) != 1)
                {
                    t = State::get_new_group(v, true, rng);
                    State::reserve_empty_group(t);
                }
                move_node(v, t, false, true);
                rs.insert(t);
                vrs.push_back(t);
            }
            State::move_nodes_parallel(vs, vrs);
            S += State::entropy();
        }
        else
        {
            for (auto& v : vs)
            {
                auto s = State::get_group(v);
                auto t = s;
                if (State::_has_b_max)
                {
                    t = State::get_b_max(v);
                }
                else if (get_wr(s) == 1)
                {
                    rs.insert(s);
                    continue;
                }
                else
                {
                    t = State::get_new_group(v, true, rng);
                }
                S += State::virtual_move(v, t);
                move_node(v, t, true);
                rs.insert(t);
            }
        }

        return S;
    }

    template <bool strict, class RNG>
    double merge_and_sweep(GSet<Group>& rs, std::vector<Node>& vs, size_t B_min,
                           size_t B_max, RNG& rng)
    {
        auto Bnext =
            std::min(std::max(std::min(rs.size() - 1,
                                       size_t(round(rs.size() * _r))),
                              B_min),
                     B_max);

        double S = 0;
        while (rs.size() > Bnext)
        {
            S += merge_sweep(rs, Bnext, _merge_sweeps, rng, strict, B_max);
            if (strict && rs.size() <= B_max)
                break;
        }

        for (size_t i = 0; i < _mh_sweeps; ++i)
        {
            double dS = 0;

            if constexpr (_has_relabel)
                dS += relabel_sweep(rs, _beta, rng);

            if (_parallel && _has_parallel_virtual_move)
                dS += pseudo_mh_sweep<true>(vs, rs, _beta, rng, Bnext, B_max);
            else
                dS += mh_sweep<true>(vs, rs, _beta, rng, Bnext, B_max);

            if constexpr (_has_extra_sweep)
                dS += State::extra_sweep(rs, _beta, _parallel, rng);

            S += dS;
            if (std::isinf(_beta) && abs(dS) < 1e-8)
                break;
        }

        return S;
    }

    template <class RNG>
    double stage_multilevel(GSet<Group>& rs, std::vector<Node>& vs, RNG& rng)
    {
        size_t N = vs.size();

        if (N == 1)
            return 0;

        if (_verbose)
            println("staging multilevel, N = {}", N);

        double S_init = State::entropy();

        size_t B_max = State::_global_moves ? std::min(N, _B_max) : std::min(N, State::_M);
        size_t B_min = State::_global_moves ? std::max(size_t(1), _B_min) : 1;

        B_min = std::min(std::max(B_min, State::get_Bmin(vs)), B_max);

        size_t B_mid;

        size_t B_max_init = B_max;
        size_t B_min_init = B_min;

        double S_best = std::numeric_limits<double>::infinity();

        map<size_t,
            std::conditional_t<!_has_relabel,
                               std::tuple<double, std::vector<Group>>,
                               std::tuple<double, std::vector<Group>,
                                          std::vector<std::tuple<Group,group_label_t>>>>> cache;

        auto has_cache =
            [&](size_t B)
            {
                return cache.find(B) != cache.end();
            };

        auto put_cache = [&](size_t B, double S)
        {
            assert(std::ranges::fold_left(rs, 0, [&](auto x, auto r) { return x + size_t(_groups[r].size() > size_t(0)); }) == B);
            assert(cache.find(B) == cache.end());
            auto& c = cache[B];
            std::get<0>(c) = S;
            std::get<1>(c).resize(vs.size());
            #pragma omp parallel for schedule(runtime)
            for (size_t i = 0; i < vs.size(); ++i)
                std::get<1>(c)[i] = State::get_group(vs[i]);
            if constexpr (_has_relabel)
            {
                for (auto r : rs)
                    std::get<2>(c).emplace_back(r, State::group_label(r));
            }
            if constexpr (_has_extra_sweep)
                State::put_cache(B);
            if (S < S_best)
                S_best = S;
        };

        auto get_cache = [&](size_t B, GSet<Group>& rs_)
        {
            assert(cache.find(B) != cache.end());

            rs_.clear();
            auto& c = cache[B];

            for (size_t i = 0; i < vs.size(); ++i)
            {
                auto& s = std::get<1>(c)[i];
                move_node(vs[i], s, false, _has_parallel_move);
                rs_.insert(s);
            }

            if constexpr (_has_parallel_move)
                State::move_nodes_parallel(vs, std::get<1>(c));

            if constexpr (_has_relabel)
            {
                for (auto& [r, u] : std::get<2>(c))
                    State::relabel_group(r, u);
            }

            if constexpr (_has_extra_sweep)
                State::get_cache(B);

            assert(rs_.size() == B);
            return std::get<0>(c);
        };

        auto get_S = [&](size_t B, bool keep_cache=true)
        {
            auto iter = cache.lower_bound(B);
            if (iter->first == B)
                return std::get<0>(iter->second);
            assert(iter != cache.end());

            double S = get_cache(iter->first, rs);

            if (_verbose)
            {
                println("bracket B = [ {}, {}, {} ]", B_min, B_mid, B_max);
                println("shrinking from B = {} to B = {}", iter->first, B);
            }

            // merge & sweep
            while (rs.size() > B)
            {
                size_t Bprev = rs.size();

                S += merge_and_sweep<false>(rs, vs, B, B_max, rng);

                if ((keep_cache && _cache_states) || rs.size() == B)
                    put_cache(rs.size(), S);

                if (_verbose)
                    println("    {} -> {}:  {:10.1f}", Bprev, rs.size(), S);
            }

            assert(rs.size() == B);
            return S;
        };

        if (State::_has_b_min)
        {
            if (B_min == _B_min)
            {
                push_b(vs);
                double S = S_init;
                if (rs.size() != B_min)
                {
                    if constexpr (_has_parallel_move)
                    {
                        for (auto& v : vs)
                        {
                            auto s = State::get_group(v);
                            auto t = State::get_b_min(v);
                            if (s == t)
                                continue;
                            move_node(v, t, false, true);
                        }
                        State::move_nodes_parallel
                            (vs,
                             vs | std::views::transform([&](auto&& v){ return State::get_b_min(v); }));
                        S = State::entropy();
                    }
                    else
                    {
                        for (auto& v : vs)
                        {
                            auto r = State::get_group(v);
                            Group t = State::get_b_min(v);
                            if (r == t)
                                continue;
                            S += State::virtual_move(v, t);
                            move_node(v, t, true);
                        }
                    }
                }
                assert(rs.size() == B_min);
                put_cache(B_min, S);
                pop_b();
            }
        }
        else if (B_min == 1)
        {
            push_b(vs);
            double S = S_init;
            if (rs.size() > 1)
            {
                auto u = uniform_sample(vs, rng);
                auto t = State::get_group(u);

                if constexpr (_has_parallel_move)
                {
                    for (auto& v : vs)
                    {
                        auto s = State::get_group(v);
                        if (s == t)
                            continue;
                        move_node(v, t, false, true);
                    }
                    State::move_nodes_parallel
                        (vs,
                         vs | std::views::transform([&](auto&&){ return t; }));
                    S = State::entropy();
                }
                else
                {
                    for (auto& v : vs)
                    {
                        auto r = State::get_group(v);
                        if (r == t)
                            continue;
                        S += State::virtual_move(v, t);
                        move_node(v, t, true);
                    }
                }
                rs.clear();
                rs.insert(t);
            }
            put_cache(B_min, S);
            pop_b();
        }

        if (!State::_has_b_max)
        {
            double S = S_init;
            push_b(vs);
            State::relax_update(true); // init heuristic has simplified search

            S += spread_groups(vs, rs, rng);

            put_cache(rs.size(), S);

            // single-node sweep initialization with B = N. This is faster than
            // using merges!
            if (std::isinf(_beta) && _init_r < 1.)
            {
                S += single_move_init(rs, vs, _init_beta, _init_min_iter,
                                      B_min, B_max, true, rng);
                B_max = B_max_init = std::min(rs.size(), B_max);
            }

            if (rs.size() >= B_min && !has_cache(rs.size()))
                put_cache(rs.size(), S);

            pop_b();
            State::relax_update(false);
            get_cache(rs.size(), rs);
        }
        else if (B_max != B_min)
        {
            double S = S_init;

            if constexpr (_has_parallel_move)
            {
                for (auto& v : vs)
                {
                    auto s = State::get_group(v);
                    auto t = State::get_b_max(v);
                    if (s == t)
                        continue;
                    move_node(v, t, false, true);
                }
                State::move_nodes_parallel
                    (vs,
                     vs | std::views::transform([&](auto&& v){ return State::get_b_max(v); }));
                S = State::entropy();
            }
            else
            {
                for (auto& v : vs)
                {
                    auto r = State::get_group(v);
                    Group t = State::get_b_max(v);
                    if (r == t)
                        continue;
                    S += State::virtual_move(v, t);
                    move_node(v, t, true);
                }
            }
            put_cache(B_max, S);
        }

        BisectionSearch<size_t> fb;
        if (State::_random_bisection)
            fb.search(B_min, B_mid, B_max, get_S, 0, 0, rng);
        else
            fb.search(B_min, B_mid, B_max, get_S);

        // add midpoint
        if (!std::isinf(_beta))
        {
            size_t Br = (State::_random_bisection ?
                         fb.get_mid(B_min_init, B_max_init, rng) :
                         fb.get_mid(B_min_init, B_max_init));
            get_S(Br, false);
        }

        // remove out of bounds
        auto iter = cache.upper_bound(B_max);
        if (iter != cache.end())
            cache.erase(iter, cache.end());
        iter = cache.lower_bound(B_min);
        if (iter != cache.end())
            cache.erase(cache.begin(), iter);

        // Sample partition
        iter = max_element(cache.begin(), cache.end(),
                           [&](auto& x, auto& y)
                           {
                               return std::get<0>(x.second) > std::get<0>(y.second);
                           });

        std::vector<size_t> Bs;
        std::vector<double> probs;
        for (auto& [B, S] : cache)
        {
            Bs.push_back(B);
            if (!std::isinf(_beta))
            {
                probs.push_back(exp(_beta * (-std::get<0>(S) +
                                              std::get<0>(iter->second))));
            }
            else
            {
                if (std::get<0>(S) == std::get<0>(iter->second))
                    probs.push_back(1);
                else
                    probs.push_back(0);
            }
        }

        Sampler<size_t> B_sampler(Bs, probs);
        size_t B = B_sampler.sample(rng);
        double S = get_cache(B, rs);

        assert(rs.size() == B);

        return S - S_init;
    }

    template <class RNG>
    void sample_rs(GSet<Group>& rs, RNG& rng)
    {
        rs.clear();
        if (State::_global_moves)
        {
            for (auto r : _rlist)
                rs.insert(r);
        }
        else
        {
            std::uniform_int_distribution<size_t>
                sample(1, std::min(State::_M, _rlist.size()));

            auto M = sample(rng);

            auto r = uniform_sample(_rlist, rng);
            _rlist.erase(r);
            rs.insert(r);

            while (rs.size() < M)
            {
                auto t = r;
                size_t count = 0;
                while (_rlist.find(t) == _rlist.end())
                {
                    auto v = uniform_sample(_groups[r], rng);
                    t = State::sample_group(v, true, false, false, rng);
                    if (++count > _sample_group_thres)
                        r = uniform_sample(_rlist, rng);
                }
                r = t;
                _rlist.erase(r);
                rs.insert(r);
            }

            for (const auto& s : rs)
                _rlist.insert(s);
        }
    }

    template <class RNG>
    std::tuple<size_t, size_t>
    move_proposal(const Node&, RNG& rng)
    {
        _dS = _a = 0;
        _vs.clear();
        _nmoves = 0;

        sample_rs(_rs, rng);
        size_t nr = _rs.size();

        _vs.clear();
        for (const auto& r : _rs)
            get_group_vs<false>(r, _vs);

        std::shuffle(_vs.begin(), _vs.end(), rng);

        size_t nnr;
        if (_refine)
        {
            for (auto r : _rs)
                _rlist.erase(r);
            int64_t Ba, Bb;
            Bb = _rs.size();
            _dS += merge_and_sweep<true>(_rs, _vs, _B_min, _B_max, rng);
            Ba = _rs.size();
            if (_verbose)
                println("resizing: {} -> {}, dS: {:10.1f}", Bb, Ba, _dS);
            for (auto r : _rs)
                _rlist.insert(r);
            nnr = _rs.size();
        }
        else if (_bisection)
        {
            push_b(_vs);
            _dS = stage_multilevel(_rs, _vs, rng);
            nnr = _rs.size();
            push_b(_vs);
            swap_b();
            pop_b();
        }
        else
        {
            for (auto r : _rs)
                _rlist.erase(r);

            push_b(_vs);

            _dS = spread_groups(_vs, _rs, rng);

            _dS += single_move_init(_rs, _vs, _init_beta, _init_min_iter,
                                   _B_min, _B_max, true, rng);

            int64_t Ba, Bb;
            do
            {
                Bb = _rs.size();
                _dS += merge_and_sweep<true>(_rs, _vs, _B_min, _B_max, rng);
                Ba = _rs.size();
                if (_verbose)
                    println("resizing: {} -> {}, dS: {:10.1f}", Bb, Ba, _dS);
            } while (std::abs(Ba - Bb)/double(max(Ba, Bb)) > _dB_epsilon);

            pop_b(_dS <= 0 || _force_accept);

            if (_dS > 0 && !_force_accept)
                _dS = 0;

            for (auto r : _rs)
                _rlist.insert(r);

            nnr = _rs.size();
        }

        if (_verbose)
            println("multilevel proposal: {} -> {} ({}), dS: {:10.1f}", nr, nnr,
                    _vs.size(), _dS);

        if (_force_accept)
            _dS = -std::numeric_limits<double>::infinity();

        return {0, _nmoves};
    }

    std::tuple<double, double>
    virtual_move_dS(const Node&, size_t)
    {
        return {_dS, _a};
    }

    void perform_move(const Node&, size_t)
    {
        if (!_refine && _bisection)
            pop_b();
    }

    void step(const Node&, size_t)
    {
        clear_b();
    }

    constexpr bool is_deterministic()
    {
        return true;
    }

    constexpr bool is_sequential()
    {
        return false;
    }

    std::array<Node, 1> _vlist = {Node()};

    auto& get_vlist()
    {
        return _vlist;
    }

    size_t get_N()
    {
        return 1; //_N;
    }

    double get_beta()
    {
        return _beta;
    }

    size_t get_niter()
    {
        return State::_niter;
    }
};

} // graph_tool namespace

#endif // MULTILEVEL_HH
