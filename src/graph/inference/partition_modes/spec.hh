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

#ifndef SPEC_HH
#define SPEC_HH

#define STATE_NAME mode_cluster

#define SPEC_params                                                            \
    ((bs, std::vector<bv_t>, 0))                                               \
    ((relabel_init, bool, 0))

#define EARGS_params

#include "../base/spec_base.hh"

#include "../support/cache.hh"

#include "partition_mode.hh"
#include "../blockmodel/partition.hh"

#include <boost/python.hpp>

#include "../../numpy_bind.hh"

namespace graph_tool
{

typedef std::vector<int64_t> b_t;
typedef PartitionModeState::bv_t bv_t;

namespace STATE_NAME
{

struct StateSpec
    : public SpecBase
{
    typedef int64_t group_t;
    typedef b_t& vmap_t;
    typedef b_t bmap_t;

    using SpecBase::entropy_args_t;

    constexpr static bool has_parallel_virtual_move = false;

    template <class State>
    struct Imp
        : public SpecBase::Imp<State>
    {
        GET_PARAMS_SPEC_TYPEDEF(SPEC_params)
        GET_PARAMS_SPEC_DEFINE(SPEC_params)

        Imp(State& state)
            : SpecBase::Imp<State>(state),
              GET_PARAMS_SPEC_INIT(SPEC_params)
              _state(state),
              _M(_bs.size()),
              _pos(_M),
              _modes(_M)
        {
            GILRelease gil_release;

            _state._N = state._b.size();
            for (size_t i = 0; i < _M; ++i)
            {
                auto r = _state._b[i];
                auto& x = _bs[i];
                _pos[i] = _modes[r].add_partition(x, _relabel_init);
            }

            for (auto r : _state._b)
                _partition_stats.add_item(r);
        }

        void post_init()
        {
            _modes.resize(_state._groups.size());
            _partition_stats.add_group(_state._groups.size() -
                                       _partition_stats.get_B());
        }

        State& _state;

        size_t _M;
        std::vector<size_t> _pos;
        std::vector<PartitionModeState> _modes;

        PartitionStats<group_t> _partition_stats;

        std::vector<std::vector<std::tuple<size_t,std::vector<std::vector<int64_t>>>>> _lstack;

        std::shared_ptr<b_t> _bp;

        PartitionModeState& get_mode(size_t r)
        {
            return _modes[r];
        }

        void move_vertex(size_t v, group_t r, group_t nr)
        {
            _modes[r].remove_partition(_pos[v]);

            auto& x = _bs[v];
            _pos[v] = _modes[nr].add_partition(x, true);

            _partition_stats.remove_item(r);
            _partition_stats.add_item(nr);
        }

        double virtual_move(size_t v, group_t r, group_t nr, const entropy_args_t&)
        {
            double dS = 0;

            auto& x = _bs[v];

            dS += _modes[r].virtual_remove_partition(x);
            dS += _modes[nr].virtual_add_partition(x);
            dS += _partition_stats.get_delta_dl(r, nr);

            return dS;
        }

        double virtual_add_partition(bv_t& x, size_t r, bool relabel = true)
        {
            double dS = _modes[r].virtual_add_partition(x, relabel);
            dS += _partition_stats.get_delta_dl(_partition_stats._null_group, r);
            return dS;
        }

        void add_partition(bv_t& x, size_t r, bool relabel = true)
        {
            auto pos = _modes[r].add_partition(x, relabel);
            _pos.push_back(pos);
            _state._b.push_back(r);
            _state._clabel.push_back(_state._bclabel[r]);
            _bs.push_back(x);
            _partition_stats.add_item(r);
            _state._groups[r].insert(_pos.size() - 1);
            _modes.emplace_back();
            _M++;
        }

        template <class RNG>
        double replace_partitions(RNG& rng)
        {
            std::vector<size_t> pos(_M);
            std::iota(pos.begin(), pos.end(), 0);
            std::shuffle(pos.begin(), pos.end(), rng);

            double dS = 0;
            for (auto j : pos)
            {
                auto& mode = _modes[_state._b[j]];
                auto& b = _bs[j];
                double ddS = mode.virtual_remove_partition(b);
                mode.remove_partition(_pos[j]);
                ddS += mode.virtual_add_partition(b);
                _pos[j] = mode.add_partition(b, ddS < 0);
                if (ddS < 0)
                    dS += ddS;
            }
            return dS;
        }

        double entropy(const entropy_args_t&)
        {
            double S = 0;
            for (auto r : _state._occupied_groups)
                S += _modes[r].entropy();
            S += _partition_stats.get_dl();
            return S;
        }

        void add_group(size_t n)
        {
            _modes.resize(_modes.size() + n);
            _partition_stats.add_group(n);
        }

        double posterior_entropy(bool MLE)
        {
            double S = 0;
            for (auto& [r, vs] : _state._groups)
            {
                if (vs.size() == 0)
                    continue;
                S += (_modes[r].posterior_entropy(MLE) * vs.size()) / _M;
                S += -xlogx(vs.size() / double(_M));
            }
            return S;
        }

        double posterior_lprob(size_t r, bv_t& b, bool MLE)
        {
            double L = log(_state._groups[r].size()) - log(_M);
            L += _modes[r].posterior_lprob(b, MLE);
            return L;
        }

        void relabel_mode(PartitionModeState& x, PartitionModeState& base)
        {
            size_t n = std::max(x._nr.size(), base._nr.size());
            x._nr.resize(n);
            x._count.resize(n);
            base._nr.resize(n);
            base._count.resize(n);

            adj_list<> g;
            vprop_map_t<int64_t> label(get(vertex_index_t(), g));
            vprop_map_t<bool> partition(get(vertex_index_t(), g));
            eprop_map_t<double> mrs(get(edge_index_t(), g));

            get_contingency_graph<true>(g, partition, label, mrs, x._nr, base._nr);

            typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
            vprop_map_t<vertex_t> match(get(vertex_index_t(), g));

            auto& u = get_undirected_adaptor(g);
            //maximum_weighted_matching(u, mrs, match);
            maximum_bipartite_weighted_matching(u, partition, mrs, match);

            idx_map<int64_t, size_t> x_vertices;
            for (auto v : vertices_range(u))
            {
                if (v < x._B)
                    x_vertices[label[v]] = v;
                else
                    break;
            }

            std::vector<int64_t> unmatched;
            int64_t max_s = 0;
            for (size_t r = 0; r < x._count.size(); ++r)
            {
                if (x._count[r] == 0)
                    continue;
                auto v = match[x_vertices[r]];
                if (v == graph_traits<adj_list<>>::null_vertex())
                    unmatched.push_back(r);
                else
                    max_s = std::max(max_s, label[v]);
            }

            std::sort(unmatched.begin(), unmatched.end(),
                      [&](auto r, auto s) { return x._count[r] > x._count[s]; });

            idx_map<int64_t, int64_t> umatch;
            for (auto r : unmatched)
                umatch[r] = ++max_s;

            for (auto& jb : x._bs)
            {
                auto bv = x.get_nested_partition(jb.first);
                auto& b = bv[0].get();
                PartitionModeState::b_t b_orig = b;
                for (auto& r : b)
                {
                    if (r == -1)
                        continue;
                    auto v = match[x_vertices[r]];
                    if (v != graph_traits<adj_list<>>::null_vertex())
                        r = label[v];
                    else
                        r = umatch[r];
                }

                if (x._coupled_state != nullptr)
                {
                    auto& c = x._coupled_state->get_partition(x._coupled_pos[jb.first]);
                    relabel_nested(b, b_orig, c);
                }
            }
            x.rebuild_nr();
            if (x._coupled_state != nullptr)
                relabel_mode(*x._coupled_state, *base._coupled_state);
        }

        void relabel_modes(double epsilon, size_t maxiter)
        {
            std::vector<size_t> modes(_M);
            std::iota(modes.begin(), modes.end(), 0);
            std::sort(modes.begin(), modes.end(),
                      [&](auto r, auto s) { return _state._groups[r].size() > _state._groups[s].size(); });
            PartitionModeState base;

            std::vector<idx_map<size_t, size_t>> pos(_M);
            for (size_t r : modes)
            {
                if ( _state._groups[r].size() == 0)
                    continue;
                auto& mode = _modes[r];
                if (!base._bs.empty())
                    relabel_mode(mode, base);
                else
                    mode.relabel();
                for (auto& jb : mode._bs)
                {
                    auto b = mode.get_nested_partition(jb.first);
                    pos[r][jb.first] = base.add_partition(b, false);
                }
            }

            double dS = epsilon + 1;
            size_t iter = 0;
            while (abs(dS) > epsilon && (maxiter == 0 || iter < maxiter))
            {
                dS = 0;
                for (size_t r : modes)
                {
                    if (_state._groups[r].size() == 0)
                        continue;
                    auto& mode = _modes[r];

                    for (auto& jb : mode._bs)
                    {
                        auto b = mode.get_nested_partition(jb.first);
                        dS += base.virtual_remove_partition(b);
                        base.remove_partition(pos[r][jb.first]);
                    }

                    if (!base._bs.empty())
                        relabel_mode(mode, base);

                    for (auto& jb : mode._bs)
                    {
                        auto b = mode.get_nested_partition(jb.first);
                        dS += base.virtual_add_partition(b, false);
                        pos[r][jb.first] = base.add_partition(b, false);
                    }
                }
                iter++;
            }
        }

        template <class RNG>
        std::pair<size_t, PartitionModeState::b_t>
        sample_partition(bool MLE, RNG& rng)
        {
            auto r = uniform_sample(_state._b, rng);
            return {r, _modes[r].sample_partition(MLE, rng)};
        }

        template <class RNG>
        std::pair<size_t, std::vector<PartitionModeState::b_t>>
        sample_nested_partition(bool MLE, bool fix_empty, RNG& rng)
        {
            auto r = uniform_sample(_state._b, rng);
            return {r, _modes[r].sample_nested_partition(MLE, fix_empty, rng)};
        }

        template <class VS>
        void push_state(VS& vs)
        {
            _lstack.emplace_back();
            auto& back = _lstack.back();
            for (auto v : vs)
            {
                auto& bv = _bs[v];
                back.emplace_back();
                get<0>(back.back()) = v;
                auto& x = get<1>(back.back());
                for (auto& b : bv)
                    x.push_back(b.get());
            }
        }

        void pop_state(bool discard)
        {
            if (!discard)
            {
                auto& back = _lstack.back();
                for (auto& vx : back)
                {
                    size_t v = get<0>(vx);
                    auto& x = get<1>(vx);
                    auto& bv = _bs[v];
                    auto r = _state._b[v];
                    _modes[r].remove_partition(_pos[v]);
                    for (size_t l = 0; l < bv.size(); ++l)
                        bv[l].get() = x[l];
                    _pos[v] = _modes[r].add_partition(bv, false);
                }
            }
            _lstack.pop_back();
        }

        void swap_state()
        {
            std::swap(_lstack[_lstack.size() - 2], _lstack[_lstack.size() - 1]);
        }

        void clear_state()
        {
            _lstack.clear();
        }
    };

    template <class State>
    static void python_export_state(auto& c)
    {
        using namespace boost::python;
        c.def("virtual_add_partition",
              +[](State& state, object ob, size_t r, bool relabel)
               {
                   auto bv = extract_param<bv_t>()(ob);
                   return state._spec.virtual_add_partition(bv, r, relabel);
               })
            .def("add_partition",
                 +[](State& state, object ob, size_t r, bool relabel)
                  {
                      auto bv = extract_param<bv_t>()(ob);
                      state._spec.add_partition(bv, r, relabel);
                  })
            .def("posterior_entropy",
                 +[](State& state, bool MLE)
                  {
                      return state._spec.posterior_entropy(MLE);
                  })
            .def("posterior_lprob",
                 +[](State& state, size_t r, object obv, bool MLE)
                  {
                      PartitionModeState::bv_t bv;
                      for (int i = 0; i < python::len(obv); ++i)
                      {
                          PartitionModeState::b_t& b =
                              python::extract<PartitionModeState::b_t&>(obv[i]);
                          bv.emplace_back(b);
                      }
                      return state._spec.posterior_lprob(r, bv, MLE);
                  })
            .def("relabel_modes",
                 +[](State& state, double epsilon, size_t maxiter)
                  {
                      return state._spec.relabel_modes(epsilon, maxiter);
                  })
            .def("replace_partitions",
                 +[](State& state, rng_t& rng)
                  {
                      return state._spec.replace_partitions(rng);
                  })
            .def("get_mode",
                 +[](State& state, size_t r) -> PartitionModeState&
                  {
                      return state._spec.get_mode(r);
                  },
                 return_internal_reference<>())
            .def("sample_partition",
                 +[](State& state, bool MLE, rng_t& rng)
                  {
                      auto rb = state._spec.sample_partition(MLE, rng);
                      return python::make_tuple(rb.first,
                                                wrap_vector_owned(rb.second));
                  })
            .def("sample_nested_partition",
                 +[](State& state, bool MLE, bool fix_empty, rng_t& rng)
                  {
                      python::list obv;
                      auto rbv = state._spec.sample_nested_partition(MLE,
                                                                    fix_empty,
                                                                    rng);
                      for (auto& b : rbv.second)
                          obv.append(wrap_vector_owned(b));
                      return python::make_tuple(rbv.first, obv);
                  });
    }
};

} // STATE_NAME

template <>
struct extract_param<std::vector<bv_t>>
{
    auto operator()(boost::python::object obs)
    {
        std::vector<bv_t> bs;
        for (int64_t i = 0; i < python::len(obs); ++i)
        {
            PartitionModeState::bv_t bv;
            for (int64_t l = 0; l < python::len(obs[i]); ++l)
            {
                PartitionModeState::b_t& b =
                    python::extract<PartitionModeState::b_t&>(obs[i][l]);
                bv.emplace_back(b);
            }
            bs.push_back(bv);
        }
        return bs;
    }
};

template <>
struct extract_param<b_t>
{
    auto operator()(boost::python::object o)
    {
        auto a = get_array<int64_t, 1>(o);
        b_t b(a.begin(), a.end());
        return b;
    }
};

} // graph_tool namespace

#endif //SPEC_HH
