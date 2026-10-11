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

#ifndef GRAPH_OVERLAP_BLOCK_MODEL_BASE_HH
#define GRAPH_OVERLAP_BLOCK_MODEL_BASE_HH

#include "config.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <print>

#include <boost/unordered/concurrent_flat_map.hpp>

#include "../spec.hh"

#include "parallel_util.hh"

#include "python_interface.hh"

#ifndef COUPLED_BLOCK_MODEL
#define COUPLED_BLOCK_MODEL block_model::BlockState
#endif

typedef decltype(hana::tuple_t<std::true_type, std::false_type>) bool_tr;

#define OVERLAP_SPEC_params                                                    \
    ((gen_overlap, bool_tr, 1))

#include "half_edge_adaptor.hh"
#include "types.hh"
#include "entropy.hh"
#include "partition.hh"
#include "partition_simple.hh"

namespace graph_tool
{
using namespace std;
using namespace hana::literals;

template <class Base = BlockStateSpec>
struct OverlapBlockStateSpec
    : public Base
{
    using count_t = Base::count_t;
    using group_t = Base::group_t;
    typedef ovweight_t vcount_t;

    static inline auto gviews =
        hana::transform(all_graph_views,
                        [](auto&& x)
                        {
                            using G = std::decay_t<typename decltype(+x)::type>;
                            return hana::type_c<half_edge_adaptor<G>&>;
                        });

    typedef decltype(gviews) g_tr;

    struct get_bg_t
        : public last_type_func_t
    {
        template <class T>
        constexpr static auto get(T)
        {
            return Base::get_bg_t::get(hana::type_c<typename std::decay_t<typename T::type>::base_t>);
        }
    };

    constexpr inline static auto get_bg_type = hana::tuple_t<get_bg_t>;
    typedef decltype(get_bg_type) bg_tr;

    typedef VPropAdaptor<eprop_map_t<std::vector<int64_t>>> bmap_t;

    typedef VPropAdaptor<eprop_map_t<std::vector<std::vector<double>>>> bfield_t;

    template <class Val>
    using vprop_t = VPropAdaptor<eprop_map_t<std::vector<Val>>>;

    typedef decltype(hana::tuple_t<vprop_t<ovweight_t>>) vweight_tr;
    typedef decltype(hana::tuple_t<vprop_map_t<ovweight_t>>) wr_tr;

    typedef idx_map<group_t, gt_hash_set<half_edge>> groups_t;

    struct entropy_args_t
        : public Base::entropy_args_t
    { };

    template <class> class dummy {};

    template <class State, template <class> class TVal = dummy>
    struct coupled_state_t
    {
        static inline auto anames = State::__param_names;
        static inline auto fargs_t =
            hana::filter(hana::zip(State::args_t, anames),
                         [](auto x) constexpr
                         {
                             constexpr auto gen_overlap = BOOST_HANA_STRING("gen_overlap");
                             return std::integral_constant<bool, (x[1_c] != gen_overlap)>{};
                         });
        static inline auto args_t =
            hana::transform(fargs_t,
                            [](auto y) constexpr
                            {
                                auto x = y[0_c];
                                using T = std::decay_t<typename decltype(x)::type>;
                                if constexpr (std::is_same_v<T, bmap_t>)
                                {
                                    return hana::type_c<typename Base::bmap_t::unchecked_t>;
                                }
                                else if constexpr (std::is_same_v<T, bfield_t>)
                                {
                                    return hana::type_c<typename Base::bfield_t::unchecked_t>;
                                }
                                else if constexpr (is_specialization_of_v<T, VPropAdaptor>)
                                {
                                    if constexpr (std::is_same_v<typename T::value_type, ovweight_t>)
                                        return hana::type_c<typename vprop_map_t<count_t>::unchecked_t>;
                                    else if constexpr (std::is_same_v<typename T::value_type, TVal<ovweight_t>>)
                                        return hana::type_c<typename vprop_map_t<TVal<count_t>>::unchecked_t>;
                                    else
                                        return hana::type_c<typename vprop_map_t<typename T::value_type>::unchecked_t>;
                                }
                                else if constexpr (std::is_same_v<T, vprop_map_t<ovweight_t>::unchecked_t>)
                                {
                                    return hana::type_c<typename vprop_map_t<count_t>::unchecked_t>;
                                }
                                else if constexpr (std::is_same_v<T, vprop_map_t<ovweight_t>>)
                                {
                                    return hana::type_c<typename vprop_map_t<count_t>::unchecked_t>;
                                }
                                else if constexpr (std::is_same_v<T, typename vprop_map_t<TVal<ovweight_t>>::unchecked_t>)
                                {
                                    return hana::type_c<typename vprop_map_t<TVal<count_t>>::unchecked_t>;
                                }
                                else if constexpr (std::is_same_v<T, vprop_map_t<TVal<ovweight_t>>>)
                                {
                                    return hana::type_c<typename vprop_map_t<TVal<count_t>>::unchecked_t>;
                                }
                                else if constexpr (is_specialization_of_v<T, half_edge_adaptor>)
                                {
                                    return get_bg_t::get(x);
                                }
                                else
                                {
                                    return x;
                                }
                            });

        constexpr static inline auto state_t =
            hana::unpack(args_t,
                         [](auto... as)
                         {
                             return hana::type_c<COUPLED_BLOCK_MODEL<typename decltype(as)::type...>>;
                         });

        typedef decltype(state_t)::type* type;
    };

    template <class State>
    struct partition_stats_t
    {
        typedef std::conditional_t<State::gen_overlap_t::value,
                                   OverlapPartitionStats<is_directed_v<typename State::g_t>>,
                                   DummyPartitionStats> type;
    };

    template <class State>
    struct deg_t
    {
        typedef odeg_t type;
    };

    template <class State>
    struct deg_stats_t
    {
        typedef std::conditional_t<State::gen_overlap_t::value,
                                   OverlapDegStats<is_directed_v<typename State::g_t>>,
                                   SimpleOverlapDegStats<is_directed_v<typename State::g_t>>> type;
    };

    static void python_export()
    {
        using namespace boost::python;

        __MOD__::def_obj<class_<entropy_args_t, bases<typename Base::entropy_args_t>>>
            (get_type_name<entropy_args_t>().c_str());
    }

    template <class State>
    static void python_export_state(auto& c)
    {
        Base::template python_export_state<State>(c);

        c.def("get_wr",
              +[](State& state, std::any wra)
               {
                   auto wr = std::any_cast<vprop_map_t<group_t>>(wra);
                   for (auto r : vertices_range(state._bg))
                       wr[r] = state._wr[r];
               });

        c.def("get_overlap_groups",
              +[](State& state, std::any abv, std::any abc_in, std::any abc_out, std::any abc_total)
              {
                   auto bv = std::any_cast<vprop_map_t<std::vector<group_t>>>(abv);
                   auto bc_in = std::any_cast<vprop_map_t<std::vector<group_t>>>(abc_in);
                   auto bc_out = std::any_cast<vprop_map_t<std::vector<group_t>>>(abc_out);
                   auto bc_total = std::any_cast<vprop_map_t<std::vector<group_t>>>(abc_total);

                   for (auto v : vertices_range(state._g))
                   {
                       auto r = state._b[v];
                       auto u = v.get_endpoint();
                       int64_t kin = is_directed_v<typename State::g_t> ? in_degree(v, state._g) : 0;
                       int64_t kout = out_degree(v, state._g);

                       auto iter = std::find(bv[u].begin(), bv[u].end(), r);
                       if (iter == bv[u].end())
                       {
                           bv[u].push_back(r);
                           bc_in[u].push_back(kin);
                           bc_out[u].push_back(kout);
                           bc_total[u].push_back(1);
                       }
                       else
                       {
                           auto pos = iter - bv[u].begin();
                           bc_in[u][pos] += kin;
                           bc_out[u][pos] += kout;
                           bc_total[u][pos] += 1;
                       }
                   }
              });

        c.def("get_maj_groups",
              +[](State& state, std::any obv, std::any obc_total, std::any ob)
               {
                   typedef vprop_map_t<int64_t> vmap_t;
                   typedef vprop_map_t<vector<int64_t>> vvmap_t;

                   vmap_t b = std::any_cast<vmap_t>(ob);
                   vvmap_t bv = std::any_cast<vvmap_t>(obv);
                   vvmap_t bc_total = std::any_cast<vvmap_t>(obc_total);
                   for (auto v : vertices_range(state._g.base()))
                   {
                       if (bv[v].empty())
                       {
                           b[v] = numeric_limits<int64_t>::max();
                           continue;
                       }
                       auto& c_ = bc_total[v];
                       auto pos = std::max_element(c_.begin(), c_.end());
                       auto r = *(bv[v].begin() + (pos - c_.begin()));
                       b[v] = r;
                   }
               });
    }

    static void init_vweight(auto& vw, auto v, odeg_t kin, odeg_t kout)
    {
        vw[v] = {kin, kout};
    }

    static void visit_group_vs(auto& wr, auto&& f)
    {
        for (auto&& [v, ds] : wr)
            f(v);
    }

    template <class State>
    struct Imp
        : public Base::template Imp<State>
    {
        using ibase_t = Base::template Imp<State>;
        using deg_t = State::deg_t;

        GET_PARAMS_SPEC_TYPEDEF(OVERLAP_SPEC_params)
        GET_PARAMS_SPEC_DEFINE(OVERLAP_SPEC_params)

        Imp(State& state)
            : ibase_t(state),
              GET_PARAMS_SPEC_INIT(OVERLAP_SPEC_params)
              _state(state)
        {
            auto& g = _state._g;
            auto& b = _state._b;
            auto& eweight = _state._eweight;
            auto& vweight = _state._vweight;

            vweight.resize(num_vertices(g));
            _in_deg.resize(num_vertices(g.base()));
            _out_deg.resize(num_vertices(g.base()));

            parallel_vertex_loop
                (g,
                 [&](auto v)
                 {
                     State::spec_t::init_vweight(vweight[v], v.get_endpoint(),
                                                 in_degree(v, g, eweight),
                                                 out_degree(v, g, eweight));

                     auto u = v.get_endpoint();

                     auto k = out_degree(v, g, eweight);
                     _out_deg[u].emplace_or_visit
                         (b[v], k,
                          [&](auto& rc)
                          {
                              rc.second += k;
                          });

                     if constexpr (is_directed_v<typename State::g_t>)
                     {
                         k = in_degree(v, g, eweight);
                         _in_deg[u].emplace_or_visit
                             (b[v], k,
                              [&](auto& rc)
                              {
                                  rc.second += k;
                              });
                     }
                 });

            for (auto e : edges_range(g))
            {
                auto u = source(e, g);
                auto v = target(e, g);
                std::tie(u, v) = get_uv(u, v);
                _grs[get_uv(u.get_endpoint(),
                            v.get_endpoint())].first[{b[u], b[v]}] += eweight[e];
            }

            boost::unordered::erase_if
                (_grs,
                 [&](auto& uvx)
                 {
                     auto& [uv, x] = uvx;
                     auto m = x.first.begin()->second;
                     if (x.first.size() == 1 && m == 1)
                     {
                         return true;
                     }
                     else
                     {
                         x.second = std::make_shared<std::mutex>();
                         return false;
                     }
                 });

            if constexpr (!State::gen_overlap_t::value)
                _state._deg_stats.set_N(hard_num_vertices(g.base()));
        }

        auto get_uv(auto u, auto v)
        {
            if constexpr (!is_directed_v<typename State::g_t>)
            {
                if (u > v)
                    std::swap(u, v);
            }
            return std::make_tuple(u, v);
        }

        double entropy(const entropy_args_t& ea) const
        {
            double S = 0;

            if (ea.adjacency)
            {
                if (ea.dense)
                {
                    throw GraphException("Dense entropy for overlapping model not implemented!");
                }
                else
                {
                    if (_state._deg_corr && ea.deg_entropy)
                    {
                        #pragma omp parallel reduction(+:S)
                        parallel_vertex_loop_no_spawn
                            (_state._g.base(),
                             [&](auto v)
                             {
                                 _out_deg[v].visit_all
                                     ([&](auto& k_c)
                                      {
                                          S += deg_entropy(k_c.second);
                                      });

                                 _in_deg[v].visit_all
                                     ([&](auto& k_c)
                                      {
                                          S += deg_entropy(k_c.second);
                                      });
                             });
                    }

                    if (ea.multigraph && ea.parallel_entropy)
                    {
                        for (auto& [uv, x] : _grs)
                        {
                            auto& [u, v] = uv;
                            for (auto& rsc : x.first)
                                S += parallel_term(u, v, rsc.second, _state._g);
                        }
                    }
                }
            }
            return S;
        }

        auto get_deg(auto& degs, auto r)
        {
            typename State::deg_t k = typename State::deg_t();
            degs.visit(r,
                       [&](auto& rk)
                       {
                           k = get<1>(rk);
                       });
            return k;
        }

        void update_deg(auto& degs, auto r, auto&& delta)
        {
            if (delta > 0)
            {
                degs.emplace_or_visit(r, delta,
                                      [&](auto& rd)
                                      {
                                          rd.second += delta;
                                      });
            }
            else
            {
                degs.erase_if(r,
                              [&](auto& rd)
                              {
                                  rd.second += delta;
                                  return rd.second == 0;
                              });
            }
        }

        template <bool concurrent = false>
        void move_vertex_concurrent(auto v, auto r, auto nr, auto& nb, auto&& skip)
        {
            auto& g = _state._g;
            auto& eweight = _state._eweight;

            if (r == nr)
                return;

            auto kout = out_degree(v, g, eweight);
            auto w = v.get_endpoint();
            update_deg(_out_deg[w], r, -kout);
            update_deg(_out_deg[w], nr, kout);
            if constexpr (is_directed_v<typename State::g_t>)
            {
                auto kin = in_degree(v, g, eweight);
                update_deg(_in_deg[w], r, -kin);
                update_deg(_in_deg[w], nr, kin);
            }

            auto u = v;
            u.swap_endpoint();

            if (skip(u))
                return;

            half_edge u_(u), v_(v);

            if constexpr (is_directed_v<typename State::g_t>)
            {
                if (!u_.is_source())
                    std::swap(u_, v_);
            }
            else
            {
                std::tie(u_, v_) = get_uv(u_, v_);
            }

            auto iter = _grs.find(get_uv(u_.get_endpoint(),
                                         v_.get_endpoint()));

            if (iter != _grs.end())
            {
                auto ew = eweight[v.get_edge()];

                do_ulock<{.enable=concurrent}>
                    ([&]()
                     {
                         auto xiter = iter->second.first.find({_state._b[u_],
                                                               _state._b[v_]});
                         xiter->second -= ew;
                         if (xiter->second == 0)
                             iter->second.first.erase(xiter);

                         iter->second.first[{(u_ == v) ? nr : nb[u_],
                                             (v_ == v) ? nr : nb[v_]}] += ew;

                     }, *iter->second.second);
            }
        }

        void move_vertex(auto v, auto r, auto nr, auto&)
        {
            move_vertex_concurrent<false>(v, r, nr, _state._b, [](auto&){ return false; });
        }

        double virtual_move(auto v, auto r, auto nr, const entropy_args_t& ea,
                               auto&&)
        {
            auto& g = _state._g;
            auto& b = _state._b;
            auto& eweight = _state._eweight;

            double dS = 0;

            if (ea.adjacency)
            {
                if (_state._deg_corr && ea.deg_entropy)
                {
                    auto kout = out_degree(v, g, eweight);
                    auto u = v.get_endpoint();
                    auto out_deg_u_r = get_deg(_out_deg[u], r);
                    auto out_deg_u_nr = get_deg(_out_deg[u], nr);
                    dS += deg_entropy(out_deg_u_r - kout) - deg_entropy(out_deg_u_r);
                    dS += deg_entropy(out_deg_u_nr + kout) - deg_entropy(out_deg_u_nr);
                    assert(!std::isinf(dS) && !std::isnan(dS));
                    if constexpr (is_directed_v<typename State::g_t>)
                    {
                        auto kin = in_degree(v, g, eweight);
                        auto in_deg_u_r = get_deg(_in_deg[u], r);
                        auto in_deg_u_nr = get_deg(_in_deg[u], nr);
                        dS += deg_entropy(in_deg_u_r - kin) - deg_entropy(in_deg_u_r);
                        dS += deg_entropy(in_deg_u_nr + kin) - deg_entropy(in_deg_u_nr);
                        assert(!std::isinf(dS) && !std::isnan(dS));
                    }
                }

                if (ea.multigraph && ea.parallel_entropy)
                {
                    auto u = v;
                    u.swap_endpoint();
                    half_edge u_(u), v_(v);
                    if constexpr (is_directed_v<typename State::g_t>)
                    {
                        if (!u_.is_source())
                            std::swap(u_, v_);
                    }
                    else
                    {
                        std::tie(u_, v_) = get_uv(u_, v_);
                    }
                    auto iter = _grs.find({u_.get_endpoint(), v_.get_endpoint()});
                    if (iter != _grs.end())
                    {
                        auto ew = eweight[v.get_edge()];
                        auto xiter = iter->second.first.find({b[u_], b[v_]});
                        size_t m = xiter->second;
                        dS -= parallel_term(u_.get_endpoint(), v_.get_endpoint(), m, g);
                        dS += parallel_term(u_.get_endpoint(), v_.get_endpoint(), m - ew, g);

                        m = 0;
                        xiter = iter->second.first.find({(u_ == v) ? nr : b[u_],
                                                         (v_ == v) ? nr : b[v_]});
                        if (xiter != iter->second.first.end())
                            m += xiter->second;
                        dS -= parallel_term(u_.get_endpoint(), v_.get_endpoint(), m, g);
                        dS += parallel_term(u_.get_endpoint(), v_.get_endpoint(), m + ew, g);
                        assert(!std::isinf(dS) && !std::isnan(dS));
                    }
                }
            }
            assert(!std::isinf(dS) && !std::isnan(dS));
            return dS;
        }

        void merge_groups(group_t s, group_t t, auto&&)
        {
            gt_hash_map<half_edge, group_t> bo;

            auto nb = make_adhoc_prop<half_edge>
                ([&](const auto& v)
                 {
                     auto iter = bo.find(v);
                     if (iter != bo.end())
                         return iter->second;
                     return _state._b[v];
                 });

            for (auto v : _state._groups[s])
            {
                move_vertex_concurrent<false>(v, s, t, nb, [](auto&){ return false; });
                bo[v] = t;
            }
        }

        double virtual_merge(group_t s, group_t t, const entropy_args_t& ea, auto&)
        {
            auto get_delta_dl =
                [&](auto& degs)
                {
                    double S_b = 0;
                    double S_a = 0;

                    gt_hash_map<group_t, deg_t> mdegs;
                    degs.visit_all
                        ([&](auto& rc)
                         {
                             auto& [r, c] =  rc;

                             S_b += deg_entropy(c);
                             if (r == s)
                                 mdegs[t] += c;
                             else
                                 mdegs[r] += c;
                         });

                    for (auto& [r, c] : mdegs)
                        S_a += deg_entropy(c);

                    return S_a - S_b;
                };

            double S = 0;

            if (ea.adjacency)
            {
                if (_state._deg_corr && ea.deg_entropy)
                {
                    State::spec_t::visit_group_vs
                        (_state._wr[s],
                         [&](auto v)
                         {
                             S += get_delta_dl(_out_deg[v]);
                             if constexpr (is_directed_v<typename State::g_t>)
                                 S += get_delta_dl(_in_deg[v]);
                         });
                }

                gt_hash_set<std::tuple<size_t, size_t>> es;
                for (auto v : _state._groups[s])
                {
                    gt_hash_set<GraphInterface::edge_t> selfs;

                    for (auto e : out_edges_range(v, _state._g))
                    {
                         if constexpr (!is_directed_v<typename State::g_t>)
                         {
                             auto iter = selfs.find(e);
                             if (iter != selfs.end())
                                 continue;
                             selfs.insert(e);
                         }

                        auto u = target(e, _state._g);
                        auto vu = get_uv(v, u);
                        auto iter = _grs.find({std::get<0>(vu).get_endpoint(),
                                               std::get<1>(vu).get_endpoint()});
                        if (iter != _grs.end())
                            es.insert({std::get<0>(vu).get_endpoint(),
                                       std::get<1>(vu).get_endpoint()});
                    }

                    for (auto e : in_edges_range(v, _state._g))
                    {
                        auto u = source(e, _state._g);
                        auto uv = get_uv(u, v);
                        auto iter = _grs.find({std::get<0>(uv).get_endpoint(),
                                               std::get<1>(uv).get_endpoint()});
                        if (iter != _grs.end())
                            es.insert({std::get<0>(uv).get_endpoint(),
                                       std::get<1>(uv).get_endpoint()});
                    }
                }

                for (auto& [u, v] : es)
                {
                    gt_hash_map<std::tuple<group_t, group_t>, size_t> nm;
                    auto iter = _grs.find({u, v});
                    for (auto& rsc : iter->second.first)
                    {
                        auto& [rs, c] = rsc;
                        S -= parallel_term(u, v, rsc.second, _state._g);
                        nm[{(std::get<0>(rs) == s) ? t : std::get<0>(rs),
                            (std::get<1>(rs) == s) ? t : std::get<1>(rs)}] += c;
                    }
                    for (auto& [rs, c] : nm)
                        S += parallel_term(u, v, c, _state._g);
                }
            }
            return S;
        }

        State& _state;

        gt_hash_map<std::tuple<size_t, size_t>,
                    std::pair<gt_hash_map<std::tuple<group_t, group_t>, count_t>,
                              std::shared_ptr<std::mutex>>> _grs;

        vprop_map_t<boost::concurrent_flat_map<group_t, deg_t>>::unchecked_t
            _out_deg, _in_deg;
    };

    decltype(auto) random_neighbors_range(auto v, auto&& g)
    {
        return get_neighborhood(v, g);
    }
};

template <class Graph>
struct extract_param<half_edge_adaptor<Graph>&>
{
    auto& operator()(boost::python::object obj)
    {
        Graph& g = extract_param<Graph&>()(obj);
        return make_half_edge_adaptor(g);
    }
};

template <class EMap>
struct extract_param<VPropAdaptor<EMap>>
{
    auto operator()(boost::python::object obj)
    {
        EMap em = extract_param<EMap>()(obj);
        return VPropAdaptor<EMap>(em);
    }
};

template <>
struct extract_param<OverlapBlockStateSpec<>::bfield_t>
{
    auto operator()(boost::python::object obj)
    {
        auto em_s = extract_param<eprop_map_t<std::vector<double>>>()(obj[0]);
        auto em_t = extract_param<eprop_map_t<std::vector<double>>>()(obj[1]);
        auto& gi = extract_param<GraphInterface&>()(obj[2]);
        OverlapBlockStateSpec<>::bfield_t bfield;

        gt_dispatch<{.gil_release=false}>()
            ([&](auto& g)
             {
                 for (auto e : edges_range(g))
                     bfield.get_prop()[e] = {em_s[e], em_t[e]};
             }, all_graph_views)
            (gi.get_graph_view());

        return bfield;
    }
};

template <>
struct extract_param<half_edge>
{
    auto operator()(boost::python::object obj)
    {
        auto& pe = extract_param<EdgeBase&>()(obj[0]);
        auto e = pe.get_descriptor();
        if (e.s == e.t)
        {
            if (obj[1] == boost::python::object())
            {
                return half_edge(e, e.s, false);
            }
            else
            {
                auto& v = extract_param<VertexBase&>()(obj[1]);
                if (v.get_descriptor() != e.s)
                    throw bad_extract("invalid edge endpoint");
                return half_edge(e, e.s, true);
            }
        }
        else
        {
            auto& v = extract_param<VertexBase&>()(obj[1]);
                if (v.get_descriptor() != e.s && v.get_descriptor() != e.t)
                    throw bad_extract("invalid edge endpoint");
            return half_edge(e, v.get_descriptor());
        }
    }
};

template <>
struct extract_param<std::vector<half_edge>>
{
    auto operator()(boost::python::object obj)
    {
        std::vector<half_edge> vs;
        for (int i = 0; i < boost::python::len(obj); ++i)
            vs.push_back(extract_param<half_edge>()(obj[i]));
        return vs;
    }
};


template <>
struct extract_param<VPropAdaptor<eprop_map_t<std::vector<ovweight_t>>>>
{
    auto operator()(boost::python::object obj)
    {
        VPropAdaptor<eprop_map_t<std::vector<ovweight_t>>> prop;
        if (!obj.is_none())
        {
            auto& gi = extract_param<GraphInterface&>()(obj);
            gt_dispatch<>()
                ([&](auto& g)
                 {
                     auto& u = make_half_edge_adaptor(g);
                     for (auto v : vertices_range(u))
                     {
                          prop[v][v.get_endpoint()] = {in_degree(v, u),
                                                       out_degree(v, u)};
                     }
                 }, all_graph_views)
                (gi.get_graph_view());
        }
        return prop;
    }
};

template <>
struct extract_param<vprop_map_t<ovweight_t>>
{
    auto operator()(boost::python::object)
    {
        return vprop_map_t<ovweight_t>();
    }
};

} //namespace graph_tool

#endif // GRAPH_OVERLAP_BLOCK_MODEL_BASE_HH
