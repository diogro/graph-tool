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

#ifndef GRAPH_MERGE_HH
#define GRAPH_MERGE_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

#include "idx_map.hh"
#include "hash_map_wrap.hh"
#include "demangle.hh"

#include <mutex>
#include <shared_mutex>

#include <boost/python/object.hpp>

namespace graph_tool
{
using namespace std;
using namespace boost;

typedef UnityPropertyMap<int, GraphInterface::edge_t> null_weight_map_t;

template <class UnionGraph, class Graph, class VertexMap, class EdgeMap,
          class EWeight1, class EWeight2>
void graph_merge(UnionGraph& ug, Graph& g, VertexMap vmap, EdgeMap emap,
                 EWeight1 ueweight, EWeight2 eweight, bool multiset,
                 bool diff, bool sym_diff, bool intersect, bool symmetric,
                 bool simple, bool parallel)
{
    typedef typename graph_traits<UnionGraph>::edge_descriptor edge_t;
    typedef typename property_traits<EWeight1>::value_type ew_t;

    GILRelease gil;

    constexpr bool is_vindex =
        !std::is_convertible_v
        <typename boost::property_traits<VertexMap>::category,
         boost::writable_property_map_tag>;

    if constexpr (!is_vindex)
    {
        for (auto v : vertices_range(g))
        {
            if (get(vmap, v) < 0)
            {
                put(vmap, v, add_vertex(ug));
            }
            else
            {
                auto w = vertex(get(vmap, v), ug);
                if (w == graph_traits<UnionGraph>::null_vertex())
                {
                    put(vmap, v, add_vertex(ug));
                }
                else
                {
                    while (w >= num_vertices(ug))
                        add_vertex(ug);
                }
            }
        }
    }
    else
    {
        while (num_vertices(ug) < num_vertices(g))
            add_vertex(ug);
    }

    constexpr bool is_weighted = !std::is_same_v<EWeight1, null_weight_map_t>;

    auto compose_weight =
        [&](auto w1, auto w2) -> ew_t
        {
            if (intersect && (w1 == 0 || w2 == 0))
                return 0;

            if (sym_diff)
                return std::abs(w1 - w2);
            else if (diff)
                return w1 - w2;
            else
                return w1 + w2;
        };

    if (multiset)
    {
        for (auto e : edges_range(g))
        {
            if (eweight[e] <= 0)
                continue;

            auto ne = add_edge(vertex(get(vmap, source(e, g)), ug),
                               vertex(get(vmap, target(e, g)), ug), ug).first;
            emap[e] = ne;
            if constexpr (is_weighted)
                ueweight[ne] = eweight[e];
        }
    }
    else
    {
        std::shared_mutex mutex;
        std::vector<std::mutex> vmutex(is_vindex ? 0 : num_vertices(ug));
        std::vector<edge_t> res;
        gt_hash_set<edge_t> res_set;

        idx_set<size_t> self_loops;
        auto eindex = get(edge_index, g);

        parallel = (parallel &&
                    (num_vertices(g) > get_openmp_min_thresh()) &&
                    (get_num_threads() > 1));

        auto lock
            = [&](auto&& f, bool shared) [[gnu::always_inline]]
              {
                  if (parallel)
                  {
                      if (shared)
                      {
                          std::shared_lock slock(mutex);
                          f();
                      }
                      else
                      {
                          std::unique_lock slock(mutex);
                          f();
                      }
                  }
                  else
                  {
                      f();
                  }
              };

        #pragma omp parallel if (parallel) firstprivate(self_loops, res, res_set)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto s)
             {
                 if (!is_directed(g))
                     self_loops.clear();

                 size_t s_u = get(vmap, s);

                 for (auto e : out_edges_range(s, g))
                 {
                     auto t = target(e, g);

                     if constexpr (!is_directed_v<Graph>)
                     {
                         if (s > t)
                             continue;
                         if (t == s)
                         {
                             auto iter = self_loops.find(eindex[e]);
                             if (iter != self_loops.end())
                                 continue;
                             self_loops.insert(eindex[e]);
                         }
                     }

                     std::decay_t<decltype(eweight[e])> w2 = 0;

                     if (simple)
                     {
                         w2 = eweight[e];
                     }
                     else
                     {
                         bool skip = false;
                         bool first = true;
                         edge_range_iter(s, t, g,
                                         [&](const auto& e_)
                                         {
                                             if (e_ != e && first)
                                             {
                                                 skip = true;
                                                 return false;
                                             }
                                             first = false;
                                             w2 += eweight[e_];
                                             return true;
                                         });
                         if (skip)
                             continue;
                     }

                     emap[e] = edge_t();

                     if (w2 == 0)
                         continue;

                     size_t t_u = get(vmap, t);

                     if (!is_vindex && parallel)
                     {
                         if (s_u != t_u)
                             std::lock(vmutex[s_u], vmutex[t_u]);
                         else
                             vmutex[s_u].lock();
                     }

                     std::decay_t<decltype(ueweight[e])> w1 = 0;
                     edge_t ge;
                     bool exists = false;

                     if (simple)
                     {
                         lock([&]()
                              {
                                  std::tie(ge, exists) = edge(s_u, t_u, ug);
                                  if (exists)
                                      w1 = ueweight[ge];
                              }, true);
                     }
                     else
                     {
                         lock([&]()
                              {
                                  edge_range_iter
                                      (s_u, t_u, ug,
                                       [&](const auto& ge_)
                                       {
                                           w1 += ueweight[ge_];
                                           if (ge == edge_t())
                                           {
                                               ge = ge_;
                                               exists = true;
                                           }
                                       });
                              }, true);
                     }

                     auto w = compose_weight(w1, w2);

                     if (w > 0)
                     {
                         if (exists)
                         {
                             if constexpr (is_weighted)
                             {
                                 lock([&]()
                                      {
                                          #pragma omp atomic
                                          ueweight[ge] += w;
                                      }, true);
                             }
                         }
                         else
                         {
                             lock([&]()
                                  {
                                      ge = add_edge(s_u, t_u, ug).first;
                                      if constexpr (is_weighted)
                                          ueweight.get_checked()[ge] = w;
                                  }, false);
                         }
                         emap[e] = ge;
                     }
                     else
                     {
                         if (exists)
                         {
                             if (simple)
                             {
                                 auto iter = res_set.find(ge);
                                 if (iter == res_set.end())
                                 {
                                     res_set.insert(ge);
                                     res.push_back(ge);
                                 }
                             }
                             else
                             {
                                 lock([&]()
                                      {
                                          edge_range_iter(s_u, t_u, ug,
                                                          [&](const auto& ge_)
                                                          {
                                                              auto iter = res_set.find(ge_);
                                                              if (iter != res_set.end())
                                                                  return;
                                                              res_set.insert(ge_);
                                                              res.push_back(ge_);
                                                          });
                                      }, true);
                             }
                         }
                         emap[e] = edge_t();
                     }

                     if (!is_vindex && parallel)
                     {
                         vmutex[s_u].unlock();
                         if (t_u != s_u)
                             vmutex[t_u].unlock();
                     }
                 }

                 if (!res.empty())
                 {
                     std::unique_lock lock(mutex);
                     for (auto e : res)
                     {
                         bool exists = false;
                         if (simple)
                         {
                             exists = edge(source(e, ug), target(e, ug), ug).second;
                         }
                         else
                         {
                             edge_range_iter(source(e, ug), target(e, ug), ug,
                                             [&](const auto& ge_)
                                             {
                                                 if (e == ge_)
                                                 {
                                                     exists = true;
                                                     return false;
                                                 }
                                                 return true;
                                             });
                         }
                         if (exists)
                             remove_edge(e, ug);
                     }
                     res.clear();
                     res_set.clear();
                 }
             });

        if (!simple)
        {
            parallel_edge_loop
            (g,
             [&](auto e)
             {
                 auto s = source(e, g);
                 auto t = target(e, g);
                 if (!is_directed(g) && s > t)
                     std::swap(s, t);
                 auto fe = edge(s, t, g).first; // first parallel edge
                 if (fe != e)
                     emap[e] = emap[fe];
             });
        }

        if (symmetric)
        {
            auto eindex = get(edge_index, ug);
            #pragma omp parallel if (num_vertices(ug) > get_openmp_min_thresh()) \
                firstprivate(self_loops, res)
            parallel_vertex_loop_no_spawn
                (ug,
                 [&](auto s)
                 {
                     std::shared_lock slock(mutex);
                     std::vector<edge_t> res;

                     if (!is_directed(g))
                         self_loops.clear();

                     for (auto e : out_edges_range(s, ug))
                     {
                         auto t = target(e, ug);

                         if (!is_directed_v<UnionGraph>)
                         {
                             if (s > t)
                                 continue;
                             if (t == s)
                             {
                                 auto iter = self_loops.find(eindex[e]);
                                 if (iter != self_loops.end())
                                     continue;
                                 self_loops.insert(eindex[e]);
                             }
                         }

                         if (edge(s, t, g).second)
                             continue;

                         std::decay_t<decltype(ueweight[e])> w1 = 0;

                         if (simple)
                         {
                             w1 = ueweight[e];
                         }
                         else
                         {
                             bool first = true;
                             bool skip = false;
                             edge_range_iter(s, t, ug,
                                             [&](const auto& ge)
                                             {
                                                 if (first && ge != e)
                                                 {
                                                     skip = true;
                                                     return false;
                                                 }
                                                 first = false;
                                                 w1 += ueweight[ge];
                                                 return true;
                                             });
                             if (skip)
                                 continue;
                         }

                         auto w = compose_weight(w1, 0);

                         if (w <= 0)
                         {
                             if (simple)
                                 res.push_back(e);
                             else
                                 edge_range_iter(s, t, ug,
                                                 [&](const auto& ge)
                                                 {
                                                     res.push_back(ge);
                                                 });
                         }
                     }

                     if (!res.empty())
                     {
                         slock.unlock();
                         std::unique_lock lock(mutex);
                         for (auto e : res)
                             remove_edge(e, ug);
                     }
                 });
        }
    }
}

enum merge_t { set, sum, diff, idx_inc, append, concat };

template <merge_t merge>
struct property_merge
{
    template <class UnionGraph, class Graph, class VertexMap, class EdgeMap,
              class UnionProp, class Prop>
    void operator()(UnionGraph& ug, Graph& g, VertexMap vmap, EdgeMap emap,
                    UnionProp uprop, Prop prop, bool simple) const
    {
        typedef typename property_traits<UnionProp>::value_type uval_t;
        constexpr bool atomic = (std::is_scalar_v<uval_t> &&
                                 (merge == merge_t::set ||
                                  merge == merge_t::sum ||
                                  merge == merge_t::diff));
        dispatch<atomic>(ug, g, vmap, emap, uprop, prop, simple,
                         std::is_same<typename property_traits<UnionProp>::key_type,
                                      typename graph_traits<Graph>::vertex_descriptor>());
    }

    template <bool atomic, class UVal, class Val>
    void dispatch_value(UVal& uval, const Val& val) const
    {
        constexpr bool is_object = std::is_same_v<Val, boost::python::object>;
        constexpr bool is_object_union = std::is_same_v<UVal, boost::python::object>;

        if constexpr (merge == merge_t::set)
        {
            if constexpr (atomic)
            {
                #pragma omp atomic write
                uval = convert<UVal>(val);
            }
            else
            {
                uval = convert<UVal>(val);
            }
        }
        else if constexpr (merge == merge_t::sum)
        {
            if constexpr (is_vector_v<Val> && is_vector_v<UVal>)
            {
                if (val.size() > uval.size())
                    uval.resize(val.size());
                for (size_t i = val.size(); i < val.size(); ++i)
                    uval[i] += convert<typename UVal::value_type>(val[i]);
            }
            else if constexpr (is_vector_v<UVal>)
            {
                for (auto& x : uval)
                    x += convert<typename UVal::value_type>(val);
            }
            else if constexpr (is_vector_v<Val>)
            {
                for (auto& x : val)
                    uval += convert<UVal>(x);
            }
            else
            {
                if constexpr (atomic)
                {
                    #pragma omp atomic
                    uval += convert<UVal>(val);
                }
                else
                {
                    uval += convert<UVal>(val);
                }
            }
        }
        else if constexpr (merge == merge_t::diff)
        {
            if constexpr (is_vector_v<Val> && is_vector_v<UVal>)
            {
                if (val.size() > uval.size())
                    uval.resize(val.size());
                for (size_t i = val.size(); i < val.size(); ++i)
                    uval[i] -= convert<typename UVal::value_type>(val[i]);
            }
            else if constexpr (is_vector_v<UVal>)
            {
                for (auto& x : uval)
                    x -= convert<typename UVal::value_type>(val);
            }
            else if constexpr (is_vector_v<Val>)
            {
                for (auto& x : val)
                    uval -= convert<UVal>(x);
            }
            else
            {
                if constexpr (atomic)
                {
                    #pragma omp atomic
                    uval -= convert<UVal>(val);
                }
                else
                {
                    uval -= convert<UVal>(val);
                }
            }
        }
        else if constexpr (merge == merge_t::idx_inc)
        {
            size_t idx;
            typename UVal::value_type diff;
            if constexpr (is_vector_v<Val>)
            {

                typedef typename Val::value_type val_t;
                val_t a = val.size() > 0 ? val[0] : val_t(0);
                if constexpr (std::is_signed_v<val_t>)
                {
                    if (a < 0)
                    {
                        size_t n = std::ceil(-a);
                        uval.resize(uval.size() + n);
                        for (size_t i = uval.size() - 1; i > n-1; --i)
                            uval[i] = uval[i-n];
                        a = 0;
                        return;
                    }
                }
                idx = a;
                diff = val.size() > 1 ? val[1] : 0;
            }
            else
            {
                if constexpr (std::is_signed_v<Val>)
                {
                    if (val < 0)
                        return;
                }
                idx = val;
                diff = 1;
            }
            if (idx >= uval.size())
                uval.resize(idx + 1);
            uval[idx] += diff;
        }
        else if constexpr (merge == merge_t::append)
        {
            if constexpr (is_object_union)
                uval.attr("append")(val);
            else
                uval.push_back(convert<typename UVal::value_type>(val));
        }
        else if constexpr (merge == merge_t::concat)
        {
            if constexpr (std::is_same_v<UVal, std::string>)
            {
                if constexpr (std::is_same_v<Val, std::string>)
                    uval += val;
                else if constexpr (std::is_same_v<Val, boost::python::object>)
                    uval += boost::python::extract<std::string>(val)();
            }
            else if constexpr (is_object_union)
            {
                if (uval == boost::python::object())
                    uval = boost::python::list();
                if constexpr (is_object)
                {
                    uval.attr("extend")(val);
                }
                else
                {
                    auto a = uval.attr("append");
                    for (auto& x : val)
                        a(x);
                }
            }
            else
            {
                if constexpr (is_object)
                {
                    for (int i = 0; i < python::len(val); ++i)
                        uval.push_back(python::extract<typename UVal::value_type>(val[i]));
                }
                else
                {
                    if constexpr (std::is_convertible_v<typename Val::value_type,
                                                        typename UVal::value_type>)
                    {
                        uval.insert(uval.end(), val.begin(), val.end());
                    }
                    else
                    {
                        for (auto& x : val)
                            uval.push_back(convert<typename UVal::value_type>(x));
                    }
                }
            }
        }
    }

    template <bool atomic, class UnionGraph, class Graph, class VertexMap,
              class EdgeMap, class UnionProp, class Prop>
    void dispatch(UnionGraph& ug, Graph& g, VertexMap vmap, EdgeMap,
                  UnionProp uprop, Prop prop, bool simple, std::true_type) const
    {
        typedef typename property_traits<Prop>::value_type val_t;
        typedef typename property_traits<UnionProp>::value_type uval_t;

        constexpr bool is_object =
            std::is_same_v<val_t, boost::python::object>;
        constexpr bool is_object_union =
            std::is_same_v<uval_t, boost::python::object>;
        constexpr bool is_vindex =
            !std::is_convertible_v
            <typename boost::property_traits<VertexMap>::category,
             boost::writable_property_map_tag>;

        GILRelease gil(!is_object && !is_object_union);

        bool parallel = (!is_object && !is_object_union &&
                         (is_vindex || simple) &&
                         (num_vertices(g) > get_openmp_min_thresh()) &&
                         (get_num_threads() > 1));

        if (!parallel)
        {
            for (auto v : vertices_range(g))
                dispatch_value<false>(uprop[vertex(get(vmap, v), ug)],
                                      get(prop, v));
        }
        else
        {
            auto dispatch_v =
                [&](auto v){ dispatch_value<atomic>(uprop[vertex(get(vmap, v), ug)],
                                                    get(prop, v)); };

            std::vector<std::mutex> vmutex((is_vindex || atomic) ? 0 : num_vertices(ug));
            std::string err;
            #pragma omp parallel
            parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     auto uv = get(vmap, v);
                     if constexpr (!is_vindex && !atomic)
                         vmutex[uv].lock();
                     if constexpr (std::is_convertible_v<val_t, uval_t> &&
                                   !is_object && !is_object_union &&
                                   !is_dynamic_map_v<Prop> &&
                                   !is_dynamic_map_v<UnionProp>)
                     {
                         dispatch_v(v);
                     }
                     else
                     {
                         if (!err.empty())
                             return;
                         try
                         {
                             dispatch_v(v);
                         }
                         catch (const ValueException& e)
                         {
                             #pragma omp critical
                             err = e.what();
                         }
                     }
                     if constexpr (!is_vindex && !atomic)
                         vmutex[uv].unlock();
                 });

            if (!err.empty())
                throw ValueException(err);
        }
    }

    template <bool atomic, class UnionGraph, class Graph, class VertexMap,
              class EdgeMap, class UnionProp, class Prop>
    void dispatch(UnionGraph& ug, Graph& g, VertexMap vmap, EdgeMap emap,
                  UnionProp uprop, Prop prop, bool simple, std::false_type) const
    {
        typedef typename property_traits<Prop>::value_type val_t;
        typedef typename property_traits<UnionProp>::value_type uval_t;

        constexpr bool is_object =
            std::is_same_v<val_t, boost::python::object>;
        constexpr bool is_object_union =
            std::is_same_v<uval_t, boost::python::object>;
        constexpr bool is_vindex =
            !std::is_convertible_v
            <typename boost::property_traits<VertexMap>::category,
             boost::writable_property_map_tag>;

        typedef typename graph_traits<UnionGraph>::edge_descriptor edge_t;

        GILRelease gil(!is_object && !is_object_union);

        bool parallel = (!is_object && !is_object_union && simple &&
                         (num_vertices(g) > get_openmp_min_thresh()) &&
                         (get_num_threads() > 1));

        if (!parallel)
        {
            for (auto e : edges_range(g))
            {
                auto& ge = emap[e];
                if (ge != edge_t())
                    dispatch_value<false>(uprop[ge], get(prop, e));
            }
        }
        else
        {
            auto dispatch_e =
                [&](const auto& e)
                {
                    auto& ge = emap[e];
                    if (ge != edge_t())
                        dispatch_value<atomic>(uprop[ge], get(prop, e));
                };

            std::vector<std::mutex> vmutex((is_vindex || atomic) ? 0 : num_vertices(ug));
            std::string err;
            #pragma omp parallel
            parallel_edge_loop_no_spawn
                (g,
                 [&](const auto& e)
                 {
                     auto s_u = get(vmap, source(e, g));
                     auto t_u = get(vmap, target(e, g));

                     if constexpr (!is_vindex && !atomic)
                     {
                         if (s_u != t_u)
                             std::lock(vmutex[s_u], vmutex[t_u]);
                         else
                             vmutex[s_u].lock();
                     }

                     if constexpr ((std::is_convertible_v<val_t, uval_t> &&
                                    !is_object && !is_object_union &&
                                    !is_dynamic_map_v<Prop> &&
                                    !is_dynamic_map_v<UnionProp>))
                     {
                         dispatch_e(e);
                     }
                     else
                     {
                         if (!err.empty())
                             return;
                         try
                         {
                             dispatch_e(e);
                         }
                         catch (const ValueException& e)
                         {
                             #pragma omp critical
                             err = e.what();
                         }
                     }

                     if constexpr (!is_vindex && !atomic)
                     {
                         vmutex[s_u].unlock();
                         if (t_u != s_u)
                             vmutex[t_u].unlock();
                     }
                 });

            if (!err.empty())
                throw ValueException(err);
        }
    }
};


} // graph_tool namespace

#endif // GRAPH_MERGE_HH
