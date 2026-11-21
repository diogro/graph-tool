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

#pragma once

#include "graph_tool.hh"

#include "../inference/support/graph_state.hh"

#include "quad_tree.hh"

#include <chrono>
#include <flat_map>
#include <memory>
#include <numbers>
#include <mutex>
#include <numeric>

// L-BFGS is only available in newer Boost versions
#if __has_include(<boost/math/optimization/lbfgs.hpp>)
#define HAVE_BOOST_LBFGS
#include <boost/math/optimization/lbfgs.hpp>
#include <boost/math/optimization/minimizer.hpp>
#endif
#include <boost/iterator/function_output_iterator.hpp>

#include <CGAL/Search_traits.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>

#include <CGAL/Convex_hull_traits_adapter_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/property_map.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/box_intersection_d.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_id_2.h>
#include <CGAL/Triangulation_face_base_with_id_2.h>
#include <CGAL/boost/graph/graph_traits_Delaunay_triangulation_2.h>
#include <CGAL/boost/graph/properties_Delaunay_triangulation_2.h>

#include <boost/graph/biconnected_components.hpp>

#ifdef HAVE_BOOST_LBFGS
namespace bopt = boost::math::optimization;
#endif

namespace graph_tool
{

// one array per level, indexed by group label (no comma, for the macros
// below)
typedef std::vector<boost::multi_array_ref<double, 1>> hfd_level_arrays_t;

#define STATE_params                                                           \
     ((g, decltype(all_graph_views), 1))                                       \
     ((pos, vprop_map_t<std::vector<double>>, 0))                              \
     ((vweight, vprop_map_t<double>, 0))                                       \
     ((eweight, eprop_map_t<double>, 0))                                       \
     ((rweight, vprop_map_t<double>, 0))                                       \
     ((K, double, 0))                                                          \
     ((rep_exp, double, 0))                                                    \
     ((att_exp, double, 0))                                                    \
     ((center_pull, double, 0))                                                \
     ((theta, double, 0))                                                      \
     ((exact_thres, size_t, 0))                                                \
     ((b, std::vector<std::vector<int64_t>>, 0))                               \
     ((group_grav, std::vector<double>, 0))                                    \
     ((group_grav_fac, hfd_level_arrays_t, 0))                                 \
     ((group_grav_offset, double, 0))                                          \
     ((group_rep, std::vector<double>, 0))                                     \
     ((group_rep_exp, double, 0))                                              \
     ((group_conf_R, hfd_level_arrays_t, 0))                                   \
     ((group_conf_exp, double, 0))                                             \
     ((group_spring, std::vector<double>, 0))                                  \
     ((group_spring_static, bool, 0))                                          \
     ((barrier_r, vprop_map_t<double>, 0))                                     \
     ((barrier_sep, double, 0))                                                \
     ((barrier_exact_thres, size_t, 0))                                        \
     ((nmin_dist, vprop_map_t<double>, 0))                                     \
     ((nmu, double, 0))                                                        \
     ((gmin_dist, vprop_map_t<std::vector<double>>, 0))                        \
     ((gmu, std::vector<double>, 0))                                           \
     ((odist, std::vector<double>, 0))                                         \
     ((omu, std::vector<double>, 0))                                           \
     ((opair_dist, std::vector<double>, 0))                                    \
     ((pmu, std::vector<double>, 0))                                           \
     ((rank, vprop_map_t<double>, 0))                                          \
     ((rmin_dist, vprop_map_t<double>, 0))                                     \
     ((rmu, double, 0))                                                        \
     ((rcm_dist, double, 0))                                                   \
     ((rcmu, double, 0))                                                       \
     ((rcm_xdist, double, 0))                                                  \
     ((rcmxu, double, 0))                                                      \
     ((nlambda, vprop_map_t<double>, 0))                                       \
     ((olambda, hfd_level_arrays_t, 0))                                        \
     ((rlambda, vprop_map_t<double>, 0))                                       \
     ((plambda, vprop_map_t<std::vector<double>>, 0))                          \
     ((rng, rng_t&, 0))

GEN_STATE_BASE(HFDStateBase, STATE_params)

// Delaunay triangulation used for the order parameters of the groups, with
// the vertex index of every point as its info. It provides the
// all_edge_iterator required by boost::filt_graph, which is otherwise the same
// CGAL triangulation, so that it can be filtered with it.
typedef CGAL::Exact_predicates_inexact_constructions_kernel hfd_dt_kernel_t;
typedef CGAL::Delaunay_triangulation_2
    <hfd_dt_kernel_t,
     CGAL::Triangulation_data_structure_2
         <CGAL::Triangulation_vertex_base_with_info_2
              <size_t, hfd_dt_kernel_t,
               CGAL::Triangulation_vertex_base_with_id_2<hfd_dt_kernel_t>>,
          CGAL::Triangulation_face_base_with_id_2<hfd_dt_kernel_t>>>
    hfd_dt_base_t;

struct hfd_dt_t
    : public hfd_dt_base_t
{
    using hfd_dt_base_t::hfd_dt_base_t;
    typedef boost::graph_traits<hfd_dt_base_t>::out_edge_iterator
        all_edge_iterator;
};

} // namespace graph_tool

namespace boost
{
template <>
struct graph_traits<graph_tool::hfd_dt_t>
    : public graph_traits<graph_tool::hfd_dt_base_t> {};

template <class Tag>
struct property_map<graph_tool::hfd_dt_t, Tag>
    : public property_map<graph_tool::hfd_dt_base_t, Tag> {};
} // namespace boost

namespace graph_tool
{

// Interface of the state of the level above a HFDState (see
// HFDState::set_upper()), whose vertices are placed at the centers of mass of
// the groups of the level below, and which contributes only the attraction and
// repulsion of its vertices (and those of the levels above it).
class HFDUpper
{
public:
    virtual ~HFDUpper() = default;

    virtual size_t ml_num_vertices() = 0;

    // sets the positions x[i] of the vertices us[i] (and those of the levels
    // above); the other vertices keep their positions
    virtual void ml_set_pos(const std::vector<int64_t>& us,
                            const std::vector<std::array<double, 2>>& x) = 0;

    // the version of the positions of this level and those above, which
    // changes whenever they are changed other than by ml_set_pos()
    virtual size_t ml_version() = 0;

    // the energy of the attraction and repulsion (and that of the levels
    // above), at the positions last set
    virtual double ml_energy() = 0;

    // computes the gradient for every vertex, returned by ml_grads()
    virtual void ml_grad_setup() = 0;
    virtual const std::vector<std::array<double, 2>>& ml_grads() = 0;
};

// defined below
inline double disk_moment(double p, double d0);

// The mean c_p of d^-p in a group of n members spread over the unit disk,
// i.e. disk_moment(p, 0), or, if it diverges (p >= 2), excluding the
// distances below the typical spacing of the members, disk_moment(p,
// 1/sqrt(n)), which is memoized by n, since many groups have the same size.
class disk_moment_cache
{
public:
    disk_moment_cache(double p)
        : _p(p), _c0((p < 2) ? disk_moment(p, 0) : 0) {}

    double operator()(size_t n)
    {
        if (_p < 2)
            return _c0;
        auto iter = _c.find(n);
        if (iter == _c.end())
            iter = _c.insert({n, disk_moment(_p, 1 / std::sqrt(n))}).first;
        return iter->second;
    }

private:
    double _p, _c0;
    gt_hash_map<size_t, double> _c;
};

template <class... Ts>
class HFDState
    : public HFDStateBase<Ts...>, public HFDUpper
{
public:
    GET_PARAMS_USING(HFDStateBase<Ts...>, STATE_params)
    GET_PARAMS_TYPEDEF(Ts, STATE_params)

    // Initialization
    // ==============

    template <class... ATs>
    HFDState(ATs&&... args)
        : HFDStateBase<Ts...>(std::forward<ATs>(args)...),
          // the vertices of zero weight are skipped, and keep their positions
          _vs(std::from_range,
              vertices_range(_g) |
              std::views::filter([&](auto v) { return _vweight[v] > 0; })),
          _x(num_vertices(_g)),
          _vel(num_vertices(_g)),
          _kd_tree(_vs.begin(), _vs.end(), typename kd_tree_t::Splitter(),
                   kd_traits_t(xmap_t(_x.begin())))
    {
        pos_load();
        init_adj();
        std::ranges::shuffle(_vs, _rng);
        _exact = _vs.size() < _exact_thres;
        reset_momentum();

        _vpos.resize(num_vertices(_g));
        for (auto [i, v] : std::views::enumerate(_vs))
            _vpos[v] = i;

        // negative weights and strengths would make the energy unbounded (and
        // vertices of negative weight would be left out of _vs, but not of
        // the adjacency)
        for (auto v : vertices_range(_g))
        {
            for (auto [name, x] : {std::pair{"vweight", _vweight[v]},
                                   {"rweight", _rweight[v]},
                                   {"nmin_dist", _nmin_dist[v]}})
            {
                if (!(x >= 0))
                    throw ValueException(std::format("vertex {} has invalid "
                                                     "{} {}", v, name, x));
            }
        }
        for (auto e : edges_range(_g))
        {
            if (!(_eweight[e] >= 0))
                throw ValueException(std::format("invalid edge weight {}",
                                                 _eweight[e]));
        }
        if (!(_nmu >= 0) || !(_rmu >= 0))
            throw ValueException("the penalties must be non-negative");
        if (!(_K >= 0) || !(_center_pull >= 0) || !(_barrier_sep >= 0))
            throw ValueException("K, center_pull and barrier_sep must be "
                                 "non-negative");
        // otherwise, the attraction decays faster than the repulsion, and
        // there is no bounded equilibrium
        if (!(_att_exp + _rep_exp >= 0))
            throw ValueException("att_exp + rep_exp must be non-negative");

        if (!(_rcmu >= 0) || !(_rcmxu >= 0))
            throw ValueException("the penalties must be non-negative");
        if ((_rmu > 0 || _rcmu > 0 || _rcmxu > 0) &&
            !std::ranges::all_of(_vs, [&](auto v)
                                      { return std::isfinite(_rank[v]); }))
            throw ValueException("the ranks must be finite");
        if (_rmu > 0 || _rcmu > 0 || _rcmxu > 0)
            init_ranks();

        init_levels();
        init_vsep();
        init_dist_scales();
        update_group_springs();

        if (has_dist())
            _dgrad.resize(_vs.size());

        if (_has_omu || _has_pmu)
            _ograd.resize(_vs.size());
    }

    void init_levels()
    {
        _L = _b.size();

        // The lists are typed explicitly only because of clang, which
        // otherwise unifies the type sugar of the members when deducing the
        // list type, at a huge cost in memory (GCC is unaffected).
        typedef std::initializer_list<std::vector<double>*> plist_t;
        for (auto* x : plist_t{&_gmu, &_omu, &_pmu, &_odist, &_opair_dist,
                               &_group_grav, &_group_rep, &_group_spring})
        {
            if (x->size() != _L)
                throw ValueException(std::format("per-level parameter has {} "
                                                 "values, but the partition "
                                                 "has {} levels", x->size(),
                                                 _L));
        }

        for (auto* x : plist_t{&_gmu, &_omu, &_pmu, &_group_grav,
                               &_group_spring})
        {
            if (!std::ranges::all_of(*x, [](auto& y) { return y >= 0; }))
                throw ValueException("the penalties, group_grav and "
                                     "group_spring must be non-negative");
        }

        auto pos = [](auto& x) { return x > 0; };
        _has_gmu = std::ranges::any_of(_gmu, pos);
        _has_omu = std::ranges::any_of(_omu, pos);
        _has_pmu = std::ranges::any_of(_pmu, pos);
        _has_group_grav = std::ranges::any_of(_group_grav, pos);
        _has_group_rep = std::ranges::any_of(_group_rep,
                                             [](auto& x) { return x != 0; });
        _group_rep_top = std::accumulate(_group_rep.begin(),
                                         _group_rep.end(), 0.);
        _group_rep_sep = _has_group_rep && _group_rep_exp != _rep_exp;

        if (_has_omu && _olambda.size() != _L)
            throw ValueException(std::format("olambda has {} levels, expected "
                                             "{}", _olambda.size(), _L));
        if (_has_group_grav && _group_grav_fac.size() != _L)
            throw ValueException(std::format("group_grav_fac has {} levels, "
                                             "expected {}",
                                             _group_grav_fac.size(), _L));

        _levels.resize(_L);
        for (auto v : vertices_range(_g))
        {
            if (_has_gmu && _gmin_dist[v].size() != _L)
                throw ValueException(std::format("vertex {} has {} values of "
                                                 "gmin_dist, expected {}", v,
                                                 _gmin_dist[v].size(), _L));
            _plambda[v].resize(_L);
        }

        // The partition is nested, as in NestedBlockState: _b[0][v] is the
        // group of vertex v, and _b[l][r] the group at level l of group r at
        // level l - 1. br[v] holds the group of v at the current level.
        std::vector<int64_t> br(num_vertices(_g));
        for (auto v : vertices_range(_g))
            br[v] = v;

        for (size_t l = 0; l < _L; ++l)
        {
            auto& bl = _b[l];
            for (auto v : vertices_range(_g))
            {
                auto r = br[v];
                if (r < 0 || std::cmp_greater_equal(r, bl.size()))
                    throw ValueException(std::format("partition level {} has "
                                                     "{} entries, but group "
                                                     "{} is referenced", l,
                                                     bl.size(), r));
                br[v] = bl[r];
            }

            auto& lev = _levels[l];
            lev.bidx.resize(num_vertices(_g));
            gt_hash_map<int64_t, size_t> rmap;
            for (auto v : vertices_range(_g))
            {
                if (!(_vweight[v] > 0))
                    continue;
                auto r = br[v];
                if (r < 0)
                    throw ValueException(std::format("vertex {} has the "
                                                     "negative group label {} "
                                                     "at level {}", v, r, l));
                auto iter = rmap.find(r);
                if (iter == rmap.end())
                {
                    iter = rmap.insert({r, rmap.size()}).first;
                    lev.groups.emplace_back();
                    lev.label.push_back(r);
                }
                lev.bidx[v] = iter->second;
                lev.groups[iter->second].push_back(v);
            }
            lev.w.assign(lev.groups.size(), 0);
            for (auto v : _vs)
                lev.w[lev.bidx[v]] += _vweight[v];

            if (has_group_trees(l))
            {
                lev.qt.resize(lev.groups.size());
                lev.src.resize(lev.groups.size());
            }

            if (_group_grav[l] > 0)
            {
                for (auto r : lev.label)
                {
                    if (std::cmp_greater_equal(r, _group_grav_fac[l].size()))
                        throw ValueException
                            (std::format("group_grav_fac at level {} has {} "
                                         "entries, but group {} exists", l,
                                         _group_grav_fac[l].size(), r));
                    lev.ggfac.push_back(_group_grav_fac[l][r]);
                }
            }

            // confinement, if a radius is given for this level
            if (l < _group_conf_R.size())
                init_conf(l);

            if (_omu[l] > 0)
            {
                for (auto r : lev.label)
                {
                    if (std::cmp_greater_equal(r, _olambda[l].size()))
                        throw ValueException(std::format("olambda at level {} "
                                                         "has {} entries, but "
                                                         "group {} exists", l,
                                                         _olambda[l].size(),
                                                         r));
                }
            }

            // Groups of two also get a maximum distance, which keeps other
            // groups from passing between them. The multiplier of a pair is
            // stored in _plambda at its first member (the one with the
            // smallest index).
            if (_pmu[l] > 0)
            {
                for (auto& vs : lev.groups)
                {
                    if (vs.size() == 2)
                        lev.pairs.push_back({vs[0], vs[1]});
                }
            }

            // the slots (in _vs) of the members of every group, for the hulls
            if (_omu[l] > 0)
            {
                lev.slots.resize(lev.groups.size());
                for (size_t r = 0; r < lev.groups.size(); ++r)
                    for (auto v : lev.groups[r])
                        lev.slots[r].push_back(_vpos[v]);
            }
        }

        for (size_t l = 0; l < _L; ++l)
        {
            if (!has_group_trees(l))
                continue;
            for (size_t r = 0; r < _levels[l].groups.size(); ++r)
                _group_tree_jobs.push_back({l, r});
        }
        // the largest first (see group_tree_loop())
        std::ranges::stable_sort(_group_tree_jobs, std::greater<>(),
                                 [&](auto& lr)
                                 {
                                     return _levels[lr[0]].groups[lr[1]].size();
                                 });
    }

    // State
    // =====

    typedef typename boost::property_traits<vweight_t>::value_type vw_t;
    typedef typename boost::property_traits<pos_t>::value_type vpos_t;
    typedef std::array<typename vpos_t::value_type, 2> p_t;

    std::vector<size_t> _vs;
    std::vector<size_t> _vpos;
    QuadTree<double, vw_t> _qt;
    // The positions are kept contiguously in _x during the minimization, and
    // copied from and to _pos by pos_load() and pos_store() (which the Python
    // bindings call around every method), since _pos stores every position
    // in a separate allocation.
    std::vector<p_t> _x, _vel;
    typedef boost::iterator_property_map
        <typename std::vector<p_t>::iterator,
         boost::typed_identity_property_map<size_t>> xmap_t;

    // The version of the positions, which is increased whenever they may have
    // been changed other than by a state below (e.g. from Python, before
    // every method), so that the states below know when the terms of the
    // levels above that they cached are stale (see ml_version()).
    size_t _version = 0;

    void pos_load()
    {
        ++_version;
        read_pos();
    }

    void read_pos()
    {
        parallel_vertex_loop
            (_g,
             [&](auto v)
             {
                 auto& p = _pos[v];
                 if (p.size() < 2)
                     p.resize(2);
                 _x[v] = {p[0], p[1]};
             });
    }

    void pos_store()
    {
        parallel_vertex_loop
            (_g,
             [&](auto v)
             {
                 _pos[v][0] = _x[v][0];
                 _pos[v][1] = _x[v][1];
             });
    }

    // Compact adjacency: the neighbours of v are _adj[_adj_begin[v]] to
    // _adj[_adj_end[v] - 1], with the edge weights, so that the attraction
    // needs a single random access (the position of the neighbour) per edge.
    // The edges of zero weight, or with an endpoint of zero weight (which is
    // not moved), exert no force, and are left out.
    std::vector<size_t> _adj_begin, _adj_end;
    std::vector<std::tuple<size_t, double>> _adj;

    // weight of the edge e of v (the vertex weights do not enter the
    // attraction, except that the edges of the vertices of zero weight exert
    // no force)
    double adj_weight(const auto& e, size_t v)
    {
        if (!(_vweight[target(e, _g)] > 0) || !(_vweight[v] > 0))
            return 0;
        return _eweight[e];
    }

    void init_adj()
    {
        _adj_begin.assign(num_vertices(_g), 0);
        _adj_end.assign(num_vertices(_g), 0);
        parallel_vertex_loop
            (_g,
             [&](auto v)
             {
                 for (auto e : _all_edges_out_range(v, _g))
                 {
                     if (adj_weight(e, v) != 0)
                         ++_adj_end[v];
                 }
             });

        // every vertex claims a contiguous block, in any order
        size_t n = 0;
        parallel_vertex_loop
            (_g,
             [&](auto v)
             {
                 size_t k = _adj_end[v], i;
                 #pragma omp atomic capture
                 {
                     i = n;
                     n += k;
                 }
                 _adj_begin[v] = i;
                 _adj_end[v] = i + k;
             });

        _adj.resize(n);
        parallel_vertex_loop
            (_g,
             [&](auto v)
             {
                 size_t i = _adj_begin[v];
                 for (auto e : _all_edges_out_range(v, _g))
                 {
                     double m = adj_weight(e, v);
                     if (m != 0)
                         _adj[i++] = {target(e, _g), m};
                 }
             });
    }

    // calls f(u, m, dx, d2) for every neighbour u of v with a different
    // position, with dx = x_u - x_v and m the product of the weights
    void visit_adj(size_t v, auto&& f)
    {
        auto& xv = _x[v];
        for (size_t i = _adj_begin[v]; i < _adj_end[v]; ++i)
        {
            auto& [u, m] = _adj[i];
            p_t dx = sub(_x[u], xv);
            double d2 = norm2(dx);
            if (d2 == 0)
                continue;
            f(u, m, dx, d2);
        }
    }

    // gradient of the attraction of v
    p_t adj_grad(size_t v)
    {
        p_t g{};
        // the exponent is resolved outside the loop
        if (_att_exp == 2)
            visit_adj(v, [&](auto, auto m, auto& dx, auto)
                         { g -= dx * (2 * m); });
        else
            visit_adj(v, [&](auto, auto m, auto& dx, auto d2)
                         { g -= dx * (att_kernel_grad(d2, _att_exp) * m); });
        return g;
    }

    // if true, the forces are computed exactly, without the quad-trees, by
    // iterating over the positions and weights of all the vertices
    bool _exact = false;
    typedef std::tuple<p_t, double> source_t;
    std::vector<source_t> _src;
    p_t _cm{};

    // kd-tree of the positions, for the distance constraints
    struct coord_iterator
    {
        auto operator()(const auto& p) const { return p.begin(); }
        auto operator()(const auto& p, int) const { return p.end(); }
    };

    using kd_traits_base_t =
        CGAL::Search_traits<double, p_t, const double*, coord_iterator,
                            CGAL::Dimension_tag<2>>;
    using kd_traits_t =
        CGAL::Search_traits_adapter<size_t, xmap_t, kd_traits_base_t>;

    // The last template argument (points cache) must stay Tag_false here.
    // With Tag_true, build() copies all coordinates into a contiguous array,
    // which speeds up queries, but invalidate_build() does not clear that
    // array (checked in CGAL 5.6 through current master), so every rebuild
    // after the first reads stale coordinates. Only use the cache with trees
    // that are built once, e.g. by constructing a fresh kd_tree_t for every
    // rebuild.
    using kd_tree_t =
        CGAL::Kd_tree<kd_traits_t, CGAL::Sliding_midpoint<kd_traits_t>,
                      CGAL::Tag_true, CGAL::Tag_false>;

    kd_tree_t _kd_tree;

    // sum_v c_v^2 of the minimum distance, by which its violation is
    // normalized in eval_constraints() (and likewise gmin_C of every level)
    double _nmin_C = 0;

    // whether any distance constraint is active
    bool has_dist()
    {
        return _nmu > 0 || _has_gmu;
    }

    void init_dist_scales()
    {
        for (auto v : _vs)
            _nmin_C += pow2(_nmin_dist[v]);
        for (size_t l = 0; l < _L; ++l)
        {
            // gmin_dist is only checked at the levels where it is used
            if (_gmu[l] == 0)
                continue;
            auto& C = _levels[l].gmin_C;
            for (auto v : _vs)
                C += pow2(_gmin_dist[v][l]);
        }
    }

    // distance constraints (neighbourhood and group)
    std::vector<p_t> _dgrad;

    // rank constraints
    std::vector<p_t> _rgrad;

    // overlap constraints
    std::vector<p_t> _ograd;

    // hierarchy levels
    size_t _L = 0;
    bool _has_gmu = false;
    bool _has_omu = false;
    bool _has_pmu = false;
    bool _has_group_grav = false;
    bool _has_group_rep = false;

    // Vector arithmetic
    // -----------------

    double dot(const auto& a, const auto& b)
    {
        return a[0] * b[0] + a[1] * b[1];
    }
    double norm(const auto& a) { return std::sqrt(norm2(a)); }
    p_t sub(const auto& a, const auto& b) { return {a[0] - b[0], a[1] - b[1]}; }
    p_t scale(double s, const auto& a) { return {s * a[0], s * a[1]}; }
    // clockwise quarter turn
    p_t rperp(const auto& a) { return {a[1], -a[0]}; }

    double pow2(auto x)
    {
        return x * x;
    }

    // Augmented Lagrangian term (max(0, l + mu c)^2 - l^2) / (2 mu) of a
    // constraint c <= 0 with multiplier l and penalty mu, and its effective
    // multiplier max(0, l + mu c), the factor of the gradient of c (and the
    // updated multiplier)
    double al_eff(double l, double mu, double c)
    {
        return std::max(0., l + mu * c);
    }

    double al_energy(double l, double mu, double c)
    {
        return (pow2(al_eff(l, mu, c)) - pow2(l)) / (2 * mu);
    }

    double norm2(const auto& x)
    {
        double n2 = 0;
        for (size_t i = 0; i < 2; ++i)
            n2 += pow2(x[i]);
        return n2;
    }

    double dist2(const auto& x, const auto& y)
    {
        double d2 = 0;
        for (size_t i = 0; i < 2; ++i)
            d2 += pow2(x[i] - y[i]);
        return d2;
    }

    // Energies and gradients
    // ======================

    // Force kernels with arbitrary exponents. The common integer exponents
    // are computed without std::pow, which is much slower; since the exponent
    // is the same for all calls, the branches are perfectly predictable.

    // Attraction
    // ----------

    // The vertices joined by an edge attract each other with the energy
    //
    //     sum_e A_e k_a(d_e),   k_a(d) = (2 / a) d^a,
    //
    // (or k_0(d) = log d^2), where the sum is over the edges e = (u, v),
    // d_e = |x_u - x_v|, A_e = _eweight[e] and a = _att_exp. The vertex
    // weights do not enter the attraction (only the repulsion), except that
    // the edges of the vertices of zero weight are left out.

    // Kernel of the attraction for the squared length d2, whose force is
    // 2 d^(a-1) for every a; att_kernel_grad() gives the factor g such that
    // the gradient with respect to x is g (x - y).
    static double att_kernel(double d2, double a)
    {
        if (a == 2)
            return d2;
        if (a == 1)
            return 2 * std::sqrt(d2);
        if (a == 0)
            return std::log(d2);
        if (a == 3)
            return (2. / 3) * d2 * std::sqrt(d2);
        if (a == 4)
            return 0.5 * d2 * d2;
        return (2 / a) * std::pow(d2, a / 2);
    }

    static double att_kernel_grad(double d2, double a)
    {
        if (a == 2)
            return 2;
        if (a == 1)
            return 2 / std::sqrt(d2);
        if (a == 0)
            return 2 / d2;
        if (a == 3)
            return 2 * std::sqrt(d2);
        if (a == 4)
            return 2 * d2;
        return 2 * std::pow(d2, a / 2 - 1);
    }

    // Attraction energy, with the same weights as the gradient (every edge
    // is in the adjacency of both endpoints, and is counted once); edges with
    // coinciding endpoints (e.g. self-loops) are skipped, since they would be
    // singular for att_exp < 2
    double att_energy()
    {
        double U = 0;
        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
            reduction(+:U)
        parallel_loop_no_spawn
            (_vs,
             [&](auto, auto v)
             {
                 visit_adj(v,
                           [&](auto u, auto m, auto&, auto d2)
                           {
                               if (u > v)
                                   U += att_kernel(d2, _att_exp) * m;
                           });
             });
        return U;
    }

    // Repulsion
    // ---------

    // All pairs of vertices repel each other with the energy
    //
    //     K sum_{i != j} m_i m_j phi_p(d_ij),   phi_p(d) = d^-p / p,
    //
    // (or phi_0(d) = -log d), where the sum is over ordered pairs (i.e. every
    // pair is counted from both endpoints), d_ij = |x_i - x_j|, and
    // p = _rep_exp. The masses m_i = w_i _rweight[i] are the weights of the
    // vertices, scaled e.g. by the radii of the groups they represent in a
    // coarse level (see rep_mass()). It is computed with the Barnes-Hut
    // approximation, unless _exact is true.

    double rep_mass(size_t v)
    {
        return _vweight[v] * _rweight[v];
    }

    // Kernel of the repulsion for the squared distance d2, whose force is
    // d^-(p+1) for every p; rep_kernel_grad() gives the factor g such that
    // the gradient with respect to x is g (x - y).
    static double rep_kernel(double d2, double p)
    {
        if (p == 1)
            return 1 / std::sqrt(d2);
        if (p == 2)
            return 0.5 / d2;
        if (p == 0)
            return -0.5 * std::log(d2);
        if (p == 3)
            return 1 / (3 * d2 * std::sqrt(d2));
        return std::pow(d2, -p / 2) / p;
    }

    static double rep_kernel_grad(double d2, double p)
    {
        if (p == 1)
        {
            double inv = 1 / std::sqrt(d2);
            return -inv * inv * inv;
        }
        if (p == 2)
            return -1 / (d2 * d2);
        if (p == 0)
            return -1 / d2;
        if (p == 3)
        {
            double inv = 1 / std::sqrt(d2);
            double inv2 = inv * inv;
            return -inv2 * inv2 * inv;
        }
        return -std::pow(d2, -p / 2) / d2;
    }

    // The repulsion of all the sources of a point (see visit_sources()) is
    // obtained by collecting them first in a buffer of the thread, and then
    // summing the kernels in a separate loop, with the exponent resolved
    // outside of it, which can be vectorized. The sources that coincide with
    // the point (e.g. itself) are skipped.

    struct sources_t
    {
        std::vector<double> x, y, w;
    };

    sources_t& gather_sources(const p_t& x)
    {
        thread_local sources_t src;
        return gather_sources(_qt, _src, x, src);
    }

    // the sources of the group tree of v at level l, in a separate buffer
    sources_t& gather_group_sources(size_t v, size_t l)
    {
        thread_local sources_t src;
        auto& lev = _levels[l];
        auto r = lev.bidx[v];
        return gather_sources(lev.qt[r], lev.src[r], _x[v], src);
    }

    sources_t& gather_sources(auto& qt, auto& srcs, const p_t& x,
                              sources_t& src)
    {
        src.x.clear();
        src.y.clear();
        src.w.clear();
        visit_sources(qt, srcs, x,
                      [&](auto& lpos, auto&& w, auto...)
                      {
                          src.x.push_back(lpos[0]);
                          src.y.push_back(lpos[1]);
                          src.w.push_back(w);
                      });
        return src;
    }

    // calls f(k) with the kernel k(d2) of exponent p, which is resolved at
    // compile time for the default p = 1
    static auto rep_dispatch(double p, auto&& kernel, auto&& f)
    {
        if (p == 1)
            return f([&](double d2) { return kernel(d2, 1.); });
        return f([&](double d2) { return kernel(d2, p); });
    }

    // sum of the repulsion energies of x with the sources
    double rep_energy_sum(const sources_t& src, const p_t& x, double p)
    {
        return rep_dispatch
            (p, [](double d2, double q) { return rep_kernel(d2, q); },
             [&](auto&& k)
             {
                 double U = 0;
                 auto xs = src.x.data(), ys = src.y.data(), ws = src.w.data();
                 size_t n = src.x.size();
                 #pragma omp simd reduction(+:U)
                 for (size_t j = 0; j < n; ++j)
                 {
                     double dx = x[0] - xs[j];
                     double dy = x[1] - ys[j];
                     double d2 = dx * dx + dy * dy;
                     U += (d2 > 0) ? k(d2) * ws[j] : 0.;
                 }
                 return U * _K;
             });
    }

    // sum of the repulsion gradients of x with the sources (the factor 2 is
    // because energy() visits every pair from both endpoints)
    p_t rep_grad_sum(const sources_t& src, const p_t& x, double p)
    {
        return rep_dispatch
            (p, [](double d2, double q) { return rep_kernel_grad(d2, q); },
             [&](auto&& k)
             {
                 double gx = 0, gy = 0;
                 auto xs = src.x.data(), ys = src.y.data(), ws = src.w.data();
                 size_t n = src.x.size();
                 #pragma omp simd reduction(+:gx, gy)
                 for (size_t j = 0; j < n; ++j)
                 {
                     double dx = x[0] - xs[j];
                     double dy = x[1] - ys[j];
                     double d2 = dx * dx + dy * dy;
                     double c = (d2 > 0) ? k(d2) * ws[j] : 0.;
                     gx += dx * c;
                     gy += dy * c;
                 }
                 return p_t{gx, gy} * (2 * _K);
             });
    }

    // Central pull
    // ------------

    // All vertices are attracted to their center of mass c by the energy
    //
    //     g sum_i m_i |x_i - c|^2,
    //
    // where g = _center_pull, with the masses m_i of the repulsion, so that c
    // is the center of mass of the repulsion, and a vertex is held against the
    // repulsion of the others independently of its mass.

    double pull_energy(const auto& x, const auto& cm)
    {
        return _center_pull * dist2(x, cm);
    }

    auto pull_grad(const auto& x, const auto& cm)
    {
        return (x - cm) * (2 * _center_pull);
    }

    // Rank constraints
    // ----------------

    // The vertices are split into classes of equal rank, and every vertex v
    // must lie at least _rmin_dist[v] above (in the y direction) all the
    // vertices of the next higher class, which orders all the classes. The
    // violation of v is
    //
    //     V_v = sum_u max(0, y_u + _rmin_dist[v] - y_v) >= 0,
    //
    // over the vertices u of the next higher class, imposed by the augmented
    // Lagrangian term w_v Phi(V_v; _rlambda[v], _rmu), with one multiplier per
    // vertex, as the minimum distances. Since only the y coordinates enter,
    // V_v and the reactions on the u are obtained for every pair of classes in
    // O(n log n), from the sorted coordinates and their prefix sums, instead
    // of from every pair of vertices.

    std::vector<std::vector<size_t>> _rclass;  // slots in _vs of every class,
                                               // by increasing rank
    std::vector<double> _rV;                   // V_v of every slot
    std::vector<size_t> _rn;                   // number of terms of V_v

    void init_ranks()
    {
        std::vector<size_t> idx(_vs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::ranges::sort(idx, {}, [&](auto i) { return _rank[_vs[i]]; });
        for (size_t k = 0; k < idx.size(); ++k)
        {
            if (k == 0 || _rank[_vs[idx[k]]] != _rank[_vs[idx[k - 1]]])
                _rclass.emplace_back();
            _rclass.back().push_back(idx[k]);
        }
        _rV.assign(_vs.size(), 0);
        _rn.assign(_vs.size(), 0);
        _rgrad.resize(_vs.size());
    }

    // y coordinates of the class c, sorted, together with a per-vertex value
    // f(i), and the prefix sums of both
    struct rank_sorted_t
    {
        std::vector<std::array<double, 2>> yf;
        std::vector<double> Sy, Sf;
    };

    void rank_sort(size_t c, rank_sorted_t& rs, auto&& f)
    {
        auto& B = _rclass[c];
        rs.yf.resize(B.size());
        for (auto [k, i] : std::views::enumerate(B))
            rs.yf[k] = {_x[_vs[i]][1], f(i)};
        std::ranges::sort(rs.yf);
        rs.Sy.assign(B.size() + 1, 0);
        rs.Sf.assign(B.size() + 1, 0);
        for (size_t k = 0; k < B.size(); ++k)
        {
            rs.Sy[k + 1] = rs.Sy[k] + rs.yf[k][0];
            rs.Sf[k + 1] = rs.Sf[k] + rs.yf[k][1];
        }
    }

    // index of the first vertex of rs strictly above t (larger y), i.e. the
    // vertices of the terms of V_v for t = y_v - rmin_v
    static size_t rank_first(const rank_sorted_t& rs, double t)
    {
        return std::ranges::upper_bound(rs.yf, t, {},
                                        [](auto& x) { return x[0]; }) -
            rs.yf.begin();
    }

    // computes _rV and _rn; with f, also calls f(i, k0, rs) for every vertex
    // i of every class but the last, where the terms of V_i are the vertices
    // k >= k0 of the next class sorted in rs (with the values g(j) of its
    // vertices)
    void rank_viol(auto&& g, auto&& f)
    {
        size_t C = _rclass.size();
        if (C > 0)
        {
            for (auto i : _rclass[C - 1])
            {
                _rV[i] = 0;
                _rn[i] = 0;
            }
        }
        std::vector<size_t> cs(C > 0 ? C - 1 : 0);
        std::iota(cs.begin(), cs.end(), 0);
        parallel_loop
            (cs,
             [&](auto, auto c)
             {
                 thread_local rank_sorted_t rs;
                 rank_sort(c + 1, rs, g);
                 size_t n = _rclass[c + 1].size();
                 for (auto i : _rclass[c])
                 {
                     auto v = _vs[i];
                     double r = _rmin_dist[v];
                     size_t k0 = rank_first(rs, _x[v][1] - r);
                     _rn[i] = n - k0;
                     _rV[i] = std::max(0., rs.Sy[n] - rs.Sy[k0] +
                                       (n - k0) * (r - _x[v][1]));
                     f(i, k0, rs);
                 }
             });
    }

    void rank_viol()
    {
        rank_viol([](auto) { return 0.; }, [](auto&&...) {});
    }

    double rank_energy(size_t i, size_t v)
    {
        return al_energy(_rlambda[v], _rmu, _rV[i]);
    }

    void rank_grads()
    {
        rank_viol();
        parallel_loop(_vs, [&](auto i, auto) { _rgrad[i] = p_t{}; });

        std::vector<size_t> cs(_rclass.empty() ? 0 : _rclass.size() - 1);
        std::iota(cs.begin(), cs.end(), 0);
        parallel_loop
            (cs,
             [&](auto, auto c)
             {
                 // the force on v, w_v leff_v per term, and its reaction on
                 // the vertices u of the next class with y_u > t_v, i.e. the
                 // sum over the t_v < y_u, from the sorted t_v
                 thread_local std::vector<std::array<double, 2>> ta;
                 thread_local std::vector<double> Sa;
                 auto& A = _rclass[c];
                 ta.clear();
                 for (auto i : A)
                 {
                     auto v = _vs[i];
                     double a = _vweight[v] *
                         al_eff(_rlambda[v], _rmu, _rV[i]);
                     if (a == 0 || _rn[i] == 0)
                         continue;
                     atomic_inc(_rgrad[i], p_t{0., -a * double(_rn[i])});
                     ta.push_back({_x[v][1] - _rmin_dist[v], a});
                 }
                 if (ta.empty())
                     return;
                 std::ranges::sort(ta);
                 Sa.assign(ta.size() + 1, 0);
                 for (size_t k = 0; k < ta.size(); ++k)
                     Sa[k + 1] = Sa[k] + ta[k][1];
                 for (auto j : _rclass[c + 1])
                 {
                     double y = _x[_vs[j]][1];
                     size_t k = std::ranges::lower_bound(ta, y, {},
                                                         [](auto& x)
                                                         { return x[0]; }) -
                         ta.begin();
                     if (k > 0)
                         atomic_inc(_rgrad[j], p_t{0., Sa[k]});
                 }
             });
    }

    // Separation of the centers of mass of the rank classes: the center of
    // mass Y_c (weighted by the vertex weights) of every class must lie at
    // least _rcm_dist above that of the next higher class, i.e.
    //
    //     g_c = Y_{c+1} + _rcm_dist - Y_c <= 0,
    //
    // imposed by Phi(g_c; _rclambda[c], _rcmu), with one multiplier per pair
    // of adjacent classes. The gradient of Y_c is w_v / W_c for its members.
    //
    // Since this squeezes the layout in the y direction, the centers of mass
    // of all the classes can also be required to lie at least
    // _rcm_xdist = d apart in the x direction (in any order), i.e.
    //
    //     c_ab = d^2 - (X_a - X_b)^2 <= 0,
    //
    // imposed by Phi(c_ab; lambda_ab, _rcmxu) for every pair a < b, with one
    // multiplier per pair, so that the energy is continuously differentiable
    // (with one multiplier per class and a summed violation, as the minimum
    // distances, the kinks of max(0, c_ab) made the step collapse). Only the
    // pairs closer than d (whose terms do not vanish without a multiplier)
    // and those with a nonzero multiplier, stored in _rcxlambda keyed by
    // a C + b, are evaluated; the close pairs are found from the classes
    // sorted by X, in O(C log C) plus their number. Coinciding centers are
    // pushed apart in a fixed direction.

    std::vector<double> _rclambda;  // multipliers, one per adjacent pair
    std::flat_map<size_t, double> _rcxlambda; // nonzero multipliers, per pair
    std::vector<double> _rcX, _rcY, _rcW; // centers of mass and weights of
                                          // the classes
    std::vector<p_t> _rcgrad;

    void rcm_setup()
    {
        size_t C = _rclass.size();
        _rcX.assign(C, 0);
        _rcY.assign(C, 0);
        _rcW.assign(C, 0);
        for (size_t c = 0; c < C; ++c)
        {
            for (auto i : _rclass[c])
            {
                auto v = _vs[i];
                _rcX[c] += _vweight[v] * _x[v][0];
                _rcY[c] += _vweight[v] * _x[v][1];
                _rcW[c] += _vweight[v];
            }
            _rcX[c] /= _rcW[c];
            _rcY[c] /= _rcW[c];
        }
        _rclambda.resize(C > 0 ? C - 1 : 0);
    }

    std::vector<size_t> _rcx_order;  // the classes sorted by X

    // calls f(a, b, lambda) for every pair a < b with a nonzero multiplier,
    // and every other pair closer than _rcm_xdist in x (with lambda = 0)
    void rcmx_pairs(auto&& f)
    {
        size_t C = _rclass.size();
        for (auto [k, l] : _rcxlambda)
            f(k / C, k % C, l);
        auto& o = _rcx_order;
        o.resize(C);
        std::iota(o.begin(), o.end(), 0);
        std::ranges::sort(o, {}, [&](auto c) { return _rcX[c]; });
        for (size_t k = 0; k < C; ++k)
        {
            for (size_t j = k + 1;
                 j < C && _rcX[o[j]] - _rcX[o[k]] < _rcm_xdist; ++j)
            {
                auto [a, b] = std::minmax(o[k], o[j]);
                if (!_rcxlambda.contains(a * C + b))
                    f(a, b, 0.);
            }
        }
    }

    // c_ab, and its derivative with respect to X_a (whose negative is that
    // with respect to X_b), for a < b; coinciding centers move apart with a
    // to the right
    std::array<double, 2> rcmx_c(size_t a, size_t b)
    {
        double dx = _rcX[a] - _rcX[b];
        return {_rcm_xdist - std::abs(dx), (dx < 0) ? 1. : -1.};
    }

    double rcm_viol(size_t c)
    {
        return _rcY[c + 1] + _rcm_dist - _rcY[c];
    }

    double rcm_energy()
    {
        rcm_setup();
        double U = 0;
        if (_rcmu > 0)
        {
            for (size_t c = 0; c < _rclambda.size(); ++c)
                U += al_energy(_rclambda[c], _rcmu, rcm_viol(c));
        }
        if (_rcmxu > 0)
        {
            rcmx_pairs([&](auto a, auto b, auto l)
                       {
                           U += al_energy(l, _rcmxu, rcmx_c(a, b)[0]);
                       });
        }
        return U;
    }

    // adds g * w_v / W_c to the component j of the gradient of the members
    // v of class c
    void rcm_scatter(size_t c, size_t j, double g)
    {
        for (auto i : _rclass[c])
            _rcgrad[i][j] += g * _vweight[_vs[i]] / _rcW[c];
    }

    void rcm_grads()
    {
        rcm_setup();
        _rcgrad.assign(_vs.size(), p_t{});
        if (_rcmu > 0)
        {
            for (size_t c = 0; c < _rclambda.size(); ++c)
            {
                double leff = al_eff(_rclambda[c], _rcmu, rcm_viol(c));
                if (leff == 0)
                    continue;
                rcm_scatter(c, 1, -leff);
                rcm_scatter(c + 1, 1, leff);
            }
        }
        if (_rcmxu > 0)
        {
            // the derivatives with respect to the centers of mass, which are
            // then distributed over the members
            std::vector<double> gX(_rclass.size());
            rcmx_pairs([&](auto a, auto b, auto l)
                       {
                           auto [c, dc] = rcmx_c(a, b);
                           double leff = al_eff(l, _rcmxu, c);
                           gX[a] += leff * dc;
                           gX[b] -= leff * dc;
                       });
            for (size_t c = 0; c < gX.size(); ++c)
            {
                if (gX[c] != 0)
                    rcm_scatter(c, 0, gX[c]);
            }
        }
    }

    boost::python::list get_rclambda()
    {
        boost::python::list l;
        for (auto x : _rclambda)
            l.append(x);
        return l;
    }

    // the nonzero multipliers of the x separation, as (a, b, lambda)
    boost::python::list get_rcxlambda()
    {
        boost::python::list l;
        size_t C = _rclass.size();
        for (auto [k, x] : _rcxlambda)
            l.append(boost::python::make_tuple(k / C, k % C, x));
        return l;
    }

    // sets the multipliers lambda from a single value or one value per
    // constraint
    void set_rcm_multipliers(std::vector<double>& lambda, double mu,
                             boost::python::object o, const char* name)
    {
        namespace py = boost::python;
        if (mu == 0)
            return;
        rcm_setup();
        py::extract<double> x(o);
        if (x.check())
        {
            std::ranges::fill(lambda, x());
            return;
        }
        if (std::cmp_not_equal(py::len(o), lambda.size()))
            throw ValueException(std::format("{} has {} values, expected {}",
                                             name, py::len(o),
                                             lambda.size()));
        for (size_t c = 0; c < lambda.size(); ++c)
            lambda[c] = py::extract<double>(o[c]);
    }

    void set_rclambda(boost::python::object o)
    {
        set_rcm_multipliers(_rclambda, _rcmu, o, "rclambda");
    }

    // accepts zero (for all pairs), or a list of (a, b, lambda)
    void set_rcxlambda(boost::python::object o)
    {
        namespace py = boost::python;
        _rcxlambda.clear();
        py::extract<double> x(o);
        if (x.check())
        {
            if (x() != 0)
                throw ValueException("rcxlambda must be zero or a list of "
                                     "(a, b, lambda)");
            return;
        }
        size_t C = _rclass.size();
        for (int i = 0; i < py::len(o); ++i)
        {
            size_t a = py::extract<size_t>(o[i][0]);
            size_t b = py::extract<size_t>(o[i][1]);
            double l = py::extract<double>(o[i][2]);
            if (a >= b || b >= C)
                throw ValueException(std::format("invalid pair of ranks "
                                                 "({}, {})", a, b));
            if (l != 0)
                _rcxlambda[a * C + b] = l;
        }
    }

    // Distance constraints
    // --------------------

    // Every vertex v must be at least c_v away from all other vertices (with
    // c_v = _nmin_dist[v]), and, at every level l, from all the vertices of
    // other groups (with c_v = _gmin_dist[v][l]). The violation of a vertex
    // is
    //
    //     V_v = sum_{u: d_vu < c_v} (c_v^2 - d_vu^2) >= 0,
    //
    // imposed by the augmented Lagrangian term
    //
    //     (max(0, lambda + mu V)^2 - lambda^2) / (2 mu),
    //
    // with V = V_v, lambda = _nlambda[v] and mu = nmu_v(v) for the minimum
    // distance, i.e. one multiplier per vertex, so that every vertex is
    // pushed away from its own neighbours as strongly as needed, and with
    // V = sum_v V_v, lambda = glambda and mu = _gmu[l] for the group distance
    // of every level. The violations are not squared, since the gradient of
    // a squared violation vanishes at the boundary d = c, so that no finite
    // multiplier would keep the pairs at the minimum distance, and the
    // updates of the multipliers alone would converge ever more slowly.

    double dist_viol(size_t v, size_t u, double c)
    {
        return std::max(0., c * c - dist2(_x[v], _x[u]));
    }

    // a unit vector that is fixed for the pair (v, u), and opposite for the
    // reversed pair, used as the direction of coinciding vertices
    static p_t pair_dir(size_t v, size_t u)
    {
        uint64_t h = (std::min(v, u) * 0x9E3779B97F4A7C15ULL) ^
            std::max(v, u);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        double theta = 2 * std::numbers::pi * double(h >> 11) * 0x1p-53;
        double sgn = (v < u) ? 1. : -1.;
        return p_t{std::cos(theta), std::sin(theta)} * sgn;
    }

    // Gradient on v of the term max(0, c^2 - |x_v - x_u|^2) owned by v; the
    // reaction on u is its negative. The gradient vanishes for coinciding
    // vertices (which the repulsion skips as well), which would then never be
    // separated, so that they get instead the gradient 2 c e of a pair at
    // the distance c along a direction e that is fixed for the pair (and
    // opposite for the reversed pair), along which the term decreases.
    p_t dist_grad(size_t v, size_t u, double c)
    {
        p_t grad;
        auto& x_v = _x[v];
        auto& x_u = _x[u];
        grad[0] = x_u[0] - x_v[0];
        grad[1] = x_u[1] - x_v[1];
        auto d2 = norm2(grad);
        if (!(c * c > d2))
            return p_t{};
        if (d2 == 0)
            return pair_dir(v, u) * (2 * c);
        return grad * 2.;
    }

    // Penalty of the minimum distance of vertex v. A pair at the distance
    // d ~ c that is pushed together by the force F, by x, has the violation
    // ~ 2 c x, so that the penalty force is ~ 4 mu c^2 x, which keeps x a
    // fraction of c if mu ~ F / c^3. The penalty is therefore given relative
    // to the force, as mu = _nmu F_v / c^3, so that _nmu has the same
    // meaning at every scale, where F_v is the magnitude of the force on v
    // without the constraints, as obtained by init_multipliers(), but at
    // least that of the repulsion between two vertices at the distance c,
    // K m_v^2 c^-(p+1) (which is also used before init_multipliers()).
    std::vector<double> _nmu_f;     // F_v, indexed by vertex

    // F_v, so that it can be restored in a copy (empty before
    // init_multipliers())
    const std::vector<double>& get_nmu_force()
    {
        return _nmu_f;
    }

    void set_nmu_force(std::vector<double> f)
    {
        if (!f.empty() && f.size() != num_vertices(_g))
            throw ValueException(std::format("nmu_force has {} values, "
                                             "expected {}", f.size(),
                                             num_vertices(_g)));
        _nmu_f = std::move(f);
    }

    double nmu_v(size_t v)
    {
        double c = _nmin_dist[v];
        if (!(c > 0))
            return _nmu;
        auto m = rep_mass(v);
        double F = _K * m * m / std::pow(c, _rep_exp + 1);
        if (!_nmu_f.empty())
            F = std::max(F, _nmu_f[v]);
        if (!(F > 0))       // e.g. a vertex without mass, not yet measured
            return _nmu;
        return _nmu * F / (c * c * c);
    }

    // Scatters the gradient of the violations of v with its neighbours us,
    // multiplied by leff, to v and (as the reaction) to the neighbours, since
    // every source only searches its own radius c_v, so that u's term is not
    // visible from v when c_v < d < c_u.
    void dist_scatter(std::vector<p_t>& out, size_t i, size_t v, double c,
                      double leff, const std::vector<size_t>& us)
    {
        p_t self{};
        for (auto u : us)
        {
            auto g = dist_grad(v, u, c) * leff;
            self += g;
            atomic_inc(out[_vpos[u]], g * -1.);
        }
        atomic_inc(out[i], self);
    }

    // Gradient of the group distance constraint at level l, multiplied by
    // leff, accumulated in out.
    void gdist_grads(std::vector<p_t>& out, size_t l, double leff)
    {
        if (!(leff > 0))
            return;
        auto& bidx = _levels[l].bidx;
        parallel_loop
            (_vs,
             [&](auto i, auto v)
             {
                 thread_local std::vector<size_t> us;
                 us.clear();
                 double c = _gmin_dist[v][l];
                 visit_radius(v, c,
                              [&](auto u)
                              {
                                  if (bidx[u] != bidx[v])
                                      us.push_back(u);
                              });
                 dist_scatter(out, i, v, c, leff, us);
             });
    }

    void dist_grads()
    {
        parallel_loop
            (_vs,
             [&](auto i, auto)
             {
                 _dgrad[i] = p_t{};
             });

        // the minimum distance, whose effective multiplier of every vertex
        // is obtained from the same neighbours as its gradient
        if (_nmu > 0)
        {
            parallel_loop
                (_vs,
                 [&](auto i, auto v)
                 {
                     thread_local std::vector<size_t> us;
                     us.clear();
                     double c = _nmin_dist[v];
                     double V = 0;
                     visit_radius(v, c,
                                  [&](auto u)
                                  {
                                      V += dist_viol(v, u, c);
                                      us.push_back(u);
                                  });
                     double leff = al_eff(_nlambda[v], nmu_v(v), V);
                     if (leff > 0)
                         dist_scatter(_dgrad, i, v, c, leff, us);
                 });
        }
        for (size_t l = 0; l < _L; ++l)
        {
            auto& lev = _levels[l];
            if (_gmu[l] > 0 && lev.glambda_eff > 0)
                gdist_grads(_dgrad, l, lev.glambda_eff);
        }
    }

    // violation sum_v V_v of the group distance at level l
    double g_viol(size_t l)
    {
        double V = 0;
        auto& bidx = _levels[l].bidx;

        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
            reduction(+:V)
        parallel_loop_no_spawn
            (_vs,
             [&](auto, auto v)
             {
                 auto c = _gmin_dist[v][l];
                 visit_radius(v, c,
                              [&](auto u)
                              {
                                  if (bidx[u] != bidx[v])
                                      V += dist_viol(v, u, c);
                              });
             });
        return V;
    }

    // violation V_v of the minimum distance of vertex v
    double n_viol(size_t v)
    {
        double V = 0;
        visit_radius(v, _nmin_dist[v],
                     [&](auto u)
                     {
                         V += dist_viol(v, u, _nmin_dist[v]);
                     });
        return V;
    }

    // multipliers of the group distance constraint, one per level
    boost::python::list get_glambda()
    {
        boost::python::list l;
        for (auto& lev : _levels)
            l.append(lev.glambda);
        return l;
    }

    // accepts a single value for all levels, or one value per level
    void set_glambda(boost::python::object o)
    {
        namespace py = boost::python;
        py::extract<double> x(o);
        if (x.check())
        {
            for (auto& lev : _levels)
                lev.glambda = x();
            return;
        }
        if (std::cmp_not_equal(py::len(o), _L))
            throw ValueException(std::format("glambda has {} values, expected "
                                             "{}", py::len(o), _L));
        for (size_t l = 0; l < _L; ++l)
            _levels[l].glambda = py::extract<double>(o[l]);
    }

    // Group gravity
    // -------------

    // The members of every group r at every level l attract each other with
    // the energy
    //
    //     g_l f_r K sum_{i != j in r} m_i m_j log(d_ij + c),
    //
    // over ordered pairs, where g_l = _group_grav[l], f_r = _group_grav_fac[l]
    // for the label of r, c = _group_grav_offset, which keeps it finite at
    // coinciding positions, and m_i are the masses of the repulsion, since
    // it is computed with the same quad-trees.
    //
    // Logarithmic attraction: the force g K / (s + c) decays more slowly than
    // the K / s^2 repulsion, so it dominates beyond s ~ 1 / g, and distant
    // parts of a group always attract each other. It is computed with the
    // sources of the tree of the group (see group_grav_energy_sum()).

    // sums of the group gravity energies and gradients of x with the
    // sources (see rep_energy_sum()); the factor 2 of the gradient is because
    // energy() visits every pair from both endpoints
    double group_grav_energy_sum(const sources_t& src, const p_t& x,
                                 double gg)
    {
        double c = _group_grav_offset;
        double U = 0;
        auto xs = src.x.data(), ys = src.y.data(), ws = src.w.data();
        size_t n = src.x.size();
        #pragma omp simd reduction(+:U)
        for (size_t j = 0; j < n; ++j)
        {
            double dx = x[0] - xs[j];
            double dy = x[1] - ys[j];
            double d2 = dx * dx + dy * dy;
            U += (d2 > 0) ? std::log(std::sqrt(d2) + c) * ws[j] : 0.;
        }
        return U * (gg * _K);
    }

    p_t group_grav_grad_sum(const sources_t& src, const p_t& x, double gg)
    {
        double c = _group_grav_offset;
        double gx = 0, gy = 0;
        auto xs = src.x.data(), ys = src.y.data(), ws = src.w.data();
        size_t n = src.x.size();
        #pragma omp simd reduction(+:gx, gy)
        for (size_t j = 0; j < n; ++j)
        {
            double dx = x[0] - xs[j];
            double dy = x[1] - ys[j];
            double d2 = dx * dx + dy * dy;
            double s = std::sqrt(d2);
            double k = (d2 > 0) ? ws[j] / (s * (s + c)) : 0.;
            gx += dx * k;
            gy += dy * k;
        }
        return p_t{gx, gy} * (2 * _K * gg);
    }

    // Group repulsion
    // ---------------

    // Different groups repel each other with the energy
    //
    //     K sum_l rho_l sum_{i, j in different groups at l}
    //         m_i m_j phi_q(d_ij),
    //
    // over ordered pairs, where rho_l = _group_rep[l], q = _group_rep_exp and
    // phi_q is the kernel of the repulsion (see above), so that the groups of
    // every level are spaced apart from all the others (not only from their
    // siblings). Writing R_l for the repulsion of a vertex from the members
    // of its own group at level l, and R_L for its total repulsion, the
    // repulsion from the other groups at level l is R_L - R_l, so that every
    // vertex gets R_L multiplied by sum_l rho_l (on top of the general
    // repulsion), and R_l by -rho_l, computed with the same quad-trees as the
    // group gravity.

    // coefficient of R_l, the repulsion from the own group at level l
    double group_rep_coef(size_t l)
    {
        return -_group_rep[l];
    }

    // coefficient of the total repulsion, and whether the group repulsion
    // has its own exponent, set by init_levels()
    double _group_rep_top = 0;
    bool _group_rep_sep = false;

    // group gravity of the group of v at level l
    double group_grav(size_t v, size_t l)
    {
        auto& lev = _levels[l];
        return _group_grav[l] * lev.ggfac[lev.bidx[v]];
    }

    // Group confinement
    // -----------------

    // The members of a group r at a level l with confinement are attracted
    // to its center of mass x_r by the potential
    //
    //     gamma_r * sum_i w_i (|x_i - x_r| / R_r)^a,
    //
    // with a = _group_conf_exp and R_r given by _group_conf_R. Since x_r
    // depends on the positions of all members, the gradient for member j is
    // w_j (t_j - T_r / w_r), where t_j is the gradient of the potential of j
    // for fixed x_r, T_r the sum of w_i t_i over the members, and w_r their
    // total weight.
    //
    // The strength gamma_r is chosen so that, in the absence of other forces,
    // the repulsion between the members spreads them over the ball of radius
    // R_r. At equilibrium, the virial theorem gives
    //
    //     gamma_r a sum_i w_i (r_i / R_r)^a = 2 K sum_{i<j} m_i m_j d_ij^-p,
    //
    // with p = _rep_exp and the masses m_i of the repulsion (the factor 2
    // since the repulsion energy counts every pair from both endpoints),
    // which for members spread uniformly over the ball becomes
    //
    //     gamma_r = K c_p (a + 2) / (2 a R_r^p) (m_r^2 - q_r) / w_r,
    //
    // where m_r and q_r are the sum of the masses and of their squares, and
    // c_p the mean of d^-p for two points uniformly distributed in the unit
    // disk.
    //
    // If _group_spring[l] > 0, the strength is instead chosen so that the
    // pull of the edges that leave the group displaces its members by R_r / G
    // on average, with G = _group_spring[l], and is updated with
    // update_group_springs() (e.g. between sweeps, so that the energy is well
    // defined during each sweep). The edges of a member i leaving the group
    // pull it with the force F_i, and the part of it that deforms the group
    // (since the whole group follows the total force) is
    // dF_i = F_i - w_i sum_j F_j / w_r. The displacement of i is that at which
    // the confinement force c a r^(a-1) / R_r^a per unit weight balances
    // |dF_i| / w_i, so that, with phi_r the root mean square of the latter,
    // weighted by w_i, a displacement R_r / G requires
    //
    //     gamma_r = phi_r R_r G^(a-1) / a.
    //
    // For a = 2 (a harmonic spring), the displacements are proportional to
    // the pulls, so that the members are not pressed against a boundary.
    //
    // The pulls only keep the members of a group with enough edges together,
    // and a group with few (or none) would be held by nothing but these. The
    // strength of the packing spring is therefore added, an attraction to the
    // center of mass with the same exponent that alone would keep the
    // members, against their repulsion, in the disk of radius
    // R_p = (K / 2) sqrt(w_r), i.e. packed at the density of a single vertex
    // in a disk of radius K / 2. In the mean field, with the edges of the
    // group, its share of the virial at the radius R_r is
    //
    //     f_r = (R_r / R_p)^(p+a)
    //
    // (which is 1 for a group without edges, whose radius is R_p, see
    // hfd_mf_radii()), so that its strength is f_r times the one above,
    // which is the whole virial.

    // initializes the confinement of the groups at level l
    void init_conf(size_t l)
    {
        auto& lev = _levels[l];
        double a = _group_conf_exp;
        size_t B = lev.groups.size();
        auto& w_r = lev.w;
        std::vector<double> m_r(B), q_r(B);
        for (auto v : _vs)
        {
            auto r = lev.bidx[v];
            auto m = rep_mass(v);
            m_r[r] += m;
            q_r[r] += m * m;
        }
        double p = _rep_exp;
        disk_moment_cache c_p(p);
        for (size_t r = 0; r < B; ++r)
        {
            auto label = lev.label[r];
            double R = std::cmp_less(label, _group_conf_R[l].size()) ?
                _group_conf_R[l][label] : 0;
            double c = 0;
            if (R > 0 && lev.groups[r].size() > 1)
            {
                if (!(a > 0))
                    throw ValueException(std::format("invalid confinement "
                                                     "exponent: {}", a));
                c = _K * c_p(lev.groups[r].size()) * (a + 2) /
                    (2 * a * std::pow(R, p)) *
                    (m_r[r] * m_r[r] - q_r[r]) / w_r[r];
            }
            // with springs, the strength is set by update_group_springs(),
            // which adds the packing spring
            if (c > 0 && _group_spring[l] > 0)
            {
                double R_p = _K / 2 * std::sqrt(w_r[r]);
                lev.conf_pack.push_back(c * std::min(1., std::pow(R / R_p,
                                                                  p + a)));
                lev.has_conf = true;
                c = 0;
            }
            else
            {
                lev.conf_pack.push_back(0);
            }
            lev.conf.push_back(c);
            lev.conf_R.push_back(R);
            lev.conf_iRa.push_back(R > 0 ? std::pow(R, -a) : 0);
            lev.has_conf = lev.has_conf || c > 0;
        }
        if (lev.has_conf)
        {
            lev.conf_cm.resize(B);
            lev.conf_T.resize(B);
        }
    }

    bool has_springs(size_t l)
    {
        return _levels[l].has_conf && _group_spring[l] > 0;
    }

    // whether group r at a level with springs has one, which is when
    // init_conf() gives it a packing spring
    bool has_spring(size_t l, size_t r)
    {
        return _levels[l].conf_pack[r] > 0;
    }

    // Sets the strengths of the confinement from the current pull of the edges
    // that leave the groups, at the levels with springs (see above). With
    // _group_spring_static, they are set only once, with the pulls computed
    // as if every vertex were at the center of mass of its group at the level
    // in question, so that they depend only on the coarse layout (e.g. that
    // of the levels above, at the start of a pass of the hierarchical
    // layout). The springs of the ancestors are not taken into account:
    // although they also compress the group (its potential decomposes over
    // its children by the parallel axis theorem), they pull the centers of
    // mass of the siblings together by the same factor, so that they do not
    // help to separate them.
    void update_group_springs()
    {
        if (_springs_done)
            return;
        bool any = false;
        for (size_t l = 0; l < _L; ++l)
            any = any || has_springs(l);
        if (!any)
            return;

        // centers of mass of the groups, for the static springs
        auto& cm = _spring_cm;
        cm.resize(_L);
        if (_group_spring_static)
        {
            for (size_t l = 0; l < _L; ++l)
            {
                if (!has_springs(l))
                    continue;
                group_cm(l, cm[l]);
            }
        }

        // pull of the external edges on every vertex, at every level; every
        // vertex sums over its own edges, so that the loop is parallel
        auto& F = _spring_F;
        F.resize(_L);
        for (size_t l = 0; l < _L; ++l)
        {
            if (has_springs(l))
                F[l].assign(num_vertices(_g), p_t{});
            else
                F[l].clear();
        }
        parallel_loop
            (_vs,
             [&](auto, auto v)
             {
                 for (size_t i = _adj_begin[v]; i < _adj_end[v]; ++i)
                 {
                     auto& [u, m] = _adj[i];
                     p_t dx = sub(_x[u], _x[v]);
                     for (size_t l = 0; l < _L; ++l)
                     {
                         auto& lev = _levels[l];
                         auto r = lev.bidx[v], s = lev.bidx[u];
                         if (r == s)
                             break;
                         if (F[l].empty())
                             continue;
                         p_t d = _group_spring_static ?
                             cm[l][s] - cm[l][r] : dx;
                         double d2 = norm2(d);
                         if (d2 > 0)
                             F[l][v] += d * (att_kernel_grad(d2, _att_exp) *
                                             m);
                     }
                 }
             });

        // root mean square phi_r of the differential pull per unit weight of
        // the members of every group, and the strengths
        double a = _group_conf_exp;
        auto& Fs = _spring_Fs;
        auto& S = _spring_S;
        for (size_t l = 0; l < _L; ++l)
        {
            if (F[l].empty())
                continue;
            auto& lev = _levels[l];
            auto& w_r = lev.w;
            group_sum(l, Fs, [&](auto, auto v) { return F[l][v]; });
            group_sum(l, S,
                      [&](auto, auto v)
                      {
                          auto r = lev.bidx[v];
                          double w = _vweight[v];
                          return norm2(F[l][v] -
                                       Fs[r] * (w / w_r[r])) / w;
                      });
            double G = _group_spring[l];
            for (size_t r = 0; r < lev.groups.size(); ++r)
            {
                if (!has_spring(l, r))
                    continue;
                double phi = std::sqrt(S[r] / w_r[r]);
                lev.conf[r] = phi * lev.conf_R[r] * std::pow(G, a - 1) / a +
                    lev.conf_pack[r];
            }
        }
        _springs_done = _group_spring_static;
    }

    bool _springs_done = false;

    // Strengths of the springs of every level, indexed by group label (zero
    // for the groups without one), so that they can be restored, e.g. in a
    // copy of a state with static springs, whose strengths would otherwise
    // be computed from the positions at its construction.
    std::vector<std::vector<double>> get_group_springs()
    {
        std::vector<std::vector<double>> gamma(_L);
        for (size_t l = 0; l < _L; ++l)
        {
            if (!has_springs(l))
                continue;
            auto& lev = _levels[l];
            auto& x = gamma[l];
            x.resize(std::ranges::max(lev.label) + 1);
            for (size_t r = 0; r < lev.groups.size(); ++r)
                x[lev.label[r]] = lev.conf[r];
        }
        return gamma;
    }

    void set_group_springs(const std::vector<std::vector<double>>& gamma)
    {
        for (size_t l = 0; l < std::min(_L, gamma.size()); ++l)
        {
            if (!has_springs(l))
                continue;
            auto& lev = _levels[l];
            for (size_t r = 0; r < lev.groups.size(); ++r)
            {
                auto label = lev.label[r];
                if (has_spring(l, r) && std::cmp_less(label, gamma[l].size()))
                    lev.conf[r] = gamma[l][label];
            }
        }
        _springs_done = _group_spring_static;
    }

    // buffers of update_group_springs()
    std::vector<std::vector<p_t>> _spring_F, _spring_cm;
    std::vector<p_t> _spring_Fs;
    std::vector<double> _spring_S;

    // gradient of the confinement potential of v (per unit weight) for a
    // fixed center of mass
    p_t conf_grad(size_t v, size_t l)
    {
        auto& lev = _levels[l];
        auto r = lev.bidx[v];
        double c = lev.conf[r];
        if (c == 0)
            return p_t{};
        p_t y = p_t{_x[v][0], _x[v][1]} - lev.conf_cm[r];
        double a = _group_conf_exp;
        // the harmonic case (that of the springs) avoids std::pow()
        if (a == 2)
            return y * (2 * c * lev.conf_iRa[r]);
        double rho = std::sqrt(norm2(y));
        if (rho == 0)
            return p_t{};
        return y * (c * a * std::pow(rho, a - 2) * lev.conf_iRa[r]);
    }

    double conf_energy(size_t v, size_t l)
    {
        auto& lev = _levels[l];
        auto r = lev.bidx[v];
        double c = lev.conf[r];
        if (c == 0)
            return 0;
        p_t y = p_t{_x[v][0], _x[v][1]} - lev.conf_cm[r];
        double a = _group_conf_exp;
        if (a == 2)
            return c * norm2(y) * lev.conf_iRa[r];
        return c * std::pow(std::sqrt(norm2(y)), a) * lev.conf_iRa[r];
    }

    // Sums f(i, v) over the members of every group at level l into s, in
    // parallel, with an accumulator per thread (kept between calls, for every
    // value type), so that it is balanced even if the groups are few.
    std::vector<std::vector<p_t>> _gsum_p;
    std::vector<std::vector<double>> _gsum_d;

    template <class T>
    auto& group_sum_acc()
    {
        if constexpr (std::is_same_v<T, p_t>)
            return _gsum_p;
        else
            return _gsum_d;
    }

    template <class T>
    void group_sum(size_t l, std::vector<T>& s, auto&& f)
    {
        auto& lev = _levels[l];
        size_t B = lev.groups.size();
        s.assign(B, T{});
        auto& acc = group_sum_acc<T>();
        acc.resize(get_num_threads());
        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh())
        {
            auto& ls = acc[get_thread_num()];
            ls.assign(B, T{});
            parallel_loop_no_spawn
                (_vs,
                 [&](auto i, auto v)
                 {
                     ls[lev.bidx[v]] += f(i, v);
                 });
            #pragma omp critical
            for (size_t r = 0; r < B; ++r)
                s[r] += ls[r];
        }
    }

    // centers of mass of the groups at level l
    void group_cm(size_t l, std::vector<p_t>& cm)
    {
        auto& w = _levels[l].w;
        group_sum(l, cm, [&](auto, auto v) { return _x[v] * _vweight[v]; });
        for (size_t r = 0; r < cm.size(); ++r)
            cm[r] /= w[r];
    }

    // computes the centers of mass of the groups with confinement, and, if
    // grad == true, the sums of the gradients
    void conf_setup(bool grad)
    {
        for (size_t l = 0; l < _L; ++l)
        {
            auto& lev = _levels[l];
            if (!lev.has_conf)
                continue;
            group_cm(l, lev.conf_cm);
            if (!grad)
                continue;
            group_sum(l, lev.conf_T,
                      [&](auto, auto v)
                      {
                          return conf_grad(v, l) * _vweight[v];
                      });
        }
    }

    // Barnes-Hut sources
    // ------------------

    bool has_group_trees(size_t l)
    {
        return _group_grav[l] > 0 || group_rep_coef(l) != 0;
    }

    // Visits the sources of the forces acting at position x, as f(lpos, w):
    // the Barnes-Hut cells of the tree qt, or, if the forces are computed
    // exactly (in which case the trees are not built), all the vertices in
    // src.
    void visit_sources(auto& qt, auto& src, const auto& x, auto&& f)
    {
        if (_exact)
        {
            for (auto& [lpos, w] : src)
                f(lpos, w);
        }
        else
        {
            qt.barnes_hut(x, f);
        }
    }

    // center of mass of the layout
    const p_t& get_cm()
    {
        return _exact ? _cm : _qt.get_cm();
    }

    // Group overlap
    // -------------

    // The energy of the overlap of the groups is the augmented Lagrangian
    // term
    //
    //     sum_l sum_r (max(0, lambda_r + mu_l D_r)^2 - lambda_r^2) / (2 mu_l),
    //
    // where D_r is the total violation of group r (see below), lambda_r its
    // multiplier, and mu_l = _omu[l].
    //
    // The convex hulls of different groups at the same level must not
    // intersect, or come closer than _odist[l]. The violation of a pair of
    // groups is the penetration depth of their hulls plus _odist[l], where
    // the depth is obtained from the separating axis theorem, as the smallest
    // overlap of the projections of both hulls onto the normals of their
    // edges (it is negative if they are disjoint). Every group has a
    // constraint on the sum D of the positive violations with all other
    // groups, with multiplier _olambda[l][r] for group label r. The hull of a
    // group with a single member is a point, and that of a group with two (or
    // only collinear) members is a segment. The violations are not squared,
    // since inside the augmented Lagrangian that gives a force that is too
    // weak to remove shallow overlaps.

    double& olambda(size_t l, size_t r)
    {
        return _olambda[l][_levels[l].label[r]];
    }

    auto& hull_pos(size_t i)
    {
        return _x[_vs[i]];
    }

    // Penetration depth of the hulls P and Q (loops of slots), with the slots
    // of the support points h and lo, so that the depth is n . (x_h - x_lo)
    // for the axis n. The axes are the normals of the hull edges, the
    // directions of the hulls that are segments, and the coordinate axes.
    // Additional axes only make the depth smaller, so that it is still
    // positive only if the hulls intersect; the coordinate axes make it
    // consistent with the test of the bounding boxes in compute_overlaps(),
    // and the directions of the segments cap their ends. For the axis of an
    // edge (a, b), n = rperp(b - a) / |b - a| (or (b - a) / |b - a| for a
    // segment direction).
    auto sat_depth(size_t r, size_t s, const auto& P, const auto& Q)
    {
        overlap_t best{r, s, std::numeric_limits<double>::infinity(),
                       0, 0, 0, 0, overlap_t::FIXED, p_t{}};
        auto project =
            [&](const auto& loop, const p_t& n)
            {
                double pmin = std::numeric_limits<double>::infinity();
                double pmax = -pmin;
                size_t imin = 0, imax = 0;
                for (auto i : loop)
                {
                    double x = dot(n, hull_pos(i));
                    if (x < pmin)
                    {
                        pmin = x;
                        imin = i;
                    }
                    if (x > pmax)
                    {
                        pmax = x;
                        imax = i;
                    }
                }
                return std::make_tuple(pmin, imin, pmax, imax);
            };
        auto test_axis =
            [&](const p_t& n, size_t ia, size_t ib, auto kind)
            {
                auto [pmin, ipmin, pmax, ipmax] = project(P, n);
                auto [qmin, iqmin, qmax, iqmax] = project(Q, n);
                double o1 = pmax - qmin, o2 = qmax - pmin;
                if (o1 <= o2 && o1 < best.depth)
                    best = {r, s, o1, ia, ib, ipmax, iqmin, kind, n};
                else if (o2 < o1 && o2 < best.depth)
                    best = {r, s, o2, ia, ib, iqmax, ipmin, kind, n};
            };
        test_axis(p_t{1, 0}, 0, 0, overlap_t::FIXED);
        test_axis(p_t{0, 1}, 0, 0, overlap_t::FIXED);
        for (auto* A : {&P, &Q})
        {
            size_t m = A->size();
            size_t ne = (m >= 3) ? m : ((m == 2) ? 1 : 0);
            for (size_t e = 0; e < ne; ++e)
            {
                size_t ia = (*A)[e], ib = (*A)[(e + 1) % m];
                p_t l = sub(hull_pos(ib), hull_pos(ia));
                double L = norm(l);
                if (L == 0)
                    continue;
                test_axis(scale(1. / L, rperp(l)), ia, ib, overlap_t::NORMAL);
                if (m == 2)
                    test_axis(scale(1. / L, l), ia, ib, overlap_t::DIRECTION);
            }
        }
        return best;
    }

    // Gradient of the penetration depth of hit t, given as f(slot, grad) for
    // the support points and, unless the axis is fixed, the endpoints of the
    // edge that defines it. These sum to zero.
    void ohit_grads(const auto& t, auto&& f)
    {
        f(t.h, t.n);
        f(t.lo, scale(-1., t.n));
        if (t.kind == overlap_t::FIXED)
            return;
        p_t l = sub(hull_pos(t.b), hull_pos(t.a));
        double L = norm(l);
        p_t w = sub(hull_pos(t.h), hull_pos(t.lo));
        double o = dot(t.n, w);
        // derivative of n . w with respect to l
        p_t gl = (t.kind == overlap_t::NORMAL) ?
            sub(scale(-1. / L, rperp(w)), scale(o / (L * L), l)) :
            sub(scale(1. / L, w), scale(o / (L * L), l));
        f(t.b, gl);
        f(t.a, scale(-1., gl));
    }

    // Finds the pairs of groups at level l whose violation is larger than
    // -margin, and the total violation D of every group.
    void compute_overlaps(size_t l, double margin = 0)
    {
        auto& lev = _levels[l];
        auto& hulls = lev.hulls;
        size_t B = hulls.loop.size();
        double ext = _odist[l] + margin;

        // candidate pairs: the bounding boxes that intersect, after being
        // enlarged by ext / 2 on every side
        auto& boxes = _obox;
        boxes.clear();
        for (size_t r = 0; r < B; ++r)
        {
            if (hulls.loop[r].empty())
                continue;
            auto& bb = hulls.bbox[r];
            double lo[2] = {bb[0] - ext / 2, bb[1] - ext / 2};
            double hi[2] = {bb[2] + ext / 2, bb[3] + ext / 2};
            boxes.emplace_back(lo, hi, r);
        }
        auto& cand = _ocand;
        cand.clear();
        std::mutex mutex;
        auto report =
            [&](const box_t& x, const box_t& y)
            {
                std::lock_guard<std::mutex> lock(mutex);
                cand.push_back({x.info(), y.info()});
            };
        // parallel if CGAL is linked with TBB
        if (boxes.size() > get_openmp_min_thresh())
        {
#ifdef CGAL_LINKED_WITH_TBB
            tbb_num_threads tbb_threads;
#endif
            CGAL::box_self_intersection_d<CGAL::Parallel_if_available_tag>
                (boxes.begin(), boxes.end(), report);
        }
        else
        {
            CGAL::box_self_intersection_d(boxes.begin(), boxes.end(), report);
        }

        auto& hits = _ohits;
        hits.resize(cand.size());
        parallel_loop
            (cand,
             [&](auto k, auto& rs)
             {
                 auto [r, s] = rs;
                 hits[k] = sat_depth(r, s, hulls.loop[r], hulls.loop[s]);
                 hits[k].depth += _odist[l];
             });
        lev.ohits.clear();
        for (auto& t : hits)
        {
            if (t.depth + margin > 0)
                lev.ohits.push_back(t);
        }

        lev.D.assign(B, 0);
        for (auto& t : lev.ohits)
        {
            if (t.depth <= 0)
                continue;
            lev.D[t.r] += t.depth;
            lev.D[t.s] += t.depth;
        }
    }

    double overlap_energy()
    {
        double U = 0;
        for (size_t l = 0; l < _L; ++l)
        {
            auto mu = _omu[l];
            if (mu == 0)
                continue;
            auto& lev = _levels[l];
            for (size_t r = 0; r < lev.D.size(); ++r)
            {
                U += al_energy(olambda(l, r), mu, lev.D[r]);
            }
        }
        return U;
    }

    // adds to _ograd, which must be zeroed first
    void overlap_grads()
    {
        for (size_t l = 0; l < _L; ++l)
        {
            auto mu = _omu[l];
            if (mu == 0)
                continue;
            auto& lev = _levels[l];
            parallel_loop
                (lev.ohits,
                 [&](auto, auto& t)
                 {
                     if (t.depth <= 0)
                         return;
                     // the pair takes part in the constraints of both groups
                     double leff = al_eff(olambda(l, t.r), mu, lev.D[t.r]) +
                         al_eff(olambda(l, t.s), mu, lev.D[t.s]);
                     if (leff == 0)
                         return;
                     ohit_grads(t,
                                [&](auto j, const p_t& g)
                                {
                                    atomic_inc(_ograd[j], g * leff);
                                });
                 });
        }
    }

    // Two-member groups
    // -----------------

    // Two-member groups: |x_u - x_v| <= _opair_dist, which keeps other groups
    // from passing between them, imposed by the augmented Lagrangian term
    //
    //     (max(0, lambda + mu c)^2 - lambda^2) / (2 mu),
    //
    // with c = |x_u - x_v| - _opair_dist[l], lambda = _plambda[u][l] and
    // mu = _pmu[l]. The constraint gradient does not vanish at the boundary,
    // so it is exact at finite _pmu.
    double pair_violation(size_t l, size_t j)
    {
        auto& [u, v] = _levels[l].pairs[j];
        return std::sqrt(dist2(_x[u], _x[v])) - _opair_dist[l];
    }

    double pair_energy(size_t l, size_t j)
    {
        double c = pair_violation(l, j);
        double lp = _plambda[_levels[l].pairs[j][0]][l];
        return al_energy(lp, _pmu[l], c);
    }

    // The hulls of the groups and their overlaps, at the levels with the
    // overlap constraint, which are not recomputed at the same positions
    // (_ov_pos is cleared by init_multipliers(), which computes the overlaps
    // with another margin).
    std::vector<p_t> _ov_pos;

    void update_overlaps()
    {
        if (!moved(_ov_pos))
            return;
        generate_hulls();
        for (size_t l = 0; l < _L; ++l)
        {
            if (_omu[l] > 0)
                compute_overlaps(l);
        }
    }

    // Adds to _ograd, after overlap_grads(); within a level, the members of a
    // pair belong to no other pair, so no atomics are needed.
    void pair_grads()
    {
        for (size_t l = 0; l < _L; ++l)
        {
            if (_pmu[l] == 0)
                continue;
            parallel_loop
                (_levels[l].pairs,
                 [&](auto, auto& e)
                 {
                     auto& [u, v] = e;
                     p_t r = sub(_x[u], _x[v]);
                     double d = norm(r);
                     if (d == 0)
                         return;
                     double leff = al_eff(_plambda[u][l], _pmu[l],
                                          d - _opair_dist[l]);
                     if (leff == 0)
                         return;
                     r = scale(leff / d, r);
                     _ograd[_vpos[u]] += r;
                     _ograd[_vpos[v]] -= r;
                 });
        }
    }

    double pairs_energy()
    {
        double U = 0;
        for (size_t l = 0; l < _L; ++l)
        {
            if (_pmu[l] == 0)
                continue;
            size_t n = _levels[l].pairs.size();
            #pragma omp parallel if (n > get_openmp_min_thresh()) \
                reduction(+:U)
            parallel_loop_no_spawn
                (_levels[l].pairs,
                 [&](auto j, auto&)
                 {
                     U += pair_energy(l, j);
                 });
        }
        return U;
    }

    // Separation of vertices with a radius
    // ------------------------------------

    // Vertices with radii r = _barrier_r in the same group at the lowest
    // level (or all of them, if there are no groups) at distance
    // d < (1 + delta) s, with s = r_a + r_b > 0 and delta = _barrier_sep,
    // repel each other by
    //
    //     K w_a w_b / s^p log^2((d - s) / (delta s)),
    //
    // which vanishes smoothly at d = (1 + delta) s and diverges at contact. It
    // is continued quadratically shortly before its singularity, so that the
    // energy remains finite if it is crossed (e.g. from an infeasible initial
    // state).

    static constexpr double _t_min = .01;   // separation extension

    // log^2(t) for t < 1 (and zero otherwise), and its derivative, continued
    // quadratically for t < _t_min
    static std::array<double, 2> sep_pot(double t)
    {
        if (t >= 1)
            return {0, 0};
        if (t >= _t_min)
        {
            double lt = std::log(t);
            return {lt * lt, 2 * lt / t};
        }
        double lt = std::log(_t_min), dt = t - _t_min;
        double f1 = 2 * lt / _t_min, f2 = (2 - 2 * lt) / (_t_min * _t_min);
        return {lt * lt + f1 * dt + f2 * dt * dt / 2, f1 + f2 * dt};
    }

    bool _has_vsep = false;         // separation of vertices with a radius
    std::vector<p_t> _bgrad;        // separation gradient of every vertex

    void init_vsep()
    {
        for (auto v : _vs)
        {
            if (!(_barrier_r[v] >= 0))
                throw ValueException(std::format("vertex {} has invalid "
                                                 "barrier radius {}", v,
                                                 _barrier_r[v]));
        }
        _has_vsep = _barrier_sep > 0 &&
            std::ranges::any_of(_vs,
                                [&](auto v) { return _barrier_r[v] > 0; });
        if (_has_vsep)
            _bgrad.resize(_vs.size());
    }

    double sep_coef(double wa, double wb, double s)
    {
        return _K * wa * wb / std::pow(s, _rep_exp);
    }

    // the disks of the vertices with a radius, keyed by their groups (only
    // the positions change between calls)
    void vertex_disks(std::vector<p_t>& x, std::vector<double>& r,
                      std::vector<double>& w, std::vector<size_t>& key)
    {
        if (r.size() != _vs.size())
        {
            r.resize(_vs.size());
            w.resize(_vs.size());
            key.resize(_vs.size());
            for (auto [i, v] : std::views::enumerate(_vs))
            {
                r[i] = _barrier_r[v];
                w[i] = _vweight[v];
                key[i] = (_L > 0) ? _levels[0].bidx[v] : 0;
            }
        }
        x.resize(_vs.size());
        for (auto [i, v] : std::views::enumerate(_vs))
            x[i] = _x[v];
    }

    // Neighbour search among disks with centers x and radii r, restricted to
    // the disks with the same key (their group). Every group with more than
    // exact_thres disks gets its own kd-tree, and the smaller ones are
    // searched exhaustively. The searches of different disks are independent,
    // and may run in parallel.
    class near_disk_search
    {
    public:
        near_disk_search(const std::vector<std::array<double, 2>>& x,
                         const std::vector<double>& r,
                         const std::vector<size_t>& key, size_t exact_thres)
            : _x(x), _r(r), _key(key)
        {
            size_t N = x.size();
            size_t B = 0;
            for (auto k : key)
                B = std::max(B, k + 1);

            // the members of every group, contiguous in _members (counting
            // sort)
            _begin.assign(B + 1, 0);
            for (auto k : key)
                ++_begin[k + 1];
            for (size_t k = 0; k < B; ++k)
                _begin[k + 1] += _begin[k];
            _members.resize(N);
            auto pos = _begin;
            for (size_t i = 0; i < N; ++i)
                _members[pos[key[i]]++] = i;

            _r_max.assign(B, 0);
            for (size_t i = 0; i < N; ++i)
                _r_max[key[i]] = std::max(_r_max[key[i]], r[i]);

            // groups without radii have no near pairs, and need no tree
            _trees.resize(B);
            std::vector<size_t> large, huge;
            for (size_t k = 0; k < B; ++k)
            {
                size_t n = _begin[k + 1] - _begin[k];
                if (n <= exact_thres || _r_max[k] == 0)
                    continue;
                if (n > get_openmp_min_thresh())
                    huge.push_back(k);
                else
                    large.push_back(k);
            }

            // the large trees are built in parallel over the groups, and the
            // huge ones one at a time, each in parallel
            parallel_loop(large, [&](auto, auto k) { build_tree(k, false); });
            for (auto k : huge)
                build_tree(k, true);
        }

        // Calls f(j) for every disk j != i with the same key as i, such that
        // |x_i - x_j| < h (r_i + r_j).
        template <class F>
        void visit_near_nodes(size_t i, double h, F&& f) const
        {
            auto k = _key[i];
            if (_r_max[k] == 0)
                return;
            auto& xi = _x[i];
            auto near = [&](size_t j)
            {
                if (j == i)
                    return;
                double dx = _x[j][0] - xi[0];
                double dy = _x[j][1] - xi[1];
                double dmax = h * (_r[i] + _r[j]);
                if (dx * dx + dy * dy < dmax * dmax)
                    f(j);
            };

            auto& tree = _trees[k];
            if (tree == nullptr)
            {
                for (size_t m = _begin[k]; m < _begin[k + 1]; ++m)
                    near(_members[m]);
                return;
            }
            // the neighbours are visited as they are found, without collecting
            // them
            auto visit = [&](const item_t& item) { near(item.second); };
            tree->search(boost::make_function_output_iterator(std::ref(visit)),
                         sphere_t(item_t(point_t(xi[0], xi[1]), i),
                                  h * (_r[i] + _r_max[k])));
        }

    private:
        using K2 = CGAL::Exact_predicates_inexact_constructions_kernel;
        using point_t = K2::Point_2;
        using item_t = std::pair<point_t, size_t>;
        using traits_t =
            CGAL::Search_traits_adapter
                <item_t, CGAL::First_of_pair_property_map<item_t>,
                 CGAL::Search_traits_2<K2>>;
        using tree_t = CGAL::Kd_tree<traits_t>;
        using sphere_t = CGAL::Fuzzy_sphere<traits_t>;

        const std::vector<std::array<double, 2>>& _x;
        const std::vector<double>& _r;
        const std::vector<size_t>& _key;
        std::vector<size_t> _begin;
        std::vector<size_t> _members;
        std::vector<double> _r_max;
        std::vector<std::unique_ptr<tree_t>> _trees;

        void build_tree(size_t k, [[maybe_unused]] bool parallel)
        {
            std::vector<item_t> items;
            items.reserve(_begin[k + 1] - _begin[k]);
            for (size_t m = _begin[k]; m < _begin[k + 1]; ++m)
            {
                auto i = _members[m];
                items.emplace_back(point_t(_x[i][0], _x[i][1]), i);
            }
            auto& tree = _trees[k];
            tree = std::make_unique<tree_t>(items.begin(), items.end());
            // the lazy build is not thread safe
#ifdef CGAL_LINKED_WITH_TBB
            if (parallel)
            {
                tbb_num_threads tbb_threads;
                tree->template build<CGAL::Parallel_tag>();
                return;
            }
#endif
            tree->build();
        }
    };

    // buffers of vsep_setup(): the disks and the energy of every vertex
    std::tuple<std::vector<p_t>, std::vector<double>, std::vector<double>,
               std::vector<size_t>, std::vector<double>> _vsep_buf;

    // Computes the separation gradients of the vertices (_bgrad), and returns
    // the separation energy (only called if _has_vsep, i.e. delta > 0).
    double vsep_setup()
    {
        std::ranges::fill(_bgrad, p_t{});
        auto& [x, r, w, key, Us] = _vsep_buf;
        vertex_disks(x, r, w, key);
        Us.assign(_vs.size(), 0);
        double delta = _barrier_sep;
        near_disk_search near(x, r, key, _barrier_exact_thres);
        parallel_loop
            (x,
             [&](size_t i, auto&)
             {
                 near.visit_near_nodes
                     (i, 1 + delta,
                      [&](size_t j)
                      {
                          double s = r[i] + r[j];
                          p_t dx = x[i] - x[j];
                          double d = std::sqrt(norm2(dx));
                          auto [psi, dpsi] = sep_pot((d - s) / (delta * s));
                          double c = sep_coef(w[i], w[j], s);
                          Us[i] += c * psi / 2;
                          // coinciding disks are separated along a
                          // direction fixed for the pair, as by dist_grad()
                          _bgrad[i] += ((d > 0) ? dx / d :
                                        pair_dir(_vs[j], _vs[i])) *
                              (c * dpsi / (delta * s));
                      });
             });
        double U = 0;
        for (auto Ui : Us)
            U += Ui;
        return U;
    }

    // Levels above
    // ------------

    // If an upper state is set (see set_upper()), its vertices are the groups
    // at the lowest level of the partition (its vertex r is the group with
    // label r), with their total weights W_r, and its edges and their weights
    // are those of the condensed graph (so that m_rs is the sum of m_uv over
    // the edges between the groups). The attraction and
    // repulsion of its vertices (with its own K and masses M_r, see
    // rep_mass()), placed at the centers of mass X_r of the groups, add the
    // energy
    //
    //     sum_{(r,s)} m_rs k_a(D_rs)
    //         + K' sum_{r != s} M_r M_s phi_p(D_rs),
    //
    // with D_rs = |X_r - X_s|, and those of the levels above it, recursively,
    // since the upper state places its own upper state in the same way. Only
    // these terms of the upper states are used (not their central pull,
    // group terms or constraints), so that a state that was laid out on its own
    // (e.g. a coarse level of the hierarchical layout) can serve as the upper
    // state of the level below. Since X_r is the weighted mean of the
    // positions of the members, the gradient for a member v is w_v / W_r
    // times the gradient for the upper vertex r, i.e. it translates the whole
    // group, which is what the hierarchical coordinates of gradient_sweep()
    // follow.

    HFDUpper* _up = nullptr;
    std::vector<p_t> _up_cm;                    // centers of mass of the
                                                // groups
    const std::vector<std::array<double, 2>>* _up_grad = nullptr;
    std::vector<std::array<double, 2>> _ml_grad; // gradient of every vertex,
                                                 // as an upper state

    void set_upper(HFDUpper& up)
    {
        if (_L == 0)
            throw ValueException("an upper state requires a partition");
        auto& lev = _levels[0];
        size_t N = up.ml_num_vertices();
        for (auto r : lev.label)
        {
            if (r < 0 || std::cmp_greater_equal(r, N))
                throw ValueException(std::format("the upper state has {} "
                                                 "vertices, but there is a "
                                                 "group with label {}", N, r));
        }
        _up = &up;
        _pg_pos.clear();
        _bt_pg_pos.clear();
    }

    // places the vertices of the upper state at the centers of mass of the
    // groups
    void up_set_pos()
    {
        group_cm(0, _up_cm);
        _up->ml_set_pos(_levels[0].label, _up_cm);
    }

    void up_setup()
    {
        up_set_pos();
        _up->ml_grad_setup();
        _up_grad = &_up->ml_grads();
    }

    // the gradient of the levels above for vertex v; up_setup() must be
    // called first
    p_t up_grad(size_t v)
    {
        auto& lev = _levels[0];
        auto r = lev.bidx[v];
        auto& g = (*_up_grad)[lev.label[r]];
        return p_t{g[0], g[1]} * (_vweight[v] / lev.w[r]);
    }

    // Interface as an upper state (HFDUpper): only the attraction and
    // repulsion, and the levels above

    size_t ml_num_vertices() override
    {
        return num_vertices(_g);
    }

    // the positions are also written to _pos, so that the upper state shows
    // the current centers of mass of the groups in Python; the other
    // vertices (e.g. of zero weight in the state below) keep their positions
    // in _pos, which may have been changed since the last call
    void ml_set_pos(const std::vector<int64_t>& us,
                    const std::vector<std::array<double, 2>>& x) override
    {
        read_pos();
        parallel_loop
            (us,
             [&](auto i, auto u)
             {
                 _x[u] = x[i];
                 _pos[u][0] = x[i][0];
                 _pos[u][1] = x[i][1];
             });
        if (_up != nullptr)
            up_set_pos();
    }

    size_t ml_version() override
    {
        return _version + ((_up != nullptr) ? _up->ml_version() : 0);
    }

    double ml_energy() override
    {
        gen_quad_tree();
        double U = att_energy();
        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
            reduction(+:U)
        parallel_loop_no_spawn
            (_vs,
             [&](auto, auto v)
             {
                 p_t pv = {_x[v][0], _x[v][1]};
                 auto& src = gather_sources(pv);
                 U += rep_energy_sum(src, pv, _rep_exp) * rep_mass(v);
             });
        if (_up != nullptr)
            U += _up->ml_energy();
        return U;
    }

    void ml_grad_setup() override
    {
        gen_quad_tree();
        if (_up != nullptr)
        {
            _up->ml_grad_setup();
            _up_grad = &_up->ml_grads();
        }
        _ml_grad.resize(num_vertices(_g));
        parallel_cell_loop
            ([&](auto, auto v)
             {
                 p_t g = adj_grad(v);
                 p_t pv = {_x[v][0], _x[v][1]};
                 auto& src = gather_sources(pv);
                 g += rep_grad_sum(src, pv, _rep_exp) * rep_mass(v);
                 if (_up != nullptr)
                     g += up_grad(v);
                 _ml_grad[v] = {g[0], g[1]};
             });
    }

    const std::vector<std::array<double, 2>>& ml_grads() override
    {
        return _ml_grad;
    }

    // Total energy and gradient
    // -------------------------

    // The energy is the sum of all the terms above: attraction, repulsion,
    // central pull, group gravity, group repulsion and group confinement, the
    // separation of the vertices with a radius, the attraction and repulsion
    // of the levels above, and the augmented Lagrangian terms of the active
    // constraints (ranks, distances, overlaps and two-member groups).

    // The terms that depend only on the positions (attraction, repulsion,
    // central pull, group gravity and repulsion, levels above and
    // separation) are computed once for every set of positions, their energy
    // (_pg_U) together with their gradient (_pg), for the next evaluation of
    // the gradient at the same positions (see pos_terms()); the others depend
    // also on the multipliers or the strengths of the springs, which can
    // change at the same positions.
    void pos_cache()
    {
        if (!pg_valid())
        {
            double U = att_energy();
            gen_quad_tree();
            if (_up != nullptr)
                up_setup();
            _pg.resize(_vs.size());

            #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                reduction(+:U)
            parallel_cell_loop_no_spawn
                ([&](auto i, auto v)
                 {
                     pos_terms<true>(v, U, _pg[i]);
                 });

            if (_up != nullptr)
                U += _up->ml_energy();   // placed by up_setup() above
            if (_has_vsep)
            {
                U += vsep_setup();
                parallel_loop(_vs,
                              [&](auto i, auto) { _pg[i] += _bgrad[i]; });
            }

            _pg_U = U;
            _pg_ver = up_version();
            store_pos(_pg_pos);
        }
    }

    double energy()
    {
        pos_cache();
        double U = _pg_U;

        if (has_dist())
            gen_kd_tree();

        for (size_t l = 0; l < _L; ++l)
        {
            auto mu = _gmu[l];
            if (mu == 0)
                continue;
            U += al_energy(_levels[l].glambda, mu, g_viol(l));
        }

        if (_nmu > 0)
        {
            double Un = 0;
            #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                reduction(+:Un)
            parallel_loop_no_spawn
                (_vs,
                 [&](auto, auto v)
                 {
                     Un += al_energy(_nlambda[v], nmu_v(v), n_viol(v));
                 });
            U += Un;
        }

        if (_has_omu)
        {
            update_overlaps();
            U += overlap_energy();
        }
        if (_has_pmu)
            U += pairs_energy();

        // ranks and confinement
        bool conf = std::ranges::any_of(_levels,
                                        [](auto& lev) { return lev.has_conf; });
        if (_rmu > 0 || conf)
        {
            conf_setup(false);
            if (_rmu > 0)
                rank_viol();
            double Uc = 0;
            #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                reduction(+:Uc)
            parallel_loop_no_spawn
                (_vs,
                 [&](auto i, auto v)
                 {
                     auto vw = _vweight[v];
                     if (_rmu > 0)
                         Uc += rank_energy(i, v) * vw;
                     for (size_t l = 0; l < _L; ++l)
                     {
                         if (_levels[l].has_conf)
                             Uc += conf_energy(v, l) * vw;
                     }
                 });
            U += Uc;
        }

        if (_rcmu > 0 || _rcmxu > 0)
            U += rcm_energy();

        return U;
    }

    // Prepares the evaluation of the gradient, and returns whether the terms
    // that depend only on the positions are available from energy().
    bool grad_setup()
    {
        bool pg = pg_valid();
        gen_quad_tree();
        conf_setup(true);
        if (_up != nullptr && !pg)
            up_setup();

        if (has_dist())
            gen_kd_tree();

        for (size_t l = 0; l < _L; ++l)
        {
            auto& lev = _levels[l];
            if (_gmu[l] == 0)
            {
                lev.glambda_eff = 0;
                continue;
            }
            lev.glambda_eff = al_eff(lev.glambda, _gmu[l], g_viol(l));
        }

        if (has_dist())
            dist_grads();

        if (_has_omu || _has_pmu)
        {
            parallel_loop(_vs, [&](auto i, auto) { _ograd[i] = p_t{}; });
            if (_has_omu)
            {
                update_overlaps();
                overlap_grads();
            }
            if (_has_pmu)
                pair_grads();
        }

        if (_rmu > 0)
            rank_grads();

        if (_rcmu > 0 || _rcmxu > 0)
            rcm_grads();

        return pg;
    }

    // Gradient (and, if E, energy) of the terms of v that depend only on the
    // positions: the attraction (only its gradient, since the energy is
    // summed over the edges), the repulsion, central pull, levels above (only
    // the gradient, see energy()) and group gravity and repulsion. Both the
    // gradient cached by energy() and the one computed directly come from
    // here, so that they agree (up to the order in which the separation is
    // added, which pos_cache() includes).
    template <bool E>
    void pos_terms(size_t v, double& U, p_t& grad)
    {
        auto& pv = _x[v];
        auto vm = rep_mass(v);
        grad = adj_grad(v);

        // repulsion, and the total repulsion with the exponent of the group
        // repulsion, if it differs
        auto& src = gather_sources(pv);
        if constexpr (E)
        {
            double Ur = rep_energy_sum(src, pv, _rep_exp) * vm;
            double Uq = _group_rep_sep ?
                rep_energy_sum(src, pv, _group_rep_exp) * vm : Ur;
            U += Ur + _group_rep_top * Uq;
        }
        p_t rgrad = rep_grad_sum(src, pv, _rep_exp) * vm;
        p_t qgrad = _group_rep_sep ?
            rep_grad_sum(src, pv, _group_rep_exp) * vm : rgrad;
        grad += rgrad + qgrad * _group_rep_top;

        if (_center_pull > 0)
        {
            if constexpr (E)
                U += pull_energy(pv, get_cm()) * vm;
            grad += pull_grad(pv, get_cm()) * vm;
        }

        if (_up != nullptr)
            grad += up_grad(v);

        // group gravity and repulsion, with the sources of the tree of the
        // group of v at every level, summed as those of the repulsion
        if (_has_group_grav || _has_group_rep)
        {
            for (size_t l = 0; l < _L; ++l)
            {
                if (!has_group_trees(l))
                    continue;
                auto& gsrc = gather_group_sources(v, l);
                if (_group_grav[l] > 0)
                {
                    auto gg = group_grav(v, l);
                    if constexpr (E)
                        U += group_grav_energy_sum(gsrc, pv, gg) * vm;
                    grad += group_grav_grad_sum(gsrc, pv, gg) * vm;
                }
                if (auto c = group_rep_coef(l); c != 0)
                {
                    if constexpr (E)
                        U += c * rep_energy_sum(gsrc, pv, _group_rep_exp) *
                            vm;
                    grad += rep_grad_sum(gsrc, pv, _group_rep_exp) *
                        (c * vm);
                }
            }
        }
    }

    // the gradient and the energy of the terms that depend only on the
    // positions, obtained by energy() at the positions _pg_pos, and the
    // version _pg_ver of the levels above
    std::vector<p_t> _pg, _pg_pos;
    double _pg_U = 0;
    size_t _pg_ver = 0;

    size_t up_version()
    {
        return (_up != nullptr) ? _up->ml_version() : 0;
    }

    // whether the cached terms of the positions are valid
    bool pg_valid()
    {
        return same_pos(_pg_pos) && _pg_ver == up_version();
    }

    bool same_pos(const std::vector<p_t>& cache)
    {
        return (cache.size() == _vs.size() &&
                std::ranges::all_of(std::views::iota(size_t(0), _vs.size()),
                                    [&](auto i)
                                    { return cache[i] == _x[_vs[i]]; }));
    }

    void store_pos(std::vector<p_t>& cache)
    {
        cache.resize(_vs.size());
        parallel_loop(_vs, [&](auto i, auto v) { cache[i] = _x[v]; });
    }

    // Gradient of v; with pg == true, the terms that depend only on the
    // positions are taken from energy() (see grad_setup()), and if
    // constraints == false, those of the constraints are left out.
    p_t vertex_grad(size_t v, bool constraints = true, bool pg = false)
    {
        p_t grad;
        if (pg)
        {
            grad = _pg[_vpos[v]];
        }
        else
        {
            double U = 0;
            pos_terms<false>(v, U, grad);
        }
        auto vw = _vweight[v];

        // group confinement
        for (size_t l = 0; l < _L; ++l)
        {
            auto& lev = _levels[l];
            if (!lev.has_conf)
                continue;
            auto r = lev.bidx[v];
            grad += (conf_grad(v, l) - lev.conf_T[r] / lev.w[r]) * vw;
        }

        if (!constraints)
            return grad;

        if (_rmu > 0)
            grad += _rgrad[_vpos[v]];

        if (_rcmu > 0 || _rcmxu > 0)
            grad += _rcgrad[_vpos[v]];

        if (has_dist())
            grad += _dgrad[_vpos[v]];

        if (_has_omu || _has_pmu)
            grad += _ograd[_vpos[v]];

        return grad;
    }

    // Minimization
    // ============

    // Gradient descent
    // ----------------

    // whether the groups are moved rigidly by gradient_sweep(), which is the
    // case if they are kept together by springs at some level (see
    // update_group_springs())
    bool hier_coords()
    {
        for (size_t l = 0; l < _L; ++l)
        {
            if (has_springs(l))
                return true;
        }
        return _up != nullptr;
    }

    // Gradient descent with Nesterov momentum, in hierarchical coordinates if
    // hier_coords() is true, and with the separation of the vertices with a
    // radius, if any.
    //
    // In hierarchical coordinates, the variables are the positions of the
    // vertices relative to rigid translations of every group: a vertex v is
    // displaced by its own step plus the steps of all its groups,
    //
    //     dx_v = d_v + sum_l D_{b_l(v)},
    //
    // where the step D_r of group r (with its own velocity) follows the mean
    // gradient of its members, so that whole groups (and their subgroups)
    // move together, which is the slowest mode of the layout otherwise. Since
    // the spring of a group only depends on the offsets of its members from
    // its center of mass, a translation of the group does not change it.
    std::vector<p_t> _gv;  // gradients of the vertices in gradient_sweep()

    void gradient_sweep(size_t n, double step, double mu, double cap_grad)
    {
        save_sweep();
        bool hier = hier_coords();

        auto cap = [&](p_t g)
        {
            if (cap_grad > 0)
            {
                auto ng2 = norm2(g);
                if (ng2 * pow2(step) > pow2(cap_grad))
                    g /= (std::sqrt(ng2) * step) / cap_grad;
            }
            return g;
        };

        // Nesterov displacement of a variable with velocity vel
        auto move = [&](auto& vel, const p_t& g)
        {
            p_t v_prev = {vel[0], vel[1]};
            p_t v_new = v_prev * mu - cap(g) * step;
            vel[0] = v_new[0];
            vel[1] = v_new[1];
            return v_new * (1 + mu) - v_prev * mu;
        };

        if (hier)
        {
            for (auto& lev : _levels)
            {
                if (lev.gvel.size() != lev.groups.size())
                    lev.gvel.assign(lev.groups.size(), p_t{});
            }
        }

        auto& gv = _gv;
        gv.resize(_vs.size());
        for (size_t it = 0; it < n; ++it)
        {
            bool pg = grad_setup();
            // otherwise, the separation is part of the cached terms
            bool vsep = _has_vsep && !pg;
            if (vsep)
                vsep_setup();

            parallel_cell_loop
                ([&](auto i, auto v)
                 {
                     gv[i] = vertex_grad(v, true, pg);
                     if (vsep)
                         gv[i] += _bgrad[i];
                 });
            if (it == 0)
                _bt_g = gv;

            // steps of the groups, from the mean gradients of their members
            if (hier)
            {
                for (size_t l = 0; l < _L; ++l)
                {
                    auto& lev = _levels[l];
                    group_sum(l, lev.gdisp,
                              [&](auto i, auto) { return gv[i]; });
                    for (size_t g = 0; g < lev.groups.size(); ++g)
                        lev.gdisp[g] = move(lev.gvel[g], lev.gdisp[g] /
                                            double(lev.groups[g].size()));
                }
            }

            parallel_loop
                (_vs,
                 [&](auto i, auto v)
                 {
                     p_t d = move(_vel[v], gv[i]);
                     if (hier)
                     {
                         for (auto& lev : _levels)
                             d += lev.gdisp[lev.bidx[v]];
                     }
                     _x[v][0] += d[0];
                     _x[v][1] += d[1];
                 });
        }
    }

    // Loops over the vertices in the order of the cells of the quadtree, so
    // that the consecutive traversals of the tree follow similar paths (which
    // is better for the branch prediction and the cache), as f(i, v), with i
    // the index of v in _vs.
    void parallel_cell_loop_no_spawn(auto&& f)
    {
        if (_exact)
        {
            parallel_loop_no_spawn(_vs, f);
            return;
        }
        parallel_loop_no_spawn
            (_qt.order(),
             [&](auto, auto i)
             {
                 f(i, _vs[i]);
             });
    }

    void parallel_cell_loop(auto&& f)
    {
        if (_exact)
        {
            parallel_loop(_vs, f);
            return;
        }
        parallel_loop
            (_qt.order(),
             [&](auto, auto i)
             {
                 f(i, _vs[i]);
             });
    }

    void reset_momentum()
    {
        parallel_loop(_vs, [&](auto, auto v) { _vel[v] = {0, 0}; });
        for (auto& lev : _levels)
            std::ranges::fill(lev.gvel, p_t{});
    }

    // The state at the start of the last sweep (positions, velocities, and
    // the gradient of every vertex, in the order of _vs), so that the sweep
    // can be undone with restore_sweep(), and its slope obtained with
    // sweep_slope(), for the backtracking of hfd_layout().
    std::vector<p_t> _bt_x, _bt_vel, _bt_g, _bt_pg, _bt_pg_pos;
    double _bt_pg_U = 0;
    size_t _bt_pg_ver = 0;
    std::vector<std::vector<p_t>> _bt_gvel;

    void save_sweep()
    {
        _bt_x = _x;
        _bt_vel = _vel;
        _bt_gvel.resize(_L);
        for (size_t l = 0; l < _L; ++l)
            _bt_gvel[l] = _levels[l].gvel;
        // overwritten by the first step of the sweep, unless there is none
        _bt_g.assign(_vs.size(), p_t{});
        // the cached terms of the positions at the start (see pos_cache()),
        // which are restored with them
        if (pg_valid())
        {
            _bt_pg = _pg;
            _bt_pg_pos = _pg_pos;
            _bt_pg_U = _pg_U;
            _bt_pg_ver = _pg_ver;
        }
        else
        {
            _bt_pg_pos.clear();
        }
    }

    // the positions are copied in place, since the kd-tree refers to them
    void restore_sweep()
    {
        check_sweep();
        std::ranges::copy(_bt_x, _x.begin());
        std::ranges::copy(_bt_vel, _vel.begin());
        for (size_t l = 0; l < _L; ++l)
            _levels[l].gvel = _bt_gvel[l];
        if (!_bt_pg_pos.empty())
        {
            std::swap(_pg, _bt_pg);
            std::swap(_pg_pos, _bt_pg_pos);
            std::swap(_pg_U, _bt_pg_U);
            std::swap(_pg_ver, _bt_pg_ver);
        }
    }

    // the saved state is only valid after a gradient_sweep()
    void check_sweep()
    {
        if (_bt_x.size() != _x.size())
            throw ValueException("no gradient sweep to undo");
    }

    // directional derivative of the energy at the start of the last sweep,
    // along its displacement, sum_v g_v . (x_v - x0_v)
    double sweep_slope()
    {
        check_sweep();
        double m = 0;
        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
            reduction(+:m)
        parallel_loop_no_spawn
            (_vs,
             [&](auto i, auto v)
             {
                 m += dot(_bt_g[i], sub(_x[v], _bt_x[v]));
             });
        return m;
    }

    // Lagrange multipliers
    // --------------------

    // Initializes the Lagrange multipliers of all the active constraints with
    // first-order estimates for the current positions. Writing a constraint as
    // c <= 0, its multiplier is the one for which the force of the constraint
    // cancels the component of the unconstrained force (that of the energy
    // without the constraint terms) along the gradient of c,
    //
    //     lambda = max(0, -grad(U) . grad(c) / |grad(c)|^2),
    //
    // if the constraint is violated or less than margin away from it, and
    // zero otherwise. The constraints are estimated independently, except
    // that the estimate is shared among the rank constraints of each vertex.
    // The multipliers of the minimum and group distance constraints are only
    // estimated if some pair is violated, since otherwise their gradient
    // vanishes.
    void init_multipliers(double margin)
    {
        pos_cache();
        conf_setup(true);
        if (has_dist())
            gen_kd_tree();
        if (_has_omu)
        {
            _ov_pos.clear();
            generate_hulls();
        }

        std::vector<p_t> ugrad(_vs.size());
        parallel_loop
            (_vs,
             [&](auto i, auto v)
             {
                 ugrad[i] = vertex_grad(v, false, true);
             });

        // the force on every vertex, which sets the penalty of its minimum
        // distance
        if (_nmu > 0)
        {
            _nmu_f.assign(num_vertices(_g), 0);
            parallel_loop
                (_vs,
                 [&](auto i, auto v)
                 {
                     _nmu_f[v] = norm(ugrad[i]);
                 });
        }

        // the estimate for the gradient of c, given as (slot, grad) pairs
        auto estimate =
            [&](const auto& cgrad)
            {
                double dot = 0, n2 = 0;
                for (auto& [i, g] : cgrad)
                {
                    dot += ugrad[i][0] * g[0] + ugrad[i][1] * g[1];
                    n2 += norm2(g);
                }
                return (n2 > 0) ? std::max(0., -dot / n2) : 0.;
            };

        if (_rcmu > 0)
        {
            // g_c, whose gradient is -w_v / W_c on the members of class c and
            // w_v / W_{c+1} on those of class c + 1
            rcm_setup();
            for (size_t c = 0; c < _rclambda.size(); ++c)
            {
                _rclambda[c] = 0;
                if (rcm_viol(c) < -margin)
                    continue;
                double dot = 0, n2 = 0;
                for (auto [k, sgn] : {std::pair{c, -1.}, {c + 1, 1.}})
                {
                    for (auto i : _rclass[k])
                    {
                        double gi = sgn * _vweight[_vs[i]] / _rcW[k];
                        dot += ugrad[i][1] * gi;
                        n2 += gi * gi;
                    }
                }
                _rclambda[c] = std::max(0., -dot / n2);
            }
        }

        if (_rcmxu > 0)
        {
            // c_ab, whose derivative dc on X_a and -dc on X_b is distributed
            // as w_v / W_c over the members; only the violated pairs are
            // estimated, since otherwise the gradient vanishes
            rcm_setup();
            size_t C = _rclass.size();
            std::vector<double> G(C), Q(C);
            for (size_t c = 0; c < C; ++c)
            {
                for (auto i : _rclass[c])
                {
                    double f = _vweight[_vs[i]] / _rcW[c];
                    G[c] += ugrad[i][0] * f;
                    Q[c] += f * f;
                }
            }
            _rcxlambda.clear();
            rcmx_pairs([&](auto a, auto b, auto)
                       {
                           auto [c, dc] = rcmx_c(a, b);
                           if (c <= 0)
                               return;
                           double dot = dc * (G[a] - G[b]);
                           double n2 = dc * dc * (Q[a] + Q[b]);
                           double l = std::max(0., -dot / n2);
                           if (l > 0)
                               _rcxlambda[a * C + b] = l;
                       });
        }

        if (_rmu > 0)
        {
            // c = V_v, whose gradient is -n on v and 1 on each of the n
            // vertices u of its terms (here, those of the next class closer
            // than margin), and whose force is multiplied by the vertex
            // weight; the sum of the y gradients of the u is obtained from
            // the prefix sums of the sorted next class
            for (auto v : _vs)
                _rlambda[v] = 0;
            rank_viol([&](auto j) { return ugrad[j][1]; },
                      [&](auto i, auto, auto& rs)
                      {
                          auto v = _vs[i];
                          size_t n = rs.yf.size();
                          double t = _x[v][1] - _rmin_dist[v] - margin;
                          size_t k0 = std::isinf(margin) ? 0 :
                              rank_first(rs, t);
                          double m = n - k0;
                          if (m == 0)
                              return;
                          double dot = -m * ugrad[i][1] + rs.Sf[n] -
                              rs.Sf[k0];
                          _rlambda[v] = std::max(0., -dot / (m * m + m)) /
                              _vweight[v];
                      });
        }

        if (_has_omu)
        {
            // c = D_r <= 0 for every group r, with the gradients of all the
            // pairs of r; with an infinite margin, only groups closer than the
            // size of the largest bounding box are considered, to avoid
            // considering all pairs
            for (size_t l = 0; l < _L; ++l)
            {
                if (_omu[l] == 0)
                    continue;
                auto& lev = _levels[l];
                auto& hulls = lev.hulls;
                size_t B = hulls.loop.size();
                double m = margin;
                if (std::isinf(m))
                {
                    m = 0;
                    for (size_t r = 0; r < B; ++r)
                    {
                        if (hulls.loop[r].empty())
                            continue;
                        auto& bb = hulls.bbox[r];
                        m = std::max({m, bb[2] - bb[0], bb[3] - bb[1]});
                    }
                }
                compute_overlaps(l, m);

                std::vector<std::vector<size_t>> ghits(B);
                for (auto [k, t] : std::views::enumerate(lev.ohits))
                {
                    ghits[t.r].push_back(k);
                    ghits[t.s].push_back(k);
                }

                parallel_loop
                    (ghits,
                     [&](auto r, auto& ks)
                     {
                         olambda(l, r) = 0;
                         if (ks.empty())
                             return;
                         std::vector<std::pair<size_t, p_t>> cgrad;
                         for (auto k : ks)
                         {
                             ohit_grads(lev.ohits[k],
                                        [&](auto j, const p_t& g)
                                        {
                                            for (auto& [i, x] : cgrad)
                                            {
                                                if (i == j)
                                                {
                                                    x += g;
                                                    return;
                                                }
                                            }
                                            cgrad.emplace_back(j, g);
                                        });
                         }
                         olambda(l, r) = estimate(cgrad);
                     });
            }
        }

        if (_has_pmu)
        {
            // c = |x_u - x_v| - opair_dist <= 0
            for (size_t l = 0; l < _L; ++l)
            {
                if (_pmu[l] == 0)
                    continue;
                parallel_loop
                    (_levels[l].pairs,
                     [&](auto j, auto& e)
                     {
                         auto& [u, v] = e;
                         _plambda[u][l] = 0;
                         p_t r = sub(_x[u], _x[v]);
                         double d = norm(r);
                         if (d == 0 || pair_violation(l, j) <= -margin)
                             return;
                         r = scale(1. / d, r);
                         std::array<std::pair<size_t, p_t>, 2>
                             cgrad{{{_vpos[u], r}, {_vpos[v], scale(-1., r)}}};
                         _plambda[u][l] = estimate(cgrad);
                     });
            }
        }

        // c = sum_v V_v <= 0 of the group distance
        auto dist_estimate =
            [&](size_t l)
            {
                std::vector<p_t> cgrad(_vs.size());
                gdist_grads(cgrad, l, 1.);
                double dot = 0, n2 = 0;
                #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                    reduction(+:dot, n2)
                parallel_loop_no_spawn
                    (_vs,
                     [&](auto i, auto)
                     {
                         dot += ugrad[i][0] * cgrad[i][0] +
                             ugrad[i][1] * cgrad[i][1];
                         n2 += norm2(cgrad[i]);
                     });
                return (n2 > 0) ? std::max(0., -dot / n2) : 0.;
            };

        if (_nmu > 0)
        {
            // c = V_v <= 0 for every vertex, with the gradients of the terms
            // of its pairs on both endpoints
            parallel_loop
                (_vs,
                 [&](auto i, auto v)
                 {
                     _nlambda[v] = 0;
                     std::vector<std::pair<size_t, p_t>> cgrad;
                     p_t self{};
                     visit_radius(v, _nmin_dist[v],
                                  [&](auto u)
                                  {
                                      auto g = dist_grad(v, u, _nmin_dist[v]);
                                      self += g;
                                      cgrad.emplace_back(_vpos[u], g * -1.);
                                  });
                     if (cgrad.empty())
                         return;
                     cgrad.emplace_back(i, self);
                     _nlambda[v] = estimate(cgrad);
                 });
        }
        for (size_t l = 0; l < _L; ++l)
        {
            if (_gmu[l] > 0)
                _levels[l].glambda = dist_estimate(l);
        }
    }

    bool has_constraints()
    {
        return _rmu > 0 || _rcmu > 0 || _rcmxu > 0 || _nmu > 0 || _has_gmu ||
            _has_omu || _has_pmu;
    }

    // multiplies the penalty parameters of all the active constraints by c
    void scale_penalties(double c)
    {
        _rmu *= c;
        _rcmu *= c;
        _rcmxu *= c;
        _nmu *= c;
        for (auto& mu : _gmu)
            mu *= c;
        for (auto& mu : _omu)
            mu *= c;
        for (auto& mu : _pmu)
            mu *= c;
    }

    // Updates the multipliers, and returns the total violation of the
    // constraints (zero when every constraint is satisfied). The violations of
    // the minimum and group distances are relative to sum_v c_v^2, i.e. the
    // squared minimum distance times the number of vertices, since they have
    // the dimension of c^2 and grow with the number of vertices; those of the
    // ranks, overlaps and pairs are lengths, summed over the constraints.
    double update_constraints()
    {
        return eval_constraints(true);
    }

    // the total violation of the constraints, without updating the
    // multipliers
    double constraint_violation()
    {
        return eval_constraints(false);
    }

    double eval_constraints(bool update)
    {
        double R = 0;
        if (_rmu > 0)
        {
            rank_viol();
            #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                reduction(+:R)
            parallel_loop_no_spawn
                (_vs,
                 [&](auto i, auto v)
                 {
                     R += _rV[i];
                     if (update)
                         _rlambda[v] = al_eff(_rlambda[v], _rmu, _rV[i]);
                 });
        }

        if (_rcmu > 0)
        {
            rcm_setup();
            for (size_t c = 0; c < _rclambda.size(); ++c)
            {
                double g = rcm_viol(c);
                R += std::max(0., g);
                if (update)
                    _rclambda[c] = al_eff(_rclambda[c], _rcmu, g);
            }
        }

        // relative to d^2, as the minimum distances; the multipliers that
        // vanish are removed
        if (_rcmxu > 0)
        {
            rcm_setup();
            size_t C = _rclass.size();
            std::flat_map<size_t, double> nl;
            rcmx_pairs([&](auto a, auto b, auto l)
                       {
                           double c = rcmx_c(a, b)[0];
                           R += std::max(0., c) / _rcm_xdist;
                           l = al_eff(l, _rcmxu, c);
                           if (l > 0)
                               nl[a * C + b] = l;
                       });
            if (update)
                _rcxlambda = std::move(nl);
        }

        if (has_dist())
            gen_kd_tree();

        for (size_t l = 0; l < _L; ++l)
        {
            if (_gmu[l] == 0)
                continue;
            auto V = g_viol(l);
            auto& lg = _levels[l].glambda;
            if (update)
                lg = al_eff(lg, _gmu[l], V);
            if (_levels[l].gmin_C > 0)
                R += V / _levels[l].gmin_C;
        }

        if (_nmu > 0)
        {
            double Rn = 0;
            #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
                reduction(+:Rn)
            parallel_loop_no_spawn
                (_vs,
                 [&](auto, auto v)
                 {
                     auto V = n_viol(v);
                     if (update)
                         _nlambda[v] = al_eff(_nlambda[v], nmu_v(v), V);
                     Rn += V;
                 });
            if (_nmin_C > 0)
                R += Rn / _nmin_C;
        }

        if (_has_omu)
        {
            update_overlaps();
            for (size_t l = 0; l < _L; ++l)
            {
                auto mu = _omu[l];
                if (mu == 0)
                    continue;

                auto& lev = _levels[l];
                for (size_t r = 0; update && r < lev.D.size(); ++r)
                {
                    auto& lo = olambda(l, r);
                    lo = al_eff(lo, mu, lev.D[r]);
                }
                for (auto& t : lev.ohits)
                    R += std::max(0., t.depth);
            }
        }

        if (_has_pmu)
        {
            for (size_t l = 0; l < _L; ++l)
            {
                auto mu = _pmu[l];
                if (mu == 0)
                    continue;
                auto& pairs = _levels[l].pairs;
                size_t n = pairs.size();
                #pragma omp parallel if (n > get_openmp_min_thresh()) \
                    reduction(+:R)
                parallel_loop_no_spawn
                    (pairs,
                     [&](auto j, auto&)
                     {
                         auto c = pair_violation(l, j);
                         if (update)
                         {
                             auto& lp = _plambda[pairs[j][0]][l];
                             lp = al_eff(lp, mu, c);
                         }
                         R += std::max(0., c);
                     });
            }
        }

        return R;
    }

    // L-BFGS
    // ------

    void pack(auto&& x)
    {
        parallel_loop
            (_vs,
             [&](auto i, auto v)
             {
                 x[2 * i] = _x[v][0];
                 x[2 * i + 1] = _x[v][1];
             });
    }

    void unpack(auto&& x)
    {
        parallel_loop
            (_vs,
             [&](auto i, auto v)
             {
                 _x[v][0] = x[2 * i];
                 _x[v][1] = x[2 * i + 1];
             });
    }

#ifdef HAVE_BOOST_LBFGS
    // Convergence when the change of the objective between iterations is
    // below rtol relative to max(1, |f|), or below atol. (This replaces
    // Boost's relative_objective_tol_policy and
    // objective_tol_convergence_policy, which call unqualified max() and
    // abs().)
    struct objective_change_policy
    {
        double rtol, atol;
        mutable double last = 0;
        mutable bool first = true;

        bool operator()(const auto&, double f) const
        {
            if (first)
            {
                first = false;
                last = f;
                return false;
            }
            double diff = std::abs(f - last);
            double scale = std::max(1., std::abs(last));
            last = f;
            return diff < rtol * scale || diff < atol;
        }
    };

    std::vector<double> _lbfgs_x;  // the variables of lbfgs_sweep()

    // Minimization with L-BFGS. The line search is either the strong Wolfe
    // one (armijo == false), with parameters c1, c2, rho (growth while
    // bracketing) and ls_max_iter, or Armijo backtracking, with parameters
    // c1, rho (backtracking factor) and ls_max_iter. It stops when the norm of
    // the gradient falls below epsilon, or the change of the objective falls
    // below ftol (relative) or fatol (absolute), or after max_iter iterations
    // or max_time seconds (if positive).
    auto lbfgs_sweep(size_t max_iter, double step, double epsilon,
                     size_t hist_size, bool armijo, double c1, double c2,
                     double rho, size_t ls_max_iter, double ftol, double fatol,
                     double max_time)
    {
        _bt_x.clear();          // a previous gradient sweep can't be undone
        auto objective =
            [&](auto&& x)
            {
                unpack(x);
                return energy();
            };

        // the terms of the positions (including the separation) are always
        // taken from the energy, which is evaluated first at the same
        // positions
        auto gradient =
            [&](auto&& x, auto& grad)
            {
                unpack(x);
                pos_cache();
                grad_setup();
                parallel_cell_loop
                    ([&](auto i, auto v)
                     {
                         auto vg = vertex_grad(v, true, true);
                         grad[2 * i] = vg[0];
                         grad[2 * i + 1] = vg[1];
                     });
            };

        auto grad_policy =
            [&](auto& f, auto& x, auto&& f_eval, auto& fx, auto& grad)
            {
                fx = f_eval(f, x);
                gradient(x, grad);
            };

        // tolerances of zero never trigger
        auto converged =
            bopt::combined_convergence_policy
                (bopt::gradient_norm_convergence_policy<double>(epsilon),
                 objective_change_policy{ftol, fatol});

        // Boost's minimize() evaluates the convergence policy again after
        // the loop, when the objective change is zero, so convergence is
        // instead taken as the loop not having been terminated
        auto start = std::chrono::steady_clock::now();
        bool terminated = false;
        auto terminate =
            [&, start](size_t iter)
            {
                if (iter >= max_iter)
                    return terminated = true;
                if (max_time <= 0)
                    return false;
                std::chrono::duration<double> t =
                    std::chrono::steady_clock::now() - start;
                return terminated = t.count() > max_time;
            };

        auto& x = _lbfgs_x;
        x.resize(2 * _vs.size());
        pack(x);

        auto run =
            [&](auto&& line_search)
            {
                auto opt =
                    bopt::make_lbfgs
                        (std::move(objective), x, hist_size,
                         [](auto) {},                                // init
                         [](auto&& obj, auto& v) { return obj(v); }, // eval
                         std::move(grad_policy),
                         std::move(line_search));
                auto result = bopt::minimize
                    (opt, bopt::unconstrained_policy<std::vector<double>>{},
                     converged, terminate, false);
                return result.num_iter;
            };

        size_t niter;
        if (armijo)
            niter = run(bopt::armijo_line_search_policy<double>
                            (step, c1, rho, ls_max_iter));
        else
            niter = run(bopt::strong_wolfe_line_search_policy<double>
                            (step, c1, c2, rho, ls_max_iter));

        // the objective value of the optimizer is the one before the last
        // step
        unpack(x);
        return boost::python::make_tuple(niter, energy(), !terminated);
    }
#else
    boost::python::tuple lbfgs_sweep(size_t, double, double, size_t, bool,
                                     double, double, double, size_t, double,
                                     double, double)
    {
        throw ValueException("lbfgs_sweep() is not available, since "
                             "graph-tool was compiled with a Boost version "
                             "without L-BFGS (boost/math/optimization)");
    }
#endif

    // Spatial indices
    // ===============

    // Quad-tree
    // ---------

    // positions for which the trees were last built
    std::vector<p_t> _qt_pos;

    // Whether any vertex has moved since the positions were last stored in
    // cache, in which case they are stored again. The trees are not rebuilt
    // otherwise, e.g. when the energy is computed before a sweep at the same
    // positions; this is usually decided by the first vertex.
    bool moved(std::vector<p_t>& cache)
    {
        if (same_pos(cache))
            return false;
        store_pos(cache);
        return true;
    }

    void gen_quad_tree()
    {
        if (!moved(_qt_pos))
            return;
        if (_exact)
            gen_sources();
        else
            build_quad_trees();
    }

    void gen_sources()
    {
        auto source = [&](auto v) -> source_t
        {
            return {{_x[v][0], _x[v][1]}, rep_mass(v)};
        };

        _src.resize(_vs.size());
        double x = 0, y = 0, w_tot = 0;
        #pragma omp parallel if (_vs.size() > get_openmp_min_thresh()) \
            reduction(+:x, y, w_tot)
        parallel_loop_no_spawn
            (_vs,
             [&](auto i, auto v)
             {
                 _src[i] = source(v);
                 auto& [p, w] = _src[i];
                 x += p[0] * w;
                 y += p[1] * w;
                 w_tot += w;
             });
        // without any mass, the central pull vanishes anyway
        _cm = (w_tot > 0) ? p_t{x / w_tot, y / w_tot} : p_t{};

        group_tree_loop
            ([&](auto l, auto r)
             {
                 auto& lev = _levels[l];
                 auto& src = lev.src[r];
                 src.clear();
                 for (auto v : lev.groups[r])
                     src.push_back(source(v));
             });
    }

    // Calls f(l, r) for every group with its own tree. The sizes of the
    // groups vary greatly, with the largest at the upper levels: these are
    // processed one at a time (so that f can be parallel inside, as the
    // construction of a quadtree), and the others in parallel with each
    // other.
    void group_tree_loop(auto&& f)
    {
        auto& jobs = _group_tree_jobs;
        size_t thresh = get_openmp_min_thresh();
        size_t i = 0;
        for (; i < jobs.size(); ++i)
        {
            auto [l, r] = jobs[i];
            if (_levels[l].groups[r].size() <= thresh)
                break;
            f(l, r);
        }
        parallel_loop(std::span(jobs).subspan(i),
                      [&](auto, auto& lr) { f(lr[0], lr[1]); }, 1);
    }

    // the pairs (l, r) of the groups with their own trees (see
    // has_group_trees()), so that they can be built in parallel, set by
    // init_levels()
    std::vector<std::array<size_t, 2>> _group_tree_jobs;

    void build_quad_trees()
    {
        // The tree is a dense array with ~4^(depth+1)/3 cells, so its depth
        // is bounded to keep the memory O(N); otherwise, points that are very
        // close relative to the size of the layout (e.g. with far outliers)
        // would increase the depth without bound.
        uint32_t max_depth =
            uint32_t(std::ceil(std::log2(std::max(_vs.size(),
                                                  size_t(2))) / 2)) + 1;
        // the depth starts from that given by the number of points (and not
        // from the last one), so that the tree depends only on the positions
        uint32_t depth = 0;
        uint32_t max_count = 0;
        do
        {
            max_count =
                _qt.init(_vs |
                         std::views::transform
                         ([&](auto v) -> std::tuple<const p_t&, vw_t>
                          {
                              return {_x[v], rep_mass(v)};
                          }), depth, max_depth);
            depth = _qt.get_depth() + 1;
        }
        while (max_count > std::max(_vs.size()/10, size_t(100)) &&
               _qt.get_depth() < max_depth);
        _qt.barnes_hut_init(_theta);

        group_tree_loop
            ([&](auto l, auto r)
             {
                 auto& lev = _levels[l];
                 lev.qt[r].init(lev.groups[r] |
                                std::views::transform
                                ([&](auto v) -> std::tuple<const p_t&, vw_t>
                                 {
                                     return {_x[v], rep_mass(v)};
                                 }));
                 lev.qt[r].barnes_hut_init(_theta);
             });
    }

    auto visit_barnes_hut(double x, double y)
    {
        gen_quad_tree();
        boost::python::list pts;

        visit_sources(_qt, _src, p_t{x, y},
                      [&](auto& lpos, auto w, auto...)
                      {
                          auto xp = boost::python::make_tuple(lpos[0],
                                                              lpos[1]);
                          pts.append(boost::python::make_tuple(xp, w));
                      });
        return pts;
    }

    // kd-tree
    // -------

    std::vector<p_t> _kd_pos;   // positions of the last kd-tree

    void gen_kd_tree()
    {
        if (!moved(_kd_pos))
            return;
        _kd_tree.invalidate_build();
#ifdef CGAL_LINKED_WITH_TBB
        tbb_num_threads tbb_threads;
        _kd_tree.template build<CGAL::Parallel_tag>();
#else
        _kd_tree.build();
#endif
    }

    // Query of the kd-tree for the points inside a disk, for its
    // fuzzy-sphere-like search interface (without the points cache, see
    // above).
    class disk_query
    {
    public:
        using Point_d   = size_t;  // the tree's key type
        using FT        = double;
        using Dimension = CGAL::Dimension_tag<2>;

        disk_query(const std::vector<p_t>& x, const p_t& c, double r)
            : _x(&x), _cx(c[0]), _cy(c[1]), _r2(r * r)
        {
        }

        // leaf test
        bool contains(size_t v) const
        {
            auto& p = (*_x)[v];
            return inside(p[0], p[1]);
        }

        // whether the disk may touch the box; otherwise, the subtree is
        // skipped
        template <class Rect>
        bool inner_range_intersects(const Rect& b) const
        {
            double dx = gap(_cx, b.min_coord(0), b.max_coord(0));
            double dy = gap(_cy, b.min_coord(1), b.max_coord(1));
            return dx * dx + dy * dy <= _r2;
        }

        // whether the whole box is inside the disk, in which case the subtree
        // is reported untested
        template <class Rect>
        bool outer_range_contains(const Rect& b) const
        {
            double dx = std::max(std::abs(_cx - b.min_coord(0)),
                                 std::abs(_cx - b.max_coord(0)));
            double dy = std::max(std::abs(_cy - b.min_coord(1)),
                                 std::abs(_cy - b.max_coord(1)));
            return dx * dx + dy * dy <= _r2;
        }

    private:
        bool inside(double x, double y) const
        {
            double dx = x - _cx, dy = y - _cy;
            return dx * dx + dy * dy <= _r2;
        }

        static double gap(double c, double lo, double hi)
        {
            return c < lo ? lo - c : (c > hi ? c - hi : 0.);
        }

        const std::vector<p_t>* _x;
        double _cx, _cy, _r2;
    };

    // visits the vertices closer than d to v (other than v itself)
    void visit_radius(size_t v, double d, auto&& visit)
    {
        auto f = [&](auto u)
        {
            if (u != v)
                visit(u);
        };
        _kd_tree.search(boost::make_function_output_iterator(std::ref(f)),
                        disk_query(_x, _x[v], d));
    }

    // Convex hulls
    // ------------

    struct hulls_t
    {
        // slots in _vs of the hull vertices (counterclockwise, for three or
        // more members)
        std::vector<std::vector<size_t>> loop;
        std::vector<std::array<double, 4>>    bbox;   // xlo, ylo, xhi, yhi
    };

    // a pair of groups whose hulls overlap (see sat_depth())
    struct overlap_t
    {
        enum axis_t { FIXED, NORMAL, DIRECTION };

        size_t r, s;            // the groups
        double depth;           // penetration depth plus _odist (violation)
        size_t a, b;            // slots of the edge that defines the axis
        size_t h, lo;           // slots of the support points
        axis_t kind;            // fixed, edge normal, or segment direction
        p_t n;                  // the axis
    };

    // buffers of compute_overlaps(): the bounding boxes of the hulls, the
    // candidate pairs, and their overlaps
    typedef CGAL::Box_intersection_d::Box_with_info_d<double, 2, size_t>
        box_t;
    std::vector<box_t> _obox;
    std::vector<std::array<size_t, 2>> _ocand;
    std::vector<overlap_t> _ohits;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel hull_kernel_t;
    std::vector<hull_kernel_t::Point_2> _hull_pts;  // positions, for the hulls

    void generate_hulls()
    {
        using K2      = hull_kernel_t;
        using Point_2 = K2::Point_2;
        using hull_traits_t = CGAL::Convex_hull_traits_adapter_2
            <K2, CGAL::Pointer_property_map<Point_2>::type>;

        auto& cp = _hull_pts;
        cp.clear();
        for (auto v : _vs)
            cp.emplace_back(_x[v][0], _x[v][1]);

        for (size_t l = 0; l < _L; ++l)
        {
            if (_omu[l] == 0)
                continue;

            auto& member = _levels[l].slots;
            auto& hulls = _levels[l].hulls;

            // the loops are cleared in place, so that they keep their memory
            hulls.loop.resize(member.size());
            hulls.bbox.resize(member.size());

            // parallel over the groups if there are many vertices in total
            size_t thresh = (_vs.size() > get_openmp_min_thresh()) ? 0 :
                std::numeric_limits<size_t>::max();
            parallel_loop
                (member,
                 [&](auto r, auto& ms)
                 {
                     // groups of one or two members, and collinear ones,
                     // have a point or a segment as their hull
                     auto& loop = hulls.loop[r];
                     loop.clear();
                     if (ms.size() < 3)
                         loop = ms;
                     else
                         CGAL::convex_hull_2
                             (ms.begin(), ms.end(), std::back_inserter(loop),
                              hull_traits_t(CGAL::make_property_map(cp)));
                     auto& bb = hulls.bbox[r];
                     bb = {+std::numeric_limits<double>::infinity(),
                           +std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
                     for (size_t i : loop)
                     {
                         bb[0] = std::min(bb[0], _x[_vs[i]][0]);
                         bb[1] = std::min(bb[1], _x[_vs[i]][1]);
                         bb[2] = std::max(bb[2], _x[_vs[i]][0]);
                         bb[3] = std::max(bb[3], _x[_vs[i]][1]);
                     }
                 }, thresh);
        }
    }

    // Hierarchy levels
    // ================

    // Everything that depends on the partition at a single hierarchy level.
    // The group terms of a level only compare the labels of vertices at that
    // level, but the levels are nested (as used by gradient_sweep(),
    // update_group_springs() and group_order_params()).
    struct level_t
    {
        std::vector<size_t> bidx;                 // dense group of each vertex
        std::vector<double> w;                    // total weight of each group
        std::vector<std::vector<size_t>> groups;  // members of each group
        std::vector<QuadTree<double, vw_t>> qt;   // group gravity/repulsion
        std::vector<double> ggfac;                // group gravity factors
        std::vector<double> conf, conf_R;         // confinement of each group
        std::vector<double> conf_iRa;             // R^-a of each group
        std::vector<double> conf_pack;            // strength of the packing
                                                  // spring
        std::vector<p_t> gvel, gdisp;             // velocity and step of the
                                                  // translation of each group
        std::vector<p_t> conf_cm, conf_T;         // center of mass, and sum of
                                                  // the confinement gradients
        bool has_conf = false;
        std::vector<std::vector<source_t>> src;   // same, if _exact
        std::vector<int64_t> label;               // label of each group
        hulls_t hulls;
        std::vector<std::vector<size_t>> slots;   // slots of the members of
                                                  // every group in _vs
        std::vector<overlap_t> ohits;             // overlapping pairs of groups
        std::vector<double> D;                    // total violation of groups
        std::vector<std::array<size_t, 2>> pairs; // two-member groups
        double glambda = 0;
        double glambda_eff = 0;
        double gmin_C = 0;
    };

    std::vector<level_t> _levels;

    // Order parameters
    // ================

    // Order parameters of the separation of the groups at every level,
    // obtained from the Delaunay triangulation of the positions:
    //
    // - kappa: the contiguity of the groups, i.e. the fraction of the weight
    //   of each group that belongs to its largest biconnected component, in
    //   the graph of the Delaunay edges between members of the same group,
    //   averaged over the groups weighted by their total weight. Since a
    //   Delaunay edge exists only if no other point lies within some circle
    //   through its endpoints, members of a group separated by other groups
    //   are not connected, but those separated only by empty space are. Edges
    //   on the convex hull are ignored, since they can join distant members
    //   along the outer boundary of the layout.
    //
    // - S: the fraction of the area that lies between groups, i.e. of the
    //   triangles with corners in different groups, relative to all the
    //   triangles with corners in the same group of the level above (or any
    //   triangle, at the top level). Triangles with an edge on the convex
    //   hull are ignored, since they mostly cover empty space outside the
    //   layout.
    //
    // - G: the mean area of the triangles between groups relative to the
    //   mean area of the triangles within groups, which is independent of the
    //   size of the groups.
    //
    // Returns kappa, S and G, indexed by level, and the contiguity of every
    // group, indexed by level and group label (NaN for unused labels). The
    // levels are computed in parallel.
    auto group_order_params()
    {
        typedef hfd_dt_kernel_t K2;
        typedef hfd_dt_t dt_t;
        typedef typename boost::graph_traits<dt_t>::edge_descriptor dt_edge_t;

        // edges between members of the same group, which are not on the
        // convex hull (the BGL requires a default-constructible predicate)
        struct keep_t
        {
            const dt_t* dt = nullptr;
            const std::vector<size_t>* bidx = nullptr;

            bool operator()(const dt_edge_t& e) const
            {
                if (dt->is_infinite(e.first) ||
                    dt->is_infinite(e.first->neighbor(e.second)))
                    return false;
                auto u = e.first->vertex(dt->cw(e.second));
                auto w = e.first->vertex(dt->ccw(e.second));
                return (*bidx)[u->info()] == (*bidx)[w->info()];
            }
        };

        std::vector<double> kappa_l(_L, 1), S_l(_L, 0),
            G_l(_L, std::numeric_limits<double>::quiet_NaN());

        // contiguity of every group, per level, indexed by group label
        std::vector<std::vector<double>> kappa_r(_L);
        for (size_t l = 0; l < _L; ++l)
        {
            auto& label = _levels[l].label;
            kappa_r[l].resize(label.empty() ? 0 :
                              *std::ranges::max_element(label) + 1,
                              std::numeric_limits<double>::quiet_NaN());
            for (auto r : label)
                kappa_r[l][r] = 1;
        }

        auto ret = [&]
        {
            return std::make_tuple(std::move(kappa_l), std::move(S_l),
                                   std::move(G_l), std::move(kappa_r));
        };

        std::vector<std::pair<typename K2::Point_2, size_t>> pts;
        pts.reserve(_vs.size());
        for (auto v : _vs)
            pts.emplace_back(typename K2::Point_2(_x[v][0], _x[v][1]),
                             v);
        dt_t dt(pts.begin(), pts.end());
        CGAL::set_triangulation_ids(dt);

        if (dt.dimension() < 2)
            return ret();

        parallel_loop
            (std::views::iota(size_t(0), _L),
             [&](auto, size_t l)
             {
                 auto& bidx = _levels[l].bidx;

                 // contiguity
                 keep_t keep{&dt, &bidx};
                 boost::filt_graph<dt_t, keep_t, boost::keep_all>
                     fg(dt, keep, boost::keep_all());

                 std::vector<size_t> comp(num_edges(dt));
                 auto comp_map =
                     boost::make_iterator_property_map
                         (comp.begin(), get(boost::edge_index, dt));
                 boost::biconnected_components(fg, comp_map);

                 // distinct (component, vertex) pairs, since articulation
                 // points belong to more than one component
                 std::vector<std::pair<size_t, size_t>> cv;
                 for (auto e : edges_range(fg))
                 {
                     auto c = comp_map[e];
                     cv.emplace_back(c, source(e, fg)->info());
                     cv.emplace_back(c, target(e, fg)->info());
                 }
                 std::ranges::sort(cv);
                 auto [first, last] = std::ranges::unique(cv);
                 cv.erase(first, last);

                 auto& groups = _levels[l].groups;
                 auto& w_r = _levels[l].w;

                 // largest component weight of every group
                 std::vector<double> wc_r(groups.size(), 0);
                 for (size_t i = 0; i < cv.size();)
                 {
                     size_t j = i;
                     double w = 0;
                     for (; j < cv.size() && cv[j].first == cv[i].first; ++j)
                         w += _vweight[cv[j].second];
                     auto r = bidx[cv[i].second];
                     wc_r[r] = std::max(wc_r[r], w);
                     i = j;
                 }

                 double kappa = 0, w_tot = 0;
                 for (size_t r = 0; r < groups.size(); ++r)
                 {
                     // a group with a single member is contiguous
                     if (groups[r].size() == 1)
                         wc_r[r] = w_r[r];
                     kappa_r[l][_levels[l].label[r]] =
                         (w_r[r] > 0) ? wc_r[r] / w_r[r] : 1;
                     kappa += wc_r[r];
                     w_tot += w_r[r];
                 }
                 kappa_l[l] = (w_tot > 0) ? kappa / w_tot : 1;

                 // area between groups
                 double A_in = 0, A_out = 0;
                 size_t n_in = 0, n_out = 0;
                 for (auto f : dt.finite_face_handles())
                 {
                     bool hull = false;
                     for (int i = 0; i < 3; ++i)
                         hull = hull || dt.is_infinite(f->neighbor(i));
                     if (hull)
                         continue;

                     std::array<size_t, 3> vs;
                     for (int i = 0; i < 3; ++i)
                         vs[i] = f->vertex(i)->info();

                     if (l + 1 < _L)
                     {
                         auto& pidx = _levels[l + 1].bidx;
                         if (pidx[vs[0]] != pidx[vs[1]] ||
                             pidx[vs[0]] != pidx[vs[2]])
                             continue;
                     }

                     double A = std::abs(dt.triangle(f).area());
                     if (bidx[vs[0]] == bidx[vs[1]] &&
                         bidx[vs[0]] == bidx[vs[2]])
                     {
                         A_in += A;
                         n_in++;
                     }
                     else
                     {
                         A_out += A;
                         n_out++;
                     }
                 }

                 S_l[l] = (A_in + A_out > 0) ? A_out / (A_in + A_out) : 0;
                 G_l[l] = (n_in > 0 && n_out > 0 && A_in > 0) ?
                     (A_out / n_out) / (A_in / n_in) :
                     std::numeric_limits<double>::quiet_NaN();
             }, 0);

        return ret();
    }
};

GEN_DISPATCH(hfd, HFDState, STATE_params)

// Mean of d^-p for two points uniformly distributed in the unit disk,
// considering only distances above d0 (which diverges for p >= 2 if
// d0 == 0). The distance has density (2 d / pi) (2 acos(d/2) - (d/2)
// sqrt(4 - d^2)) on [0, 2], which is integrated in log(d).
inline double disk_moment(double p, double d0)
{
    auto f = [](double d)   // density divided by d
    {
        // the last point may exceed d = 2 by rounding
        d = std::min(d, 2.);
        return (2 / std::numbers::pi) *
            (2 * std::acos(d / 2) - (d / 2) * std::sqrt(4 - d * d));
    };
    double u0 = std::log(std::max(d0, 1e-12)), u1 = std::log(2.);
    size_t n = 4000;                       // even, for Simpson's rule
    double h = (u1 - u0) / n, S = 0;
    for (size_t i = 0; i <= n; ++i)
    {
        double d = std::exp(u0 + i * h);
        double y = std::pow(d, 2 - p) * f(d);
        S += y * ((i == 0 || i == n) ? 1 : (i % 2 == 1) ? 4 : 2);
    }
    double m = S * h / 3;
    if (d0 > 0)
    {
        // normalized over the distances that are considered
        double Z = 0;
        for (size_t i = 0; i <= n; ++i)
        {
            double d = std::exp(u0 + i * h);
            Z += d * d * f(d) *
                ((i == 0 || i == n) ? 1 : (i % 2 == 1) ? 4 : 2);
        }
        m /= Z * h / 3;
    }
    return m;
}

// per-group values for every level (the multipliers of the overlap
// constraint and the group gravity factors), given as a list of arrays (one
// per level), which are accessed in place
template <>
struct extract_param<hfd_level_arrays_t>
{
    auto operator()(boost::python::object o)
    {
        hfd_level_arrays_t xs;
        for (int64_t l = 0; l < boost::python::len(o); ++l)
            xs.push_back(get_array<double, 1>(o[l]));
        return xs;
    }
};

// hierarchical partition, given as a list of arrays
template <>
struct extract_param<std::vector<std::vector<int64_t>>>
{
    auto operator()(boost::python::object obs)
    {
        std::vector<std::vector<int64_t>> bs;
        for (int64_t l = 0; l < boost::python::len(obs); ++l)
        {
            auto a = get_array<int64_t, 1>(obs[l]);
            bs.emplace_back(a.begin(), a.end());
        }
        return bs;
    }
};

} // namespace graph_tool
