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

#include "graph_tool.hh"

#include <boost/python.hpp>

#include "hfd.hh"
#include "numpy_bind.hh"

#include <CGAL/Orthogonal_incremental_neighbor_search.h>

using namespace boost;
using namespace graph_tool;

// Auxiliary functions for the hierarchical version of hfd_layout(). Every
// level of the hierarchy is a graph whose vertices are the groups of the level
// below, with the following properties (see _HFDLevel in draw/__init__.py):
//
//   w:    total weight of the original vertices
//   R:    radius of the group in the mean field (see hfd_mf_radii())
//   gmin: mean gmin_dist of the original vertices, for every partition level
//   e:    total weight of the original edges (those with an endpoint of zero
//         weight are left out, since they exert no force)
//
// A level is passed from Python as the tuple (w, R, gmin, e). The
// minimum distance between the vertices (nmin_dist) applies only to the
// original graph, and is not propagated to the coarse levels.

namespace
{

// Wraps a method of HFDState so that the positions are copied from the
// property map before the call, and back after it (see
// HFDState::pos_load()), since they may be read or modified from Python
// between calls.
template <auto F, bool store>
struct pos_synced;

template <class S, class R, class... As, R (S::*F)(As...), bool store>
struct pos_synced<F, store>
{
    static R call(S& s, As... as)
    {
        s.pos_load();
        if constexpr (std::is_void_v<R>)
        {
            (s.*F)(std::forward<As>(as)...);
            if constexpr (store)
                s.pos_store();
        }
        else
        {
            R r = (s.*F)(std::forward<As>(as)...);
            if constexpr (store)
                s.pos_store();
            return r;
        }
    }
};

template <auto F>
constexpr auto synced = &pos_synced<F, true>::call;

// for the methods that do not move the vertices
template <auto F>
constexpr auto loaded = &pos_synced<F, false>::call;

typedef vprop_map_t<double> vdmap_t;
typedef vprop_map_t<std::vector<double>> vvmap_t;
typedef eprop_map_t<double> edmap_t;

template <class T>
T get_prop(boost::python::object o)
{
    return extract_param<T>()(o);
}

// The property maps are created in Python with the right sizes, so they can be
// accessed unchecked, which is also needed for the parallel loops below.
template <class T>
auto get_uprop(boost::python::object o)
{
    return get_prop<T>(o).get_unchecked();
}

struct hfd_level
{
    hfd_level(boost::python::object o)
        : w(get_uprop<vdmap_t>(o[0])), R(get_uprop<vdmap_t>(o[1])),
          gmin(get_uprop<vvmap_t>(o[2])), e(get_uprop<edmap_t>(o[3])) {}

    vdmap_t::unchecked_t w, R;
    vvmap_t::unchecked_t gmin;
    edmap_t::unchecked_t e;
};

} // anonymous namespace

#define __MOD__ layout
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace graph_tool;
    using namespace boost::python;
    using namespace __MOD__;

    def("make_hfd_state",
        +[](boost::python::object ostate)
         {
             python::object state;
             hfd::make_dispatch(ostate,
                                [&](auto& s){state = python::object(s);});
             return state;
         });

    // the states of every type can be the upper state of any other (see
    // HFDState::set_upper())
    def_obj<class_<HFDUpper, boost::noncopyable>>("HFDUpper", no_init)
        ([](auto&){});

    hfd::dispatch
        ([&](auto* s)
         {
             typedef std::decay_t<decltype(*s)> state_t;

             auto& c = get_class<state_t, bases<HFDUpper>,
                                 std::shared_ptr<state_t>, boost::noncopyable>
                 (get_type_name<state_t>().c_str(), no_init);

             c.def("gradient_sweep", synced<&state_t::gradient_sweep>);
             c.def("reset_momentum", &state_t::reset_momentum);
             c.def("restore_sweep", synced<&state_t::restore_sweep>);
             c.def("sweep_slope", loaded<&state_t::sweep_slope>);
             c.def("update_group_springs",
                   loaded<&state_t::update_group_springs>);
             c.def("lbfgs_sweep", synced<&state_t::lbfgs_sweep>);
             c.def("update_constraints",
                   loaded<&state_t::update_constraints>);
             c.def("has_constraints", &state_t::has_constraints);
             c.def("scale_penalties", &state_t::scale_penalties);
             c.def("constraint_violation",
                   loaded<&state_t::constraint_violation>);
             c.def("energy", loaded<&state_t::energy>);
             c.def("set_upper", &state_t::set_upper);
             c.def("visit_barnes_hut", loaded<&state_t::visit_barnes_hut>);
             c.def("get_glambda", &state_t::get_glambda);
             c.def("set_glambda", &state_t::set_glambda);
             c.def("get_rclambda", &state_t::get_rclambda);
             c.def("set_rclambda", &state_t::set_rclambda);
             c.def("get_rcxlambda", &state_t::get_rcxlambda);
             c.def("set_rcxlambda", &state_t::set_rcxlambda);
             c.def("init_multipliers", loaded<&state_t::init_multipliers>);
             c.def("get_group_springs",
                   +[](state_t& state)
                    {
                        python::list gamma;
                        for (auto& x : state.get_group_springs())
                            gamma.append(wrap_vector_owned(x));
                        return gamma;
                    });
             c.def("set_group_springs",
                   +[](state_t& state, python::object ogamma)
                    {
                        std::vector<std::vector<double>> gamma;
                        for (int l = 0; l < python::len(ogamma); ++l)
                        {
                            auto x = get_array<double, 1>(ogamma[l]);
                            gamma.emplace_back(x.begin(), x.end());
                        }
                        state.set_group_springs(gamma);
                    });
             c.def("get_nmu_force",
                   +[](state_t& state)
                    {
                        return wrap_vector_owned(state.get_nmu_force());
                    });
             c.def("set_nmu_force",
                   +[](state_t& state, python::object of)
                    {
                        auto x = get_array<double, 1>(of);
                        state.set_nmu_force({x.begin(), x.end()});
                    });
             c.def("group_order_params",
                   +[](state_t& state)
                    {
                        state.pos_load();
                        auto [kappa, S, G, kappa_r] =
                            state.group_order_params();
                        python::list kr;
                        for (auto& x : kappa_r)
                            kr.append(wrap_vector_owned(x));
                        return python::make_tuple(wrap_vector_owned(kappa),
                                                  wrap_vector_owned(S),
                                                  wrap_vector_owned(G), kr);
                    });
         });


    // Initializes the attraction weights of the edges of the lowest level,
    // given the edge weights ew.
    def("hfd_init_level",
        +[](GraphInterface& gi, object oew, object of)
         {
             auto ew = get_uprop<edmap_t>(oew);
             hfd_level f(of);
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      parallel_edge_loop
                          (g,
                           [&](const auto& e)
                           {
                               f.e[e] = (f.w[source(e, g)] > 0 &&
                                         f.w[target(e, g)] > 0) ? ew[e] : 0;
                           });
                  })();
         });

    // Computes the mean gmin_dist of the groups, for the coarse graph cgi of
    // the level f given by the partition b, where coarse vertex r corresponds
    // to group r, and whose w and e were already obtained with
    // condensation_graph().
    def("hfd_coarsen",
        +[](GraphInterface& gi, GraphInterface& cgi, object ob, object of,
            object oc)
         {
             auto b = get_array<int64_t, 1>(ob);
             hfd_level f(of);
             hfd_level c(oc);
             auto& cg = cgi.get_graph();
             size_t B = num_vertices(cg);
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      parallel_vertex_loop<true>
                          (g,
                           [&](auto v)
                           {
                               if (v >= b.size() || b[v] < 0 ||
                                   std::cmp_greater_equal(b[v], B))
                                   throw ValueException
                                       (std::format("invalid group for "
                                                    "vertex {}", v));
                           });

                      size_t L = 0;
                      for (auto v : vertices_range(g))
                      {
                          L = f.gmin[v].size();
                          break;
                      }
                      for (auto v : vertices_range(g))
                      {
                          if (f.gmin[v].size() != L)
                              throw ValueException
                                  (std::format("vertex {} has {} values of "
                                               "gmin_dist, expected {}", v,
                                               f.gmin[v].size(), L));
                      }

                      parallel_vertex_loop
                          (cg,
                           [&](auto r)
                           {
                               c.gmin[r].assign(L, 0);
                           });

                      parallel_vertex_loop
                          (g,
                           [&](auto v)
                           {
                               auto w = f.w[v];
                               auto r = b[v];
                               for (size_t k = 0; k < L; ++k)
                                   atomic_inc(c.gmin[r][k], w * f.gmin[v][k]);
                           });

                      parallel_vertex_loop
                          (cg,
                           [&](auto r)
                           {
                               auto w = c.w[r];
                               if (w <= 0)
                                   return;
                               for (auto& x : c.gmin[r])
                                   x /= w;
                           });
                  })();
         });

    // Computes the parameters of hfd_layout() for the given level, with the
    // group constraints starting at partition level l: the edge weights, and
    // the minimum distances to the vertices of other groups (gmin_dist),
    // which are increased by twice the radius of the groups. Returns the mean
    // radius of the groups, weighted by their weights.
    def("hfd_level_params",
        +[](GraphInterface& gi, object of, object oeweight,
            object ogmin_dist, size_t l)
         {
             hfd_level f(of);
             auto eweight = get_uprop<edmap_t>(oeweight);
             auto gmin_dist = get_uprop<vvmap_t>(ogmin_dist);
             double wR = 0, w_tot = 0;
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      size_t N = num_vertices(g);
                      // local copies, since clang rejects references to the
                      // variables of the enclosing lambda from the parallel
                      // region below
                      auto& gdist = gmin_dist;
                      size_t l0 = l;
                      #pragma omp parallel if (N > get_openmp_min_thresh()) \
                          reduction(+:wR, w_tot)
                      parallel_vertex_loop_no_spawn
                          (g,
                           [&](auto v)
                           {
                               auto d = 2 * f.R[v];
                               auto& gm = f.gmin[v];
                               gdist[v].clear();
                               for (size_t j = l0; j < gm.size(); ++j)
                                   gdist[v].push_back(gm[j] + d);
                               wR += f.w[v] * f.R[v];
                               w_tot += f.w[v];
                           });
                      parallel_edge_loop
                          (g,
                           [&](const auto& e)
                           {
                               eweight[e] = f.e[e];
                           });
                  })();
             return (w_tot > 0) ? wR / w_tot : 0.;
         });

    // Radii of the groups of every level of the nested partition bs (in the
    // format of NestedBlockState.get_bs()) in the mean field, with the
    // vertices of every group spread uniformly over a disk, and its external
    // edges assumed at the same distance as its internal ones, and held
    // together by its edges and by the packing spring (see
    // HFDState::init_conf() and hfd_layout()). The virial theorem gives
    //
    //     K c_p (m_r^2 - q_r) R^-p = e_r R^a + k_r w_r R^2,
    //
    // with e_r = 2 mu_a e^in_r + e^ex_r, i.e.
    //
    //     (R / R_e)^(a + p) + (R / R_p)^(p + 2) = 1,
    //
    // with the radius given by the edges alone,
    //
    //     R_e = (K c_p (m_r^2 - q_r) / e_r)^(1 / (a + p)),
    //
    // and the strength k_r of the spring chosen so that alone it gives the
    // packing radius R_p = (K / 2) sqrt(w_r). Here w_r is the sum of the
    // weights of the members, m_r and q_r the sums of their masses in the
    // repulsion (vweight * rweight) and of their squares, e^in_r and e^ex_r
    // the total weights ew of the internal and
    // external edges, c_p = <d^-p> (excluding the distances below the
    // typical spacing of the members if it diverges) and mu_a = <d^a>. The
    // radius is R_p for the groups without edges or with a single member
    // with weight, and (K / 2) sqrt(n_r) for those without weight. For
    // a + p = 0, the edges give no length scale, and the first term is
    // replaced by e_r / (K c_p (m_r^2 - q_r)). The repulsion K is given for
    // every level, i.e. that between the members of its groups. Returns the
    // radii of the groups, at every level.
    def("hfd_mf_radii",
        +[](GraphInterface& gi, object obs, object ovweight, object orweight,
            object oeweight, object oK, double p, double a)
         {
             auto vweight = get_uprop<vdmap_t>(ovweight);
             auto rweight = get_uprop<vdmap_t>(orweight);
             auto eweight = get_uprop<edmap_t>(oeweight);
             auto Ks = get_array<double, 1>(oK);
             std::vector<multi_array_ref<int64_t, 1>> bs;
             for (int l = 0; l < python::len(obs); ++l)
                 bs.push_back(get_array<int64_t, 1>(obs[l]));
             size_t L = bs.size();
             if (!(a + p >= 0))
                 throw ValueException("att_exp + rep_exp must be "
                                      "non-negative");
             if (Ks.size() != L)
                 throw ValueException(std::format("K has {} values, but the "
                                                  "partition has {} levels",
                                                  Ks.size(), L));
             std::vector<std::vector<double>> Rs(L);
             disk_moment_cache cp(p);
             double mu_a = disk_moment(-a, 0);
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      // the group of every vertex at the current level
                      std::vector<int64_t> lab(num_vertices(gi.get_graph()));
                      for (auto v : vertices_range(g))
                          lab[v] = v;
                      for (size_t l = 0; l < L; ++l)
                      {
                          auto& b = bs[l];
                          for (auto v : vertices_range(g))
                          {
                              auto r = lab[v];
                              if (r < 0 || std::cmp_greater_equal(r, b.size()))
                                  throw ValueException
                                      (std::format("partition level {} has "
                                                   "{} entries, but group {} "
                                                   "is referenced", l,
                                                   b.size(), r));
                              if (b[r] < 0)
                                  throw ValueException
                                      (std::format("negative group label at "
                                                   "partition level {}", l));
                              lab[v] = b[r];
                          }
                          size_t B = 0;
                          for (auto r : b)
                          {
                              if (r >= 0)
                                  B = std::max(B, size_t(r + 1));
                          }

                          std::vector<double> w_r(B), m_r(B), q_r(B),
                              e_in(B), e_ex(B);
                          std::vector<size_t> n_r(B);
                          for (auto v : vertices_range(g))
                          {
                              auto r = lab[v];
                              double w = vweight[v];
                              double m = w * rweight[v];
                              w_r[r] += w;
                              m_r[r] += m;
                              q_r[r] += m * m;
                              ++n_r[r];
                          }
                          for (auto e : edges_range(g))
                          {
                              auto u = source(e, g);
                              auto v = target(e, g);
                              // self-loops exert no force
                              if (u == v)
                                  continue;
                              // the edges of vertices of zero weight
                              // exert no force
                              if (!(vweight[u] > 0) || !(vweight[v] > 0))
                                  continue;
                              double m = eweight[e];
                              auto r = lab[u], s = lab[v];
                              if (r == s)
                              {
                                  e_in[r] += m;
                              }
                              else
                              {
                                  e_ex[r] += m;
                                  e_ex[s] += m;
                              }
                          }

                          auto& R = Rs[l];
                          R.assign(B, 0);
                          double K = Ks[l];
                          for (size_t r = 0; r < B; ++r)
                          {
                              // the members without weight exert no
                              // forces, and are spread as if of unit
                              // weight, so that they do not coincide
                              if (!(w_r[r] > 0))
                              {
                                  R[r] = K / 2 * std::sqrt(n_r[r]);
                                  continue;
                              }
                              double c_p = cp(n_r[r]);
                              double e_r = 2 * e_in[r] * mu_a + e_ex[r];
                              double R_p = K / 2 * std::sqrt(w_r[r]);
                              // without a repulsion between the members
                              // (e.g. a single one with weight), only the
                              // packing spring remains
                              double S = K * c_p *
                                  (m_r[r] * m_r[r] - q_r[r]);
                              if (!(S > 0) || !(e_r > 0))
                              {
                                  R[r] = R_p;
                                  continue;
                              }
                              if (a + p == 0)
                              {
                                  // the edges give no length scale, and only
                                  // weaken the repulsion; if they overcome
                                  // it, the group collapses, and it gets the
                                  // radius of a single member, so that the
                                  // members do not coincide
                                  double x = std::max(1 - e_r / S, 0.);
                                  R[r] = R_p *
                                      std::max(std::pow(x, 1 / (p + 2)),
                                               1 / std::sqrt(n_r[r]));
                                  continue;
                              }
                              double R_e = std::pow(S / e_r, 1 / (a + p));
                              // the left-hand side increases with R, from
                              // zero, and exceeds one at min(R_e, R_p), so
                              // that the root is found by bisection
                              double lo = 0, hi = std::min(R_e, R_p);
                              for (size_t i = 0;
                                   i < 100 && hi - lo > 1e-12 * hi; ++i)
                              {
                                  double x = (lo + hi) / 2;
                                  if (std::pow(x / R_e, a + p) +
                                      std::pow(x / R_p, p + 2) > 1)
                                      hi = x;
                                  else
                                      lo = x;
                              }
                              R[r] = (lo + hi) / 2;
                          }
                      }
                  })();
             python::list oR;
             for (size_t l = 0; l < L; ++l)
                 oR.append(wrap_vector_owned(Rs[l]));
             return oR;
         });

    // Places the vertices of gi inside the disks of their groups s = b[v] in
    // the coarse graph cgi, laid out at cpos, with radius R[s]. The radius is
    // bounded by half the distance of the group to the nearest other one, so
    // that the disks of different groups never overlap, and their members
    // are not mixed, regardless of the coarse layout. Every disk is divided
    // into as many cells of equal area as the group has members, and every
    // member is placed at the center of a different cell, chosen uniformly at
    // random, so that no two members coincide, even if the groups of the
    // coarse layout do (the cells of every group are rotated at random).
    def("sample_group_positions",
        +[](GraphInterface& gi, GraphInterface& cgi, object opos, object ob,
            object ocpos, object oR, rng_t& rng)
         {
             auto pos = get_uprop<vvmap_t>(opos);
             auto cpos = get_uprop<vvmap_t>(ocpos);
             auto b = get_array<int64_t, 1>(ob);
             auto R = get_array<double, 1>(oR);
             auto& cg = cgi.get_graph();
             if (R.size() < num_vertices(cg))
                 throw ValueException("too few group radii");

             std::vector<std::vector<size_t>> members(num_vertices(cg));
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      for (auto v : vertices_range(g))
                      {
                          if (v >= b.size() || b[v] < 0 ||
                              std::cmp_greater_equal(b[v], members.size()) ||
                              cpos[b[v]].size() < 2)
                              throw ValueException(std::format("invalid group "
                                                               "for vertex {}",
                                                               v));
                          members[b[v]].push_back(v);
                      }
                  })();

             // half the distance of every group to the nearest other one
             typedef CGAL::Exact_predicates_inexact_constructions_kernel K2;
             typedef std::pair<K2::Point_2, size_t> item_t;
             typedef CGAL::Search_traits_adapter
                 <item_t, CGAL::First_of_pair_property_map<item_t>,
                  CGAL::Search_traits_2<K2>> traits_t;
             typedef CGAL::Orthogonal_incremental_neighbor_search<traits_t>
                 knn_t;
             std::vector<double> R_max(num_vertices(cg),
                                       std::numeric_limits<double>::infinity());
             std::vector<item_t> items;
             // only the groups with members, since the others (e.g. filtered
             // out) are not placed
             for (auto s : vertices_range(cg))
             {
                 if (!members[s].empty())
                     items.emplace_back(K2::Point_2(cpos[s][0], cpos[s][1]),
                                        s);
             }
             typename knn_t::Tree tree(items.begin(), items.end());
             tree.build();
             parallel_loop
                 (items,
                  [&](auto, auto& item)
                  {
                      // the nearest group at a nonzero distance, since a
                      // coinciding one would give a zero radius, and all
                      // members would coincide
                      knn_t search(tree, item.first);
                      for (auto [nb, d2] : search)
                      {
                          if (d2 == 0)
                              continue;
                          R_max[item.second] = std::sqrt(d2) / 2;
                          break;
                      }
                  });

             // The cells are arranged in m concentric rings, the first i of
             // which enclose N_i = n (i/m)^2 cells (rounded), with radius
             // rho sqrt(N_i/n), so that every cell has area pi rho^2 / n, and
             // the ring width is close to the arc length of its cells.
             parallel_vertex_loop_rng
                 (cg,
                  [&](auto s, auto, auto& trng)
                  {
                      auto& vs = members[s];
                      size_t n = vs.size();
                      if (n == 0)
                          return;
                      std::shuffle(vs.begin(), vs.end(), trng);
                      // a random rotation, so that the cells of coinciding
                      // groups do not coincide
                      std::uniform_real_distribution<>
                          phase(0, 2 * std::numbers::pi);
                      double phi = phase(trng);
                      double rho = std::min(R[s], R_max[s]);
                      auto& x = cpos[s];
                      size_t m =
                          std::max(std::lround(std::sqrt(n / std::numbers::pi)),
                                   1l);
                      size_t j = 0, N_prev = 0;
                      for (size_t i = 0; i < m; ++i)
                      {
                          size_t N = std::lround(double(n) * (i + 1) * (i + 1) /
                                                 (m * m));
                          size_t k = N - N_prev;
                          double r0 = rho * std::sqrt(double(N_prev) / n);
                          double r1 = rho * std::sqrt(double(N) / n);
                          // the radius halving the area of the ring, or the
                          // center of the disk if it is a single cell
                          double rc = (N_prev == 0 && k == 1) ?
                              0 : std::sqrt((r0 * r0 + r1 * r1) / 2);
                          for (size_t c = 0; c < k; ++c, ++j)
                          {
                              double theta = phi +
                                  2 * std::numbers::pi * (c + .5) / k;
                              pos[vs[j]] = {x[0] + rc * std::cos(theta),
                                            x[1] + rc * std::sin(theta)};
                          }
                          N_prev = N;
                      }
                  }, rng);
         });
}, 0);
