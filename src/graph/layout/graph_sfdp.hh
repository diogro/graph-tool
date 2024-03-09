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

#ifndef GRAPH_FDP_HH
#define GRAPH_FDP_HH

#include <limits>
#include <iostream>

#include "idx_map.hh"

#include "../generation/sampler.hh"

#include "quad_tree.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Pos1, class Pos2>
double dist(const Pos1& p1, const Pos2& p2)
{
    double r = 0;
    for (size_t i = 0; i < 2; ++i)
        r += pow2(double(p1[i] - p2[i]));
    return sqrt(r);
}

template <class Pos1, class Pos2>
double f_r(double C, double K, double p, const Pos1& p1, const Pos2& p2)
{
    double d = dist(p1, p2);
    if (d == 0)
        return 0;
    return -C * pow(K, 1 + p) / pow(d, p);
}

template <class Pos1, class Pos2>
double fs_r(double C, double l, const Pos1& p1, const Pos2& p2)
{
    double d = dist(p1, p2);
    if (d == 0)
        return 0;
    return -C * exp(-d/l);
}

template <class Pos1, class Pos2>
double f_a(double K, const Pos1& p1, const Pos2& p2)
{
    return pow2(dist(p1, p2)) / K;
}

template <class Pos1, class Pos2, class Pos3>
double get_diff(const Pos1& p1, const Pos2& p2, Pos3& r)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        r[i] = p1[i] - p2[i];
        abs += r[i] * r[i];
    }
    if (abs == 0)
        abs = 1;
    abs = sqrt(abs);
    for (size_t i = 0; i < 2; ++i)
        r[i] /= abs;
    return abs;
}

template <class Pos>
double norm(Pos& x)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
        abs += pow2(x[i]);
    for (size_t i = 0; i < 2; ++i)
        x[i] /= sqrt(abs);
    return sqrt(abs);
}

template <class Graph, class PosMap, class VertexWeightMap,
          class EdgeWeightMap, class PinMap, class GroupMaps, class CMap,
          class OrderMap,
          class RNG>
void get_sfdp_layout(Graph& g, PosMap pos, VertexWeightMap vweight,
                     EdgeWeightMap eweight, PinMap pin, GroupMaps& groups,
                     double C, double K, double p, double theta, double gamma,
                     double mu, double kappa, double r, size_t kc, CMap c, double R,
                     OrderMap yorder, double init_step, double step_schedule,
                     size_t max_level, double epsilon, size_t max_iter,
                     bool simple, bool verbose, RNG& rng)
{
    typedef typename property_traits<PosMap>::value_type::value_type val_t;
    typedef std::array<val_t, 2> pos_t;

    typedef typename property_traits<VertexWeightMap>::value_type vweight_t;

    vector<size_t> vertices;
    idx_map<size_t, size_t> rs;

    double ocenter = 0;
    double omax = -std::numeric_limits<double>::max();
    double omin = std::numeric_limits<double>::max();

    int HN = 0;
    vweight_t W = 0;
    for (auto v : vertices_range(g))
    {
        if (pin[v] == 0)
            vertices.push_back(v);
        pos[v].resize(2, 0);
        HN++;

        rs[groups[0][v]]++;

        ocenter += yorder[v] * get(vweight, v);
        omax = max(yorder[v], omax);
        omin = min(yorder[v], omin);
        W += get(vweight, v);
    }

    PosMap temp = pos.copy();

    ocenter /= W;
    double o_rg = omax - omin;

    val_t delta = epsilon * K + 1, E = 0, E0;
    E0 = numeric_limits<val_t>::max();
    size_t n_iter = 0;
    val_t step = init_step;
    size_t progress = 0;

    vector<pos_t> ccm;
    vector<size_t> csize;

    vector<size_t> cs;
    vector<double> cprobs;
    for (auto v : vertices_range(g))
    {
        auto r = c[v];
        if (r >= cs.size())
        {
            cs.resize(r + 1);
            cprobs.resize(r + 1);
        }
        cprobs[r] += 1;
    }
    Sampler<size_t> csample(cs, cprobs);

    while (delta > epsilon * K && (max_iter == 0 || n_iter < max_iter))
    {
        delta = 0;
        E0 = E;
        E = 0;

        pos_t ll{numeric_limits<val_t>::max(), numeric_limits<val_t>::max()},
            ur{-numeric_limits<val_t>::max(), -numeric_limits<val_t>::max()};

        ccm.clear();
        csize.clear();

        double ycenter = 0;
        val_t ymax = -std::numeric_limits<val_t>::max();
        val_t ymin = std::numeric_limits<val_t>::max();

        for (auto v : vertices_range(g))
        {
            for (size_t j = 0; j < 2; ++j)
            {
                ll[j] = min(pos[v][j], ll[j]);
                ur[j] = max(pos[v][j], ur[j]);
            }

            size_t s = c[v];
            if (s >= ccm.size())
            {
                ccm.resize(s + 1);
                csize.resize(s + 1, 0);
            }

            csize[s] += get(vweight, v);
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] += pos[v][j] * get(vweight, v);

            ycenter += pos[v][1] * get(vweight, v);
            ymin = min(ymin, pos[v][1]);
            ymax = max(ymax, pos[v][1]);
        }
        ycenter /= W;
        double y_rg = ymax - ymin;

        for (size_t s = 0; s < ccm.size(); ++s)
        {
            if (csize[s] == 0)
                continue;
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] /= csize[s];
        }

        QuadTree<val_t, vweight_t> qt(ll, ur, max_level, num_vertices(g));
        idx_map<size_t, QuadTree<val_t, vweight_t>> qtr;
        for (auto& [r, nr] : rs)
            qtr[r] = QuadTree<val_t, vweight_t>(ll, ur, max_level, nr);

        for (auto v : vertices_range(g))
        {
            if (rs.size() > 1)
                qtr[groups[0][v]].put_pos(0, pos[v], vweight[v]);
            qt.put_pos(0, pos[v], vweight[v]);
        }

        std::shuffle(vertices.begin(), vertices.end(), rng);

        size_t nmoves = 0;
        vector<size_t> Q;
        Q.reserve(num_vertices(g));

        size_t nopen = 0;

        auto get_rf_bh =
            [&](auto v, auto& qt, auto& Q, auto& ftot, bool groups, bool intra)
            {
                pos_t cm{0, 0}, diff{0, 0};

                Q.push_back(0);
                while (!Q.empty())
                {
                    size_t q = Q.back();
                    Q.pop_back();

                    auto& dleafs = qt.get_dense_leafs(q);
                    if (!dleafs.empty())
                    {
                        for (auto& dleaf : dleafs)
                        {
                            val_t d = get_diff(get<0>(dleaf), pos[v], diff);
                            if (d == 0)
                                continue;
                            val_t f;
                            if (groups)
                            {
                                if (intra)
                                    f = -fs_r(gamma, mu, pos[v], get<0>(dleaf));
                                else
                                    f = fs_r(gamma, mu, pos[v], get<0>(dleaf)) + f_r(C, K, p, pos[v], get<0>(dleaf));
                            }
                            else
                            {
                                f = f_r(C, K, p, pos[v], get<0>(dleaf));
                            }
                            f *= get<1>(dleaf) * get(vweight, v);
                            for (size_t l = 0; l < 2; ++l)
                                ftot[l] += f * diff[l];

                            nopen++;
                        }
                    }
                    else
                    {
                        double w = qt[q].get_w();
                        qt[q].get_cm(cm);
                        double d = get_diff(cm, pos[v], diff);
                        if (w > theta * d)
                        {
                            auto leaf = qt.get_leafs(q);
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (qt[leaf].get_count() > 0)
                                    Q.push_back(leaf);
                                ++leaf;
                            }
                        }
                        else
                        {
                            if (d > 0)
                            {
                                val_t f;
                                if (groups)
                                {
                                    if (intra)
                                        f = -fs_r(gamma, mu, pos[v], cm);
                                    else
                                        f = fs_r(gamma, mu, pos[v], cm) + f_r(C, K, p, pos[v], cm);
                                }
                                else
                                {
                                    f = f_r(C, K, p, pos[v], cm);
                                }
                                f *= qt[q].get_count() * get(vweight, v);
                                for (size_t l = 0; l < 2; ++l)
                                    ftot[l] += f * diff[l];

                                nopen++;
                            }
                        }
                    }
                }
            };

        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH)   \
            private(Q) reduction(+:E, delta, nmoves)
        parallel_loop_no_spawn
            (vertices,
             [&](size_t, auto v)
             {
                 pos_t diff{0, 0}, ftot{0, 0};

                 // global repulsive forces
                 if (rs.size() == 1)
                 {
                     get_rf_bh(v, qt, Q, ftot, false, false);
                 }
                 else
                 {
                     get_rf_bh(v, qt, Q, ftot, true, false);
                     get_rf_bh(v, qtr[groups[0][v]], Q, ftot, true, true);
                 }

                 // local attractive forces
                 auto& pos_v = pos[v];
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto& pos_u = pos[u];
                     get_diff(pos_u, pos_v, diff);
                     val_t f = f_a(K, pos_u, pos_v);
                     f *= get(eweight, e) * get(vweight, u) * get(vweight, v);
                     if (groups[0][v] == groups[0][u])
                         f *= kappa;
                     for (size_t l = 0; l < 2; ++l)
                         ftot[l] += f * diff[l];
                 }

                 // inter-component attractive forces
                 if (r > 0 && ccm.size() > 1)
                 {
                     for (size_t i = 0; i < std::min(kc, ccm.size() - 1); ++i)
                     {
                         auto s = csample(rng);
                         if (csize[s] == 0)
                             continue;
                         if (s == size_t(c[v]))
                             continue;
                         val_t d = get_diff(ccm[s], pos[v], diff);
                         if (d == 0)
                             continue;
                         double Kp = K * pow2(HN);
                         val_t f = f_a(Kp, ccm[s], pos[v]) * r * csize[s] * get(vweight, v);
                         for (size_t l = 0; l < 2; ++l)
                             ftot[l] += f * diff[l];
                     }
                 }

                 // yorder repulsive force
                 if (R > 0)
                 {
                     double dz = (yorder[v] - ocenter)/o_rg;
                     double dp = (pos[v][1] - ycenter)/y_rg;
                     ftot[1] +=  R * (dz - dp);
                 }

                 E += pow2(norm(ftot));

                 for (size_t l = 0; l < 2; ++l)
                 {
                     ftot[l] *= step;
                     temp[v][l] += ftot[l];
                 }

                 delta += norm(ftot);
                 nmoves++;
             });

        parallel_loop
            (vertices,
             [&](size_t, auto v)
             {
                 pos[v] = temp[v];
             });

        n_iter++;
        delta /= nmoves;

        if (verbose)
            cout << n_iter << " " << E << " " << step << " "
                 << delta << " " << max_level << " "
                 << nopen / double(HN) << endl;

        if (simple)
        {
            step *= step_schedule;
        }
        else
        {
            if (E < E0)
            {
                ++progress;
                if (progress >= 5)
                {
                    progress = 0;
                    step /= step_schedule;
                }
            }
            else
            {
                progress = 0;
                step *= step_schedule;
            }
        }
    }
}

} // namespace graph_tool


#endif // GRAPH_FDP_HH
