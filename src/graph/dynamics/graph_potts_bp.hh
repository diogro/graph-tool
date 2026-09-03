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

#ifndef GRAPH_BP_HH
#define GRAPH_BP_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "random.hh"
#include "parallel_rng.hh"

#include "../inference/support/util.hh"
#include "samplers.hh"

namespace graph_tool
{

class PottsBPState
{
public:

    typedef eprop_map_t<std::vector<double>>::unchecked_t emmap_t;
    typedef vprop_map_t<std::vector<double>>::unchecked_t vmmap_t;
    typedef eprop_map_t<double>::unchecked_t emap_t;
    typedef vprop_map_t<std::vector<double>>::unchecked_t vmap_t;
    typedef vprop_map_t<uint8_t>::unchecked_t vfmap_t;

    template <class Graph, class RNG>
    PottsBPState(Graph& g, boost::multi_array_ref<double, 2> f, emap_t x,
                 vmap_t theta, emmap_t m, emmap_t em, vmmap_t vm,
                 bool marginal_init, vfmap_t frozen, bool init_vm, bool init_em,
                 bool init_m, RNG& rng)
        : _f(f), _x(x), _theta(theta), _m(m), _em(em), _vm(vm), _q(f.shape()[0]),
          _frozen(frozen)
    {
        std::uniform_real_distribution<> rand;

        if (init_vm)
        {
            for (auto v : vertices_range(g))
            {
                if (_vm[v].empty())
                {
                    for (size_t r = 0; r < _q; ++r)
                        _vm[v].push_back(log(rand(rng)));
                }
                _vm[v].resize(_q + 1);
                double lZ = _vm[v][_q] = log_Zm(_vm[v].begin());
                for (size_t r = 0; r < _q; ++r)
                    _vm[v][r] -= lZ;
            }
        }

        if (init_em || init_m)
        {
            for (auto e : edges_range(g))
            {
                if (init_em)
                    _em[e].resize(_q * _q);

                if (init_m)
                {
                    _m[e].resize(2 * (_q + 1));

                    auto u = source(e, g);
                    auto v = target(e, g);
                    auto m_uv = get_message(g, e, _m, u);
                    auto m_vu = get_message(g, e, _m, v);

                    if (marginal_init)
                    {
                        for (size_t r = 0; r < _q + 1; ++r)
                        {
                            m_uv[r] = _vm[u][r];
                            m_vu[r] = _vm[v][r];
                        }
                    }
                    else
                    {
                        for (size_t r = 0; r < _q; ++r)
                        {
                            m_uv[r] = log(rand(rng));
                            m_vu[r] = log(rand(rng));
                        }
                        m_uv[_q] = log_Zm(m_uv);
                        m_vu[_q] = log_Zm(m_vu);

                        if (_frozen[u])
                        {
                            for (size_t r = 0; r < _q + 1; ++r)
                                m_uv[r] = _vm[u][r];
                        }

                        if (_frozen[v])
                        {
                            for (size_t r = 0; r < _q + 1; ++r)
                                m_vu[r] = _vm[v][r];
                        }
                    }
                }
            }
        }

        _temp = _m.copy();
    };

    template <class Graph, class Edge, class ME>
    std::vector<double>::iterator
    get_message(Graph& g, const Edge& e, ME& me, size_t s)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        if (u > v)
            std::swap(u, v);
        auto& m = me[e];
        if (s == u)
            return m.begin();
        else
            return m.begin() + _q + 1;
    }

    template <class Iter>
    double log_Zm(Iter iter)
    {
        double lZ = -_inf;
        for (size_t r = 0; r < _q; ++r)
            lZ = log_sum_exp(iter[r], lZ);
        return lZ;
    }

    template <class Graph, class Iter>
    double update_message(Graph& g, Iter m, size_t s, size_t t)
    {
        std::vector<double> nm(_q);
        for (size_t r = 0; r < _q; ++r)
        {
            nm[r] = _theta[s][r];
            for (auto e : out_edges_range(s, g))
            {
                auto u = target(e, g);
                if (u == t)
                    continue;
                auto m_u = get_message(g, e, _m, u);
                auto w = _x[e];
                double temp = -_inf;
                auto fr = _f[r];
                for (size_t x = 0; x < _q; ++x)
                    temp = log_sum_exp(m_u[x] + w * fr[x], temp);
                nm[r] += temp;
            }
        }
        double lZ = log_Zm(nm.begin());
        double delta = 0;
        for (size_t r = 0; r < _q; ++r)
        {
            auto nm_r = nm[r] - lZ;
            auto& m_r =  m[r];
            delta += abs(nm_r - m_r);
            m_r = nm_r;
        }
        m[_q] = lZ;
        return delta;
    }

    template <class Graph, class Edge, class ME>
    double update_edge(Graph& g, const Edge& e, ME& me)
    {
        auto u = source(e, g);
        auto v = target(e, g);
        auto muv = get_message(g, e, me, u);
        auto mvu = get_message(g, e, me, v);
        double delta = 0;
        if (!_frozen[u])
            delta += update_message(g, muv, u, v);
        if (!_frozen[v])
            delta += update_message(g, mvu, v, u);
        return delta;
    }

    template <class Graph>
    void update_marginals(Graph& g)
    {
        parallel_vertex_loop
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 auto m = _vm[v].begin();
                 update_message(g, m, v, _null);
             });

        parallel_edge_loop
            (g,
             [&] (auto e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (u > v)
                     std::swap(u, v);

                 auto m_vu = get_message(g, e, _m, v);
                 auto m_uv = get_message(g, e, _m, u);

                 auto& m = _em[e];
                 auto x = _x[e];
                 for (size_t r = 0; r < _q; ++r)
                     for (size_t s = 0; s < _q; ++s)
                         m[r + s * _q] = x * _f[r][s] + m_uv[r] + m_vu[s];

                 auto lZ = log_sum_exp(m);
                 for (auto& y : m)
                     y -= lZ;
             });
    }

    template <class Graph>
    double iterate(Graph& g, size_t niter)
    {
        double delta = 0;
        for (size_t i = 0; i < niter; ++i)
        {
            delta = 0;
            for (auto e : edges_range(g))
                delta += update_edge(g, e, _m);
        }
        return delta;
    }

    template <class Graph>
    double iterate_parallel(Graph& g, size_t niter)
    {
        double delta = 0;
        for (size_t i = 0; i < niter; ++i)
        {
            delta = 0;
            #pragma omp parallel reduction(+:delta)
            parallel_edge_loop_no_spawn
                (g,
                 [&] (const auto& e)
                 {
                     _temp[e] = _m[e];
                     delta += update_edge(g, e, _temp);
                 });

            parallel_edge_loop
                (g,
                 [&] (const auto& e)
                 {
                     _m[e] = _temp[e];
                 });
        }
        return delta;
    }

    template <class Graph>
    double log_Z(Graph& g)
    {
        double lZ = 0;

        #pragma omp parallel reduction(+:lZ)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 update_message(g, _vm[v].begin(), v, _null);
                 lZ += _vm[v][_q];
             });

        #pragma omp parallel reduction(+:lZ)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 //auto v = target(e, g);
                 auto muv = get_message(g, e, _m, u);
                 lZ -= _vm[u][_q] - muv[_q];
             });

        return lZ;
    }

    template <class Graph>
    double energy(Graph& g)
    {
        double U = 0;

        #pragma omp parallel reduction(+:U)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 for (size_t r = 0; r < _q; ++r)
                     U -= exp(_vm[v][r]) * _theta[v][r];
             });

        #pragma omp parallel reduction(+:U)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto x = _x[e];
                 auto& em = _em[e];
                 for (size_t r = 0; r < _q; ++r)
                     for (size_t s = 0; s < _q; ++s)
                         U -= exp(em[r + s * _q]) * (x * _f[r][s]);
             });

        return U;
    }

    template <class Graph>
    double entropy(Graph& g)
    {
        double S = 0;

        #pragma omp parallel reduction(+:S)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 double S_v = 0;
                 for (size_t r = 0; r < _q; ++r)
                     S_v -= _vm[v][r] * exp(_vm[v][r]);
                 S += (1. - out_degree(v, g)) * S_v;
             });

        #pragma omp parallel reduction(+:S)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 for (auto lp : _em[e])
                     S -= lp * exp(lp);
             });

        return S;
    }

    template <class Graph>
    double marginal_pair_lprob(Graph& g, size_t u, size_t v, size_t r, size_t s)
    {
        auto [e, has_edge] = edge(u, v, g);

        if (!has_edge)
            return _vm[u][r] + _vm[v][s];
        if (u > v)
            std::swap(s, r);
        return _em[e][r + s * _q];
    }

    template <class Graph, class VMap>
    double bethe_lprob(Graph& g, VMap s)
    {
        double L = 0;

        #pragma omp parallel reduction(+:L)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 L += (1. - out_degree(v, g)) * _vm[v][s[v]];
             });

        #pragma omp parallel reduction(+:L)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (u > v)
                     std::swap(u, v);
                 L += _em[e][s[u] + s[v] * _q];
             });

        return L;
    }

    template <class Graph>
    double marginal_entropy(Graph& g)
    {
        double S = 0;

        #pragma omp parallel reduction(+:S)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 for (size_t r = 0; r < _q; ++r)
                     S -= _vm[v][r] * exp(_vm[v][r]);
             });

        return S;
    }

    template <class Graph>
    double marginal_mean(Graph& g)
    {
        double M = 0;
        size_t N = 0;

        #pragma omp parallel reduction(+:M, N)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 for (size_t r = 0; r < _q; ++r)
                     M += r * exp(_vm[v][r]);
                 N++;
             });

        return M / N;
    }

    template <class Graph, class VMap>
    double sample_energy(Graph& g, VMap s)
    {
        double H = 0;
        #pragma omp parallel reduction(+:H)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 H -= _theta[v][s[v]];
             });

        #pragma omp parallel reduction(+:H)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (_frozen[u] && _frozen[v])
                     return;
                 H -= _x[e] * _f[s[u]][s[v]];
             });

        return H;
    }

    template <class Graph, class VMap>
    double sample_energies(Graph& g, VMap ss)
    {
        double H = 0;
        #pragma omp parallel reduction(+:H)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 for (auto s : ss[v])
                     H += _theta[v][s];
             });

        #pragma omp parallel reduction(+:H)
        parallel_edge_loop_no_spawn
            (g,
             [&] (const auto& e)
             {
                 auto u = source(e, g);
                 auto v = target(e, g);
                 if (_frozen[u] && _frozen[v])
                     return;
                 auto& s_u = ss[u];
                 auto& s_v = ss[v];
                 auto xe = _x[e];
                 for (size_t m = 0; m < s_u.size(); ++m)
                     H += xe * _f[s_u[m]][s_v[m]];
             });

        return H;
    }

    template <class Graph, class VMap>
    double marginal_lprob(Graph& g, VMap s)
    {
        double L = 0;
        #pragma omp parallel reduction(+:L)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 L += _vm[v][s[v]];
             });
        return L;
    }

    template <class Graph, class VMap>
    double marginal_lprobs(Graph& g, VMap ss)
    {
        double L = 0;
        #pragma omp parallel reduction(+:L)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 if (_frozen[v])
                     return;
                 for (auto s : ss[v])
                     L += _vm[v][s];
             });
        return L;
    }

    template <class Graph, class VMap, class RNG>
    void marginal_sample(Graph& g, VMap s, RNG& rng_)
    {
        parallel_rng<rng_t> prng(rng_);

        std::vector<int> vals(_q);
        std::vector<double> probs(_q);
        for (size_t r = 0; r < _q; ++r)
            vals[r] = r;

        #pragma omp parallel firstprivate(probs)
        parallel_vertex_loop_no_spawn
            (g,
             [&] (auto v)
             {
                 auto& rng = prng.get(rng_);
                 for (size_t r = 0; r < _q; ++r)
                     probs[r] = exp(_vm[v][r]);
                 Sampler<int> sampler(vals, probs);
                 s[v] = sampler(rng);
             });
    }

    template <class Graph, class SMap, class VMap, class RNG>
    void sample(Graph& g, VMap vorder, SMap s, bool parallel, size_t maxiter,
                double epsilon, RNG& rng)
    {
        auto f_tmp = _frozen.copy();
        auto m_tmp = _m.copy();
        auto em_tmp = _em.copy();
        auto vm_tmp = _vm.copy();

        std::vector<int> vals(_q);
        std::vector<double> probs(_q);
        for (size_t r = 0; r < _q; ++r)
            vals[r] = r;

        for (auto v : vorder)
        {
            double delta = epsilon + 1;
            size_t iter = 0;
            while (delta > epsilon && iter++ < maxiter)
                delta = parallel ? iterate_parallel(g, 1) : iterate(g, 1);

            update_marginals(g);

            for (size_t r = 0; r < _q; ++r)
                probs[r] = exp(_vm[v][r]);
            Sampler<int> sampler(vals, probs);

            s[v] = sampler(rng);

            _frozen[v] = true;

            for (size_t r = 0; r < _q; ++r)
                _vm[v][r] = -_inf;
            _vm[v][s[v]] = 0;

            for (auto e : out_edges_range(v, g))
            {
                auto muv = get_message(g, e, _m, v);
                for (size_t r = 0; r < _q; ++r)
                    muv[r] = _vm[v][r];
            }
        }

        for (auto v : vertices_range(g))
        {
            _frozen[v] = f_tmp[v];
            _vm[v] = vm_tmp[v];
        }

        for (auto e : edges_range(g))
        {
            _m[e] = m_tmp[e];
            _em[e] = em_tmp[e];
        }
    }

private:
    boost::multi_array_ref<double, 2> _f;
    emap_t _x;
    vmap_t _theta;
    emmap_t _m;
    emmap_t _temp;
    emmap_t _em;
    vmmap_t _vm;
    size_t _q;
    vfmap_t _frozen;
    constexpr static size_t _null = std::numeric_limits<size_t>::max();
    constexpr static double _inf = std::numeric_limits<double>::infinity();
};

} // namespace graph_tool

#endif // GRAPH_BP_HH
