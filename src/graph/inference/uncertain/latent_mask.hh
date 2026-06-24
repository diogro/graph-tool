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

#ifndef GRAPH_BLOCKMODEL_MEASURED_HH
#define GRAPH_BLOCKMODEL_MEASURED_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "uncertain.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define LATENT_MASK_STATE_params                                               \
    ((g, decltype(all_graph_views), 1))                                        \
    ((ecount, eprop_map_t<int64_t>, 0))                                        \
    ((f, eprop_map_t<double>, 0))                                              \
    ((f_default, double, 0))                                                   \
    ((max_m, int64_t, 0))                                                      \
    ((init, double, 0))

template <class BlockState>
struct LatentMask
{
    GEN_STATE_BASE(LatentMaskStateBase, LATENT_MASK_STATE_params)

    template <class... Ts>
    class LatentMaskState
        : public LatentMaskStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(LatentMaskStateBase<Ts...>,
                         LATENT_MASK_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, LATENT_MASK_STATE_params)

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        LatentMaskState(BlockState& block_state, ATs&&... args)
            : LatentMaskStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state)
        {
            GILRelease gil_release;

            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                get_u_edge<true>(source(e, _u), target(e, _u)) = e;
                _E += _eweight[e];
            }

            _edges.resize(num_vertices(_g));
            auto lf = _lf.get_checked();

            for (auto e : edges_range(_g))
            {
                get_edge<true>(source(e, _g), target(e, _g)) = e;
                lf[e] = log1p(-_f[e]);

                auto m = get_u_count(source(e, _g), target(e, _g));
                if (_f[e] == 0)
                {
                    if (_ecount[e] > 0)
                        throw ValueException("positive edge counts inconsistent with f_ij = 0!");
                }
                else if (_f[e] == 1)
                {
                    if (m < size_t(_ecount[e]))
                    {
                        if (_ecount[e] - m > 0)
                            add_edge(source(e, _g), target(e, _g), _ecount[e] - m);
                    }
                    else if (m > size_t(_ecount[e]))
                    {
                        if (m - _ecount[e]> 0)
                            remove_edge(source(e, _g), target(e, _g), m - _ecount[e]);
                    }
                }
                else if (m < size_t(_ecount[e]))
                {
                    if (_ecount[e] - m > 0)
                        add_edge(source(e, _g), target(e, _g), _ecount[e] - m);
                }
            }

            if (_init > 0)
            {
                for (auto e : edges_range(_g))
                {
                    if (_f[e] == 0)
                        continue;
                    auto m_g = _ecount[e];
                    auto k = std::max(std::min(size_t(m_g/_f[e]),
                                               size_t(_max_m)),
                                      size_t(m_g));
                    auto m = get_u_count(source(e, _g), target(e, _g));
                    if (k > m)
                        add_edge(source(e, _g), target(e, _g), k - m);
                    else if (k < m)
                        remove_edge(source(e, _g), target(e, _g), m - k);
                }
            }

            _lf_default = log1p(-_f_default);
        }

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typename BlockState::g_t& _u = _block_state._g;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;

        size_t _E = 0;

        f_t _lf;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _u_edges;
        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _edges;

        double _lf_default;

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];
            if (insert)
                return qe[v];
            auto iter = qe.find(v);
            if (iter != qe.end())
                return iter->second;
            return _null_edge;
        }

        template <bool insert=false>
        auto& get_u_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _u, _u_edges);
        }

        template <bool insert=false>
        auto& get_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _g, _edges);
        }

        auto get_g_count(size_t u, size_t v)
        {
            auto e = get_edge(u, v);
            if (e == _null_edge)
                return std::make_tuple(int64_t(0), _f_default, _lf_default);
            return std::make_tuple(_ecount[e], _f[e], _lf[e]);
        }

        size_t get_u_count(size_t u, size_t v)
        {
            auto e = get_u_edge(u, v);
            if (e == _null_edge)
                return 0;
            return _block_state._eweight[e];
        }

        double entropy(const uentropy_args_t& ea)
        {
            double S = 0;

            if (ea.latent_edges)
            {
                for (auto e : edges_range(_u))
                {
                    auto [m_g, f, lf] = get_g_count(source(e, _g), target(e, _g));
                    auto m_u = _block_state._eweight[e];

                    if (m_u < m_g)
                    {
                        cout << source(e, _g) << " " << target(e, _g) << " "
                             << m_u << " " << m_g  << endl;
                        S = -std::numeric_limits<double>::infinity();
                        break;
                    }

                    S += lbinom_fast(m_u, m_g);

                    if (f == 0)
                    {
                        if (m_g > 0)
                        {
                            S = -std::numeric_limits<double>::infinity();
                            break;
                        }
                    }
                    else if (f == 1)
                    {
                        if (m_g != m_u)
                        {
                            S = -std::numeric_limits<double>::infinity();
                            break;
                        }
                    }
                    else
                    {
                        S += m_g * log(f) + (m_u - m_g) * lf;
                    }
                }
            }

            if (ea.density)
                S += -(_E * log(ea.aE)) + lgamma_fast(_E + 1) - ea.aE;

            return -S;
        }

        double remove_edge_dS(size_t u, size_t v, int64_t dm, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);

            double dS = _block_state.modify_edge_dS(u, v, e, -dm, ea);

            if (ea.density)
            {
                dS += log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 - dm) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                auto [m_g, f, lf] = get_g_count(u, v);
                auto m = _block_state._eweight[e];

                if (m - dm < m_g)
                    return std::numeric_limits<double>::infinity();

                if (f == 1)
                {
                    if (m - dm == m_g)
                        return -std::numeric_limits<double>::infinity();
                    return std::numeric_limits<double>::infinity();
                }
                else if (f > 0)
                {
                    dS += lbinom_fast(m, m_g);
                    dS -= lbinom_fast(m - dm, m_g);
                    dS += dm * lf;
                }
            }

            return dS;
        }

        double add_edge_dS(size_t u, size_t v, int64_t dm, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);

            auto m = (e == _null_edge) ? 0 : _eweight[e];

            if (m + dm > _max_m)
                return numeric_limits<double>::infinity();

            double dS = _block_state.modify_edge_dS(u, v, e, dm, ea);

            if (ea.density)
            {
                dS -= log(ea.aE) * dm;
                dS += lgamma_fast(_E + 1 + dm) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                auto [m_g, f, lf] = get_g_count(u, v);

                if (f == 1)
                {
                    if (m == m_g)
                        return std::numeric_limits<double>::infinity();
                    if (m + dm == m_g)
                        return -std::numeric_limits<double>::infinity();
                }
                else if (f > 0)
                {
                    dS += lbinom_fast(m, m_g);
                    dS -= lbinom_fast(m + dm, m_g);
                    dS -= dm * lf;
                }
            }
            return dS;
        }

        void remove_edge(size_t u, size_t v, int64_t dm)
        {
            auto& e = get_u_edge(u, v);
            _block_state.template modify_edge<false>(u, v, e, dm);
            _E -= dm;
        }

        void add_edge(size_t u, size_t v, int64_t dm)
        {
            auto& e = get_u_edge<true>(u, v);
            _block_state.template modify_edge<true>(u, v, e, dm);
            _E += dm;
        }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MEASURED_HH
