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

#ifndef MCMC_LOOP_HH
#define MCMC_LOOP_HH

#include "config.h"

#include <iostream>

#include <tuple>

#include "hash_map_wrap.hh"
#include "parallel_rng.hh"
#include "graph_python_interface.hh"
#include "samplers.hh"

#ifdef _OPENMP
#include <omp.h>
#endif
namespace graph_tool
{

struct MetropolisStateBase
{
    // required
    auto get_vlist() { return std::array<size_t,0>(); };
    size_t get_niter() { return 0; };

    template <class Node>
    constexpr size_t node_state(Node&&) { return 0; };

    template <class Node, class RNG>
    constexpr size_t move_proposal(Node&&, RNG) { return 0; };

    constexpr static size_t _null_move = std::numeric_limits<size_t>::max();

    template <class Node, class State>
    constexpr std::tuple<double,double> virtual_move_dS(Node&&, State&&)
    { return {0,0}; };

    template <class Node, class State>
    constexpr void perform_move(Node&&, State&&) { };

    // optional
    double get_beta() { return 1; };
    constexpr bool is_determinisitic() { return false; };
    constexpr bool is_sequential() { return true; };

    template <class RNG>
    constexpr double init_iter(RNG&) { return 0; };

    constexpr size_t get_N() { return 0; };

    template <class Node>
    constexpr bool skip_node(Node&&) { return false; };

    template <class Node, class State>
    constexpr void step(Node&&, State&&) { };

    constexpr static bool _force_accept = false;
    constexpr static bool _verbose = false;
};


template <class RNG>
bool metropolis_accept(double dS, double mP, double beta, RNG& rng)
{
    if (std::isinf(beta))
    {
        return dS < 0;
    }
    else
    {
        double a = -dS * beta + mP;
        if (a > 0)
        {
            return true;
        }
        else
        {
            std::uniform_real_distribution<> sample;
            return sample(rng) < exp(a);
        }
    }
}


template <class MCMCState, class RNG>
auto mcmc_sweep(MCMCState& state, RNG& rng)
{
    GILRelease gil;

    auto& vlist = state.get_vlist();
    auto beta = state.get_beta();

    typedef std::remove_const_t<decltype(state._null_move)> move_t;
    constexpr bool single_step =
        std::is_same_v<std::remove_reference_t<decltype(state.move_proposal(vlist.front(), rng))>,
                       move_t>;

    double S = 0;
    size_t nattempts = 0;
    size_t nmoves = 0;

    for (size_t iter = 0; iter < state.get_niter(); ++iter)
    {
        if (state.is_sequential() && !state.is_deterministic())
            std::shuffle(vlist.begin(), vlist.end(), rng);

        size_t nsteps = 1;
        auto get_N =
            [&]
            {
                if constexpr (single_step)
                    return vlist.size();
                else
                    return state.get_N();
            };

        S += state.init_iter(rng);

        for (size_t vi = 0; vi < get_N(); ++vi)
        {
            auto& v = (state.is_sequential()) ?
                vlist[vi] : uniform_sample(vlist, rng);

            if (state.skip_node(v))
                continue;

            if (state._verbose > 1)
            {
                auto&& r = state.node_state(v);
                std::print("{}: {}", v, r);
            }

            auto&& ret = state.move_proposal(v, rng);

            auto get_s =
                [&]() -> move_t&
                {
                    if constexpr (single_step)
                    {
                        return ret;
                    }
                    else
                    {
                        nsteps = get<1>(ret);
                        return get<0>(ret);
                    }
                };

            move_t& s = get_s();

            if (s == state._null_move)
            {
                if (state._verbose > 1)
                    std::println(" (null proposal)");
                continue;
            }

            double dS, mP;
            std::tie(dS, mP) = state.virtual_move_dS(v, s);

            nattempts += nsteps;

            bool accept = false;
            if (metropolis_accept(dS, mP, beta, rng) || state._force_accept)
            {
                state.perform_move(v, s);
                nmoves += nsteps;
                S += dS;
                accept = true;
            }

            state.step(v, s);

            if (state._verbose > 1)
                std::println(" -> {} accept: {}, dS: {}, mP: {}, a: {}, S: {}",
                             s, accept, dS, mP, -dS * beta + mP, S);
        }

        if (state.is_sequential() && state.is_deterministic())
            std::reverse(vlist.begin(), vlist.end());
    }
    return make_tuple(S, nattempts, nmoves);
}

} // graph_tool namespace

#endif //MCMC_LOOP_HH
