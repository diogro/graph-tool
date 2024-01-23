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

#ifndef PARALLEL_PSEUDO_MCMC_HH
#define PARALLEL_PSEUDO_MCMC_HH

#include "config.h"

#include <iostream>

#include <tuple>

#include "parallel_rng.hh"
#include "mcmc_loop.hh"

namespace graph_tool
{

template <class MCMCState, class RNG>
auto pseudo_mcmc_sweep(MCMCState& state, RNG& rng_)
{
    GILRelease gil;

    auto& vlist = state.get_vlist();
    auto beta = state.get_beta();

    typedef std::remove_const_t<decltype(state._null_move)> move_t;

    size_t nattempts = 0;
    size_t nmoves = 0;

    parallel_rng<rng_t> prng(rng_);

    std::mutex move_lock;
    std::vector<std::tuple<move_t, bool>> moves(vlist.size(),
                                                {state._null_move, false});

    double Sa = state.entropy();

    for (size_t iter = 0; iter < state.get_niter(); ++iter)
    {
        state.init_iter(rng_);

        if (!state.is_deterministic())
            std::shuffle(vlist.begin(), vlist.end(), rng_);

        #pragma omp parallel for schedule(runtime) reduction(+: nattempts)
        for (size_t i = 0; i < vlist.size(); ++i)
        {
            auto& rng = prng.get(rng_);

            auto v = vlist[i];

            bool locked = state.stage_proposal(v, rng);

            if (!locked)
            {
                moves[i] = {state._null_move, false};
                continue;
            }

            move_t s = state.move_proposal(v, rng);

            nattempts++;

            if (s == state._null_move)
            {
                moves[i] = {s, false};
                continue;
            }

            auto [dS, mP] = state.virtual_move_dS(v, s);

            moves[i] = {s, metropolis_accept(dS, mP, beta, rng)};
        }

        #pragma omp parallel for schedule(runtime) reduction(+: nmoves)
        for (size_t i = 0; i < vlist.size(); ++i)
        {
            auto v = vlist[i];
            auto [s, accept] = moves[i];

            state.proposal_unlock(v);

            if (!accept)
                continue;

            {
                std::unique_lock lock(move_lock);
                state.perform_move(v, s, lock); // optional: unlock internally before parallel tail
            }

            nmoves++;
        }
    }

    double Sb = state.entropy();

    return make_tuple(Sb - Sa, nattempts, nmoves);
}

} // graph_tool namespace

#endif //PARALLEL_PSEUDO_MCMC_HH
