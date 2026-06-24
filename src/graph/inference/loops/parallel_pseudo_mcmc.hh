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

    size_t nattempts = 0;
    size_t nmoves = 0;

    parallel_rng<rng_t> prng(rng_);

    double S = 0;

    for (size_t iter = 0; iter < state.get_niter(); ++iter)
    {
        S += state.init_iter(rng_);

        if (!state.is_deterministic())
            std::shuffle(vlist.begin(), vlist.end(), rng_);

        #pragma omp parallel for schedule(runtime) \
            reduction(+: S, nattempts, nmoves)
        for (size_t i = 0; i < vlist.size(); ++i)
        {
            auto& rng = prng.get(rng_);

            auto& v = vlist[i];

            bool locked = state.stage_proposal(v, rng);

            if (!locked)
                continue;

            auto s = state.move_proposal(v, rng);

            nattempts++;

            if (s == state._null_move)
            {
                state.proposal_unlock(v);
                continue;
            }

            state.lock_move();

            auto [dS, mP] = state.virtual_move_dS(v, s);

            if (metropolis_accept(dS, mP, beta, rng))
            {
                nmoves++;
                state.perform_move(v, s); // should unlock
                S += dS;
            }
            else
            {
                state.proposal_unlock(v);
                state.unlock_move();
            }
        }
    }

    return make_tuple(S, nattempts, nmoves);
}

} // graph_tool namespace

#endif //PARALLEL_PSEUDO_MCMC_HH
