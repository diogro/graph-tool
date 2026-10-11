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

#include "config.h"
#include "util.hh"

#include <random>

namespace graph_tool
{

template <class RNG>
size_t sample_geometric(double mean, RNG& rng)
{
    double p = 1./(mean + 1);
    std::geometric_distribution<size_t> sample(p);
    return sample(rng);
}

template <class Val>
double geometric_lprob(size_t x, Val mean)
{
    return x * log(mean) - (x + 1) * log1p(mean);
}

template <class RNG>
size_t sample_neg_binomial(size_t r, double p, RNG& rng)
{
    std::negative_binomial_distribution<size_t> sample(r, p);
    return sample(rng);
}

template <class V1, class V2>
double neg_binomial_lprob(V1 m, size_t r, V2 p)
{
    return lbinom(m + r - 1, m) + m * log1p(-p) + r * log(p);
}

} // namespace graph_tool
