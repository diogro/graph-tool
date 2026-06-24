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

#ifndef BOX_SAMPLER_HH
#define BOX_SAMPLER_HH

#include "samplers.hh"

namespace graph_tool
{

class BoxSampler
{
public:
    BoxSampler(double alpha, size_t max)
        : _alpha(alpha), _max(max)
    {
        setup();
    }

    void setup()
    {
        for (size_t m = 0; m < _max; ++m)
        {
            _items.push_back(m + 1);
            _probs.push_back(std::pow(m + 1, -_alpha));
            if (m == 0)
                _cums.push_back(_probs.back());
            else
                _cums.push_back(_probs.back() + _cums.back());
        }
        _sampler = Sampler<size_t,false>(_items, _probs);
    }

    template <class RNG>
    size_t sample(RNG& rng)
    {
        return _sampler.sample(rng);
    }

    double prob(size_t m)
    {
        return _probs[m-1]/_cums.back();
    }

    double cum_prob(size_t m)
    {
        return _cums[m-1]/_cums.back();
    }

private:
    double _alpha;
    size_t _max;
    std::vector<size_t> _items;
    std::vector<double> _probs;
    std::vector<double> _cums;

    Sampler<size_t, mpl::false_> _sampler;
};

}
#endif // BOX_SAMPLER_HH
