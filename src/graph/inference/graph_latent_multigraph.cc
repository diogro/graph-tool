// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_latent_multigraph.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

void latent_multigraph(GraphInterface& gi, std::any aw, std::any atheta_out,
                       std::any atheta_in, double epsilon, size_t max_niter,
                       bool verbose)
{
    typedef eprop_map_t<double> emap_t;
    typedef vprop_map_t<double> vmap_t;
    auto w = std::any_cast<emap_t>(aw).get_unchecked();
    auto theta_out = std::any_cast<vmap_t>(atheta_out).get_unchecked();
    auto theta_in = std::any_cast<vmap_t>(atheta_in).get_unchecked();

    run_action<>()
        (gi, [&](auto& g){ get_latent_multigraph(g, w, theta_out, theta_in,
                                                 epsilon, max_niter, verbose); })();
}

using namespace boost::python;


using namespace boost::python;

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("latent_multigraph", &latent_multigraph);
 });
