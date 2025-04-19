#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from .. import Graph, _get_rng, _prop
from . base_states import *
from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np

@entropy_state_signature
class PottsState(MCMCState, MultiflipMCMCState,
                 GibbsMCMCState, DrawBlockState):
    r"""Sample from a generalized Potts model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    f : :class:`~numpy.ndarray`
        :math:`q\times q` spin iteraction strengths.
    w : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights. If not supplied, it will be
        assummed to be unity.
    theta : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map of type ``vector<double>`` with the node fields.
        If not supplied, it will be assummed to be zero.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial spin labels. If not supplied, a random distribution will be
        used.

    """

    def __init__(self, g, f, w=None, theta=None, b=None, **kwargs):
        EntropyState.__init__(self, entropy_args={})
        self.g = g
        self.f = np.asarray(f, dtype="double")
        self.q = f.shape[0]
        if b is None:
            b = g.new_vp("int64_t",
                         vals=np.random.randint(0, self.q, g.num_vertices()))
        elif b.value_type() != "int64_t":
            b = b.copy("int64_t")
        self.b = b
        if theta is None:
            theta = g.new_vp("vector<double>")
        elif theta.value_type() != "vector<double>":
            theta = theta.copy("vector<double>")
        self.theta = theta
        if w is None:
            w = g.new_ep("double", val=1)
        elif w.value_type() != "double":
            w = w.copy("double")
        self.w = w
        self._state = libinference.make_potts_state(self)

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        args = self.__getstate__()
        args.update(**kwargs)
        return PottsState(**args)

    def __getstate__(self):
        state = EntropyState.__getstate__(self)
        return dict(state, g=self.g, f=self.f, b=self.b, w=self.w,
                    theta=self.theta)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<PottsState object with %d spins, for graph %s, at 0x%x>" % \
            (self.q, str(self.g), id(self))

    @copy_state_wrap
    def _entropy(self, **kwargs):
        r"""Returns the energy of generalized Potts model.

        Notes
        -----

        The energy of the generalized Potts model is given by

        .. math::

             = -\sum_{i<j}w_{ij}A_{ij}f_{b_i,b_j} - \sum_i\theta^{(i)}_{b_i}.

        """

        eargs = self._get_entropy_args(locals())

        S = self._state.entropy(eargs)

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return S

    def _gen_eargs(self, args):
        return libinference.potts_eargs()

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.potts_mcmc_sweep(mcmc_state, self._state,
                                              _get_rng())

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.potts_multiflip_mcmc_sweep(mcmc_state,
                                                        self._state,
                                                        _get_rng())

    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.potts_gibbs_sweep(gibbs_state, self._state,
                                               _get_rng())

