#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2026 Tiago de Paula Peixoto <tiago@skewed.de>
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

from .. import Graph, GraphView, _get_rng, _prop
from . base_states import *
from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np

from . blockmodel import *

@entropy_state_signature
class PottsState(CLabelState, MCMCState, MultiflipMCMCState, MultilevelMCMCState,
                 GibbsMCMCState, DrawBlockState):
    r"""Sample from a generalized Potts model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    f : :class:`~numpy.ndarray`
        :math:`q\times q` spin iteraction strengths.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge property map with the edge weights. If not supplied, it will be
        assummed to be unity.
    theta : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map of type ``vector<double>`` with the node fields.
        If not supplied, it will be assummed to be zero.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial spin labels. If not supplied, a random distribution will be
        used.

    """

    def __init__(self, g, f, b=None, eweight=None, theta=None, clabel=None, entropy_args={}):
        EntropyState.__init__(self, entropy_args=entropy_args)
        CLabelState.__init__(self, g=g, clabel=clabel)
        self.g = GraphView(g, directed=False)
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
        if eweight is None:
            eweight = g.new_ep("double", val=1)
        self.eweight = eweight
        self._state = libinference.make_potts_state(self)

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

            H = -\sum_{i<j}w_{ij}A_{ij}f_{b_i,b_j} - \sum_i\theta^{(i)}_{b_i}.

        """
        return EntropyState._entropy_dispatch(**dict(locals(), **kwargs))
