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

from .. import Graph, GraphView, _get_rng, _prop, PropertyMap, \
    perfect_prop_hash, Vector_size_t, group_vector_property
from . blockmodel import DictState
from . util import *
import numpy as np

from . base_states import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

@entropy_state_signature
class NormCutState(CLabelState, MCMCState, MultiflipMCMCState,
                   MultilevelMCMCState, GibbsMCMCState, DrawBlockState):
    r"""Obtain the partition of a network according to the normalized cut.

    .. warning::

       Do not use this approach in the analysis of networks without
       understanding the consequences. This algorithm is included only for
       comparison purposes. In general, the inference-based approaches based on
       :class:`~graph_tool.inference.BlockState`,
       :class:`~graph_tool.inference.NestedBlockState`, and
       :class:`~graph_tool.inference.PPBlockState` should be
       universally preferred.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be partitioned.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.

    """

    def __init__(self, g, b=None, eweight=None, clabel=None, entropy_args={}):
        EntropyState.__init__(self, entropy_args=entropy_args)
        self.g = g = GraphView(g, directed=False)
        CLabelState.__init__(self, g=g, b=b, clabel=clabel)
        if eweight is None:
            eweight = g.new_ep("int64_t", 1)
        else:
            eweight = g.own_property(eweight)
        self.eweight = eweight

        self._state = libinference.make_norm_cut_state(self)

    def __repr__(self):
        return "<NormCutState object with %d blocks, for graph %s, at 0x%x>" % \
            (self.get_B(), str(self.g), id(self))

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b.fa))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b.fa), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*log(w)).sum())

    @copy_state_wrap
    def _entropy(self, **kwargs):
        r"""Return the normalized cut. See :meth:`~NormCutState.norm_cut`."""
        return EntropyState._entropy_dispatch(**dict(locals(), **kwargs))

    def norm_cut(self):
        r"""Return the normalized cut.

        Notes
        -----

        The normalized cut is defined as

        .. math::

           B - \sum_r \frac{e_{rr}}{e_r}

        Where :math:`B` is the number of groups, :math:`e_{rr}` is twice
        the number of edges between nodes of group :math:`r`, and :math:`e_r` is
        the sum of degrees of nodes in group :math:`r`.

        """

        return self.entropy()
