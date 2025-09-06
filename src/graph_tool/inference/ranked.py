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

from .. import Graph, GraphView, _get_rng, _prop, PropertyMap, \
    edge_endpoint_property, Vector_int64_t
from . blockmodel import BlockState
from . base_states import *
from . util import *
from .. generation import contract_parallel_edges

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math
from scipy.stats import rankdata

@entropy_state_signature
class RankedBlockState(BlockState):
    r"""Obtain the ordered partition of a network according to the ranked
    stochastic block model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    u : :class:`~graph_tool.PropertyMap` or iterable (optional, default: ``None``)
        Ordering of the group labels. It should contain a map from each group
        label to an integer label, inidicating how the groups
        should be ordered. If not supplied, the numeric values of the group
        lalbels will be used to initialize the ordering.
    distance: ``bool`` (optional, default: ``False``)
        If ``True``, the distance-based version of the model will be used.
    entropy_args: ``dict`` (optional, default: ``{}``)
        Override default arguments for :meth:`~RankedBlockState.entropy()`
        method and releated operations.

    References
    ----------
    .. [peixoto-ordered-2022] Tiago P. Peixoto, "Ordered community detection in
       directed networks", Phys. Rev. E 106, 024305 (2022),
       :doi:`10.1103/PhysRevE.106.024305`, :arxiv:`2203.16460`

    """

    def __init__(self, g, b=None, u=None, distance=False, unique=False, M=None, **kwargs):
        if not g.is_directed():
            raise ValueError("RankedBlockState can only be used with directed graphs")

        BlockState.__init__(self, g, b=b, **kwargs)
        self.distance = distance
        self.unique = unique
        if M is None:
            M = g.num_vertices()
        self.M = M

        if u is None:
            u = Vector_int64_t()

        if isinstance(u, PropertyMap):
            u = Vector_int64_t(init=u.fa)
        elif not isinstance(u, Vector_int64_t):
            u = Vector_int64_t(init=u)

        self.u = u
        self.rng = _get_rng()

        if type(self) is RankedBlockState:
            with self._mk_state():
                self._state = libinference.make_ranked_block_model_state(self)

    def _repr_extra(self):
        return " %d upstream, %d downstream, and %d lateral edges,%s%s" % \
            (self.get_Es()[0],
             self.get_Es()[2],
             self.get_Es()[1],
             " distance-based variant," if self.distance else "",
             " unique positions," if self.unique else f" {self.M} shared position")

    def _get_block_state_type(self):
        return BlockState

    def _prepare_bg(self, bg, eweight, vweight):
        bg = GraphView(bg, directed=False)
        bg.ep.weight = eweight
        bg.vp.vweight = vweight
        bg = bg.copy()
        eweight = contract_parallel_edges(bg, weight=bg.ep.weight)
        vweight = bg.vp.vweight
        return bg, eweight, vweight

    def get_block_order(self):
        """Returns an array indexed by the group label containing its rank order."""
        idx = self.wr.fa == 0
        u = self.u.a.copy()
        u[idx] = u.max() + 1
        return rankdata(u, method='ordinal') - 1

    def get_vertex_order(self):
        """Returns a vertex :class:`~graph_tool.PropertyMap` with the rank order
        for every vertex."""
        u = self.b.copy()
        pmap(u, self.get_block_order())
        return u

    def get_vertex_position(self):
        """Returns a vertex :class:`~graph_tool.PropertyMap` with vertex
        positions in the unit interval :math:`[0,1]`."""
        u = self.get_vertex_order()
        u = u.copy("double")
        u.fa /= max((u.fa.max(), 1))
        return u

    def _get_groups(self, vals=None):
        if vals is None:
            groups = self.bg.vertex_index.copy().fa
            wr = self.wr.fa
            groups = groups[wr > 0]
        else:
            groups = vals
        return Vector_int64_t(init=vals)

    @mcmc_sweep_wrap
    def label_mcmc_sweep(self, beta=1.,  niter=1, entropy_args={},
                   sequential=True, deterministic=False,
                   groups=None, verbose=False, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection
        MCMC to sample group positions.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``1.``)
            Inverse temperature.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps to perform. During each sweep, a move attempt is
            made for each node.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.BlockState.entropy`.
        sequential : ``bool`` (optional, default: ``True``)
            If ``sequential == True`` each vertex move attempt is made
            sequentially, where vertices are visited in random order. Otherwise
            the moves are attempted by sampling vertices randomly, so that the
            same vertex can be moved more than once, before other vertices had
            the chance to move.
        deterministic : ``bool`` (optional, default: ``False``)
            If ``sequential == True`` and ``deterministic == True`` the
            vertices will be visited in deterministic order.
        groups : ``list`` of ints (optional, default: ``None``)
            If provided, this should be a list of groups which will be
            moved. Otherwise, all groups will.
        verbose : ``bool`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E)` complexity, where :math:`E` is the
        number of edges (independent of the number of groups).
        """

        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = self._get_entropy_args(entropy_args)
        mcmc_state.groups = self._get_groups(vals=groups)
        mcmc_state.state = self._state
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return self._state.label_mcmc_sweep(mcmc_state, _get_rng())

    def collect_vertex_marginals(self, p=None, b=None, update=1):
        r"""Collect the vertex marginal histogram, which counts the number of times a
        node was assigned to a given block.

        Parameters
        ----------
        p : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with vector-type values, storing the previous block
            membership counts. If not provided, an empty histogram will be created.
        b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with group partition. If not provided, the
            state's partition will be used.
        update : int (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.VertexPropertyMap`
            Vertex property map with vector-type values, storing the accumulated
            block membership counts.
        """

        if p is None:
            p = self.g.new_vp("vector<int>")
        if b is None:
            b = self.get_vertex_order()
        libinference.vertex_marginals(self.g._Graph__graph,
                                      _prop("v", self.g, b),
                                      _prop("v", self.g, p),
                                      update)
        return p

    def get_edge_dir(self):
        """Return an edge :class:`~graph_tool.PropertyMap` containing the edge
        direction: ``-1`` (downstream), ``0`` (lateral), ``+1`` (upstream).
        """
        u = self.get_vertex_order()
        u_s = edge_endpoint_property(self.g, u, "source")
        u_t = edge_endpoint_property(self.g, u, "target")
        edir = self.g.new_ep("int")
        edir.a = u_s.a < u_t.a
        idx = edir.a == 0
        edir.a[idx] = (u_s.a > u_t.a)[idx]
        edir.a[idx] *= -1
        return edir

    def get_Es(self):
        """Return the number of dowstream, lateral, and upstream edges."""
        return self._state.get_Es()

    def _entropy(self, label_dl=True, **kwargs):
        r"""Return the description length (negative joint log-likelihood). See
        :meth:`BlockState.entropy` for documentation."""

        return BlockState._entropy(**dict(locals(), **kwargs))

    def get_edge_colors(self):
        """Return :class:`~graph_tool.EdgePropertyMap` containing the edge
        colors according to their rank direction: upstream (blue), downstream
        (red), lateral (grey).
        """
        edir = self.get_edge_dir()
        ecolor = self.g.new_ep("vector<double>")
        for e in self.g.edges():
            if edir[e] == 0:
                ecolor[e] = (0.1, 0.1, 0.3, 0.6)
            elif edir[e] == 1:
                ecolor[e] = (0.2823529411764706, 0.47058823529411764, 0.8156862745098039, .6)
            else:
                ecolor[e] = (0.8392156862745098, 0.37254901960784315, 0.37254901960784315, .8)
        return ecolor

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.graph_draw` that
        draws the state of the graph as colors on the vertices and edges."""

        edir = self.get_edge_dir()
        ecolor = self.get_edge_colors()

        Es = self.get_Es()
        if Es[-1] < Es[0]:
            edir.a *= -1
        edir.a[edir.a == 0] = 2

        return super().draw(**dict(dict(edge_gradient=[], edge_color=ecolor,
                                        eorder=edir), **kwargs))
