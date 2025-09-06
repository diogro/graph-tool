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

from .. import _prop, Graph, libcore, _get_rng, PropertyMap

import numpy as np

from .. import group_vector_property, ungroup_vector_property

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . blockmodel import *
from . layered_blockmodel import *

@entropy_state_signature
class OverlapBlockState(BlockState):
    r"""The stochastic block model state of a given graph."""

    @kwargs_wrap
    def __init__(self, g, b=None, clabel=None, pclabel=None, bfield=None,
                 vweight=None, gen_overlap=False, **kwargs):

        BlockState.__init__(self, g, **kwargs)

        if pclabel is None:
            pclabel = g.new_ep("vector<int64_t>", val=[0, 0])
        self.pclabel = pclabel

        if clabel is None:
            clabel = pclabel
        self.clabel = clabel

        if b is None:
            b = clabel.copy()
        self.b = b

        if vweight is None:
            vweight = g.new_ep("vector<int64_t>", val=[1, 1])
        self._vweight = vweight

        if bfield is None:
            bfield = (g.new_ep("vector<double>"),
                      g.new_ep("vector<double>"),
                      g._Graph__graph)
        self.bfield = bfield

        self.gen_overlap = gen_overlap

        if type(self) is OverlapBlockState:
            with self._mk_state():
                self._state = libinference.make_overlap_block_model_state(self)

    @property
    def vweight(self):
        if self._mk_override:
            return self.g
        return self._vweight

    @property
    def B(self):
        b_s, b_t = ungroup_vector_property(self.b)
        return max(b_s.fa.max(), b_t.fa.max()) + 1

    def _repr_extra(self):
        return " with generalized overlapping partition," if self.gen_overlap else ""

    def _get_block_state_type(self):
        return BlockState

    def _get_vertices(self, vals=None):
        if vals is None:
            vertices = []
        else:
            vertices = vals
        return vertices

    def _get_b_min(self):
        sidx = self.g.edge_index.copy("int64_t")
        tidx = sidx.copy()
        tidx.fa += sidx.fa.max() + 1
        return group_vector_property([sidx, tidx])

    def _get_b_max(self):
        zs = self.g.new_ep("int64_t")
        return group_vector_property([zs, zs])

    def _check_clabel(self, b=None, clabel=None):
        return True      # FIXME: add implementation

    def get_N(self):
        r"Returns the total number of nodes."
        return self.g.num_vertices()

    def get_wr(self):
        wr = self.bg.new_vp("int64_t")
        self._state.get_wr(wr._get_any())
        return wr

    def _get_bweight(self):
        return self.bg.new_vp("int64_t", self.get_wr().a > 0)

    def get_bclabel(self, clabel=None):
        r"""Returns a :class:`~graph_tool.VertexPropertyMap` corresponding to constraint
        labels for the block graph."""
        bclabel = self.bg.new_vertex_property("int64_t")
        b_s, b_t = ungroup_vector_property(self.b)
        b = np.concatenate([b_s.fa, b_t.fa])
        reverse_map(b, bclabel)
        if clabel is None:
            clabel = self.clabel
        c_s, c_t = ungroup_vector_property(clabel)
        c = np.concatenate([c_s.fa, c_t.fa])
        pmap(bclabel, c)
        return bclabel

    def get_overlap_groups(self):
        r"""Returns the mixed membership of each vertex.

        Returns
        -------
        bv : :class:`~graph_tool.VertexPropertyMap`
           A vector-valued vertex property map containing the block memberships
           of each node.
        bc_in : :class:`~graph_tool.VertexPropertyMap`
           The labelled in-degrees of each node, i.e. how many in-edges belong
           to each group, in the same order as the ``bv`` property above.
        bc_out : :class:`~graph_tool.VertexPropertyMap`
           The labelled out-degrees of each node, i.e. how many out-edges belong
           to each group, in the same order as the ``bv`` property above.
        bc_total : :class:`~graph_tool.VertexPropertyMap`
           The labelled total degrees of each node, i.e. how many incident edges
           belong to each group, in the same order as the ``bv`` property above.

        """
        bv = self.g.new_vertex_property("vector<int64_t>")
        bc_in = self.g.new_vertex_property("vector<int64_t>")
        bc_out = self.g.new_vertex_property("vector<int64_t>")
        bc_total = self.g.new_vertex_property("vector<int64_t>")
        self._state.get_overlap_groups(bv, bc_in, bc_out, bc_total)
        return bv, bc_in, bc_out, bc_total

    def get_majority_groups(self):
        r"""Returns a scalar-valued vertex property map with the majority block
        membership of each node."""

        bv = self.get_overlap_groups()
        bv, bc = bv[0], bv[-1]
        b = self.g.new_vertex_property("int64_t")
        self._state.get_maj_groups(bv, bc, b)
        return b

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.graph_draw` that
        draws the state of the graph as colors on the vertices and edges."""

        bv, bc_in, bc_out, bc_total = self.get_overlap_groups()
        pie_fractions = bc_total.copy("vector<double>")

        return DrawBlockState.draw(self, vertex_fill_color=None, vertex_color=None,
                                   vertex_shape=kwargs.get("vertex_shape", "pie"),
                                   vertex_pie_colors=kwargs.get("vertex_pie_colors", bv),
                                   vertex_pie_fractions=kwargs.get("vertex_pie_fractions",
                                                                   pie_fractions),
                                   **kwargs)


@entropy_state_signature
class WeightedOverlapBlockState(WeightedBlockState, OverlapBlockState):

    @kwargs_wrap
    def __init__(self, g, **kwargs):
        WeightedBlockState.__init__(self, g, **kwargs)
        OverlapBlockState.__init__(self, g, **kwargs)

        if type(self) is WeightedOverlapBlockState:
            with self._mk_state():
                self._state = libinference.make_weighted_overlap_block_model_state(self)

    def _get_block_state_type(self):
        return WeightedBlockState

@entropy_state_signature
class LayeredOverlapBlockState(OverlapBlockState, LayeredBlockState):

    @kwargs_wrap
    def __init__(self, g, **kwargs):
        OverlapBlockState.__init__(self, g, **kwargs)
        LayeredBlockState.__init__(self, g, **kwargs)

        if type(self) is LayeredOverlapBlockState:
            with self._mk_state():
                self._state = libinference.make_layered_overlap_block_model_state(self)

    @property
    def cvweight(self):
        if self._mk_override:
            return self.g
        return self._cvweight

    @property
    def vweight(self):
        if self._mk_override:
            return self.g
        vweight = self.g.new_ep("vector<int64_t>")
        self._state.get_vweight(vweight)
        return vweight

    def _repr_extra(self):
        return LayeredBlockState._repr_extra(self) + \
            OverlapBlockState._repr_extra(self)

    def _get_block_state_type(self):
        return LayeredBlockState

@entropy_state_signature
class LayeredWeightedOverlapBlockState(OverlapBlockState,
                                       LayeredWeightedBlockState):

    @kwargs_wrap
    def __init__(self, g, **kwargs):
        OverlapBlockState.__init__(self, g, **kwargs)
        LayeredWeightedBlockState.__init__(self, g, **kwargs)

        if type(self) is LayeredWeightedOverlapBlockState:
            with self._mk_state():
                self._state = libinference.make_layered_weighted_overlap_block_model_state(self)

    @property
    def cvweight(self):
        if self._mk_override:
            return self.g
        return self._cvweight

    def _repr_extra(self):
        return LayeredWeightedBlockState._repr_extra(self) + \
            OverlapBlockState._repr_extra(self)

    def _get_block_state_type(self):
        return LayeredWeightedBlockState


def get_block_edge_gradient(g, be, cmap=None):
    r"""Get edge gradients corresponding to the block membership at the endpoints of
    the edges given by the ``be`` edge property map.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    be : :class:`~graph_tool.EdgePropertyMap`
        Vector-valued edge property map with the block membership at each
        endpoint.
    cmap : :class:`matplotlib.colors.Colormap` (optional, default: ``default_cm``)
        Color map used to construct the gradient.

    Returns
    -------
    cp : :class:`~graph_tool.EdgePropertyMap`
       A vector-valued edge property map containing a color gradient.
    """

    if cmap is None:
        from .. draw import default_cm
        cmap = default_cm

    cp = g.new_edge_property("vector<double>")
    rg = [np.inf, -np.inf]
    for e in g.edges():
        s, t = be[e]
        rg[0] = min((s, rg[0]))
        rg[0] = min((t, rg[0]))
        rg[1] = max((s, rg[1]))
        rg[1] = max((t, rg[1]))

    for e in g.edges():
        if int(e.source()) < int(e.target()) or g.is_directed():
            s, t = be[e]
        else:
            t, s = be[e]
        cs = cmap((s - rg[0]) / max((rg[1] - rg[0], 1)))
        ct = cmap((t - rg[0]) / max((rg[1] - rg[0], 1)))
        cp[e] = [0] + list(cs) + [1] + list(ct)
    return cp
