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

from .. import _prop, Graph, GraphView, libcore, _get_rng, openmp_context
from .. import group_vector_property, ungroup_vector_property, perfect_prop_hash

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . base_states import _bm_test

from . blockmodel import *

import numpy

@entropy_state_signature
class LayeredBlockState(BlockState):
    r"""The stochastic block model state of a given graph.

    """

    @kwargs_wrap
    def __init__(self, g, ec, independent=True, **kwargs):

        if hasattr(self, "independent"):
            return

        BlockState.__init__(self, g, **kwargs)

        self.independent = independent
        self.ec = ec
        if "vector" not in self.ec.value_type():
            self.ec = group_vector_property([ec],
                                            value_type="int64_t")
        if isinstance(self._eweight, PropertyMap):
            if "vector" not in self._eweight.value_type():
                self._eweight = group_vector_property([self._eweight],
                                                  value_type="int64_t")
        if self.ec.value_type() != "vector<int64_t>":
            self.ec = self.ec.copy("vector<int64_t>")
        if isinstance(self._eweight, PropertyMap):
            if self._eweight.value_type() != "vector<int64_t>":
                self._eweight = self._eweight.copy("vector<int64_t>")

        self.b_ec = self.bg.new_ep("vector<int64_t>")
        self._mrs = self.bg.new_ep("vector<int64_t>")

        self._cvweight = self._vweight

        if type(self) is LayeredBlockState:
            with self._mk_state():
                self._state = libinference.make_layered_block_model_state(self)

    @property
    def eweight(self):
        if self._mk_override:
            if isinstance(self._eweight, libcore.any):
                return self._eweight
            return (self.g, self.ec, self._eweight, self.independent)
        return self._eweight

    @property
    def vweight(self):
        if self._mk_override:
            return self._vweight
        vweight = self.g.new_vp("int64_t")
        self._state.get_vweight(vweight)
        return vweight

    @property
    def cvweight(self):
        return self._cvweight

    @property
    def wr(self):
        if self._mk_override or not self.independent:
            return self._wr
        wr = self.bg.new_vp("int64_t")
        self._state.get_wr(wr)
        return wr

    @property
    def mrs(self):
        if self._mk_override:
            return (self.bg, self.b_ec, self._mrs, self.independent)
        return self._mrs

    def _repr_extra(self):
        return f" with {pl(self._state.get_C(), ('independent' if self.independent else 'conditioned') + ' layer')},"

    def get_nonempty_B(self):
        r"Returns the total number of nonempty blocks."
        return self._state.get_B()

    def get_wr(self):
        wr = self.bg.new_vp("int64_t")
        self._state.get_wr(wr)
        return wr

    def get_lvweight(self):
        ls = self.g.new_vp("vector<int64_t>")
        cs = self.g.new_vp("vector<int64_t>")
        self._state.get_lvweight(ls, cs)
        return ls, cs

    def get_lwr(self):
        ls = self.bg.new_vp("vector<int64_t>")
        cs = self.bg.new_vp("vector<int64_t>")
        self._state.get_lwr(ls, cs)
        return ls, cs

    def _get_bweight(self):
        return self.get_wr().t(lambda x: x > 0)

    def _get_block_state(self, b, final=False, couple=True, **kwargs):
        state = BlockState._get_block_state(self, b, final=final,
                                            ec=self.b_ec,
                                            eweight=self.mrs,
                                            independent=self.independent,
                                            couple=False, **kwargs)
        state.update_entropy_args(lmemb=False)
        if couple:
            self._couple_state(state, state.get_entropy_args())
        return state

    def get_N(self):
        return self._state.get_N()

    def _entropy(self, lmemb=True, **kwargs):
        r"""Calculate the description length (a.k.a. negative joint
        log-likelihood) associated with the current block partition.
        """

        return BlockState._entropy(**dict(locals(), **kwargs))


@entropy_state_signature
class LayeredWeightedBlockState(LayeredBlockState, WeightedBlockState):

    @kwargs_wrap
    def __init__(self, g, rec=[], **kwargs):
        recdx = kw_pop(kwargs, "recdx", libinference.rcount_vec())
        Lrecdx = kw_pop(kwargs, "Lrecdx", libinference.rcount_vec())

        for i in range(len(rec)):
            if "vector" not in rec[i].value_type():
                rec[i] = group_vector_property([rec[i]], value_type="double")

        LayeredBlockState.__init__(self, g, **kwargs)
        WeightedBlockState.__init__(self, g, rec=rec, rvtype="vector<double>",
                                    **kwargs)

        self.recdx = recdx
        self.Lrecdx = Lrecdx

        if type(self) is LayeredWeightedBlockState:
            with self._mk_state():
                self._state = libinference.make_layered_weighted_block_model_state(self)

    @property
    def eweight(self):
        if self._mk_override:
            if isinstance(self._eweight, libcore.any):
                return self._eweight
            ew = (self.g, self.ec, self._eweight,
                  WeightedBlockState.eweight.fget(self))
            return (self.g, self.ec, ew, self.independent)
        return LayeredBlockState.eweight.fget(self)

    @property
    def mrs(self):
        if self._mk_override:
            b_ew = (self.bg, self.b_ec, self._mrs,
                    WeightedBlockState.mrs.fget(self))
            return (self.bg, self.b_ec, b_ew, self.independent)
        return LayeredBlockState.mrs.fget(self)

    def _repr_extra(self):
        return LayeredBlockState._repr_extra(self) + WeightedBlockState._repr_extra(self)

    def _get_block_state(self, b, final=False, couple=True):
        state = WeightedBlockState._get_block_state(self, b, final=final,
                                                    ec=self.b_ec,
                                                    eweight=self._state.get_mrs(),
                                                    independent=self.independent,
                                                    couple=False)
        state.update_entropy_args(lmemb=False)
        if couple:
            self._couple_state(state, state.get_entropy_args())
        return state
