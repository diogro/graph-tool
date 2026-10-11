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

from .. import Graph, GraphView, _get_rng, Vector_size_t, PropertyMap, \
    group_vector_property
from . base_states import *
from . util import *
from . blockmodel import BlockState

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math

@entropy_state_signature
class PPBlockState(CLabelState, MCMCState, MultiflipMCMCState,
                   MultilevelMCMCState, GibbsMCMCState, DrawBlockState):
    r"""Obtain the partition of a network according to the Bayesian planted partition
    model.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge multiplicities (for multigraphs or block graphs).
    clabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Constraint labels on the vertices. If supplied, vertices with different
        label values will not be clustered in the same group.
    pclabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Partition constraint labels on the vertices. This has the same
        interpretation as ``clabel``, but will be used to compute the partition
        description length.
    uniform : ``bool`` (optional, default: ``False``)
        If ``True``, the uniform model variant will be assumed.
    deg_corr : ``bool`` (optional, default: ``True``)
        If ``True``, the degree-corrected version of the blockmodel ensemble will
        be assumed, otherwise the traditional variant will be used.
    entropy_args : ``dict`` (optional, default: ``{}``)
        Override default arguments for :meth:`~PPBlockState.entropy()`
        method and related operations.

    References
    ----------
    .. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
       inference of assortative community structures", Phys. Rev. Research 2
       043271 (2020), :doi:`10.1103/PhysRevResearch.2.043271`, :arxiv:`2006.14493`

    """

    @kwargs_wrap
    def __init__(self, g, b=None, eweight=None, clabel=None, pclabel=None,
                 uniform=False, deg_corr=True, entropy_args={}, **kwargs):
        EntropyState.__init__(self, entropy_args=entropy_args)
        CLabelState.__init__(self, g=g, b=b, clabel=clabel, pclabel=pclabel)
        self.g = g
        self.uniform = uniform
        self.deg_corr = deg_corr

        if eweight is None:
            eweight = g.new_ep("int64_t", 1)
        else:
            eweight = g.own_property(eweight)
        self.eweight = eweight

        self._state = libinference.make_planted_partition_state(self)

    @property
    def B(self):
        r"""Nominal number of groups"""
        return int(self.b.fa.max()) + 1

    def _repr_extra(self):
        if self.uniform:
            return " uniform assortativity,"
        else:
            return " non-uniform assortativity,"

    def __repr__(self):
        return BlockState.__repr__(self)

    def get_blocks(self):
        r"""Returns the property map which contains the block labels for each vertex."""
        return self.b

    def get_state(self):
        """Alias to :meth:`~PPBlockState.get_blocks`."""
        return self.get_blocks()

    def get_B(self):
        r"Returns the total number of non-empty groups."
        return len(np.unique(self.b.fa))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group :math:`r`.
        """
        w = np.array(np.bincount(self.b.fa), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return np.exp(-(w*np.log(w)).sum())

    @copy_state_wrap
    def _entropy(self, adjacency=True, parallel_entropy=True,
                 degree_entropy=True, dl=True, partition_dl=True,
                 degree_dl=True, edges_dl=True, degree_dl_kind="distributed",
                 constants=True, **kwargs):
        r"""Return the model entropy (negative log-likelihood).

        Parameters
        ----------
        adjacency : ``bool`` (optional, default: ``True``)
            If ``True``, the adjacency term of the description length will be
            included.
        parallel_entropy : ``bool`` (optional, default: ``True``)
            If ``True``, the entropy term relating to the multiplicities of
            parallel edges, which is independent of the network partition, will
            be included.
        degree_entropy : ``bool`` (optional, default: ``True``)
            If ``True``, the degree entropy term that is independent of the
            network partition will be included (for degree-corrected models).
        dl : ``bool`` (optional, default: ``True``)
            If ``True``, the description length for the parameters will be
            included.
        partition_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the partition description length
            will be included.
        degree_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the degree sequence description
            length will be included (for degree-corrected models).
        edges_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the edge count description length
            will be included.
        degree_dl_kind : ``str`` (optional, default: ``"distributed"``)
            This specifies the prior used for the degree sequence. It must be
            one of: ``"uniform"`` or ``"distributed"`` (default).
        constants : ``bool`` (optional, default: ``True``)
            If ``True``, the terms that are independent of the network
            partition (controlled by ``degree_entropy`` and
            ``parallel_entropy``) will be included.

        Notes
        -----

        The "entropy" of the state is the negative log-likelihood of the
        microcanonical SBM, that includes the generated graph
        :math:`\boldsymbol{A}` and the model parameters :math:`e_{\text{in}}`,
        :math:`e_{\text{out}}`, :math:`\boldsymbol{k}` and
        :math:`\boldsymbol{b}`,

        .. math::

           \Sigma &= - \ln P(\boldsymbol{A},e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}) \\
                  &= - \ln P(\boldsymbol{A}|e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}) - \ln P(e_{\text{in}},e_{\text{out}},\boldsymbol{k},\boldsymbol{b}).

        This value is also called the `description length
        <https://en.wikipedia.org/wiki/Minimum_description_length>`_ of the data,
        and it corresponds to the amount of information required to describe it
        (in `nats <https://en.wikipedia.org/wiki/Nat_(unit)>`_).

        For the uniform version of the model, the likelihood is

        .. math::

            P(\boldsymbol{A}|\boldsymbol{k},\boldsymbol{b}) = \frac{e_{\text{in}}!e_{\text{out}}!}
            {\left(\frac{B}{2}\right)^{e_{\text{in}}}{B\choose 2}^{e_{\text{out}}}(E+1)^{1-\delta_{B,1}}\prod_re_r!}\times
            \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_i A_{ii}!!}.

        where :math:`e_{\text{in}}` and :math:`e_{\text{out}}` are the number of
        edges inside and outside communities, respectively, and :math:`e_r` is
        the sum of degrees in group :math:`r`.

        For the non-uniform model we have instead:

        .. math::

            P(\boldsymbol{A}|\boldsymbol{k},\boldsymbol{b}) = \frac{e_{\text{out}}!\prod_re_{rr}!!}
            {{B\choose 2}^{e_{\text{out}}}(E+1)^{1-\delta_{B,1}}\prod_re_r!}\times{B + e_{\text{in}} - 1 \choose e_{\text{in}}}^{-1}\times
            \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_i A_{ii}!!}.


        Here there are two options for the prior on the degrees:

        1. ``degree_dl_kind == "uniform"``

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\left(\!\!{n_r\choose e_r}\!\!\right)^{-1}.

            This corresponds to a noninformative prior, where the degrees are
            sampled from a uniform distribution.

        2. ``degree_dl_kind == "distributed"`` (default)

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\frac{\prod_k\eta_k^r!}{n_r!} \prod_r q(e_r, n_r)^{-1}

            with :math:`\eta_k^r` being the number of nodes with degree
            :math:`k` in group :math:`r`, and :math:`q(n,m)` being the number of
            `partitions
            <https://en.wikipedia.org/wiki/Partition_(number_theory)>`_ of
            integer :math:`n` into at most :math:`m` parts.

            This corresponds to a prior for the degree sequence conditioned on
            the degree frequencies, which are themselves sampled from a uniform
            hyperprior. This option should be preferred in most cases.


        For the partition prior :math:`P(\boldsymbol{b})` please refer to
        :meth:`~graph_tool.inference.BlockState.entropy`.

        References
        ----------
        .. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
           inference of assortative community structures", Phys. Rev. Research 2
           043271 (2020), :doi:`10.1103/PhysRevResearch.2.043271`, :arxiv:`2006.14493`
        """

        return EntropyState._entropy_dispatch(**dict(locals(), **kwargs))
