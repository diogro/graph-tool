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

from .. import _prop, Graph, libcore, _get_rng, PropertyMap, \
    EdgePropertyMap, VertexPropertyMap, Vector_size_t, Vector_double, \
    group_vector_property, perfect_prop_hash
from .. generation import condensation_graph, random_rewire, generate_sbm, \
    solve_sbm_fugacities, generate_maxent_sbm, remove_parallel_edges, \
    remove_self_loops
from .. spectral import adjacency
import numpy as np
import math
import copy
import collections.abc
import itertools
import warnings
from contextlib import contextmanager

from . util import *

from . base_states import *
from . base_states import _bm_test

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . libgraph_tool_inference import PartitionHist, BlockPairHist

@entropy_state_signature
class BlockState(CLabelState, MCMCState, MultiflipMCMCState,
                 MultilevelMCMCState, GibbsMCMCState, ExhaustiveSweepState,
                 DrawBlockState):
    r"""The stochastic block model state of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial block labels on the vertices. If not supplied, it will be
        randomly sampled.
    B : ``int`` (optional, default: ``None``)
        Number of blocks (or vertex groups). If not supplied it will be obtained
        from the parameter ``b``.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge multiplicities (for multigraphs or block graphs).
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex multiplicities (for block graphs).
    clabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Constraint labels on the vertices. If supplied, vertices with different
        label values will not be clustered in the same group.
    pclabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Partition constraint labels on the vertices. This has the same
        interpretation as ``clabel``, but will be used to compute the partition
        description length.
    bfield : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Local field acting as a prior for the node partition. This should be a
        vector property map of type ``vector<double>``, and contain the
        log-probability for each node to be placed in each group.
    deg_corr : ``bool`` (optional, default: ``True``)
        If ``True``, the degree-corrected version of model will be assumed,
        otherwise the non-degree-corrected variant will be used.
    flat_map : ``bool`` (optional, default: ``False``)
        If ``True``, flat maps will be used instead of hash tables, which will
        be slower, but will use less memory.
    entropy_args: ``keyword parameters`` (optional, default: ``empty``)
        Override default arguments for :meth:`~BlockState.entropy()` method and
        releated operations.

    """

    @kwargs_wrap
    def __init__(self, g, b=None, B=None, eweight=None, vweight=None,
                 clabel=None, pclabel=None, bfield=None, Bfield=None,
                 deg_corr=True, flat_map=False, entropy_args={}, **kwargs):

        if hasattr(self, "g"):
            return

        EntropyState.__init__(self, entropy_args=entropy_args)
        CLabelState.__init__(self, g=g, clabel=clabel, pclabel=pclabel)

        self.g = g
        self.deg_corr = deg_corr

        if eweight is None:
            self._eweight = g.new_ep("int64_t", 1)
        else:
            self._eweight = eweight

        if vweight is None:
            self._vweight = g.new_vp("int64_t", 1)
        else:
            self._vweight = vweight

        if B is None and b is None:
            B = 1

        # ensure we have at most as many blocks as nodes
        if B is not None and b is None:
            B = min((B, self.g.num_vertices()))

        if b is None:
            # create a random partition into B blocks.
            if B is None:
                if self.clabel.fa.max() == 0:
                    raise ValueError("either 'b' or 'B' must be specified")
                else:
                    self.b = self.clabel.copy()
            else:
                B = min((B, self.g.num_vertices()))
                ba = np.random.randint(0, B, self.g.num_vertices())
                ba[:B] = np.arange(B)        # avoid empty blocks
                if B < self.g.num_vertices():
                    np.random.shuffle(ba)
                b = g.new_vp("int64_t")
                b.fa = ba
                self.b = self._clabel_combine(b, self.clabel)
        else:
            # if a partition is available, we will incorporate it.
            if isinstance(b, np.ndarray):
                self.b = g.new_vp("int64_t")
                ba = np.zeros(len(self.b.fa), dtype="int64")
                ba[:min((len(ba), len(b)))] = b[:min((len(ba), len(b)))]
                self.b.fa = ba
            else:
                self.b = g.own_property(b)
            if B is None:
                B = int(self.b.fa.max()) + 1

        # Internal structures
        self.bg = Graph(directed=g.is_directed())

        self._mrs = self.bg.new_ep("int64_t")
        self._wr = self.bg.new_vp("int64_t")
        self._mrp = self.bg.new_vp("int64_t")
        if g.is_directed():
            self._mrm = self.bg.new_vp("int64_t")
        else:
            self._mrm = self.mrp

        self._coupled_state = None

        self.flat_map = flat_map

        self.bfield = g.new_vp("vector<double>") if bfield is None else bfield
        if Bfield is None:
            self.Bfield = Vector_double()
        else:
            self.Bfield = Vector_double(len(Bfield))
            self.Bfield.a = Bfield

        self._mk_override = False

        if type(self) is BlockState:
            with self._mk_state():
                self._check_params()
                self._state = libinference.make_block_model_state(self)

    @property
    def eweight(self):
        r""":class:`~graph_tool.EdgePropertyMap` with edge multiplicities"""
        return self._eweight

    @property
    def vweight(self):
        r""":class:`~graph_tool.VertexPropertyMap` with node multiplicities"""
        return self._vweight

    @property
    def mrs(self):
        r""":class:`~graph_tool.EdgePropertyMap` with with edge counts between groups"""
        return self._mrs

    @property
    def wr(self):
        r""":class:`~graph_tool.VertexPropertyMap` with with group sizes"""
        return self._wr

    @property
    def mrp(self):
        r""":class:`~graph_tool.VertexPropertyMap` with with sums of out-degrees per group"""
        return self._mrp

    @property
    def mrm(self):
        r""":class:`~graph_tool.VertexPropertyMap` with with sums of in-degrees per group"""
        return self._mrm

    @property
    def B(self):
        r"""Nominal number of groups"""
        return int(self.b.fa.max()) + 1

    @contextmanager
    def _mk_state(self):
        self._mk_override = True
        yield
        self._mk_override = False

    def _check_params(self):
        ptypes = {"int64_t": ["eweight", "vweight", "clabel", "pclabel", "b"],
                  "vector<double>": ["bfield"]}
        for t, ps in ptypes.items():
            for p in ps:
                if getattr(self, p).value_type() != t:
                    setattr(self, p, getattr(self, p).copy(value_type=t))
        if not self._check_clabel():
            raise ValueError("provided clabel is inconsistent with node partition")
        if not self._check_clabel(clabel=self.pclabel):
            raise ValueError("provided pclabel is inconsistent with node partition")
        if not self._check_clabel(b=self.pclabel, clabel=self.clabel):
            raise ValueError("provided pclabel and clabel are inconsistent")

    def _repr_extra(self):
        return ""

    def __repr__(self):
        extra = ' degree corrected,' if self.deg_corr else ''
        extra += self._repr_extra()
        return f"<{self.__class__.__name__} object with {pl(self.get_B(), 'group')} ({self.B} nominal),{extra} for graph {str(self.g)}, at 0x{id(self):x}>"

    def _get_bweight(self):
        return self.bg.new_vp("int64_t", self.wr.a > 0)

    def _prepare_bg(self, *args):
        return args

    def _get_block_state_type(self):
        return self.__class__

    def _get_block_state(self, b, final=False, couple=True, **kwargs):
        bg = self.bg
        eweight = self.mrs
        vweight = self._get_bweight()
        bg, eweight, vweight = self._prepare_bg(bg, eweight, vweight)
        bstate_t = self._get_block_state_type()
        state = bstate_t(**dict(dict(g=bg,
                                     eweight=eweight,
                                     vweight=vweight,
                                     b=b,
                                     deg_corr=False,
                                     clabel=self.get_bclabel() if not final else None,
                                     pclabel=self.get_bpclabel() if not final else None,
                                     flat_map=self.flat_map),
                                **kwargs))
        state.update_entropy_args(dense=True, edges_dl=final)
        if couple:
            self._couple_state(state, state.get_entropy_args())
        return state

    def get_E(self):
        r"Returns the total number of edges."
        return int(self.eweight.fa.sum())

    def get_N(self):
        r"Returns the total number of nodes."
        return int(self.vweight.fa.sum())

    def get_B(self):
        r"Returns the total number of nonempty groups."
        return self._state.get_actual_B()

    def get_Be(self):
        r"""Returns the effective number of groups, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(self.wr.a, dtype="double")
        w = w[w > 0]
        w /= w.sum()
        return np.exp(-(w*np.log(w)).sum())

    def get_bclabel(self, clabel=None):
        r"""Returns a :class:`~graph_tool.VertexPropertyMap` corresponding to constraint
        labels for the block graph."""

        bclabel = self.bg.new_vertex_property("int64_t")
        idx = self.vweight.fa > 0
        reverse_map(self.b.fa[idx], bclabel)
        if clabel is None:
            clabel = self.clabel
        pmap(bclabel, clabel.fa[idx])
        return bclabel

    def _set_bclabel(self, bstate):
        self.bclabel.a = bstate.b.a
        self.clabel.a = self.b.a
        pmap(self.clabel, self.bclabel)

    def get_bpclabel(self):
        r"""Returns a :class:`~graph_tool.VertexPropertyMap` corresponding to partition
        constraint labels for the block graph."""
        return self.get_bclabel(self.pclabel)

    def _couple_state(self, state, entropy_args):
        self._coupled_state = (state, entropy_args)
        eargs = state._get_entropy_args(entropy_args)
        self._state.couple_state(state._state, eargs)
        self.update_entropy_args(edges_dl=False)

    def get_blocks(self):
        r"""Returns the property map which contains the block labels for each vertex."""
        return self.b

    def get_state(self):
        """Alias to :meth:`~graph_tool.inference.BlockState.get_blocks`."""
        return self.get_blocks()

    def set_state(self, b, parallel=False):
        r"""Sets the internal partition of the state. If ``parallel`` is
        ``True``, the update is done in parallel.
        """
        if not isinstance(b, VertexPropertyMap):
            b = self.g.new_vp("int64_t", vals=b)
        if b.value_type() != "int64_t":
            b = b.copy("int64_t")
        self._state.set_partition(_prop("v", self.g, b), parallel)

    def get_bg(self):
        r"""Returns the block graph."""
        return self.bg

    def get_ers(self):
        r"""Returns the edge property map of the block graph which contains the
        :math:`e_{rs}` matrix entries.  For undirected graphs, the diagonal
        values (self-loops) contain :math:`e_{rr}/2`."""
        return self.mrs

    def get_er(self):
        r"""Returns the vertex property map of the block graph which contains the number
        :math:`e_r` of half-edges incident on block :math:`r`. If the graph is
        directed, a pair of property maps is returned, with the number of
        out-edges :math:`e^+_r` and in-edges :math:`e^-_r`, respectively."""
        if self.bg.is_directed():
            return self.mrp, self.mrm
        else:
            return self.mrp

    def get_nr(self):
        r"""Returns the vertex property map of the block graph which contains the block
        sizes :math:`n_r`."""
        return self.wr

    @copy_state_wrap
    def _entropy(self, adjacency=True, dl=True, partition_dl=True,
                 degree_dl=True, degree_dl_kind="distributed", edges_dl=True,
                 dense=False, multigraph=True, deg_entropy=True,
                 parallel_entropy=True, beta_dl=1., Bfield=True, propagate=True,
                 constants=True, **kwargs):
        r"""Calculate the description length (a.k.a. negative joint
        log-likelihood) associated with the current block partition.

        Parameters
        ----------
        adjacency : ``bool`` (optional, default: ``True``)
            If ``True``, the adjacency term of the description length will be
            included.
        dl : ``bool`` (optional, default: ``True``)
            If ``True``, the description length for the parameters will be
            included.
        partition_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the partition description length
            will be included.
        degree_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the degree sequence description
            length will be included (for degree-corrected models).
        degree_dl_kind : ``str`` (optional, default: ``"distributed"``)
            This specifies the prior used for the degree sequence. It must be
            one of: ``"uniform"`` or ``"distributed"`` (default).
        edges_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the edge matrix description length
            will be included.
        dense : ``bool`` (optional, default: ``False``)
            If ``True``, the "dense" variant of the entropy will be computed.
        multigraph : ``bool`` (optional, default: ``True``)
            If ``True``, the multigraph entropy will be used.
        deg_entropy : ``bool`` (optional, default: ``True``)
            If ``True``, the degree entropy term that is independent of the
            network partition will be included (for degree-corrected models).
        parallel_entropy : ``bool`` (optional, default: ``True``)
            If ``True``, the entropy term relating to the multiplcities of
            parallel edges, which is independent of the network partition, will
            be included (for muligraph models).
        beta_dl : ``double`` (optional, default: ``1.``)
            Prior inverse temperature.
        Bfield : ``bool`` (optional, default: ``False``)
            If ``True``, the ``Bfield`` parameter passed to the construtor will be
            taken into account.

        Notes
        -----

        The "entropy" of the state is the negative log-likelihood of the
        microcanonical SBM, that includes the generated graph
        :math:`\boldsymbol{A}` and the model parameters
        :math:`\boldsymbol{\theta}`,

        .. math::

           \Sigma &= - \ln P(\boldsymbol{A},\boldsymbol{\theta}) \\
                  &= - \ln P(\boldsymbol{A}|\boldsymbol{\theta}) - \ln P(\boldsymbol{\theta}).

        This value is also called the `description length
        <https://en.wikipedia.org/wiki/Minimum_description_length>`_ of the data,
        and it corresponds to the amount of information required to describe it
        (in `nats <https://en.wikipedia.org/wiki/Nat_(unit)>`_).

        For the non-degree-corrected blockmodel (``deg_corr == False``), the model
        parameters are :math:`\boldsymbol{\theta} = \{\boldsymbol{e},
        \boldsymbol{b}\}`, where :math:`\boldsymbol{e}` is the matrix of edge
        counts between blocks, and :math:`\boldsymbol{b}` is the partition of the
        nodes into blocks. For the degree-corrected blockmodel (``deg_corr ==
        True``), we have an additional set of parameters, namely the degree
        sequence :math:`\boldsymbol{k}`.

        For the non-degree-corrected blockmodel, the model likelihood is

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!}{\prod_rn_r^{e_r}}\times \frac{1}{\prod_{i<j}A_{ij}!\prod_iA_{ii}!!},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \frac{\prod_{rs}e_{rs}!}{\prod_rn_r^{e_r}}\times \frac{1}{\prod_{ij}A_{ij}!},

        for undirected and directed graphs, respectively, where :math:`e_{rs}`
        is the number of edges from block :math:`r` to :math:`s` (or the number
        of half-edges for the undirected case when :math:`r=s`), and :math:`n_r`
        is the number of vertices in block :math:`r`.

        For the degree-corrected variant the equivalent expressions are

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b},\boldsymbol{k}) &= \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!}{\prod_re_r!}\times \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_iA_{ii}!!},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b},\boldsymbol{k}) &= \frac{\prod_{rs}e_{rs}!}{\prod_re_r^+!\prod_re_r^-!}\times \frac{\prod_ik_i^+!\prod_ik_i^-!}{\prod_{ij}A_{ij}!},

        where :math:`e_r = \sum_se_{rs}` is the number of half-edges incident on
        block :math:`r`, and :math:`e^+_r = \sum_se_{rs}` and :math:`e^-_r =
        \sum_se_{sr}` are the numbers of out- and in-edges adjacent to block
        :math:`r`, respectively.

        If ``dense == True``, the likelihood for the non-degree-corrected model
        becomes instead

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \left[\prod_{r<s}{n_rn_s\choose e_{rs}}\prod_r{{n_r\choose 2}\choose e_{rr}/2}\right]^{-1},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \left[\prod_{rs}{n_rn_s\choose e_{rs}}\right]^{-1}

        if ``multigraph == False``, otherwise we replace :math:`{n\choose
        m}\to\left(\!\!{n\choose m}\!\!\right)` above, where
        :math:`\left(\!\!{n\choose m}\!\!\right) = {n+m-1\choose m}`.  A "dense"
        entropy for the degree-corrected model is not available, and if
        requested will raise a :exc:`NotImplementedError`.

        If ``dl == True``, the description length :math:`\mathcal{L} = -\ln
        P(\boldsymbol{\theta})` of the model parameters will be returned as
        well. The terms :math:`P(\boldsymbol{e})` and :math:`P(\boldsymbol{b})`
        are described in described as follows.

        For an undirected graph, the number of distinct :math:`e_{rs}` matrices is
        given by,

        .. math::

           \Omega_m = \left(\!\!{B(B+1)/2 \choose E}\!\!\right)

        and for a directed graph,

        .. math::
           \Omega_m = \left(\!\!{B^2 \choose E}\!\!\right)


        where :math:`\left(\!{n \choose k}\!\right) = {n+k-1\choose k}` is the
        number of :math:`k` combinations with repetitions from a set of size
        :math:`n`. Hence, we have the description length of the edge counts

        .. math::

            -\ln P(\boldsymbol{e}) = \ln \Omega_m.

        For the node partition :math:`\boldsymbol{b}` we assume a two-level Bayesian
        hierarchy, where first the group size histogram is generated, and
        conditioned on it the partition, which leads to a description length:

        .. math::

           -\ln P(\boldsymbol{b}) = \ln {N - 1 \choose B - 1} + \ln N! - \sum_r \ln n_r!.

        where :math:`n_r` is the number of nodes in block :math:`r`.

        The total information necessary to describe the model is then,

        .. math::

           -\ln P(\boldsymbol{e}, \boldsymbol{b}) = -\ln P(\boldsymbol{e}) - \ln P(\boldsymbol{b}).

        If ``nr`` is ``None``, it is assumed :math:`n_r=N/B`. If ``nr`` is
        ``False``, the partition term :math:`-\ln P(\boldsymbol{b})` is omitted
        entirely.

        For the degree-corrected model we need to specify the prior
        :math:`P(\boldsymbol{k})` for the degree sequence as well. Here there
        are two options:

        1. ``degree_dl_kind == "uniform"``

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\left(\!\!{n_r\choose e_r}\!\!\right)^{-1}.

            This corresponds to a noninformative prior, where the degrees are
            sampled from a uniform distribution.

        2. ``degree_dl_kind == "distributed"`` (default)

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\frac{\prod_k\eta_k^r!}{n_r!} \prod_r q(e_r, n_r)^{-1}

            with :math:`\eta_k^r` being the number of nodes with degree
            :math:`k` in group :math:`r`, and :math:`q(m,n)` being the number of
            `partitions
            <https://en.wikipedia.org/wiki/Partition_(number_theory)>`_ of
            integer :math:`m` into at most :math:`n` parts. This is given by the
            recurrence

            .. math::

                q(m, n) = q(m, n-1) + q(m-n, n),

            with boundary conditions :math:`q(m,n) = q(m,m)` for :math:`n>m`,
            :math:`q(m,0) = 0` for :math:`m>0`, and :math:`q(0,0) = 1`.

            This corresponds to a prior for the degree sequence conditioned on
            the degree counts, which are themselves sampled from a uniform
            hyperprior. This option should be preferred in most cases.

        For the directed case, the above expressions are duplicated for the in-
        and out-degrees.

        References
        ----------

        .. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
           Bayesian inference of the microcanonical stochastic block model",
           Phys. Rev. E 95 012317 (2017), :doi:`10.1103/PhysRevE.95.012317`,
           :arxiv:`1610.02703`
        .. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
           structures and high-resolution model selection in large networks ",
           Phys. Rev. X 4, 011047 (2014), :doi:`10.1103/PhysRevX.4.011047`,
           :arxiv:`1310.4377`.
        """

        return EntropyState._entropy_dispatch(**dict(locals(), **kwargs))

    def get_matrix(self):
        r"""Returns the block matrix (as a sparse :class:`~scipy.sparse.csr_matrix`),
        which contains the number of edges between each block pair.

        .. warning::

           This corresponds to the adjacency matrix of the block graph, which by
           convention includes twice the amount of edges in the diagonal entries
           if the graph is undirected.

        Examples
        --------

        .. testsetup:: get_matrix

           np.random.seed(42)
           gt.seed_rng(42)
           from pylab import *

        .. doctest:: get_matrix

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=5, deg_corr=True)
           >>> state.mcmc_sweep(niter=1000)
           (...)
           >>> m = state.get_matrix()
           >>> figure()
           <...>
           >>> matshow(m.todense())
           <...>
           >>> savefig("bloc_mat.svg")

        .. figure:: bloc_mat.*
           :align: center

           A  5x5 block matrix.

       """

        return adjacency(self.bg, weight=self.mrs)

    def virtual_vertex_move(self, v, s, **kwargs):
        r"""Computes the entropy difference if vertex ``v`` is moved to block ``s``. The
        remaining parameters are the same as in
        :meth:`graph_tool.inference.BlockState.entropy`."""
        return self._state.virtual_move(int(v), s,
                                        self._get_entropy_args(dict(self._entropy_args,
                                                                    **kwargs)))

    def move_vertex(self, v, s, parallel=False):
        r"""Move vertex ``v`` to block ``s``.

        This optionally accepts a list of vertices and blocks to move
        simultaneously. In this case, if ``parallel`` is ``True``, the updates
        are done in parallel.
        """
        if not isinstance(v, collections.abc.Iterable):
            self._state.move_vertex(int(v), s)
        else:
            self._state.move_vertices(np.asarray(v, dtype="uint64"),
                                      np.asarray(s, dtype="int64"),
                                      parallel)

    def virtual_merge(self, r, s, **kwargs):
        r"""Computes the entropy difference if group ``r`` is merged with block ``s``. The
        remaining parameters are the same as in
        :meth:`graph_tool.inference.BlockState.entropy`."""
        return self._state.virtual_merge(r, s,
                                         self._get_entropy_args(dict(self._entropy_args,
                                                                     **kwargs)))

    def merge_groups(self, r, s):
        r"""Merges group ``r`` with ``s``."""
        return self._state.merge_groups(r, s)

    def move_vertex(self, v, s, parallel=False):
        r"""Move vertex ``v`` to block ``s``.

        This optionally accepts a list of vertices and blocks to move
        simultaneously. In this case, if ``parallel`` is ``True``, the updates
        are done in parallel.
        """
        if not isinstance(v, collections.abc.Iterable):
            self._state.move_vertex(int(v), s)
        else:
            self._state.move_vertices(np.asarray(v, dtype="uint64"),
                                      np.asarray(s, dtype="int64"),
                                      parallel)

    def remove_vertex(self, v):
        r"""Remove vertex ``v`` from its current group.

        This optionally accepts a list of vertices to remove.

        .. warning::

           This will leave the state in an inconsistent state before the vertex
           is returned to some other group, or if the same vertex is removed
           twice.
        """
        if isinstance(v, collections.abc.Iterable):
            if not isinstance(v, np.ndarray):
                v = list(v)
            self._state.remove_vertices(np.asarray(v, dtype="uint64"))
        else:
            self._state.remove_vertex(int(v))

    def add_vertex(self, v, r):
        r"""Add vertex ``v`` to block ``r``.

        This optionally accepts a list of vertices and blocks to add.

        .. warning::

           This can leave the state in an inconsistent state if a vertex is
           added twice to the same group.
        """
        if isinstance(v, collections.abc.Iterable):
            if not isinstance(v, np.ndarray):
                v = list(v)
            if not isinstance(r, np.ndarray):
                r = list(r)
            self._state.add_vertices(np.asarray(v, dtype="uint64"),
                                     np.asarray(r, dtype="uint64"))
        else:
            self._state.add_vertex(int(v), r)


    def sample_vertex_move(self, v, c=1., d=.1):
        r"""Sample block membership proposal of vertex ``v`` according to real-valued
        sampling parameters ``c`` and ``d``: For :math:`c\to 0` the blocks are sampled
        according to the local neighborhood and their connections; for
        :math:`c\to\infty` the blocks are sampled randomly. With a probability
        ``d``, a new (empty) group is sampled.
        """
        return self._state.sample_block(int(v), c, d, _get_rng())

    def get_move_prob(self, v, s, c=1., d=.1, reverse=False):
        r"""Compute the log-probability of a move proposal for vertex ``v`` to block ``s``
        according to sampling parameters ``c`` and ``d``, as obtained with
        :meth:`graph_tool.inference.BlockState.sample_vertex_move`. If ``reverse
        == True``, the reverse probability of moving the node back from block
        ``s`` to its current one is obtained.
        """
        if not reverse:
            return self._state.get_move_prob(int(v), self.b[v], s, c, d, False)
        else:
            return self._state.get_move_prob(int(v), s, self.b[v], c, d, True)

    def modify_edge_dS(self, u, v, dm, entropy_args={}):
        """Computes the difference in the description length if edge :math:`(u,
        v)` would have its multiplicity changed by an integer difference ``dm``,
        taking into account the ``entropy_args`` parameters as described in
        :meth:`~BlockState.entropy`.
        """
        ea = self._get_entropy_args(dict(self._entropy_args, **entropy_args))
        return self._state.modify_edge_dS(u, v, dm, ea)

    def modify_edge(self, u, v, dm):
        """Changes the multiplicity of edge ;math:`(u, v)` by a integer
        difference ``dm``."""
        return self._state.modify_edge(u, v, dm)

    def _clear_egroups(self):
        self._state.clear_egroups()

    def collect_edge_marginals(self, p=None, update=1):
        r"""Collect the edge marginal histogram, which counts the number of times
        the endpoints of each node have been assigned to a given block pair.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        p : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            Edge property map with edge marginals to be updated.  If not
            provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.EdgePropertyMap`
            Edge property map with updated edge marginals.

        Examples
        --------
        .. doctest:: collect_edge_marginals

           >>> np.random.seed(42)
           >>> gt.seed_rng(42)
           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> pe = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     pe = state.collect_edge_marginals(pe)
           >>> gt.bethe_entropy(g, pe)[0]
           -19.622404...

        """

        if p is None:
            p = self.g.new_ep("python::object",
                              vals=[libinference.BlockPairHist()
                                    for i in range(self.g.num_edges())])

        libinference.edge_marginals(self.g._Graph__graph,
                                    _prop("v", self.g, self.b),
                                    _prop("e", self.g, p),
                                    update)
        return p

    def collect_vertex_marginals(self, p=None, b=None, unlabel=False, update=1):
        r"""Collect the vertex marginal histogram, which counts the number of times a
        node was assigned to a given block.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        p : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with vector-type values, storing the previous block
            membership counts. If not provided, an empty histogram will be created.
        b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with group partition. If not provided, the
            state's partition will be used.
        unlabel : bool (optional, default: ``False``)
            If ``True``, a canonical labelling of the groups will be used, so
            that each partition is uniquely represented.
        update : int (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.VertexPropertyMap`
            Vertex property map with vector-type values, storing the accumulated
            block membership counts.

        Examples
        --------
        .. doctest:: cvm

           >>> np.random.seed(42)
           >>> gt.seed_rng(42)
           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> pv = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     pv = state.collect_vertex_marginals(pv)
           >>> gt.mf_entropy(g, pv)
           33.094673...
           >>> gt.graph_draw(g, pos=g.vp["pos"], vertex_shape="pie",
           ...               vertex_pie_fractions=pv, output="polbooks_blocks_soft_B4.svg")
           <...>

        .. figure:: polbooks_blocks_soft_B4.*
           :align: center

           "Soft" block partition of a political books network with :math:`B=4`.

        """

        if p is None:
            p = self.g.new_vp("vector<int>")
        if b is None:
            b = self.b
        if unlabel:
            b = perfect_prop_hash([b])[0]
        libinference.vertex_marginals(self.g._Graph__graph,
                                      _prop("v", self.g, b),
                                      _prop("v", self.g, p),
                                      update)
        return p

    def collect_partition_histogram(self, h=None, update=1, unlabel=True):
        r"""Collect a histogram of partitions.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Partition histogram. If not provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.
        unlabel : bool (optional, default: ``True``)
            If ``True``, a canonical labelling of the groups will be used, so
            that each partition is uniquely represented.

        Returns
        -------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Updated Partition histogram.

        Examples
        --------
        .. doctest:: cvm

           >>> np.random.seed(42)
           >>> gt.seed_rng(42)
           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> ph = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     ph = state.collect_partition_histogram(ph)
           >>> gt.microstate_entropy(ph)
           137.136857...

        """

        if h is None:
            h = PartitionHist()
        libinference.collect_partitions(_prop("v", self.g, self.b),
                                        h, update, unlabel)
        return h

    def sample_graph(self, canonical=False, multigraph=True, self_loops=True,
                     sample_params=False, max_ent=False, n_iter=1000):
        r"""Sample a new graph from the fitted model.

        Parameters
        ----------
        canonical : ``bool`` (optional, default: ``False``)
            If ``canonical == True``, the graph will be sampled from the
            maximum-likelihood estimate of the canonical stochastic block
            model. Otherwise, it will be sampled from the microcanonical model.
        multigraph : ``bool`` (optional, default: ``True``)
            If ``True``, parallel edges will be allowed.
        self-loops : ``bool`` (optional, default: ``True``)
            If ``True``, self-loops will be allowed.
        sample_params : ``bool`` (optional, default: ``True``)
            If ``True``, and ``canonical == True`` and ``max_ent == False``,
            the count parameters (edges between groups and node degrees) will be
            sampled from their posterior distribution conditioned on the actual
            state. Otherwise, their maximum-likelihood values will be used.
        max_ent : ``bool`` (optional, default: ``False``)
            If ``True``, maximum-entropy model variants will be used.
        n_iter : ``int`` (optional, default: ``1000``)
            Number of iterations used (only relevant if ``canonical == False``
            and ``max_ent == True``).

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            Generated graph.

        Notes
        -----
        This function is just a convenience wrapper to
        :func:`~graph_tool.generation.generate_sbm`. However, if
        ``max_ent==True`` and ``canonical == False`` it wraps
        :func:`~graph_tool.generation.random_rewire` instead.

        Examples
        --------
        .. doctest:: gen_sbm

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.minimize_blockmodel_dl(g, multilevel_mcmc_args=dict(B_max=3))
           >>> u = state.sample_graph(canonical=True, self_loops=False, multigraph=False)
           >>> ustate = gt.BlockState(u, b=state.b)
           >>> state.draw(pos=g.vp.pos, output="polbooks-sbm.svg")
           <...>
           >>> ustate.draw(pos=u.own_property(g.vp.pos), output="polbooks-sbm-sampled.svg")
           <...>

        .. image:: polbooks-sbm.svg
            :width: 40%
        .. image:: polbooks-sbm-sampled.svg
            :width: 40%

        *Left:* Political books network. *Right:* Sample from the degree-corrected
        SBM fitted to the original network.

        """
        in_degs = out_degs = None
        eweight = self.eweight
        if self.deg_corr:
            out_degs = self.g.degree_property_map("out", weight=eweight).fa
            if self.g.is_directed():
                in_degs = self.g.degree_property_map("in", weight=eweight).fa
            else:
                in_degs = None
        probs = adjacency(self.bg, weight=self.mrs).T
        if not max_ent:
            if canonical and sample_params:
                rs = self.wr.a > 0
                B = rs.sum()
                if self.g.is_directed():
                    p = self.g.num_edges() / B ** 2
                if not self.g.is_directed():
                    p = 2 * self.g.num_edges() / ((B + 1) * B)
                idx = probs.nonzero()
                probs[idx] = np.random.gamma(probs[idx] + 1, p/(p + 1))
                for r in rs:
                    idx = self.b.fa == r
                    out_degs[idx] = np.random.dirichlet(out_degs[idx] + 1)
                    if in_degs is not None:
                        in_degs[idx] = np.random.dirichlet(in_degs[idx] + 1)

            g = generate_sbm(b=self.b.fa, probs=probs,
                             in_degs=in_degs, out_degs=out_degs,
                             directed=self.g.is_directed(),
                             micro_ers=not canonical,
                             micro_degs=not canonical and self.deg_corr)

            if not multigraph:
                remove_parallel_edges(g)
            if not self_loops:
                remove_self_loops(g)
        else:
            if canonical:
                ret = solve_sbm_fugacities(self.b.fa, probs, out_degs, in_degs,
                                           multigraph=multigraph,
                                           self_loops=self_loops)
                if in_degs is None:
                    mrs, theta_out = ret
                    theta_in = None
                else:
                    mrs, theta_out, theta_in = ret
                g = generate_maxent_sbm(self.b.fa, mrs, theta_out, theta_in,
                                        directed=self.g.is_directed(),
                                        multigraph=multigraph,
                                        self_loops=self_loops)
            else:
                g = self.g.copy()
                if self.deg_corr:
                    random_rewire(g, model="constrained-configuration",
                                  block_membership=g.own_property(self.b),
                                  configuration=False, parallel_edges=multigraph,
                                  self_loops=self_loops, n_iter=n_iter)
                else:
                    random_rewire(g, model="blockmodel-micro",
                                  block_membership=g.own_property(self.b),
                                  configuration=False, parallel_edges=multigraph,
                                  self_loops=self_loops, n_iter=n_iter)
        return g


@entropy_state_signature
class WeightedBlockState(BlockState):
    r"""The weighted stochastic block model state of a given graph (i.e. with
    edge covariates).

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    rec : list of :class:`~graph_tool.EdgePropertyMap` instances (optional, default: ``[]``)
        List of real or discrete-valued edge covariates.
    rec_types : list of edge covariate types (optional, default: ``[]``)
        List of types of edge covariates. The possible types are:
        ``"real-exponential"``, ``"real-normal"``, ``"discrete-geometric"``,
        ``"discrete-poisson"`` or ``"discrete-binomial"``.
    rec_params : list of ``dict`` (optional, default: ``[]``)
        Model hyperparameters for edge covariates. This should be a list of
        ``dict`` instances, or the string `"microcanonical"` (the default if
        nothing is specified). The keys depend on the type of edge covariate:

        ``"real-exponential"`` or ``"discrete-poisson"``
            The parameter list is ``["r", "theta"]``, corresponding to the
            parameters of the `Gamma
            <https://en.wikipedia.org/wiki/Gamma_distribution>`_ prior
            distribution.  If unspecified, the default is the "empirical Bayes"
            choice: ``r = 1.0`` and ``theta`` is the global average of the edge
            covariate.

        ``"discrete-geometric"``
            The parameter list is ``["alpha", "beta"]``, corresponding to the
            parameters of the `Beta
            <https://en.wikipedia.org/wiki/Beta_distribution>`_ prior
            distribution. If unspecified, the default is the noninformative
            choice: ``alpha = beta = 1.0``

        ``"discrete-binomial"``
            The parameter list is ``["N", "alpha", "beta"]``, corresponding to
            the number of trials ``N`` and the parameters of the `Beta
            <https://en.wikipedia.org/wiki/Beta_distribution>`_ prior
            distribution. If unspecified, the default is the noninformative
            choice, ``alpha = beta = 1.0``, and ``N`` is taken to be the maximum
            edge covarite value.

        ``"real-normal"``
            The parameter list is ``["m0", "k0", "v0", "nu0"]`` corresponding to
            the `normal
            <https://en.wikipedia.org/wiki/Normal_distribution>`_-`inverse-chi-squared
            <https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution>`_
            prior. If unspecified, the defaults are: ``m0 = rec.fa.mean()``,
            ``k0 = 1``, ``v0 = rec.fa.std() ** 2``, and ``nu0 = 3``, where
            ``rec`` is the corresponding edge covariate property map.
    **kwargs : ``(keywork parameters)`` (optional, default: ``(none)``)
        Parameters to be passed to :class:`~graph_tool.inference.BlockState`.
    """

    @kwargs_wrap
    def __init__(self, g, rec=[], rec_types=[], rec_params=[], **kwargs):

        if hasattr(self, "rec"):
            return

        BlockState.__init__(self, g, **kwargs)

        self.drec = kw_pop(kwargs, "drec", None)
        self.Lrecdx = kw_pop(kwargs, "Lrecdx", None)
        self.epsilon = kw_pop(kwargs, "epsilon", None)
        self.is_base = kw_pop(kwargs, "is_base", True)
        rvtype = kw_pop(kwargs, "rvtype", "double")
        self.rvtype = rvtype

        self.rec = [self.g.own_property(p) for p in rec]

        for i in range(len(self.rec)):
            if self.rec[i].value_type() != rvtype:
                self.rec[i] = self.rec[i].copy(rvtype)

        if self.drec is None:
            self.drec = []
            for _ in self.rec:
                self.drec.append(self.g.new_ep(rvtype))
        else:
            self.drec = [self.g.own_property(p) for p in self.drec]

        rec_types = list(rec_types)
        rec_params = list(rec_params)

        if len(self.rec) == 0 or rec_types[0] != libinference.rec_type.count:
            rec_types.insert(0, libinference.rec_type.count)
            rec_params.insert(0, {})
            self.rec.insert(0, self._eweight.copy(rvtype))
            self.drec.insert(0, self.g.new_ep(rvtype))

        self._init_rec(self.rec, rec_types, rec_params)
        self.recdx = libcore.Vector_double(len(self.rec))
        if self.Lrecdx is None:
            self.Lrecdx = libcore.Vector_double(len(self.rec)+1)
            self.Lrecdx[0] = -1
        self.Lrecdx.resize(len(self.rec)+1)
        if self.epsilon is None:
            self.epsilon = libcore.Vector_double()

        if type(self) is WeightedBlockState:
            with self._mk_state():
                self._state = libinference.make_weighted_block_model_state(self)

    @property
    def eweight(self):
        r""":class:`~graph_tool.EdgePropertyMap` with edge multiplicities"""
        if self._mk_override:
            if isinstance(self._eweight, libcore.any):
                return self._eweight
            return (self._eweight, self.rec, self.drec, self.rec_types)
        ew = self.g.new_ep("int64_t")
        self._state.get_eweight(ew)
        return ew

    @property
    def mrs(self):
        r""":class:`~graph_tool.EdgePropertyMap` with with edge counts between groups"""
        if self._mk_override:
            return (self._mrs, self.brec, self.bdrec, self.rec_types)
        return self._mrs

    def _init_rec(self, rec, rec_types, rec_params):
        if len(rec_types) != len(rec):
            raise ValueError("The size of 'rec_types' (%d) must be the same of 'rec' (%d)" %
                             (len(rec_types), len(rec)))
        self.rec_types = libcore.Vector_int32_t()
        for rec_type in rec_types:
            if rec_type == "real-exponential":
                rt = libinference.rec_type.real_exponential
            elif rec_type == "real-normal":
                rt = libinference.rec_type.real_normal
            elif rec_type == "discrete-geometric":
                rt = libinference.rec_type.discrete_geometric
            elif rec_type == "discrete-poisson":
                rt = libinference.rec_type.discrete_poisson
            elif rec_type == "discrete-binomial":
                rt = libinference.rec_type.discrete_binomial
            else:
                rt = rec_type
            self.rec_types.append(rt)

        self.brec = [self.bg.new_ep(self.rvtype) for p in self.rec]
        self.bdrec = [self.bg.new_ep(self.rvtype) for p in self.drec]

        self.rec_params = rec_params = list(rec_params)
        while len(rec_params) < len(self.rec_types):
            rec_params.append("microcanonical")
        self.wparams = libcore.Vector_Vector_double()

        for i, rt in enumerate(self.rec_types):
            ps = Vector_double()
            if rt == libinference.rec_type.count:
                defaults = {}
            elif rt in [libinference.rec_type.real_exponential,
                        libinference.rec_type.discrete_poisson]:
                if rec_params[i] != "microcanonical":
                    defaults = {"alpha": 1,
                                "beta": self.rec[i].fa.mean()}
                else:
                    defaults = {"alpha": np.nan,
                                "beta": np.nan}
            elif rt == libinference.rec_type.real_normal:
                if rec_params[i] != "microcanonical":
                    defaults = {"m0": self.rec[i].fa.mean(),
                                "k0": 1,
                                "v0": self.rec[i].fa.std() ** 2,
                                "nu0": 3}
                else:
                    defaults = {"m0": np.nan,
                                "k0": np.nan,
                                "v0": np.nan,
                                "nu0": np.nan}
            elif rt == libinference.rec_type.discrete_geometric:
                if rec_params[i] != "microcanonical":
                    defaults = {"alpha": 1,
                                "beta": 1}
                else:
                    defaults = {"alpha": np.nan,
                                "beta": np.nan}
            elif rt == libinference.rec_type.discrete_binomial:
                if rec_params[i] != "microcanonical":
                    if "N" in rec_params[i] and len(rec_params[i]) == 1:
                        defaults = {"N" : rec_params[i]["N"],
                                    "alpha": np.nan,
                                    "beta": np.nan}
                    else:
                        defaults = {"N": self.rec[i].fa.max(),
                                    "alpha": 1,
                                    "beta": 1}
                else:
                    defaults = {"N": self.rec[i].fa.max(),
                                "alpha": np.nan,
                                "beta": np.nan}

            ks = list(defaults.keys())
            if rec_params[i] != "microcanonical":
                defaults.update(rec_params[i])
            rec_params[i] = defaults.copy()
            for k in ks:
                ps.append(defaults.pop(k))
            if len(defaults) > 0:
                raise ValueError("unknown parameters for weight type: " +
                                 str(list(defaults.keys())))
            self.wparams.append(ps)

    def get_rec_params(self):
        """Get model hyperparameters for edge covariates."""
        params = []
        for rt, ps in zip(self.rec_types, self.rec_params):
            if rt == libinference.rec_type.count:
                continue
            p = dict(ps)
            params.append(p)
        return params

    def set_rec_params(self, params):
        """Update model hyperparameters for edge covariates."""
        for ps, ws, nps in zip(self.rec_params[1:], self.wparams[1:], params):
            ps.update(nps)
            for i, (k, v) in enumerate(ps.items()):
                ws[i] = v

    def _repr_extra(self):
        return " with " + pl(len(self.rec_types) - 1, "edge covariate") + ","

    def _get_entropy_args(self, kwargs, consume=False):
        ea = BlockState._get_entropy_args(self, kwargs, consume)
        dl = kwargs.get("rec_dl", self._entropy_args["dl"])
        if not dl:
            ea.rec_dl = False
        return ea

    def _entropy(self, rec=True, rec_dl=True, **kwargs):
        r"""Calculate the description length (a.k.a. negative joint
        log-likelihood) associated with the current block partition.

        Parameters
        ----------
        rec : ``bool`` (optional, default: ``True``)
            If ``True``, the likelihood for real or discrete-valued edge
            covariates is computed.
        rec_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the edge covariate description
            length will be included.

        References
        ----------
        .. [peixoto-weighted-2017] Tiago P. Peixoto, "Nonparametric weighted
           stochastic block models", Phys. Rev. E 97, 012306 (2018),
           :doi:`10.1103/PhysRevE.97.012306`, :arxiv:`1708.01432`

        """

        return BlockState._entropy(**dict(locals(), **kwargs))

    def _get_block_state(self, b, final=False, couple=True, **kwargs):
        rec = []
        rec_types = []
        rec_params = []
        for rt, rp, r in zip(self.rec_types, self.wparams, self.brec):
            if rt == libinference.rec_type.count:
                rec.append(self.bg.new_ep(self.rvtype))
                rec_types.append(rt)
                rec_params.append("microcanonical")
            elif rt in [libinference.rec_type.discrete_geometric,
                        libinference.rec_type.discrete_binomial,
                        libinference.rec_type.discrete_poisson]:
                rec.append(r)
                rec_types.append(libinference.rec_type.discrete_geometric)
                rec_params.append("microcanonical")
            elif rt == libinference.rec_type.real_exponential:
                rec.append(r)
                rec_types.append(rt)
                rec_params.append("microcanonical")
            elif rt == libinference.rec_type.real_normal:
                rec.append(r)
                rec_types.append(rt)
                rec_params.append("microcanonical")

        state = BlockState._get_block_state(self, b, final=final, couple=False,
                                            rec_types=rec_types,
                                            rec=rec,
                                            drec=None,
                                            rec_params=rec_params,
                                            Lrecdx=self.Lrecdx,
                                            epsilon=self.epsilon, is_base=False,
                                            **kwargs)

        state.update_entropy_args(dense=True, rec_dl=final, edges_dl=final)
        if couple:
            self._couple_state(state, state.get_entropy_args())
        return state

    def _couple_state(self, state, entropy_args):
        BlockState._couple_state(self, state, entropy_args)
        self.update_entropy_args(rec_dl=False)

def bethe_entropy(g, p):
    r"""Compute the Bethe entropy given the edge block membership marginals.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    p : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with edge marginals.

    Returns
    -------
    H : ``float``
        The Bethe entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__)
    Hmf : ``float``
        The "mean field" entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__),
        as would be returned by the :func:`mf_entropy` function.
    pv : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with vector-type values, storing the accumulated
        block membership counts. These are the node marginals, as would be
        returned by the
        :meth:`~graph_tool.inference.BlockState.collect_vertex_marginals`
        method.

    Notes
    -----

    The Bethe entropy is defined as,

    .. math::

        H = -\sum_{i<j}A_{ij}\sum_{rs}\pi_{ij}(r,s)\ln\pi_{ij}(r,s) - \sum_i(1-k_i)\sum_r\pi_i(r)\ln\pi_i(r),

    where :math:`\pi_{ij}(r,s)` is the marginal probability that vertices
    :math:`i` and :math:`j` belong to blocks :math:`r` and :math:`s`,
    respectively, and :math:`\pi_i(r)` is the marginal probability that vertex
    :math:`i` belongs to block :math:`r`, and :math:`k_i` is the degree of
    vertex :math:`i` (or total degree for directed graphs).

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`

    """
    H = 0
    pv =  g.new_vertex_property("vector<double>")

    H, Hmf  = libinference.bethe_entropy(g._Graph__graph,
                                         _prop("e", g, p),
                                         _prop("v", g, pv))
    return H, Hmf, pv


def mf_entropy(g, p):
    r"""Compute the "mean field" entropy given the vertex block membership marginals.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    p : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with vector-type values, storing the accumulated block
        membership counts.

    Returns
    -------
    Hmf : ``float``
        The "mean field" entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__).

    Notes
    -----

    The "mean field" entropy is defined as,

    .. math::

        H = - \sum_{i,r}\pi_i(r)\ln\pi_i(r),

    where :math:`\pi_i(r)` is the marginal probability that vertex :math:`i`
    belongs to block :math:`r`.

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`
    """

    return libinference.mf_entropy(g._Graph__graph,
                                   _prop("v", g, p))

def microstate_entropy(h, unlabel=True):
    r"""Compute microstate entropy given a histogram of partitions.

    Parameters
    ----------
    h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
        Partition histogram.
    unlabel : bool (optional, default: ``True``)
        If ``True``, a canonical labelling of the groups will be used, so that
        each partition is uniquely represented. However, the entropy computed
        will still correspond to the full distribution over labelled partitions,
        where all permutations are assumed to be equally likely.

    Returns
    -------
    H : ``float``
        The microstate entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__).

    Notes
    -----

    The microstate entropy is defined as,

    .. math::

        H = - \sum_{\boldsymbol b}p({\boldsymbol b})\ln p({\boldsymbol b}),

    where :math:`p({\boldsymbol b})` is observed frequency of labelled partition
    :math:`{\boldsymbol b}`.

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`

    """

    return libinference.partitions_entropy(h, unlabel)
