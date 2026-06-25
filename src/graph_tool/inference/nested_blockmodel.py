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

from .. import _prop, Graph, GraphView

from . base_states import _bm_test
from . base_states import *

from . blockmodel import *

import numpy as np
import copy

class NestedBlockState(object):
    r"""The nested stochastic block model state of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modeled.
    bs : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`numpy.ndarray` (optional, default: ``None``)
        Hierarchical node partition. If not provided it will correspond to a
        single-group hierarchy of length :math:`\lceil\log_2(N)\rceil`.
    base_state : ``type`` (optional, default: :class:`~graph_tool.inference.BlockState`)
        State type for lowermost level
        (one of :class:`~graph_tool.inference.BlockState`,
        :class:`~graph_tool.inference.WeightedBlockState`,
        :class:`~graph_tool.inference.OverlapBlockState`,
        :class:`~graph_tool.inference.LayeredBlockState`,
        :class:`~graph_tool.inference.LayeredOverlapBlockState`,
        :class:`~graph_tool.inference.WeightedOverlapBlockState`,
        :class:`~graph_tool.inference.LayeredWeightedOverlapBlockState`)
    base_state_args : ``dict`` (optional, default: ``{}``)
        Keyword arguments to be passed to base type constructor.
    **kwargs :  keyword arguments
        Keyword arguments to be passed to base type constructor. The
        ``base_state_args`` parameter overrides this.
    """

    def __init__(self, g, bs=None, base_state=BlockState, base_state_args={},
                 **kwargs):
        self.g = g

        self.base_state = base_state
        self.base_state_args = dict(kwargs, **base_state_args)

        self.levels = [base_state(g, b=bs[0] if bs is not None else None,
                                 **self.base_state_args)]

        if bs is None:
            N = self.levels[0].get_N()
            L = int(np.ceil(np.log2(N)))
            bs = [None] * (L + 1)

        for i, b in enumerate(bs[1:]):
            state = self.levels[-1]
            bstate = state._get_block_state(b=b, final=i == len(bs[1:]) - 1)
            self.levels.append(bstate)

        if _bm_test():
            self._consistency_check()

    def __repr__(self):
        return "<NestedBlockState object, with base %s, and %d levels of sizes %s at 0x%x>" % \
            (repr(self.levels[0]), len(self.levels),
             str([(s.get_N(), s.get_nonempty_B()) for s in self.levels]), id(self))

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the block state. The parameters override the state properties,
        and have the same meaning as in the constructor."""
        state = dict(self.__getstate__(), **kwargs)
        return NestedBlockState(**state)

    def __getstate__(self):
        base_state = self.levels[0].__getstate__()
        base_state_args = dict(self.base_state_args, **base_state)
        base_state_args.pop("g", None)
        base_state_args.pop("b", None)
        state = dict(g=self.g, bs=self.get_bs(),
                     base_state=type(self.levels[0]),
                     base_state_args=base_state_args)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def get_bs(self):
        """Get hierarchy levels as a list of :class:`numpy.ndarray` and
        :class:`~graph_tool.VertexPropertyMap` objects with the group
        memberships at each level.
        """
        return [s.b.copy() if l == 0 else s.b.fa.copy() for l, s in enumerate(self.levels)]

    def get_state(self):
        """Alias to :meth:`~NestedBlockState.get_bs`."""
        return self.get_bs()

    def set_state(self, bs):
        r"""Sets the internal nested partition of the state."""
        for i in range(len(bs)):
            self.levels[i].set_state(bs[i])

    def get_levels(self):
        """Get hierarchy levels as a list of :class:`~graph_tool.inference.BlockState`
        instances."""
        return self.levels

    def project_partition(self, j, l):
        """Project partition of level ``j`` onto level ``l``, and return it."""
        b = self.levels[l].b.copy()
        for i in range(l + 1, j + 1):
            clabel = self.levels[i].b.copy()
            pmap(b, clabel)
        return b

    def _consistency_check(self):
        for l in range(1, len(self.levels)):
            b = self.levels[l].b.fa.copy()
            state = self.levels[l-1]
            bstate = state._get_block_state(b=b,
                                            final=l == len(self.levels) - 1,
                                            couple=False)
            b2 = bstate.b.fa.copy()
            b = contiguous_map(b)
            b2 = contiguous_map(b2)
            assert (b == b2).all(), \
                "inconsistent level %d (%s,  %s): %s" % \
                (l, str(bstate), str(self.levels[l]), str(self))

    def level_entropy(self, l, bstate=None, **kwargs):
        """Compute the entropy of level ``l``."""
        if bstate is None:
            bstate = self.levels[l]
        if l > 0:
            eargs = {}
        else:
            eargs = kwargs
        S = bstate.entropy(test=kwargs.get("test", True), propagate=False,
                           **eargs)
        if l > 0:
            S *= kwargs.get("beta_dl", 1.)
        return S

    @copy_state_wrap
    def entropy(self, **kwargs):
        """Obtain the description length (i.e. negative joint log-likelihood)
        for the hierarchical partition.

        The keyword arguments are passed to the ``entropy()`` method of the
        underlying state objects
        (e.g. :class:`graph_tool.inference.BlockState.entropy`,
        :class:`graph_tool.inference.OverlapBlockState.entropy`, or
        :class:`graph_tool.inference.LayeredBlockState.entropy`).

        """
        S = self.levels[0].entropy(**dict(kwargs, test=False))
        return S

    def move_vertex(self, v, s):
        r"""Move vertex ``v`` to block ``s``."""
        self.levels[0].move_vertex(v, s)

    def remove_vertex(self, v):
        r"""Remove vertex ``v`` from its current group.

        This optionally accepts a list of vertices to remove.

        .. warning::

           This will leave the state in an inconsistent state before the vertex
           is returned to some other group, or if the same vertex is removed
           twice.
        """
        self.levels[0].remove_vertex(v)

    def add_vertex(self, v, r):
        r"""Add vertex ``v`` to block ``r``.

        This optionally accepts a list of vertices and blocks to add.

        .. warning::

           This can leave the state in an inconsistent state if a vertex is
           added twice to the same group.
        """
        self.levels[0].add_vertex(v, r)

    def get_bstack(self):
        """Return the nested levels as individual graphs.

        This returns a list of :class:`~graph_tool.Graph` instances
        representing the inferred hierarchy at each level. Each graph has two
        internal vertex and edge property maps named "count" which correspond to
        the vertex and edge counts at the lower level, respectively. Additionally,
        an internal vertex property map named "b" specifies the block partition.
        """

        bstack = []
        for l, bstate in enumerate(self.levels):
            cg = bstate.g
            if l == 0:
                cg = GraphView(cg, skip_properties=True)
            try:
                cg.vp["b"] = bstate.b.copy()
            except ValueError:
                cg.ep["b"] = bstate.b.copy()
            cg.ep["count"] = cg.own_property(bstate.eweight.copy())
            try:
                cg.vp["count"] = cg.own_property(bstate.vweight.copy())
            except ValueError:
                pass

            bstack.append(cg)
            if bstate.get_N() == 1:
                break
        return bstack

    def project_level(self, l):
        """Project the partition at level ``l`` onto the lowest level, and return the
        corresponding state."""
        b = self.project_partition(l, 0)
        return self.levels[0].copy(b=b)

    def print_summary(self):
        """Print a hierarchy summary."""
        for l, state in enumerate(self.levels):
            print("l: %d, N: %d, B: %d" % (l, state.get_N(),
                                           state.get_nonempty_B()))
            if state.get_N() == 1:
                break

    def _clear_egroups(self):
        for lstate in self.levels:
            lstate._clear_egroups()

    def _h_sweep_gen(self, **kwargs):

        verbose = kwargs.get("verbose", False)
        c = kwargs.get("c", None)
        eargs = kwargs.get("entropy_args", {})

        lrange = list(kwargs.pop("ls", range(len(self.levels))))
        if kwargs.pop("ls_shuffle", False):
            np.random.shuffle(lrange)
        for l in lrange:
            if check_verbose(verbose):
                print(verbose_pad(verbose) + "level:", l)

            if c is None:
                args = dict(kwargs)
            else:
                args = dict(kwargs, c=c[l])

            if l == 0:
                args["entropy_args"] = eargs

            if l > 0:
                args.pop("vertices", None)
                if "beta_dl" in eargs:
                    args = dict(args, beta=args.get("beta", 1.) * eargs["beta_dl"])

            yield l, self.levels[l], args

    def _h_sweep(self, algo, **kwargs):
        entropy_args = kwargs.get("entropy_args", {})

        dS = 0
        nattempts = 0
        nmoves = 0

        for l, lstate, args in self._h_sweep_gen(**kwargs):
            ret = algo(self.levels[l], **dict(args, test=False))
            if l > 0 and "beta_dl" in entropy_args:
                dS += ret[0] * entropy_args["beta_dl"]
            else:
                dS += ret[0]
            nattempts += ret[1]
            nmoves += ret[2]

        return dS, nattempts, nmoves

    def _h_sweep_states(self, algo, **kwargs):
        entropy_args = kwargs.get("entropy_args", {})
        for l, lstate, args in self._h_sweep_gen(**kwargs):
            beta_dl = entropy_args.get("beta_dl", 1) if l > 0 else 1
            yield l, lstate, algo(self.levels[l], dispatch=False, **args), beta_dl

    @mcmc_sweep_wrap
    def mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection
        MCMC to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            return self._h_sweep(lambda s, **a: s.mcmc_sweep(**a), c=c,
                                 **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.mcmc_sweep(**a),
                                        c=c, **kwargs)

    @mcmc_sweep_wrap
    def multiflip_mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection MCMC
        with multiple moves to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.multiflip_mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        kwargs["psingle"] = kwargs.get("psingle", self.levels[0].get_N())

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            def dispatch_level(s, **a):
                if s is not self.levels[0]:
                    a = dict(**a)
                    a.pop("B_min", None)
                    a.pop("B_max", None)
                    a.pop("b_min", None)
                    a.pop("b_max", None)
                    a.pop("vertices", None)
                return s.multiflip_mcmc_sweep(**a)

            return self._h_sweep(dispatch_level, c=c, **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.multiflip_mcmc_sweep(**a),
                                        c=c, **kwargs)

    @mcmc_sweep_wrap
    def multilevel_mcmc_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection MCMC
        with multilevel moves to sample hierarchical network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.multilevel_mcmc_sweep`.

        If the parameter ``c`` is a scalar, the values used at each level are
        ``c * 2 ** l`` for ``l`` in the range ``[0, L-1]``. Optionally, a list
        of values may be passed instead, which specifies the value of ``c[l]``
        to be used at each level.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """

        c = kwargs.pop("c", 1)
        if not isinstance(c, collections.abc.Iterable):
            c = [c * 2 ** l for l in range(0, len(self.levels))]

        if kwargs.pop("dispatch", True):
            def dispatch_level(s, **a):
                if s is not self.levels[0]:
                    a = dict(**a)
                    a.pop("B_min", None)
                    a.pop("B_max", None)
                    a.pop("b_min", None)
                    a.pop("b_max", None)
                    a.pop("vertices", None)
                return s.multilevel_mcmc_sweep(**a)
            return self._h_sweep(dispatch_level, c=c, **kwargs)
        else:
            return self._h_sweep_states(lambda s, **a: s.multilevel_mcmc_sweep(**a),
                                        c=c, **kwargs)

    @mcmc_sweep_wrap
    def gibbs_sweep(self, **kwargs):
        r"""Perform ``niter`` sweeps of a rejection-free Gibbs sampling MCMC
        to sample network partitions.

        The arguments accepted are the same as in
        :meth:`graph_tool.inference.BlockState.gibbs_sweep`.

        .. warning::

           This function performs ``niter`` sweeps at each hierarchical level
           once. This means that in order for the chain to equilibrate, we need
           to call this function several times, i.e. it is not enough to call
           it once with a large value of ``niter``.

        """
        return self._h_sweep(lambda s, **a: s.gibbs_sweep(**a),
                             **kwargs)

    def collect_partition_histogram(self, h=None, update=1):
        r"""Collect a histogram of partitions.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.NestedBlockState.mcmc_sweep` function.

        Parameters
        ----------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Partition histogram. If not provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        h : :class:`~graph_tool.inference.PartitionHist` (optional, default: ``None``)
            Updated Partition histogram.

        """

        if h is None:
            h = PartitionHist()
        bs = [_prop("v", state.g, state.b) for state in self.levels]
        libinference.collect_hierarchical_partitions(bs, h, update)
        return h

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.draw_hierarchy` that
        draws the hierarchical state."""
        import graph_tool.draw
        return graph_tool.draw.draw_hierarchy(self,
                                              vcmap=kwargs.pop("vcmap",
                                                               graph_tool.draw.default_cm),
                                              **kwargs)

def get_hierarchy_tree(state, empty_branches=False):
    r"""Obtain the nested hierarchical levels as a tree.

    This transforms a :class:`~graph_tool.inference.NestedBlockState` instance
    into a single :class:`~graph_tool.Graph` instance containing the hierarchy
    tree.

    Parameters
    ----------
    state : :class:`~graph_tool.inference.NestedBlockState`
       Nested block model state.
    empty_branches : ``bool`` (optional, default: ``False``)
       If ``empty_branches == False``, dangling branches at the upper layers
       will be pruned.

    Returns
    -------

    tree : :class:`~graph_tool.Graph`
       A directed graph, where vertices are blocks, and a directed edge points
       to an upper to a lower level in the hierarchy.
    label : :class:`~graph_tool.VertexPropertyMap`
       A vertex property map containing the block label for each node.
    order : :class:`~graph_tool.VertexPropertyMap`
       A vertex property map containing the relative ordering of each layer
       according to the total degree of the groups at the specific levels.
    """

    bstack = state.get_bstack()

    g = bstack[0]
    try:
        b = g.vp["b"]
    except KeyError:
        b = state.levels[0].get_majority_groups()
    bstack = bstack[1:]

    def get_ew(g):
        ew = g.ep.count
        if "vector" in g.ep.count.value_type():
            ew = ew.t(sum, no_array=True, value_type="int64_t")
        return ew

    if bstack[-1].num_vertices() > 1:
        bg = Graph(directed=g.is_directed())
        bg.add_vertex()
        e = bg.add_edge(0, 0)
        bg.vp.count = bg.new_vp("int64_t", 1)
        ew = get_ew(g)
        bg.ep.count = bg.new_ep("int64_t", ew.fa.sum())
        bg.vp.b = bg.new_vp("int64_t", 0)
        bstack.append(bg)

    t = Graph()

    if g.get_vertex_filter() is None:
        t.add_vertex(g.num_vertices())
    else:
        t.add_vertex(g.num_vertices(ignore_filter=True))
        filt = g.get_vertex_filter()
        t.set_vertex_filter(t.own_property(filt.copy()))
    label = t.vertex_index.copy("int64_t")

    order = t.own_property(g.degree_property_map("total").copy())
    t_vertices = list(t.vertices())

    last_pos = 0
    for l, s in enumerate(bstack):
        pos = t.num_vertices()
        if s.num_vertices() > 1:
            t_vertices.extend(t.add_vertex(s.num_vertices()))
        else:
            t_vertices.append(t.add_vertex(s.num_vertices()))
        label.a[-s.num_vertices():] = np.arange(s.num_vertices())

        # relative ordering based on total degree
        count = get_ew(s).copy("double")
        for e in s.edges():
            if e.source() == e.target():
                count[e] /= 2
        vs = []
        pvs = {}
        for vi in range(pos, t.num_vertices()):
            vs.append(t_vertices[vi])
            pvs[vs[-1]] = vi - pos
        vs = sorted(vs, key=lambda v: (s.vertex(pvs[v]).out_degree(count) +
                                       s.vertex(pvs[v]).in_degree(count)))
        for vi, v in enumerate(vs):
            order[v] = vi

        for vi, v in enumerate(g.vertices()):
            w = t_vertices[vi + last_pos]
            if s.num_vertices() == 1:
                u = t_vertices[pos]
            else:
                u = t_vertices[b[v] + pos]
            t.add_edge(u, w)

        last_pos = pos
        g = s
        if empty_branches:
            if g.num_vertices() == 1:
                break
        else:
            if g.vp.count.fa.sum() == 1:
                break
        b = g.vp["b"]

    if not empty_branches:
        vmask = t.new_vertex_property("bool", True)
        t = GraphView(t, vfilt=vmask)
        vmask = t.get_vertex_filter()
        N = t.num_vertices()
        for vi, v in enumerate(list(t.vertices())):
            if vi < state.g.num_vertices():
                continue
            if v.out_degree() == 0:
                vmask[v] = False
        label = t.own_property(label)
        order = t.own_property(order)

    return t, label, order

from . minimize import *
