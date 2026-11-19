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

import numpy
from . util import *
from . blockmodel import *
from . nested_blockmodel import *
from .. decorators import _parallel

@_parallel
def minimize_blockmodel_dl(g, state=BlockState, state_args={},
                           multilevel_mcmc_args={}, refine=True,
                           epsilon=0.001):
    r"""Fit the stochastic block model, by minimizing its description length using an
    agglomerative heuristic.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    state : SBM-like state class (optional, default: :class:`~graph_tool.inference.BlockState`)
        Type of model that will be used. Must be derived from :class:`~graph_tool.inference.MultilevelMCMCState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to appropriate state constructor (e.g.
        :class:`~graph_tool.inference.BlockState`)
    multilevel_mcmc_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.
    refine : ``bool`` (optional, default: ``True``)
        If ``True``, a refinement loop will be run as a final step, controlled
        by the relative convergence criterion ``epsilon``.
    epsilon : ``float`` (optional, default: ``0.001``)
        If ``refine == True``, this value determines the relative convergence
        criterion of the refinement step.

    Returns
    -------
    min_state : type given by parameter ``state``
        State with minimum description length.

    Notes
    -----

    This function is a convenience wrapper around
    :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.

    See [peixoto-efficient-2014]_ for details on the algorithm.

    This algorithm has a complexity of :math:`O(V \ln^2 V)`, where :math:`V` is
    the number of nodes in the network.

    @parallel@

    Examples
    --------

    .. testsetup:: mdl

       gt.seed_rng(43)
       np.random.seed(43)

    .. doctest:: mdl

       >>> g = gt.collection.data["polbooks"]
       >>> state = gt.minimize_blockmodel_dl(g)
       >>> state.draw(pos=g.vp["pos"], vertex_shape=state.get_blocks(),
       ...            output="polbooks_blocks_mdl.svg")
       <...>

    .. figure:: polbooks_blocks_mdl.*
       :align: center

       Block partition of a political books network, which minimizes the
       description length of the network according to the degree-corrected
       stochastic blockmodel.


    .. testsetup:: mdl_overlap

       gt.seed_rng(42)
       np.random.seed(42)

    .. doctest:: mdl_overlap

       >>> g = gt.collection.data["polbooks"]
       >>> state = gt.minimize_blockmodel_dl(g, state=gt.OverlapBlockState)
       >>> state.draw(pos=g.vp["pos"], output="polbooks_overlap_blocks_mdl.svg")
       <...>

    .. figure:: polbooks_overlap_blocks_mdl.*
       :align: center

       Overlapping partition of a political books network, which minimizes the
       description length of the network according to the overlapping
       degree-corrected stochastic blockmodel.

    .. doctest:: mdl_pp

       >>> g = gt.collection.data["celegansneural"]
       >>> state = gt.minimize_blockmodel_dl(g, state=gt.PPBlockState)
       >>> state.draw(output="celegans_mdl_pp.pdf")
       <...>

    .. testcleanup:: mdl_pp

       conv_png("celegans_mdl_pp.pdf")

    .. figure:: celegans_mdl_pp.png
       :align: center
       :width: 60%

       Assortative partition of the *C. elegans* neural network, which minimizes
       the description length of the network according to the degree-corrected
       planted-partition blockmodel.

    References
    ----------
    .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and greedy
       heuristic for the inference of stochastic block models", Phys. Rev. E 89,
       012804 (2014), :doi:`10.1103/PhysRevE.89.012804`, :arxiv:`1310.4378`.

    """

    state = state(g, **state_args)

    args = dict(niter=1, beta=numpy.inf, force_accept=True)
    args.update(multilevel_mcmc_args)

    state.multilevel_mcmc_sweep(**args)

    if refine:
        refine_loop(state, args, epsilon)

    return state

@_parallel
def minimize_nested_blockmodel_dl(g, state=NestedBlockState,
                                  base_state=BlockState, state_args={},
                                  base_state_args={}, multilevel_mcmc_args={},
                                  simple_init=False, simple_init_reverse=False,
                                  base_factor=10.,  B_min_base=1000, top_factor=2.,
                                  strict=False, refine=True, epsilon=0.001):
    r"""Fit the nested stochastic block model, by minimizing its description length
    using an agglomerative heuristic.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    state : nested SBM state class (optional, default: :class:`~graph_tool.inference.NestedBlockState`)
        Type of nested model that will be used.
    base_state : base SBM state class (optional, default: :class:`~graph_tool.inference.BlockState`)
        Type of model that will be used at the bottom of the hierarchy.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to appropriate state constructor (e.g.
        :class:`~graph_tool.inference.NestedBlockState`)
    base_state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to appropriate base state constructor (e.g.
        :class:`~graph_tool.inference.BlockState`)
    multilevel_mcmc_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to :meth:`~graph_tool.inference.MultilevelMCMCState.multilevel_mcmc_sweep`.
    simple_init : ``bool`` (optional, default: ``False``)
        If ``True``, a simple initialization scheme will be used, where the
        hierarchy shape will be upper bounded by an initial guess based on the
        number of nodes and the parameters ``base_factor``, ``top_factor``, and
        ``B_min_base``. This is meant to be used with ``refine == True`` for an
        alternative to the more expensive multilevel minimization algorithm.
    simple_init_reverse : ``bool`` (optional, default: ``False``)
        If ``True``, the initialization scheme will run from top to bottom in
        the hierarchy, otherwise from bottom to top.
    base_factor : ``float`` (optional, default: ``10.``)
        If ``simple_init == True``, this will determine the minimum number of
        groups at the lowest level of the hierarchy as ``N/base_factor``, where
        ``N`` is the number of vertices at the base level.
    B_min_base : ``int`` (optional, default: ``1000``)
        If ``simple_init == True``, this will determine the smallest minimum number
        of groups at the lowest level during initialization, regardless of the
        value of ``base_factor``.
    top_factor : ``float`` (optional, default: ``2.``)
        If ``simple_init == True``, this will determine the minimum number of
        groups at the upper levels of the hierarchy as ``N/top_factor``, where
        ``N`` is the number of vertices at a particular level.
    strict : ``bool`` (optional, default: ``False``)
        If ``simple_init == True``, the initial phase will set the number of
        groups exactly, otherwise they will only be lower bounds.
    refine : ``bool`` (optional, default: ``True``)
        If ``True``, a refinement loop will be run as a final step, controlled
        by the relative convergence criterion ``epsilon``.
    epsilon : ``float`` (optional, default: ``0.001``)
        If ``refine == True``, this value determines the relative convergence
        criterion of the refinement step.

    Returns
    -------
    min_state : type given by parameter ``state``
        State with minimum description length.

    Notes
    -----
    This function is a convenience wrapper around
    :meth:`~graph_tool.inference.NestedBlockState.multilevel_mcmc_sweep`.

    See [peixoto-hierarchical-2014]_ for details on the algorithm.

    This algorithm has a complexity of :math:`O(E \ln^2 V)`, where :math:`E` and
    :math:`V` are the number of edges and nodes in the network, respectively.

    @parallel@

    Examples
    --------
    .. testsetup:: nested_mdl

       gt.seed_rng(43)
       np.random.seed(43)

    .. doctest:: nested_mdl

       >>> g = gt.collection.data["power"]
       >>> state = gt.minimize_nested_blockmodel_dl(g)
       >>> state.draw(output="power_nested_mdl.pdf")
       (...)

    .. testcleanup:: nested_mdl

       conv_png("power_nested_mdl.pdf")

    .. figure:: power_nested_mdl.png
       :align: center
       :width: 60%

       Hierarchical Block partition of a power-grid network, which minimizes
       the description length of the network according to the nested
       (degree-corrected) stochastic blockmodel.


    .. doctest:: nested_mdl_overlap

       >>> g = gt.collection.data["celegansneural"]
       >>> state = gt.minimize_nested_blockmodel_dl(g, base_state=gt.OverlapBlockState)
       >>> state.draw(output="celegans_nested_mdl_overlap.pdf")
       (...)

    .. testcleanup:: nested_mdl_overlap

       conv_png("celegans_nested_mdl_overlap.pdf")

    .. figure:: celegans_nested_mdl_overlap.png
       :align: center
       :width: 60%

       Overlapping block partition of the *C. elegans* neural network, which
       minimizes the description length of the network according to the nested
       overlapping degree-corrected stochastic blockmodel.

    References
    ----------
    .. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
       structures and high-resolution model selection in large networks",
       Phys. Rev. X 4, 011047 (2014), :doi:`10.1103/PhysRevX.4.011047`,
       :arxiv:`1310.4377`.

    """

    state = state(g, base_state=base_state,
                  base_state_args=base_state_args, **state_args)

    args = dict(niter=1, beta=numpy.inf)
    args.update(multilevel_mcmc_args)

    if simple_init:
        levels = enumerate(state.levels)
        if simple_init_reverse:
            levels = reversed(list(levels))
        for l, s in levels:
            if l == 0:
                B = int(max(s.get_N() / base_factor, B_min_base))
            else:
                B = int(s.get_N() / top_factor)
            # at least one group for each constraint label (e.g. both sides
            # of a bipartite network, with pclabel)
            B = min(max(B, s._get_B_min_clabel()), s.get_N())
            largs = dict(args)
            if l > 0:
                # as in NestedBlockState.multilevel_mcmc_sweep(), the vertex
                # subset refers only to the bottom level
                largs.pop("vertices", None)
            s.multilevel_mcmc_sweep(**dict(largs,
                                           B_min=B,
                                           B_max=B if strict else s.get_N(),
                                           force_accept=True,
                                           bisection=False))
    else:
        l = 0
        while l >= 0:
            ret = state.multilevel_mcmc_sweep(ls=[l], **args)
            if args.get("verbose", False):
                print(l, ret, state)
            if abs(ret[0]) < 1e-8:
                l -= 1
            else:
                l = min((l + 1, len(state.levels) - 1))

    if refine:
        refine_loop(state, args, epsilon)

    return state

def refine_loop(state, mcmc_args, epsilon):
    eargs = mcmc_args.get("entropy_args", {})
    delta = epsilon + 1
    while delta > epsilon:
        delta = state.multilevel_mcmc_sweep(**dict(mcmc_args, refine=True,
                                                   force_accept=False))[0]
        S = state.entropy(**eargs)
        delta = abs(delta / S if S != 0 else delta)
