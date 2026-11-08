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

"""
``graph_tool.draw``
-------------------

This module contains functions for graph drawing and layout.

Layout algorithms
=================

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   sfdp_layout
   hfd_layout
   fruchterman_reingold_layout
   arf_layout
   radial_tree_layout
   planar_layout
   random_layout
   HFDState

Graph drawing
=============

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   graph_draw
   draw_hierarchy
   prop_to_size
   get_hierarchy_control_points


Low-level graph drawing
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   cairo_draw
   interactive_window

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   GraphWidget
   GraphWindow

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   GraphArtist

"""

from .. import Graph, GraphView, _check_prop_vector, _check_prop_scalar, \
    group_vector_property, ungroup_vector_property, infect_vertex_property, \
    _prop, _get_rng, VertexPropertyMap, EdgePropertyMap, perfect_prop_hash
from .. topology import max_cardinality_matching, max_independent_vertex_set, \
    label_components, shortest_distance, make_maximal_planar, is_planar
from .. generation import predecessor_tree, condensation_graph
from .. inference import minimize_blockmodel_dl, BlockState, ModularityState, \
    PickleState
from .. inference.util import nested_contiguous_map
from .. decorators import _parallel, _copy_params_doc
import numpy.random
from numpy import sqrt
from collections.abc import Iterable
import os.path
import time
import inspect
import types

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_layout")


__all__ = ["graph_draw", "fruchterman_reingold_layout",
           "arf_layout", "sfdp_layout", "hfd_layout", "planar_layout",
           "random_layout",
           "radial_tree_layout", "HFDState", "cairo_draw", "prop_to_size",
           "get_hierarchy_control_points", "default_cm", "default_cm_bin",
           "default_clrs"]


def random_layout(g, shape=None, pos=None, dim=2):
    r"""Performs a random layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    shape : tuple or list (optional, default: ``None``)
        Rectangular shape of the bounding area. The size of this parameter must
        match `dim`, and each element can be either a pair specifying a range,
        or a single value specifying a range starting from zero. If ``None`` is
        passed, a square of linear size :math:`\sqrt{N}` is used.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.
    dim : int (optional, default: ``2``)
        Number of coordinates per vertex.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V)`.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.random_graph(100, lambda: (3, 3))
    >>> shape = [[50, 100], [1, 2], 4]
    >>> pos = gt.random_layout(g, shape=shape, dim=3)
    >>> pos[g.vertex(0)].a
    array([68.72700594,  1.03142919,  2.56812658])

    """

    if pos is None:
        pos = g.new_vertex_property("vector<double>")
        _check_prop_vector(pos, name="pos")

    pos = ungroup_vector_property(pos, list(range(0, dim)))

    if shape is None:
        shape = [sqrt(g.num_vertices())] * dim

    for i in range(dim):
        if hasattr(shape[i], "__len__"):
            if len(shape[i]) != 2:
                raise ValueError("The elements of 'shape' must have size 2.")
            r = [min(shape[i]), max(shape[i])]
        else:
            r = [min(shape[i], 0), max(shape[i], 0)]
        d = r[1] - r[0]

        # deal with filtering
        p = pos[i].fa
        pos[i].fa = numpy.random.random(len(p)) * d + r[0]

    pos = group_vector_property(pos)
    return pos


@_parallel
def planar_layout(g, pos=None):
    r"""Performs a canonical layout of a planar graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Planar graph to be used.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V + E)`.

    @parallel@

    Examples
    --------
    >>> g = gt.lattice([10, 10])
    >>> pos = gt.planar_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="lattice-planar.pdf")
    <...>

    .. testcleanup::

       conv_png("lattice-planar.pdf")


    .. figure:: lattice-planar.png
        :align: center
        :width: 60%

        Straight-line drawing of planar graph (a 2D square lattice).

    References
    ----------
    .. [straight-line-boost] http://www.boost.org/doc/libs/release/libs/graph/doc/straight_line_drawing.html
    .. [chrobak-linear-1995] M. Chrobak, T. Payne, "A Linear-time Algorithm for
       Drawing a Planar Graph on the Grid", Information Processing Letters 54:
       241-246, (1995), :doi:`10.1016/0020-0190(95)00020-D`
    """

    if g.num_vertices() < 3:
        raise ValueError("Graph must have at least 3 vertices.")
    if not is_planar(g):
        raise ValueError("Graph is not planar.")
    u = Graph(GraphView(g, directed=False, skip_properties=True))
    make_maximal_planar(u)
    embed = is_planar(u, embedding=True)[1]
    if pos is None:
        pos = u.new_vp("vector<double>")
    else:
        pos = u.own_property(pos)
    libgraph_tool_layout.planar_layout(u._Graph__graph,
                                       _prop("v", u, embed),
                                       _prop("v", u, pos))
    pos = g.own_property(pos)
    return pos


def fruchterman_reingold_layout(g, weight=None, a=None, r=1., scale=None,
                                circular=False, grid=True, t_range=None,
                                n_iter=100, pos=None):
    r"""Calculate the Fruchterman-Reingold spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    a : float (optional, default: :math:`V`)
        Attracting force between adjacent vertices.
    r : float (optional, default: 1.0)
        Repulsive force between vertices.
    scale : float (optional, default: :math:`\sqrt{V}`)
        Total scale of the layout (either square side or radius).
    circular : bool (optional, default: ``False``)
        If ``True``, the layout will have a circular shape. Otherwise the shape
        will be a square.
    grid : bool (optional, default: ``True``)
        If ``True``, the repulsive forces will only act on vertices which are on
        the same site on a grid. Otherwise they will act on all vertex pairs.
    t_range : tuple of floats (optional, default: ``(scale / 10, scale / 1000)``)
        Temperature range used in annealing. The temperature limits the
        displacement at each iteration.
    n_iter : int (optional, default: ``100``)
        Total number of iterations.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored. If
        provided, this will also be used as the initial position of the
        vertices.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [fruchterman-reingold]_, and has
    complexity :math:`O(\text{n-iter}\times V^2)` if `grid=False` or
    :math:`O(\text{n-iter}\times (V + E))` otherwise.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(300)
    >>> pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-fr.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-fr.pdf")

    .. figure:: graph-draw-fr.png
       :align: center
       :width: 60%

       Fruchterman-Reingold layout of a Price network.

    References
    ----------
    .. [fruchterman-reingold] Fruchterman, Thomas M. J.; Reingold, Edward M.
       "Graph Drawing by Force-Directed Placement". Software - Practice & Experience
       (Wiley) 21 (11): 1129-1164. (1991) :doi:`10.1002/spe.4380211102`
    """

    if pos is None:
        pos = random_layout(g, dim=2)
    _check_prop_vector(pos, name="pos", floating=True)

    if a is None:
        a = float(g.num_vertices())

    if scale is None:
        scale = sqrt(g.num_vertices())

    if t_range is None:
        t_range = (scale / 10, scale / 1000)

    ug = GraphView(g, directed=False)
    libgraph_tool_layout.fruchterman_reingold_layout(ug._Graph__graph,
                                                     _prop("v", g, pos),
                                                     _prop("e", g, weight),
                                                     a, r, not circular, scale,
                                                     grid, t_range[0],
                                                     t_range[1], n_iter)
    return pos


@_parallel
def arf_layout(g, weight=None, d=.5, a=10, dt=0.001, epsilon=1e-6,
               max_iter=1000, pos=None, dim=2):
    r"""Calculate the ARF spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    d : float (optional, default: ``0.5``)
        Opposing force between vertices.
    a : float (optional, default: ``10``)
        Attracting force between adjacent vertices.
    dt : float (optional, default: ``0.001``)
        Iteration step size.
    epsilon : float (optional, default: ``1e-6``)
        Convergence criterion.
    max_iter : int (optional, default: ``1000``)
        Maximum number of iterations. If this value is ``0``, it runs until
        convergence.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vector vertex property maps where the coordinates should be stored.
    dim : int (optional, default: ``2``)
        Number of coordinates per vertex.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [geipel-self-organization-2007]_, and has
    complexity :math:`O(V^2)`.

    @parallel@

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(300)
    >>> pos = gt.arf_layout(g, a=5)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-arf.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-arf.pdf")

    .. figure:: graph-draw-arf.png
       :align: center
       :width: 60%

       ARF layout of a Price network.

    References
    ----------
    .. [geipel-self-organization-2007] Markus M. Geipel, "Self-Organization
       applied to Dynamic Network Layout", International Journal of Modern
       Physics C vol. 18, no. 10 (2007), pp. 1537-1549,
       :doi:`10.1142/S0129183107011558`, :arxiv:`0704.1748v5`
    .. _arf: http://www.sg.ethz.ch/research/graphlayout
    """

    if pos is None:
        pos = random_layout(g, dim=dim, shape=[[0, 1]] * dim)
    _check_prop_vector(pos, name="pos", floating=True)

    ug = GraphView(g, directed=False)
    libgraph_tool_layout.arf_layout(ug._Graph__graph, _prop("v", g, pos),
                                    _prop("e", g, weight), d, a, dt, max_iter,
                                    epsilon, dim)
    return pos


def _coarse_graph(g, vweight, eweight, kind="matching", groups=None):
    mivs = None
    if groups is None:
        if kind == "mivs":
            mivs = max_independent_vertex_set(g, high_deg=True)
            u = GraphView(g, vfilt=mivs, directed=False)
            c = label_components(u)[0]
            c.fa += 1
            u = GraphView(g, directed=False)
            infect_vertex_property(u, c,
                                   list(range(1, c.fa.max() + 1)))
            c = g.own_property(c)
        elif kind == "ec":
            m = max_cardinality_matching(GraphView(g, directed=False),
                                         heuristic=True, weight=eweight,
                                         minimize=False, edges=True)
            u = GraphView(g, efilt=m, directed=False)
            c = label_components(u)[0]
            c = g.own_property(c)
            u = GraphView(g, directed=False)
        elif kind == "modularity":
            B = max(g.num_vertices() // 2, 1)
            state = minimize_blockmodel_dl(GraphView(g, directed=False),
                                           state=ModularityState,
                                           state_args=dict(eweight=eweight),
                                           multilevel_mcmc_args=dict(B_min=B,
                                                                     B_max=B))
            c = state.b.copy()
        elif kind in ["sbm", "dcsbm"]:
            B = max(g.num_vertices() // 2, 1)
            w = eweight.copy("int") if eweight is not None else None
            state = minimize_blockmodel_dl(GraphView(g, directed=False),
                                           state=BlockState,
                                           state_args=dict(eweight=w,
                                                           deg_corr=kind == "dcsbm"),
                                           multilevel_mcmc_args=dict(B_min=B,
                                                                     B_max=B))
            c = state.b.copy()
        else:
            raise ValueError("invalid coarsening method:", kind)
    else:
        c = g.new_vp("int", vals=groups)
    c = perfect_prop_hash([c])[0]
    cg, vcount, ecount = condensation_graph(g, c, vweight, eweight)
    return cg, cg.vertex_index.copy("int32_t"), vcount, ecount, c, mivs


def _propagate_pos(g, cg, c, cc, cpos, delta, mivs):
    pos = g.new_vertex_property(cpos.value_type())

    if mivs is not None:
        g = GraphView(g, vfilt=mivs)
    libgraph_tool_layout.propagate_pos(g._Graph__graph,
                                       cg._Graph__graph,
                                       _prop("v", g, c),
                                       _prop("v", cg, cc),
                                       _prop("v", g, pos),
                                       _prop("v", cg, cpos),
                                       delta if mivs is None else 0,
                                       _get_rng())

    if mivs is not None:
        g = g.base
        u = GraphView(g, directed=False)
        libgraph_tool_layout.propagate_pos_mivs(u._Graph__graph,
                                                _prop("v", u, mivs),
                                                _prop("v", u, pos),
                                                delta, _get_rng())
    return pos


def _avg_edge_distance(g, pos):
    libgraph_tool_layout.sanitize_pos(g._Graph__graph, _prop("v", g, pos))
    ad = libgraph_tool_layout.avg_dist(g._Graph__graph, _prop("v", g, pos))
    if numpy.isnan(ad) or ad == 0:
        ad = 1.
    return ad


def coarse_graphs(g, method="hybrid", mivs_thres=0.9, ec_thres=0.75,
                  weighted_coarse=False, eweight=None, vweight=None,
                  groups=None, gamma=None, verbose=False):
    cg = [[g, None, None, None, None, None]]
    if weighted_coarse:
        cg[-1][2], cg[-1][3] = vweight, eweight
    hybrid = False
    if method == "hybrid":
        method = "ec"
        hybrid = True
    if verbose:
        print(f"Coarse level (none): 0, N={g.num_vertices()}")
    while True:
        if groups is None or len(groups) < len(cg):
            b = None
        else:
            b = groups[len(cg)-1]
        u = _coarse_graph(cg[-1][0], cg[-1][2], cg[-1][3], method, b)
        thres = mivs_thres if method == "mivs" else ec_thres
        if u[0].num_vertices() >= thres * cg[-1][0].num_vertices():
            if hybrid and method == "ec":
                method = "mivs"
            else:
                break
        if u[0].num_vertices() <= 2:
            break
        cg.append(u)
        if verbose:
            print(f"Coarse level ({method}): {len(cg)}, N={u[0].num_vertices()}")
    cg.reverse()
    Ks = []
    pos = random_layout(cg[0][0], dim=2)
    for i in range(len(cg)):
        if i == 0:
            u = cg[i][0]
            K = _avg_edge_distance(u, pos)
            if K == 0:
                K = 1.
            Ks.append(K)
            continue
        if weighted_coarse:
            r = 1.
        else:
            r = 0.75
        Ks.append(Ks[-1] * r)

    if gamma is None:
        gamma = [0] * len(cg)
    else:
        if not isinstance(gamma, Iterable):
            gamma = [gamma]
        gamma = list(gamma) + [0] * (len(cg) - len(gamma))

    for i in range(len(cg)):
        u, cc, vcount, ecount, c, mivs = cg[i]
        if groups is None or len(cg) - i > len(groups):
            b = None
        else:
            b = groups[len(cg) - 1 - i:]
        yield u, pos, Ks[i], vcount, ecount, b, gamma[len(cg)-i-1:]

        if verbose:
            print("avg edge distance:", _avg_edge_distance(u, pos))

        if i < len(cg) - 1:
            if verbose:
                print("propagating...", end=' ')
                print(mivs.a.sum() if mivs is not None else "")
            pos = _propagate_pos(cg[i + 1][0], u, c, cc, pos,
                                 #Ks[i] / 1000.,
                                 1e-6,
                                 mivs)


@_parallel
def sfdp_layout(g, vweight=None, eweight=None, pin=None, C=0.2, K=None, p=2.,
                theta=0.6, max_level=15, r=1., kc=10, groups=None, gamma=.1,
                mu=2., kappa=1., rmap=None, R=1, init_step=None,
                cooling_step=0.95, adaptive_cooling=True, epsilon=1e-2,
                max_iter=0, pos=None, multilevel=None, coarse_method="hybrid",
                mivs_thres=0.9, ec_thres=0.75, weighted_coarse=False,
                verbose=False):
    r"""Obtain the SFDP spring-block layout of the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        A vertex property map with the respective weights.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        An edge property map with the respective weights.
    pin : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        A vertex property map with boolean values, which, if given,
        specify the vertices which will not have their positions modified.
    C : float (optional, default: ``0.2``)
        Relative strength of repulsive forces.
    K : float (optional, default: ``None``)
        Optimal edge length. If not provided, it will be taken to be the average
        edge distance in the initial layout.
    p : float (optional, default: ``2``)
        Repulsive force exponent.
    theta : float (optional, default: ``0.6``)
        Quadtree opening parameter, a.k.a. Barnes-Hut opening criterion.
    max_level : int (optional, default: ``15``)
        Maximum quadtree level.
    r : float (optional, default: ``1.``)
        Strength of attractive force between connected components.
    kc : int (optional, default: ``10``)
        Number of connected components to subsample.
    groups : :class:`~graph_tool.VertexPropertyMap` or list of list of integers (optional, default: ``None``)
        A vertex property map with group assignments. Vertices belonging to the
        same group will be put close together. Optionally, this can take a
        hierarchical partition, represented by a list of lists, where the list
        above corresponds to the partition of the list below.
    gamma : float or list of floats (optional, default: ``.1``)
        Strength of the attractive force between nodes of the same group to
        their center of mass. In case of a hierarchical partition, this should
        correspond to a list of floats, containing the strengths for each
        hierarchical level. This option has no effect if ``groups`` is ``None``.
    rmap : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex rank used to order the vertices preferentially in the
        :math:`y` direction.
    R : float (optional, default: ``1.0``)
        Strength of the rank ordering in the :math:`y` direction.
    init_step : float (optional, default: ``None``)
        Initial update step. If not provided, it will be chosen automatically.
    cooling_step : float (optional, default: ``0.95``)
        Cooling update step.
    adaptive_cooling : bool (optional, default: ``True``)
        Use an adaptive cooling scheme.
    epsilon : float (optional, default: ``0.01``)
        Relative convergence criterion.
    max_iter : int (optional, default: ``0``)
        Maximum number of iterations. If this value is ``0``, it runs until
        convergence.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial vertex layout. If not provided, it will be randomly chosen.
    multilevel : bool (optional, default: ``None``)
        Use a multilevel layout algorithm. If ``None`` is given, it will be
        activated based on the size of the graph.
    coarse_method : str (optional, default: ``"hybrid"``)
        Coarsening method used if ``multilevel == True``. Allowed methods are
        ``"hybrid"``, ``"mivs"``, ``"ec"``, ``"modularity"``, ``"sbm"`` and
        ``"dcsbm"``.
    mivs_thres : float (optional, default: ``0.9``)
        If the relative size of the MIVS coarse graph is above this value, the
        coarsening stops.
    ec_thres : float (optional, default: ``0.75``)
        If the relative size of the EC coarse graph is above this value, the
        coarsening stops.
    weighted_coarse : bool (optional, default: ``False``)
        Use weighted coarse graphs.
    verbose : bool (optional, default: ``False``)
        Provide verbose information.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm is defined in [hu-multilevel-2005]_, and has
    complexity :math:`O(V\log V)`.

    @parallel@

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(3000)
    >>> pos = gt.sfdp_layout(g)
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-sfdp.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-sfdp.pdf")

    .. figure:: graph-draw-sfdp.png
       :align: center
       :width: 60%

       SFDP layout of a Price network.

    References
    ----------
    .. [hu-multilevel-2005] Yifan Hu, "Efficient, High-Quality Force-Directed
       Graph Drawing", Mathematica Journal, vol. 10, Issue 1, pp. 37-71, (2005)
       http://www.mathematica-journal.com/issue/v10i1/graph_draw.html

    """

    if pos is None:
        pos = random_layout(g, dim=2)
    _check_prop_vector(pos, name="pos", floating=True)

    if isinstance(groups, VertexPropertyMap):
        groups = [numpy.asarray(groups.a.copy(), dtype="int32")]
    elif groups is not None:
        if isinstance(groups[0], VertexPropertyMap):
            groups = [numpy.asarray(groups[0].a.copy(), dtype="int32")] + list(groups[1:])
        groups = nested_contiguous_map(groups)

    if groups is None:
        gamma = [0]
    else:
        if not isinstance(gamma, Iterable):
            gamma = [gamma]
        if len(gamma) < len(groups):
            gamma = list(gamma) + [0] * (len(groups) - len(gamma))

    g_ = g
    g = GraphView(g, directed=False)

    if pin is not None:
        if pin.value_type() != "bool":
            raise ValueError("'pin' property must be of type 'bool'.")
    else:
        pin = g.new_vertex_property("bool")

    if K is None:
        K = _avg_edge_distance(g, pos)

    mu *= K

    if rmap is None:
        rmap = g.new_vp("double")
        R = 0
    elif rmap.value_type() != "double":
        rmap = rmap.copy("double")

    if init_step is None:
        init_step = 2 * max(_avg_edge_distance(g, pos), K)

    if multilevel is None:
        multilevel = g.num_vertices() > 1000

    if multilevel:
        if eweight is not None or vweight is not None:
            weighted_coarse = True
        cgs = coarse_graphs(g, method=coarse_method,
                            mivs_thres=mivs_thres,
                            ec_thres=ec_thres,
                            weighted_coarse=weighted_coarse,
                            eweight=eweight,
                            vweight=vweight,
                            groups=groups,
                            gamma=gamma,
                            verbose=verbose)
        for count, (u, pos, K_, vcount, ecount, groups, gamma) in enumerate(cgs):
            if verbose:
                print(f"Positioning level: {count}, {u.num_vertices()}", end=" ")
                print(f"with K={K_}, gamma={gamma} ...")
            pos = sfdp_layout(u, pos=pos,
                              vweight=vcount if weighted_coarse else None,
                              eweight=ecount if weighted_coarse else None,
                              groups=groups,
                              C=C, K=K_, p=p,
                              theta=theta,
                              gamma=gamma,
                              epsilon=epsilon,
                              max_iter=max_iter,
                              cooling_step=cooling_step,
                              adaptive_cooling=False,
                              # init_step=max(2 * K,
                              #               _avg_edge_distance(u, pos)),
                              multilevel=False,
                              verbose=False)
        pos = g_.own_property(pos)
        return pos

    if g.num_vertices() <= 1:
        return pos
    if g.num_vertices() == 2:
        vs = [g.vertex(0, False), g.vertex(1, False)]
        pos[vs[0]] = [0, 0]
        pos[vs[1]] = [1, 1]
        return pos
    if g.num_vertices() <= 50:
        max_level = 0

    if groups is None:
        groups = [numpy.zeros(g.num_vertices(True), dtype="int32")]

    c = label_components(g)[0]

    libgraph_tool_layout.sanitize_pos(g._Graph__graph, _prop("v", g, pos))
    libgraph_tool_layout.sfdp_layout(g._Graph__graph, _prop("v", g, pos),
                                     _prop("v", g, vweight),
                                     _prop("e", g, eweight),
                                     _prop("v", g, pin),
                                     (C, K, p, gamma, groups, r, kc,
                                      _prop("v", g, c), R, _prop("v", g, rmap)),
                                     theta, init_step, cooling_step, max_level,
                                     epsilon, max_iter, not adaptive_cooling,
                                     verbose, _get_rng())
    pos = g_.own_property(pos)
    return pos


def _hfd_prop(x, g, key="v", vtype="double", val=None):
    """A property map of type vtype: x if it already has it, a converted copy
    of it otherwise, or a new one with value val if x is None."""
    if x is None:
        return g.new_property(key, vtype, val=val)
    return x if x.value_type() == vtype else x.copy(vtype)


def _hfd_vprop(x, g):
    """A scalar or per-vertex value as a ``double`` vertex property map."""
    if isinstance(x, VertexPropertyMap):
        return x.copy("double")
    return g.new_vp("double", val=x)


def _hfd_level_vprop(x, g, L):
    """A per-level (and possibly per-vertex) value as a ``vector<double>``
    vertex property map with L values for every vertex."""
    if not isinstance(x, VertexPropertyMap):
        return g.new_vp("vector<double>", val=HFDState._per_level(x, L))
    if x.value_type().startswith("vector"):
        return x.copy("vector<double>")
    if L == 0:
        return g.new_vp("vector<double>")
    return group_vector_property([x] * L, value_type="double")


def _hfd_halving(x, L):
    """Per-level values for L levels: a single value x is halved at every
    level, and a list of values is used as given."""
    if isinstance(x, Iterable):
        return HFDState._per_level(x, L)
    return [float(x) / 2 ** l for l in range(L)]


def _hfd_level_arrays(x, bs, val):
    """Per-group values for every level of the partition bs, as a list of
    arrays indexed by the group labels, created with value val if x is None,
    and copied otherwise. A single array is taken as a list of one level."""
    if x is None:
        return [numpy.full(b.max() + 1 if len(b) > 0 else 0, val,
                           dtype="double") for b in bs]
    if isinstance(x, numpy.ndarray) and x.ndim == 1:
        x = [x]
    return [numpy.array(y, dtype="double") for y in x]


def _hfd_order_labels(bs, args):
    """Compact the labels of every level of the nested partition bs, so that
    they are contiguous, without empty groups, keeping their order, and move
    the per-group parameters in args accordingly. Returns the new partition
    and arguments."""
    nbs, used = [], []
    u = None                # old labels kept at the level below
    for l, b in enumerate(bs):
        if u is not None:
            if len(u) > 0 and u[-1] >= len(b):
                raise ValueError(f"partition level {l} has {len(b)} " +
                                 f"entries, but group {u[-1]} is referenced")
            b = b[u]
        nb = b.copy()
        k = b >= 0
        u, nb[k] = numpy.unique(b[k], return_inverse=True)
        nbs.append(nb)
        used.append(u)

    def take(x, u, val):
        y = numpy.full(len(u), val, dtype="double")
        k = u < len(x)
        y[k] = x[u[k]]
        return y

    args = dict(args, b=nbs)
    for k, val in [("group_conf_R", 0), ("group_grav_fac", 1)]:
        if args[k] is not None:
            xs = _hfd_level_arrays(args[k], bs, val)
            args[k] = [take(x, used[l], val) if l < len(used) else x
                       for l, x in enumerate(xs)]
    return nbs, args


def _hfd_bs(b):
    """Normalize a single or hierarchical partition into a list of arrays."""
    if b is None:
        b = []
    elif isinstance(b, (VertexPropertyMap, numpy.ndarray)):
        b = [b]
    return [numpy.array(x.a if isinstance(x, VertexPropertyMap) else x,
                        dtype="int64") for x in b]


@_parallel
class HFDState(PickleState):
    r"""State of a hierarchical force-directed (HFD) layout of a graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be laid out.
    pos : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial positions, as a ``vector<double>`` property map, which is
        updated in place. If not provided, they are chosen at random. They
        should be distinct, since the repulsion vanishes between coinciding
        vertices.
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex weights. Vertices with zero weight are ignored, and keep their
        positions. If not provided, all weights are ``1``.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge weights. If not provided, all weights are ``1``.
    rweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Factor of the vertex weights in the repulsion and gravity (e.g. the
        relative sizes of the groups represented by the vertices of a coarse
        graph). If not provided, all factors are ``1``.
    K : float (optional, default: ``1``)
        Strength of the repulsion, which sets the length scale of the layout.
    rep_exp : float (optional, default: ``1``)
        Exponent :math:`p` of the repulsion, whose force decays as
        :math:`d^{-(p+1)}` with the distance :math:`d` (logarithmic for
        :math:`p = 0`).
    att_exp : float (optional, default: ``2``)
        Exponent :math:`a` of the attraction along the edges, whose force
        grows as :math:`d^{a-1}` (logarithmic for :math:`a = 0`).
    grav : float (optional, default: ``1e-4``)
        Strength of the gravity toward the center of mass, which keeps
        disconnected parts of the layout together.
    theta : float (optional, default: ``0.6``)
        Opening criterion of the Barnes-Hut approximation. Smaller values are
        more accurate and slower.
    exact_thres : int (optional, default: ``500``)
        Graphs with fewer vertices are computed exactly, in time
        :math:`O(V^2)`, instead of with the Barnes-Hut approximation.
    b : :class:`~graph_tool.VertexPropertyMap`, :class:`numpy.ndarray` or list of these (optional, default: ``None``)
        Group labels, as a single partition, or a hierarchical one in the
        format of :meth:`~graph_tool.inference.NestedBlockState.get_bs`
        (``b[0][v]`` is the group of vertex ``v``, and ``b[l][r]`` the group
        at level ``l`` of group ``r`` of level ``l - 1``). The group forces
        and constraints act at every level.
    group_grav : float or list of floats (optional, default: ``0.1``)
        Strength of the logarithmic attraction between the members of every
        group, which dominates the repulsion beyond distances of about
        ``1 / group_grav``. A list gives one value per level. It is ignored
        by :func:`~graph_tool.draw.hfd_layout` with ``hierarchical == True``
        and ``grouped == True``, where the groups are kept together by
        springs instead.
    group_grav_fac : :class:`numpy.ndarray` or list of :class:`numpy.ndarray` (optional, default: ``None``)
        Factor of the group gravity of every group, indexed by its label, with
        one array per level.
    group_grav_offset : float (optional, default: ``0.01``)
        Offset :math:`c` of the distance in the group gravity,
        :math:`\log(d + c)`, which keeps it finite.
    group_rep : float or list of floats (optional, default: ``0``)
        Additional repulsion between vertices of different groups, relative to
        the general one. A single value is halved at every level above.
    group_rep_exp : float (optional, default: ``1``)
        Exponent of the group repulsion, as ``rep_exp``.
    group_conf_R : :class:`numpy.ndarray` or list of :class:`numpy.ndarray` (optional, default: ``None``)
        Radius of every group, indexed by its label, with one array per
        level. The members of a group with a positive radius are confined
        around its center of mass, within about this radius.
    group_conf_exp : float (optional, default: ``6``)
        Exponent of the confinement potential. The larger it is, the more
        sharply the members are kept inside the radius (``2`` gives a
        harmonic spring).
    group_spring : float or list of floats (optional, default: ``1``)
        If positive, the strength of the confinement of every group is chosen
        so that the edges leaving the group displace its members by
        ``group_conf_R / group_spring`` on average, and is updated with
        :meth:`~HFDState.update_group_springs`. Otherwise, it is chosen so
        that the repulsion fills the radius.
    group_spring_static : bool (optional, default: ``False``)
        If ``True``, the strengths of the springs are computed once, from the
        positions of the groups, and not updated.
    upper : :class:`~HFDState` (optional, default: ``None``)
        State of the level above, whose vertices are the groups of the lowest
        level of ``b``, with their total weights, and whose edge weights are
        those of the condensed graph divided by the weights of the endpoints.
        Its attraction and repulsion act on the centers of mass of the
        groups, and those of its own upper state on the levels above.
    barrier_r : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Radius of every vertex. Vertices of the same group are kept from
        overlapping by a barrier.
    barrier_sep : float (optional, default: ``0.2``)
        Range of the barrier, relative to the sum of the radii. It is
        disabled if zero.
    barrier_exact_thres : int (optional, default: ``100``)
        The overlapping vertices are found exhaustively in groups up to this
        size, and with a kd-tree otherwise.
    nmin_dist : float or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Minimum distance between any two vertices. If given, the minimum
        distance constraint is enabled.
    nmu : float (optional, default: ``100``)
        Penalty of the minimum distance constraint, relative to the force that
        pushes the vertices together.
    gmin_dist : float, list of floats or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Minimum distance between vertices of different groups, for every level
        (a list, or a ``vector<double>`` property map, gives one value per
        level). If given, the group distance constraint is enabled.
    gmu : float or list of floats (optional, default: ``10``)
        Penalty of the group distance constraint. It is disabled at the levels
        where it is zero.
    odist : float or list of floats (optional, default: ``None``)
        Minimum distance between the convex hulls of different groups. If
        given, the overlap constraint is enabled.
    omu : float or list of floats (optional, default: ``10``)
        Penalty of the overlap constraint. It is disabled at the levels where
        it is zero.
    opair_dist : float or list of floats (optional, default: ``None``)
        Maximum distance between the two members of groups of size two, which
        keeps other groups from passing between them. If given, the pair
        constraint is enabled.
    pmu : float or list of floats (optional, default: ``10``)
        Penalty of the pair constraint. It is disabled at the levels where it
        is zero.
    rank : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex ranks, ordered along the :math:`y` direction, with higher ranks
        at smaller :math:`y`. If given, the rank constraint is enabled.
    rmin_dist : float or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0``)
        Minimum separation along :math:`y` of every vertex from all the
        vertices of the next higher rank, so that the vertices of different
        ranks are strictly ordered.
    rmu : float (optional, default: ``1``)
        Penalty of the rank constraint.
    rcm_dist : float (optional, default: ``None``)
        Minimum separation along :math:`y` of the center of mass of the
        vertices of every rank from that of the next higher rank. If given
        (together with ``rank``), this constraint is enabled, independently
        of the one given by ``rmin_dist`` (which can be disabled with
        ``rmu=0``).
    rcmu : float (optional, default: ``1000``)
        Penalty of the constraint given by ``rcm_dist``.
    rcm_xdist : float (optional, default: ``None``)
        Minimum separation along :math:`x` between the centers of mass of
        the vertices of every pair of ranks (in any order), which compensates
        the squeezing of the layout along :math:`y` by ``rcm_dist``. If given
        (together with ``rank``), this constraint is enabled.
    rcmxu : float (optional, default: ``1000``)
        Penalty of the constraint given by ``rcm_xdist``.
    nlambda : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Lagrange multipliers of the minimum distance constraint, one per
        vertex, updated in place.
    glambda : float or list of floats (optional, default: ``0``)
        Lagrange multipliers of the group distance constraint, one per level.
    rclambda : float or list of floats (optional, default: ``0``)
        Lagrange multipliers of the constraint given by ``rcm_dist``, one per
        pair of adjacent ranks.
    rcxlambda : ``0`` or list of tuples (optional, default: ``0``)
        Nonzero Lagrange multipliers of the constraint given by ``rcm_xdist``,
        one per pair of ranks, as a list of ``(a, b, lambda)`` with the
        indices ``a < b`` of the ranks in increasing order.
    olambda : list of :class:`numpy.ndarray` (optional, default: ``None``)
        Lagrange multipliers of the overlap constraint, one per group and
        level, indexed by the group labels (which are copied, and updated in
        ``HFDState.olambda``).
    rlambda : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Lagrange multipliers of the rank constraint, one per vertex, updated
        in place.
    plambda : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Lagrange multipliers of the distance of the groups of size two, one
        per level, stored at the member with the smallest index, updated in
        place.
    group_spring_strength : list of :class:`numpy.ndarray` (optional, default: ``None``)
        Strengths of the springs, one array per level indexed by the group
        labels, used instead of computing them (e.g. to restore the static
        springs of a copied state).
    nmu_force : :class:`numpy.ndarray` (optional, default: ``None``)
        Magnitudes of the forces on the vertices that set the penalties of
        their minimum distances (see ``nmu``), as computed by
        :meth:`~HFDState.init_multipliers`.

    Notes
    -----
    The layout minimizes the energy

    .. math::

        U(\boldsymbol x) = \sum_{(u,v)\in E}w_{uv}w_uw_v A_a(d_{uv})
                         + K\sum_{u\neq v}m_um_v R_p(d_{uv})
                         + g\sum_u m_u|\boldsymbol x_u - \boldsymbol x_{\text{cm}}|^2
                         + K\sum_l g_l\sum_{u\neq v}\delta_{b^l_u,b^l_v}f^l_{b^l_u}m_um_v\log(d_{uv} + c)
                         + K\sum_l r_l\sum_{u\neq v}(1-\delta_{b^l_u,b^l_v})m_um_v R_q(d_{uv}),

    with the distances :math:`d_{uv}`, the attraction
    :math:`A_a(d) = (2/a)d^a` and repulsion :math:`R_p(d) = d^{-p}/p` (whose
    forces are :math:`2d^{a-1}` and :math:`d^{-(p+1)}`), the edge and vertex
    weights :math:`w_{uv}` and :math:`w_u`, the masses
    :math:`m_u = w_u\rho_u` with :math:`\rho_u` given by ``rweight``, the
    gravity :math:`g` (``grav``), the groups :math:`b^l_u`, the group
    gravity :math:`g_l f^l_r` (``group_grav`` and ``group_grav_fac``) and
    the group repulsion :math:`r_l` (``group_rep``), with the exponent
    :math:`q` given by ``group_rep_exp``. The repulsive terms are computed
    with the Barnes-Hut approximation.

    The energy may have additional terms:

    - the attraction and repulsion of the vertices of the ``upper`` state,
      placed at the centers of mass of the groups, which move the groups
      rigidly; in this case, :meth:`~HFDState.gradient_sweep` also moves
      every group as a whole;
    - the confinement of the members of every group around its center of
      mass, :math:`\gamma_r\sum_u w_u(|\boldsymbol x_u - \boldsymbol
      x_r|/R_r)^\alpha`, with :math:`R_r` given by ``group_conf_R`` and the
      strength :math:`\gamma_r` set by ``group_spring``; with springs, a
      packing term keeps also the groups with few or no edges together;
    - a barrier between the vertices with radii (``barrier_r``), which is not
      supported by :meth:`~HFDState.lbfgs_sweep`.

    The constraints (minimum distances, group overlap, ranks) are imposed
    with the augmented Lagrangian method: the energy is minimized with
    :meth:`~HFDState.gradient_sweep` or :meth:`~HFDState.lbfgs_sweep` for
    fixed Lagrange multipliers, which are then updated with
    :meth:`~HFDState.update_constraints`, until the violation is small
    enough. The minimum distance has one multiplier per vertex, so that
    every vertex is pushed away from its own neighbors as strongly as
    needed. The function :func:`~graph_tool.draw.hfd_layout` carries out
    this whole procedure.

    Each evaluation of the energy or its gradient has complexity
    :math:`O(V\log V + E)`, or :math:`O(V^2 + E)` if
    :math:`V <` ``exact_thres``.

    @parallel@

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.collection.data["football"]
    >>> b = g.vp.value
    >>> R = np.sqrt(np.bincount(b.a)) / 2     # radii of the conferences
    >>> state = gt.HFDState(g, b=b, group_conf_R=R, group_conf_exp=2,
    ...                     group_spring=10)
    >>> for i in range(20):
    ...     ret = state.lbfgs_sweep(max_iter=100)
    ...     R = state.update_constraints()
    ...     state.update_group_springs()
    >>> gt.graph_draw(g, pos=state.pos, vertex_fill_color=b,
    ...               output="graph-draw-hfd.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-hfd.pdf")

    .. figure:: graph-draw-hfd.png
       :align: center
       :width: 60%

       Force-directed layout of the football network, with the
       vertices of every conference kept together by a harmonic spring
       towards its center of mass.

    """

    def __init__(self, g, pos=None, vweight=None, eweight=None, rweight=None,
                 K=1, rep_exp=1, att_exp=2, grav=1e-4, theta=.6,
                 exact_thres=500, b=None, group_grav=.1, group_grav_fac=None,
                 group_grav_offset=.01, group_rep=0, group_rep_exp=1,
                 group_conf_R=None, group_conf_exp=6, group_spring=1,
                 group_spring_static=False, upper=None, barrier_r=None,
                 barrier_sep=.2, barrier_exact_thres=100, nmin_dist=None,
                 nmu=100, gmin_dist=None, gmu=10, odist=None, omu=10,
                 opair_dist=None, pmu=10, rank=None, rmin_dist=0, rmu=1,
                 rcm_dist=None, rcmu=1000, rcm_xdist=None, rcmxu=1000,
                 nlambda=None, glambda=0, rclambda=0, rcxlambda=0,
                 olambda=None, rlambda=None, plambda=None,
                 group_spring_strength=None, nmu_force=None):
        PickleState.__init__(self)
        # these live only in the C++ state
        self._ignore_state_args = ["glambda", "rclambda", "rcxlambda",
                                   "group_spring_strength", "nmu_force"]
        self.g = g
        self.theta = theta
        self.exact_thres = exact_thres
        if isinstance(K, Iterable):
            raise ValueError("K must be a single value (a list is only " +
                             "supported by the hierarchical layout)")
        self.K = K
        self.rep_exp = rep_exp
        self.att_exp = att_exp
        self.grav = grav
        self.vweight = _hfd_prop(vweight, g, val=1)
        self.rweight = _hfd_prop(rweight, g, val=1)
        self.eweight = _hfd_prop(eweight, g, key="e", val=1)
        if pos is None:
            pos = random_layout(g)
        _check_prop_vector(pos, name="pos", floating=True)
        if pos.value_type() != "vector<double>":
            pos = pos.copy("vector<double>")
        self.pos = pos

        # the constraints are enabled by their distances (or ranks), and are
        # otherwise disabled in the C++ state (see _cpp_args())
        self.nmin_dist = None if nmin_dist is None else \
            _hfd_vprop(nmin_dist, g)
        self.nmu = nmu

        self.b = b = _hfd_bs(b)
        # the C++ state checks the consistency of the levels
        L = len(b)

        self.gmin_dist = None if gmin_dist is None else \
            _hfd_level_vprop(gmin_dist, g, L)
        self.gmu = self._per_level(gmu, L)
        self.group_grav = self._per_level(group_grav, L)
        self.group_grav_fac = _hfd_level_arrays(group_grav_fac, b, 1)
        self.group_grav_offset = group_grav_offset
        self.group_conf_R = _hfd_level_arrays(group_conf_R, b, 0)
        self.group_conf_exp = group_conf_exp
        self.group_spring = self._per_level(group_spring, L)
        self.group_spring_static = group_spring_static
        self.barrier_r = _hfd_prop(barrier_r, g)
        self.barrier_sep = barrier_sep
        self.barrier_exact_thres = barrier_exact_thres
        self.group_rep = _hfd_halving(group_rep, L)
        self.group_rep_exp = group_rep_exp
        self.odist = None if odist is None else self._per_level(odist, L)
        self.omu = self._per_level(omu, L)
        self.opair_dist = None if opair_dist is None else \
            self._per_level(opair_dist, L)
        self.pmu = self._per_level(pmu, L)

        self.rank = None if rank is None else rank.copy("double")
        self.rmin_dist = _hfd_vprop(rmin_dist, g)
        self.rmu = rmu
        self.rcm_dist = rcm_dist
        self.rcmu = rcmu
        self.rcm_xdist = rcm_xdist
        self.rcmxu = rcmxu

        self.olambda = _hfd_level_arrays(olambda, b, 0)
        self.rlambda = _hfd_prop(rlambda, g)
        self.plambda = _hfd_prop(plambda, g, vtype="vector<double>")
        self.nlambda = _hfd_prop(nlambda, g)

        self.upper = upper
        self.rng = _get_rng()
        # also those of the disabled constraints, which the C++ state does not
        # see, since they would be used by copies
        if any(not (x >= 0) for x in [self.nmu, self.rmu, self.rcmu,
                                      self.rcmxu, *self.gmu,
                                      *self.omu, *self.pmu]):
            raise ValueError("the penalties must be non-negative")
        self._state = libgraph_tool_layout.make_hfd_state(self._cpp_args())
        if upper is not None:
            self._state.set_upper(upper._state)
        self.set_glambda(glambda)
        self._state.set_rclambda(rclambda)
        self._state.set_rcxlambda(rcxlambda)
        if group_spring_strength is not None:
            self._state.set_group_springs([numpy.asarray(x, dtype="double")
                                           for x in group_spring_strength])
        if nmu_force is not None:
            self._state.set_nmu_force(numpy.asarray(nmu_force,
                                                    dtype="double"))

    def _cpp_args(self):
        """The parameters of the C++ state, where the constraints whose
        distances (or ranks) are not given are disabled by a zero penalty."""
        g, L = self.g, len(self.b)
        args = types.SimpleNamespace(**self.__dict__)
        if self.nmin_dist is None:
            args.nmin_dist, args.nmu = g.new_vp("double"), 0
        if self.gmin_dist is None:
            args.gmin_dist, args.gmu = g.new_vp("vector<double>"), [0.] * L
        if self.odist is None:
            args.odist, args.omu = [0.] * L, [0.] * L
        if self.opair_dist is None:
            args.opair_dist, args.pmu = [0.] * L, [0.] * L
        if self.rank is None:
            args.rank, args.rmu = g.new_vp("double"), 0
        if self.rank is None or self.rcm_dist is None:
            args.rcm_dist, args.rcmu = 0, 0
        if self.rank is None or self.rcm_xdist is None:
            args.rcm_xdist, args.rcmxu = 0, 0
        return args

    @staticmethod
    def _per_level(x, L):
        if isinstance(x, Iterable):
            return [float(y) for y in x]
        return [float(x)] * L

    def __getstate__(self):
        return dict(PickleState.__getstate__(self),
                    glambda=self.get_glambda(),
                    rclambda=self._state.get_rclambda(),
                    rcxlambda=self._state.get_rcxlambda(),
                    group_spring_strength=self._state.get_group_springs(),
                    nmu_force=self._state.get_nmu_force())

    _constraint_args = {"pos", "theta", "exact_thres", "nmin_dist", "nmu",
                        "gmin_dist", "gmu", "odist", "omu", "opair_dist",
                        "pmu", "rank", "rmin_dist", "rmu", "rcm_dist",
                        "rcmu", "rclambda", "rcm_xdist", "rcmxu",
                        "rcxlambda", "nlambda",
                        "glambda", "olambda", "rlambda", "plambda",
                        "group_spring_strength", "nmu_force"}

    def copy(self, **kwargs):
        r"""Copies the state. The parameters override the state properties, and
        have the same meaning as in the constructor.

        The Lagrange multipliers, the strengths of the springs and the forces
        that set the penalties of the minimum distances are copied as well,
        unless the parameters they depend on are overridden, in which case
        they are reset (as are the per-group parameters ``group_conf_R`` and
        ``group_grav_fac``, if ``b`` is overridden). The partition ``b`` can
        only be replaced by one with the same number of levels, since the
        per-level parameters are kept.
        """
        if "g" in kwargs:
            kwargs = dict(dict(glambda=0, nlambda=None, olambda=None,
                               rlambda=None, plambda=None), **kwargs)
        elif "b" in kwargs:
            L = len(_hfd_bs(kwargs["b"]))
            if L != len(self.b):
                raise ValueError(f"the partition has {L} levels, but the " +
                                 f"state has {len(self.b)}; a state with " +
                                 "another number of levels must be " +
                                 "constructed anew")
            kwargs = dict(dict(glambda=0, olambda=None, plambda=None,
                               group_grav_fac=None, group_conf_R=None),
                          **kwargs)
        if "rank" in kwargs:
            kwargs = dict(dict(rlambda=None, rclambda=0, rcxlambda=0),
                          **kwargs)
        # the forces of the vertices without the constraints, which set the
        # penalties of the minimum distances, depend on every parameter
        # except those of the constraints (and the positions, whose change
        # only makes them less accurate)
        if any(k not in self._constraint_args for k in kwargs):
            kwargs = dict(dict(nmu_force=None), **kwargs)
        # the strengths of the springs depend on the edges and the repulsion
        forces = ["g", "vweight", "eweight", "rweight", "K", "rep_exp",
                  "att_exp"]
        if any(k in kwargs for k in forces + ["b", "group_conf_R",
                                              "group_conf_exp",
                                              "group_spring",
                                              "group_spring_static"]):
            kwargs = dict(dict(group_spring_strength=None), **kwargs)
        return PickleState.copy(self, **kwargs)

    @_parallel
    def energy(self):
        r"""Return the value of the augmented Lagrangian, i.e. the energy plus
        the constraint penalties for the current Lagrange multipliers."""
        return self._state.energy()

    @_parallel
    def gradient_sweep(self, niter=10, step=.01, mu=.9, cap_grad=0.1):
        r"""Perform ``niter`` steps of gradient descent with Nesterov momentum,
        for fixed Lagrange multipliers.

        @parallel@

        Parameters
        ----------
        niter : int (optional, default: ``10``)
            Number of steps.
        step : float (optional, default: ``0.01``)
            Step size.
        mu : float (optional, default: ``0.9``)
            Momentum parameter, in the interval :math:`[0, 1)`. If ``mu == 0``,
            plain gradient descent is used.
        cap_grad : float (optional, default: ``0.1``)
            If positive, the gradient contribution to the displacement of each
            vertex in a single step is capped at this value.
        """
        self._state.gradient_sweep(niter, step, mu, cap_grad)

    @_parallel
    def group_order_params(self):
        r"""Return order parameters that measure how well the groups are
        separated at every level of the partition, obtained from the Delaunay
        triangulation of the positions.

        Returns
        -------
        kappa : :class:`numpy.ndarray`
            Contiguity of the groups at every level: the fraction of the
            weight of each group that belongs to its largest biconnected
            component, in the graph of the Delaunay edges between members of
            the same group, averaged over the groups weighted by their total
            weight. It is ``1`` if all groups are contiguous.
        S : :class:`numpy.ndarray`
            Fraction of the area between groups at every level, i.e. of the
            Delaunay triangles with corners in different groups, relative to
            all the triangles with corners in the same group of the level
            above (or all triangles, at the top level).
        G : :class:`numpy.ndarray`
            Mean area of the triangles between groups at every level, relative
            to the mean area of the triangles within groups.
        kappa_groups : list of :class:`numpy.ndarray`
            Contiguity of every group, at every level, indexed by the group
            labels as they appear in ``b`` (``nan`` for unused labels).

        Notes
        -----
        Since a Delaunay edge exists only if no other vertex lies within some
        circle through its endpoints, members of a group that are separated by
        other groups are not connected, whereas those separated only by empty
        space are. The edges and triangles on the convex hull of the layout
        are ignored, since they can join distant vertices along its outer
        boundary.

        This has complexity :math:`O(V\log V)`, and the levels are computed
        in parallel.
        """
        return self._state.group_order_params()

    def update_group_springs(self):
        r"""Update the strengths of the confinement of the groups from the
        current pull of the edges that leave them, if ``group_spring > 0``
        (see :class:`~HFDState`)."""
        self._state.update_group_springs()

    @_parallel
    def reset_momentum(self):
        r"""Set the velocities used by the momentum of
        :meth:`~HFDState.gradient_sweep` to zero."""
        self._state.reset_momentum()

    @_parallel
    def lbfgs_sweep(self, max_iter=10, step=1., epsilon=1e-3, hist_size=10,
                    line_search="wolfe", c1=1e-4, c2=.9, rho=None,
                    ls_max_iter=20, ftol=0, fatol=0, max_time=None):
        r"""Minimize the augmented Lagrangian with the L-BFGS algorithm, for
        fixed Lagrange multipliers.

        @parallel@

        Parameters
        ----------
        max_iter : int (optional, default: ``10``)
            Maximum number of iterations.
        step : float (optional, default: ``1.``)
            Initial step size of the line search.
        epsilon : float (optional, default: ``1e-3``)
            The minimization stops when the norm of the gradient falls below
            this value.
        hist_size : int (optional, default: ``10``)
            Number of past updates used to approximate the inverse Hessian.
        line_search : ``"wolfe"`` or ``"armijo"`` (optional, default: ``"wolfe"``)
            Line search used to choose the step size: one that satisfies the
            strong Wolfe conditions, or Armijo backtracking.
        c1 : float (optional, default: ``1e-4``)
            Sufficient decrease parameter of the line search (Armijo
            condition).
        c2 : float (optional, default: ``0.9``)
            Curvature parameter of the strong Wolfe line search, with
            ``0 < c1 < c2 < 1``. It is ignored by the Armijo line search.
        rho : float (optional, default: ``None``)
            For the strong Wolfe line search, the factor by which the step
            grows while bracketing an acceptable step (default: ``2``). For
            the Armijo line search, the factor by which it shrinks while
            backtracking (default: ``0.5``).
        ls_max_iter : int (optional, default: ``20``)
            Maximum number of iterations of each line search.
        ftol : float (optional, default: ``0``)
            The minimization stops when the change of the objective between
            iterations falls below this value, relative to the maximum of ``1``
            and its absolute value. It is disabled if ``ftol == 0``.
        fatol : float (optional, default: ``0``)
            The minimization stops when the absolute change of the objective
            between iterations falls below this value. It is disabled if
            ``fatol == 0``.
        max_time : float (optional, default: ``None``)
            If given, the minimization stops after this many seconds.

        Returns
        -------
        niter : int
            Number of iterations performed.
        E : float
            Final value of the augmented Lagrangian.
        converged : bool
            ``True`` if one of the convergence criteria (``epsilon``, ``ftol``
            or ``fatol``) was met before the minimization was stopped.
        """
        if line_search not in ["wolfe", "armijo"]:
            raise ValueError(f"invalid line search: {line_search}")
        armijo = line_search == "armijo"
        if rho is None:
            rho = .5 if armijo else 2.
        if armijo:
            if not 0 < c1 < 1 or not 0 < rho < 1:
                raise ValueError("the Armijo line search requires 0 < c1 < 1 " +
                                 f"and 0 < rho < 1, got c1 = {c1}, rho = {rho}")
        else:
            if not 0 < c1 < c2 < 1 or rho <= 1:
                raise ValueError("the strong Wolfe line search requires " +
                                 "0 < c1 < c2 < 1 and rho > 1, got " +
                                 f"c1 = {c1}, c2 = {c2}, rho = {rho}")
        return self._state.lbfgs_sweep(max_iter, step, epsilon, hist_size,
                                       armijo, c1, c2, rho, ls_max_iter, ftol,
                                       fatol,
                                       0 if max_time is None else max_time)

    @_parallel
    def update_constraints(self):
        r"""Update the Lagrange multipliers of all active constraints.

        @parallel@

        Returns
        -------
        R : float
            Total constraint violation, which is zero if all constraints are
            satisfied. The violations of the minimum and group distances,
            :math:`\sum_v V_v`, are relative to :math:`\sum_v c_v^2`, i.e.
            the squared distance times the number of vertices, while those of
            the ranks, overlaps and pairs are lengths, summed over the
            constraints.
        """
        return self._state.update_constraints()

    @_parallel
    def constraint_violation(self):
        r"""Return the total violation of all active constraints, without
        updating the Lagrange multipliers.

        @parallel@

        Returns
        -------
        R : float
            Total constraint violation, which is zero if all constraints are
            satisfied.
        """
        return self._state.constraint_violation()

    def scale_penalties(self, c):
        r"""Multiply the penalty parameters of all the constraints (``nmu``,
        ``gmu``, ``omu``, ``pmu``, ``rmu``, ``rcmu`` and ``rcmxu``) by
        ``c``."""
        self.nmu *= c
        self.rmu *= c
        self.rcmu *= c
        self.rcmxu *= c
        self.pmu = [x * c for x in self.pmu]
        self.gmu = [x * c for x in self.gmu]
        self.omu = [x * c for x in self.omu]
        self._state.scale_penalties(c)

    @_parallel
    def init_multipliers(self, margin=numpy.inf):
        r"""Initialize the Lagrange multipliers of all active constraints with
        estimates obtained from the current positions.

        @parallel@

        Parameters
        ----------
        margin : float (optional, default: ``numpy.inf``)
            Constraints that are satisfied by a margin larger than this value
            get a multiplier of zero. By default, all constraints get an
            estimate.

        Notes
        -----
        Writing each constraint as :math:`c(\boldsymbol x) \le 0`, its
        multiplier is estimated as

        .. math::

            \lambda = \max\left(0, -\frac{\nabla U\cdot\nabla c}{|\nabla c|^2}\right),

        where :math:`U` is the energy without the constraints, so that the
        force of the constraint cancels the other forces along its gradient.
        Otherwise, the constraints exert no force until they are violated.
        The forces on the vertices obtained here also set the penalties of
        their minimum distances (see ``nmu``).
        """
        self._state.init_multipliers(margin)

    def get_glambda(self):
        r"""Return the Lagrange multipliers of the group distance constraint, as
        a list with one value per hierarchy level."""
        return self._state.get_glambda()

    def set_glambda(self, glambda):
        r"""Set the Lagrange multipliers of the group distance constraint, given
        as a single value for all hierarchy levels, or a list with one value
        per level."""
        self._state.set_glambda(glambda)

    def get_nlambda(self):
        r"""Return the Lagrange multipliers of the minimum distance constraint,
        as a vertex property map with one value per vertex."""
        return self.nlambda

    def set_nlambda(self, l):
        r"""Set the Lagrange multipliers of the minimum distance constraint,
        given as a vertex property map, or a single value for all
        vertices."""
        if isinstance(l, VertexPropertyMap):
            self.nlambda.a = l.a
        else:
            self.nlambda.a = l

    def visit_barnes_hut(self, x, y):
        r"""Return the cells used in the Barnes-Hut approximation of the
        repulsion felt at point :math:`(x, y)`.

        Parameters
        ----------
        x : float
            :math:`x` coordinate of the point.
        y : float
            :math:`y` coordinate of the point.

        Returns
        -------
        cells : list of tuples
            List of ``((x, y), w)`` tuples, with the center of mass and total
            vertex weight of each cell.
        """
        return self._state.visit_barnes_hut(x, y)


@_parallel
@_copy_params_doc(HFDState)
def hfd_layout(g, pos=None, vweight=None, eweight=None, rweight=None, K=1,
               rep_exp=1, att_exp=2, grav=1e-4, theta=.6, exact_thres=500,
               b=None, group_grav=.1, group_grav_fac=None,
               group_grav_offset=.01, group_rep=0, group_rep_exp=1,
               group_conf_R=None, group_conf_exp=6, group_spring=1,
               group_spring_static=False, upper=None, barrier_r=None,
               barrier_sep=.2, barrier_exact_thres=100, nmin_dist=None,
               nmu=100, gmin_dist=None, gmu=10, odist=None, omu=10,
               opair_dist=None, pmu=10, rank=None, rmin_dist=0, rmu=1,
               rcm_dist=None, rcmu=1000, rcm_xdist=None, rcmxu=1000,
               hierarchical=True, grouped=None,
               spring_G=20, spring_decay=2, grav_c=1, grav_L="mf",
               method="gradient", max_iter=None, epsilon=None, patience=3,
               sweep_args={}, adaptive_cooling=True, cooling_step=.9,
               momentum_restart=True, backtrack=True, init_multipliers=True,
               tol=1e-3, update_interval=3, draw_levels=False,
               return_state=False, verbose=False):
    r"""Obtain a hierarchical force-directed (HFD) layout of the graph.

    Parameters
    ----------
    @params@
    hierarchical : bool (optional, default: ``True``)
        If ``True``, the layout is obtained hierarchically, from the top of
        the partition ``b`` to the bottom (see below). If ``b`` is not given,
        a hierarchical partition is generated.
    grouped : bool (optional, default: ``None``)
        If ``hierarchical == True``, whether the groups are kept together and
        apart from each other in the final layout. Otherwise, the hierarchy
        only guides the initial placement. If ``None``, it is ``True`` if and
        only if ``b`` is given. If ``True``, the groups are kept together by
        springs set by ``spring_G`` and ``spring_decay`` (which replace
        ``group_conf_R``, ``group_conf_exp``, ``group_spring`` and
        ``group_spring_static``), and ``group_grav`` is ignored.
    spring_G : float or list of floats (optional, default: ``20``)
        Strength of the springs that keep the groups of the lowest level
        together, relative to the pull of their external edges. Larger
        values give more compact groups. A list gives one value per level,
        and ``spring_decay`` is then ignored.
    spring_decay : float (optional, default: ``2``)
        Factor by which the strength of the springs decreases at every level
        above. Smaller values make the groups of the upper levels more
        compact and separated.
    grav_c : float (optional, default: ``1``)
        If not ``None`` and ``grouped == True``, the gravity is set relative
        to the size of the layout, so that a vertex without edges stays
        within this multiple of the radius of the layout, and ``grav`` is
        ignored. Smaller values give more compact
        layouts.
    grav_L : ``"mf"`` or ``"measured"`` (optional, default: ``"mf"``)
        Radius of the layout used for ``grav_c``: estimated in the mean field,
        or measured from the layout of the level above.
    method : ``"gradient"`` or ``"lbfgs"`` (optional, default: ``"gradient"``)
        Minimization method, :meth:`~HFDState.gradient_sweep` or
        :meth:`~HFDState.lbfgs_sweep`.
    max_iter : int (optional, default: ``None``)
        Maximum number of sweeps (``1000`` for ``"gradient"`` and ``200`` for
        ``"lbfgs"``, if ``None``).
    epsilon : float (optional, default: ``None``)
        Tolerance for the relative change of the energy between sweeps
        (``1e-6`` with adaptive cooling, ``1e-5`` without it, and ``1e-4``
        for ``"lbfgs"``, if ``None``).
    patience : int (optional, default: ``3``)
        Number of consecutive sweeps within ``epsilon`` for the minimization
        to be considered converged.
    sweep_args : ``dict`` (optional, default: ``{}``)
        Additional parameters passed to every sweep.
    adaptive_cooling : bool (optional, default: ``True``)
        If ``True``, the step size of :meth:`~HFDState.gradient_sweep`
        starts proportional to :math:`\sqrt{V}`, and, if ``backtrack ==
        False``, is adapted to the progress of the energy
        [hu-multilevel-2005]_.
    cooling_step : float (optional, default: ``0.9``)
        Factor by which the step size is reduced when the energy increases,
        and whose inverse increases it otherwise. With ``backtrack == True``,
        it only sets the increase.
    momentum_restart : bool (optional, default: ``True``)
        If ``True``, the momentum is reset whenever a sweep increases the
        energy (which is always done with ``backtrack == True``).
    backtrack : bool (optional, default: ``True``)
        If ``True``, a sweep that increases the energy is undone and repeated
        with a smaller step.
    init_multipliers : bool (optional, default: ``True``)
        If ``True``, the Lagrange multipliers are initialized from the initial
        positions with :meth:`~HFDState.init_multipliers`.
    tol : float (optional, default: ``1e-3``)
        Tolerance for the total constraint violation (see
        :meth:`~HFDState.update_constraints`).
    update_interval : int (optional, default: ``3``)
        Number of sweeps between the updates of the Lagrange multipliers.
    draw_levels : bool or ``dict`` (optional, default: ``False``)
        If ``True``, the graph of every level of the hierarchy is drawn with
        :func:`~graph_tool.draw.graph_draw` before and after its layout,
        colored by the groups of the level above, and sized by the weights it
        represents. A ``dict`` is passed as additional arguments to
        :func:`~graph_tool.draw.graph_draw`; an ``output`` file name gets the
        level and stage before its extension (e.g. ``levels-l2-before.pdf``
        and ``levels-l2-after.pdf``), so that every drawing is kept.
    return_state : bool (optional, default: ``False``)
        If ``True``, the :class:`~HFDState` of the (lowest) layout is also
        returned. In the hierarchical layout, the labels of its partition are
        made contiguous (keeping their order).
    verbose : bool (optional, default: ``False``)
        If ``True``, the progress of the minimization is printed.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        Vector-valued vertex property map with the coordinates of the
        vertices.
    state : :class:`~HFDState`
        The state of the layout, which can be used to refine it further. Only
        returned if ``return_state == True``.

    Notes
    -----
    The energy of a :class:`~HFDState` with the given parameters is
    minimized until its relative change falls below ``epsilon`` for
    ``patience`` consecutive sweeps. With constraints, the Lagrange
    multipliers are updated every ``update_interval`` sweeps, and whenever
    the minimization converges: the layout then stops if the violation is
    below ``tol``, and otherwise the minimization is restarted, with the
    penalties doubled (up to a factor :math:`2^{10}` in total) if the
    violation did not decrease enough.

    **Hierarchical layout.** The graph is coarse-grained according to every
    level of the partition, by merging every group into a vertex that carries
    the weights it represents. The coarse graphs are laid out from the top
    down, each one placing the members of every group evenly inside a disk
    around its position, with the radius of the group in the mean field.
    The coarse graphs are minimized until the relative change of the energy
    falls below :math:`10^{-4}` (``epsilon`` and ``patience`` apply to the
    original graph).

    If ``grouped == True``, the groups of every level are kept apart by the
    attraction and repulsion of their centers of mass, given by the coarse
    graphs above, with the repulsion between two groups proportional to
    their relative sizes; and kept together by springs whose strengths
    follow from the pull of their external edges (``spring_G`` and
    ``spring_decay``). In this case, ``K`` can be a list with one value for
    the vertices followed by one for the groups of every level, which sets
    how strongly they are pushed apart (the gravity of every level, and the
    radii of the groups whose members are its vertices, follow its value). The gravity of every level is set by
    ``grav_c``. The group gravity (``group_grav``) is not used in this case.

    If no partition is given, one with at most :math:`\lceil\log_2 V\rceil`
    levels is generated by halving the number of groups at every level, with
    :func:`~graph_tool.inference.minimize_blockmodel_dl` and
    :class:`~graph_tool.inference.ModularityState`.

    If ``pos`` is given, it is used as the initial layout, and only the
    original vertices are laid out, e.g. to refine an existing layout with
    different parameters.

    Each sweep has complexity :math:`O(V\log V + E)`.

    @parallel@

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.collection.data["pgp-strong-2009"]
    >>> state = gt.minimize_nested_blockmodel_dl(g, simple_init=True)
    >>> bs = state.get_bs()
    >>> b = state.project_level(6).get_blocks()
    >>> cmap = matplotlib.colormaps["tab20"]
    >>> def draw(pos, output):
    ...     return gt.graph_draw(g, pos=pos, vertex_fill_color=b, vcmap=cmap,
    ...                          vcnorm=lambda x: np.asarray(x) % cmap.N,
    ...                          vertex_size=1.5, vertex_pen_width=0,
    ...                          edge_pen_width=.3,
    ...                          edge_color=[.2, .2, .2, .06],
    ...                          output_size=(800, 800), output=output)
    >>> pos0 = gt.hfd_layout(g, b=bs)
    >>> draw(pos0, "graph-draw-hfd-layout-default.pdf")
    <...>

    The layout with the default parameters is then used as the starting
    point of the layouts with other parameters, so that their effect can be
    compared directly:

    >>> for c in [.25, 5]:
    ...     pos = gt.hfd_layout(g, b=bs, pos=pos0, grav_c=c)
    ...     draw(pos, f"graph-draw-hfd-layout-{c}.pdf")
    <...>
    >>> for G in [2, 200]:
    ...     pos = gt.hfd_layout(g, b=bs, pos=pos0, spring_G=G)
    ...     draw(pos, f"graph-draw-hfd-layout-G-{G}.pdf")
    <...>
    >>> for f in [1.2, 4]:
    ...     pos = gt.hfd_layout(g, b=bs, pos=pos0, spring_decay=f)
    ...     draw(pos, f"graph-draw-hfd-layout-decay-{f}.pdf")
    <...>
    >>> for k in [.1, 10]:
    ...     pos = gt.hfd_layout(g, b=bs, pos=pos0, K=[1] + [k] * len(bs))
    ...     draw(pos, f"graph-draw-hfd-layout-K-{k}.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-hfd-layout-default.pdf")
       for x in ["0.25", "5", "G-2", "G-200", "decay-1.2", "decay-4",
                 "K-0.1", "K-10"]:
           conv_png(f"graph-draw-hfd-layout-{x}.pdf")

    .. image:: graph-draw-hfd-layout-0.25.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-default.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-5.png
       :width: 32%

    .. image:: graph-draw-hfd-layout-G-2.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-default.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-G-200.png
       :width: 32%

    .. image:: graph-draw-hfd-layout-decay-1.2.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-default.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-decay-4.png
       :width: 32%

    .. image:: graph-draw-hfd-layout-K-0.1.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-default.png
       :width: 32%
    .. image:: graph-draw-hfd-layout-K-10.png
       :width: 32%

    Hierarchical layout of the PGP web of trust, grouped by the nested
    stochastic block model, and colored by its groups at the sixth level.
    The middle column has the default parameters, and the others a smaller
    (*left*) and larger (*right*) value of one parameter, from top to
    bottom: ``grav_c`` (:math:`0.25` and :math:`5`), which compresses or
    loosens the layout; ``spring_G`` (:math:`2` and :math:`200`), which
    makes the groups diffuse or compact; ``spring_decay`` (:math:`1.2` and
    :math:`4`), which makes the upper levels compact and separated, or
    blended; and ``K`` of the groups (:math:`0.1` and :math:`10`), which
    lets them crowd or pushes them apart.

    References
    ----------
    .. [hu-multilevel-2005] Yifan Hu, "Efficient, High-Quality Force-Directed
       Graph Drawing", Mathematica Journal, vol. 10, Issue 1, pp. 37-71, (2005)
       http://www.mathematica-journal.com/issue/v10i1/graph_draw.html

    """

    return _hfd_run(_hfd_layout_iter(locals()))


def _hfd_run(it):
    """Run the generator it to the end, and return its value."""
    while True:
        try:
            next(it)
        except StopIteration as e:
            return e.value


def _hfd_args(**kwargs):
    """The arguments of hfd_layout(), with their defaults overridden by
    kwargs."""
    params = inspect.signature(hfd_layout).parameters
    for k in kwargs:
        if k not in params:
            raise TypeError(f"invalid parameter for hfd_layout(): {k}")
    return {k: kwargs.get(k, p.default) for k, p in params.items()}


def _hfd_layout_iter(args):
    """Generator version of hfd_layout() with the arguments args (as given by
    _hfd_args()), which yields the HFDState being minimized after every sweep
    (of the coarse graphs of every level first, in the hierarchical layout),
    and returns the value of hfd_layout()."""
    if args["hierarchical"]:
        return (yield from _hfd_hierarchical(args))

    (g, method, max_iter, epsilon, adaptive_cooling, init_multipliers,
     momentum_restart, backtrack, sweep_args, cooling_step, group_spring,
     group_conf_R, group_spring_static, patience, tol, update_interval,
     verbose, return_state) = \
        (args[k] for k in ["g", "method", "max_iter", "epsilon",
                           "adaptive_cooling", "init_multipliers",
                           "momentum_restart", "backtrack", "sweep_args",
                           "cooling_step", "group_spring", "group_conf_R",
                           "group_spring_static", "patience", "tol",
                           "update_interval", "verbose", "return_state"])

    state_params = inspect.signature(HFDState.__init__).parameters
    state = HFDState(**{k: v for k, v in args.items() if k in state_params})

    # defaults chosen so that both methods stop near the point where further
    # sweeps no longer reduce the energy appreciably
    if method == "gradient":
        sweep = state.gradient_sweep
        max_iter = 1000 if max_iter is None else max_iter
        if epsilon is None:
            epsilon = 1e-6 if adaptive_cooling else 1e-5
    elif method == "lbfgs":
        sweep = state.lbfgs_sweep
        max_iter = 200 if max_iter is None else max_iter
        epsilon = 1e-4 if epsilon is None else epsilon
    else:
        raise ValueError(f"invalid method: {method}")

    if init_multipliers:
        state.init_multipliers()

    cooling = adaptive_cooling and method == "gradient"
    restart = momentum_restart and method == "gradient"
    backtrack = backtrack and method == "gradient"
    if cooling or backtrack:
        sweep_args = dict(sweep_args)
        # the initial steps should be commensurate with the size of the layout
        scale = numpy.sqrt(g.num_vertices()) if cooling else 1
        step = sweep_args.pop("step", .01 * scale)
        cap_grad = sweep_args.pop("cap_grad", .1 * scale)
    f = 1               # factor of the step size and gradient cap
    progress = 0        # consecutive sweeps that decreased the energy
    springs = numpy.any(numpy.asarray(group_spring) > 0) and \
        group_conf_R is not None and not group_spring_static
    constrained = state._state.has_constraints()

    E = state.energy()
    nconv = 0           # consecutive sweeps within epsilon
    since = 0           # sweeps since the last update of the multipliers
    R_prev = None       # violation at the last restart
    pen_factor = 1      # increase of the penalties

    def log(i, E_new, R, step, extra=""):
        # scientific notation with the same number of digits, left-aligned
        # with a column for the sign, so that the decimal points line up
        def num(x):
            return f"{x:< 13.6e}" if x is not None else f"{' -':13}"
        msg = f"{i:6d}: E = {num(E_new)}  dE = {num(E_new - E)}  " + \
            f"R = {num(R)}"
        if cooling or backtrack:
            msg += f"  step = {num(step)}"
        if extra:
            msg += "  " + extra
        print(msg)

    def do_sweep():
        if cooling or backtrack:
            sweep(step=step * f, cap_grad=cap_grad * f, **sweep_args)
        else:
            sweep(**sweep_args)
        return state.energy()

    for i in range(max_iter):
        if backtrack:
            # any sweep that does not increase the energy is accepted; a
            # sufficient decrease relative to the slope (Armijo) cannot be
            # met when the energy is nearly singular, e.g. with nearly
            # coinciding vertices, since the slope then diverges
            for k in range(20):
                E_new = do_sweep()
                if E_new <= E:
                    break
                # the sweep is undone, and repeated without momentum, with the
                # step at the minimum of the quadratic interpolation of the
                # energy along the displacement, bounded to [0.1, 0.5] of it
                m = state._state.sweep_slope()
                state._state.restore_sweep()
                state.reset_momentum()
                q = E_new - E - m
                r = min(max(-m / (2 * q) if m < 0 and q > 0 else .5, .1), .5)
                if verbose:
                    log(i, E_new, None, step * f, "(rejected)")
                f *= r
            else:
                E_new = E   # no acceptable sweep: the state is restored
            # the steps that are too large are undone, so that the step can
            # grow after every sweep accepted at the first attempt
            if k == 0:
                f /= cooling_step
        else:
            E_new = do_sweep()
            if cooling:
                if E_new < E:
                    progress += 1
                    if progress >= 5:
                        f /= cooling_step
                        progress = 0
                else:
                    f *= cooling_step
                    progress = 0
            if restart and E_new > E:
                state.reset_momentum()
        if springs:
            state.update_group_springs()
        dE = E_new - E
        # the energy is minimized for fixed multipliers until it converges; a
        # single small change can be due to a transiently small step, or a
        # slow collective rearrangement, so we require several in a row
        if abs(dE) <= epsilon * abs(E_new):
            nconv += 1
        else:
            nconv = 0
        converged = nconv >= patience
        R = None
        extra = ""
        done = converged and not constrained
        updated = False
        if constrained:
            since += 1
            if converged or since >= update_interval:
                R = state.update_constraints()
                done = converged and R <= tol
        if R is not None and not done:
            since = 0
            updated = True
            extra = "(multipliers updated)"
            if converged:
                # if the violation did not decrease enough since the last
                # restart, the multipliers alone are converging too slowly,
                # and the penalties are increased, but only at the restarts
                # (between the periodic updates it rarely decreases by much),
                # and at most by a factor 2^10 in total, since too stiff
                # penalties freeze the layout when vertices are jammed
                if (R_prev is not None and R > .25 * R_prev and
                    pen_factor < 2 ** 10):
                    state.scale_penalties(2)
                    pen_factor *= 2
                    extra = "(multipliers updated, penalties x2)"
                R_prev = R
                # the minimization is restarted from the initial step, since
                # the step may have shrunk so much that it would only seem to
                # converge
                nconv = 0
                f = 1
                progress = 0
                state.reset_momentum()
        if verbose:
            # the violation is shown after every sweep, even if the
            # multipliers were not updated
            if R is None:
                R = state.constraint_violation() if constrained else 0
            log(i, E_new, R, step * f if cooling or backtrack else None,
                extra)
        if done:
            break
        if updated or springs:
            # the energy has changed with the multipliers or springs
            E_new = state.energy()
        E = E_new
        yield state

    if return_state:
        return state.pos, state
    return state.pos


class _HFDLevel(object):
    """A level of the hierarchy used by the hierarchical hfd_layout(): a graph
    whose vertices are the groups of the level below (obtained with
    condensation_graph()), with their total weight w, mean-field radius R and
    mean gmin_dist, and the total attraction weight e of its edges (see
    hfd.cc)."""

    def __init__(self, g):
        self.g = g
        self.w, self.R = g.new_vp("double"), g.new_vp("double")
        self.gmin = g.new_vp("vector<double>")
        self.e = g.new_ep("double")

    def props(self):
        return (self.w, self.R, self.gmin, self.e)


def _hfd_partition(g, eweight=None):
    """Hierarchical partition with ceil(log2(N)) levels at most, where every
    level halves the number of vertices of the graph of the groups below it,
    via minimize_blockmodel_dl() with ModularityState (or at random, without
    any edges)."""
    bs = []
    for l in range(int(numpy.ceil(numpy.log2(max(g.num_vertices(), 2))))):
        B = g.num_vertices() // 2
        if B > 1 and g.num_edges() > 0:
            state = minimize_blockmodel_dl(g, state=ModularityState,
                                           state_args=dict(eweight=eweight),
                                           multilevel_mcmc_args=dict(B_min=B,
                                                                     B_max=B))
            b = state.b.copy("int64_t")
        elif B > 1:
            b = g.new_vp("int64_t",
                         vals=numpy.random.permutation(g.num_vertices()) % B)
        else:
            b = g.new_vp("int64_t")
        b.fa = numpy.unique(b.fa, return_inverse=True)[1]  # contiguous labels
        bs.append(b.a.copy())
        if B <= 1:
            break
        g, vcount, eweight = condensation_graph(g, b, eweight=eweight)
    return bs


def _hfd_hierarchical(args):
    """Hierarchical version of hfd_layout(): the coarse graphs of the
    hierarchy are laid out from the top down, each initializing the next."""
    g = args["g"]
    log = print if args["verbose"] else lambda *a, **k: None
    if args["upper"] is not None:
        raise ValueError("the hierarchical layout builds its own upper " +
                         "states; an upper state can only be given with " +
                         "hierarchical=False")
    if g.num_vertices() == 0:
        return (yield from _hfd_layout_iter(dict(args, hierarchical=False)))
    bs = _hfd_bs(args["b"])
    generated = len(bs) == 0
    if generated:
        bs = _hfd_partition(g, args["eweight"])
    else:
        # without gaps in the labels, so that the coarse graphs have no
        # empty groups
        bs, args = _hfd_order_labels(bs, args)
    # whether the groups are kept together, by the springs and the forces
    # between their centers of mass; otherwise, the hierarchy only guides the
    # initial placement, so that the group constraints and group forces of
    # its coarse levels are not active
    grouped = (not generated) if args["grouped"] is None else \
        bool(args["grouped"])
    # if the groups are kept together, the springs do it, and the group
    # gravity is not used
    if grouped:
        args = dict(args, group_grav=0)
        if generated:
            args = dict(args, b=bs)
    largs = args                # arguments for the coarse levels
    if not grouped:
        largs = dict(args, gmin_dist=None, odist=None, opair_dist=None,
                     group_grav=0, group_rep=0, group_grav_fac=None,
                     group_conf_R=None)
    L = len(bs)
    per_level = lambda x: numpy.asarray(HFDState._per_level(x, L))
    group_grav = per_level(largs["group_grav"])

    # the repulsion of the vertices (K[0]) and of the coarse graph of every
    # level (K[l]), whose vertices are the groups of the level below
    K_l = numpy.asarray(HFDState._per_level(args["K"], L + 1), dtype="double")
    if len(K_l) != L + 1:
        raise ValueError(f"K has {len(K_l)} values, but there are {L} " +
                         "hierarchy levels (it must have one value, or one " +
                         "more than the levels)")
    K = K_l[0]

    levels = [_HFDLevel(g)]
    f = levels[0]
    f.w = _hfd_vprop(1 if args["vweight"] is None else args["vweight"], g)
    gmin = largs["gmin_dist"]
    f.gmin = _hfd_level_vprop(0 if gmin is None else gmin, g, L)
    ew = _hfd_prop(args["eweight"], g, key="e", val=1)
    libgraph_tool_layout.hfd_init_level(g, ew, f.props())

    # radii of the groups in the mean field, used to place their members and
    # as the reference of their springs; the extra level at the top has a
    # single group with all the vertices, whose radius is that of the whole
    # layout
    root = numpy.zeros(bs[-1].max() + 1, dtype="int64")
    R_mf = libgraph_tool_layout.hfd_mf_radii(g, bs + [root], f.w,
                                             _hfd_prop(args["rweight"], g,
                                                       val=1),
                                             ew, K_l,
                                             args["rep_exp"],
                                             args["att_exp"])
    L_top = R_mf.pop()[0]

    for l in range(L):
        f = levels[-1]
        b = f.g.new_vp("int64_t")
        b.a = bs[l][:len(b.a)]
        cg, w_c, e_c = condensation_graph(f.g, b, vweight=f.w, eweight=f.e)
        c = _HFDLevel(cg)
        c.w, c.e = w_c, e_c
        c.R.a = R_mf[l][:len(c.R.a)]
        libgraph_tool_layout.hfd_coarsen(f.g, cg, b.a, f.props(), c.props())
        levels.append(c)

    log(f"hierarchy with {L + 1} levels " +
        f"({'generated' if generated else 'given'} partition):")
    for l, c in enumerate(levels):
        log(f"    level {l}: {c.g.num_vertices()} vertices, " +
            f"{c.g.num_edges()} edges")

    draw = args["draw_levels"]
    draw_args = draw if isinstance(draw, dict) else {}

    def draw_out(l, stage):
        """The arguments of the drawing of level l at the given stage, with
        an output file name of its own, so that the drawings are kept."""
        out = draw_args.get("output")
        if not isinstance(out, str):
            return draw_args
        root, ext = os.path.splitext(out)
        return dict(draw_args, output=f"{root}-l{l}-{stage}{ext}")

    def layout(l, **kwargs):
        G = levels[l].g
        log(f"laying out level {l} ({G.num_vertices()} vertices)...")
        if draw:
            # vertices colored by their groups in the level above, and sized
            # according to the number of vertices and edges they represent
            # (unless they are all the same)
            lev = levels[l]
            dargs = {}
            w_l, e_l = lev.w.fa, lev.e.fa
            if len(w_l) > 0 and w_l.max() > w_l.min():
                dargs["vertex_size"] = prop_to_size(lev.w, mi=3, ma=20)
            if len(e_l) > 0 and e_l.max() > e_l.min():
                dargs["edge_pen_width"] = prop_to_size(lev.e, mi=.3, ma=5)
            if l < L:
                b = G.new_vp("int64_t")
                b.a = bs[l][:len(b.a)]
                # the groups are colored with the tab20 colormap, cycled
                import matplotlib
                cmap = matplotlib.colormaps["tab20"]
                dargs["vertex_fill_color"] = b
                dargs["vcmap"] = cmap
                dargs["vcnorm"] = lambda x: numpy.asarray(x) % cmap.N
            graph_draw(G, pos=kwargs["pos"],
                       **dict(dargs, **draw_out(l, "before")))
        if (kwargs["adaptive_cooling"] and kwargs["method"] == "gradient" and
            (l < L or pos0 is not None)):
            # the initial placement only needs to be refined, so we start
            # with the default steps
            kwargs["sweep_args"] = dict(dict(step=.01, cap_grad=.1),
                                        **kwargs["sweep_args"])
        t = time.perf_counter()
        ret = yield from _hfd_layout_iter(dict(kwargs, draw_levels=False))
        log(f"level {l} done in {time.perf_counter() - t:.2f}s")
        if draw:
            graph_draw(G, pos=ret[0] if isinstance(ret, tuple) else ret,
                       **dict(dargs, **draw_out(l, "after")))
        return ret

    if args["grav_L"] not in ("mf", "measured"):
        raise ValueError("invalid grav_L: " + str(args["grav_L"]))

    # harmonic springs that keep the groups together, with the mean-field
    # radii as reference, and strengths decaying upwards
    if isinstance(args["spring_G"], Iterable):
        spring_G = numpy.asarray(args["spring_G"], dtype="double")
        if len(spring_G) != L:
            raise ValueError(f"spring_G has {len(spring_G)} values, but " +
                             f"there are {L} hierarchy levels (it must " +
                             "have one value, or one per level)")
    else:
        spring_G = numpy.array([args["spring_G"] / args["spring_decay"] ** l
                                for l in range(L)])

    def spring_args(l):
        """Arguments of hfd_layout() for the springs of the levels l and
        above."""
        if not grouped:
            return {}
        return dict(group_conf_R=R_mf[l:], group_conf_exp=2,
                    group_spring=spring_G[l:], group_spring_static=True)

    # if the groups are kept together, the repulsion of the centers of mass
    # of the groups is multiplied by the radii of both groups, estimated as
    # those of the groups packed in a disk, i.e. proportional to the square
    # root of their weights; they are relative to their mean at the level,
    # since otherwise the factor compounds up the hierarchy and the upper
    # levels dominate the layout
    def rweight_of(l):
        c = levels[l]
        if l == 0:
            return _hfd_prop(args["rweight"], c.g, val=1)
        rw = c.g.new_vp("double", val=1)
        if grouped:
            # the mean of the groups with weight (e.g. not filtered out)
            rw.fa = numpy.sqrt(c.w.fa)
            k = c.w.fa > 0
            if k.any():
                rw.fa /= rw.fa[k].mean()
        return rw

    # gravity relative to the size of the layout; since the gravity of a
    # vertex is proportional to its mass in the repulsion, the latter is
    # balanced at the same distance for every vertex, with the total mass
    # and the repulsion strength K_l of the level
    p = args["rep_exp"]

    def grav_of(L_lay, l):
        if not grouped or args["grav_c"] is None or not (L_lay > 0):
            return {}
        c = levels[l]
        m_tot = (c.w.fa * rweight_of(l).fa).sum()
        gr = K_l[l] * m_tot / (args["grav_c"] * L_lay) ** (p + 2)
        log(f"gravity {gr:.3g} for the radius {L_lay:.3g}")
        return dict(grav=gr)

    def layout_radius(c, pos_c):
        # only the (non-empty) groups in the view
        vs = c.g.get_vertices()
        x = pos_c.get_2d_array([0, 1]).T[vs]
        wc = c.w.a[vs]
        cm = (x * wc[:, None]).sum(axis=0) / wc.sum()
        return numpy.sqrt(2 * (wc * ((x - cm) ** 2).sum(axis=1)).sum() /
                          wc.sum())

    # the radius of the whole layout in the mean field (L_top) is used for
    # the initial positions of the top level, and its gravity
    L_grav = L_top

    # if a layout is given, the initial positions of every level are the
    # centers of mass of its groups (and those of the vertices, at the
    # lowest level)
    pos0 = args["pos"]
    def initial_pos(l):
        if l == 0:
            return pos0     # updated in place, as without the hierarchy
        G = levels[l].g
        pos = G.new_vp("vector<double>")
        vs = g.get_vertices()
        # only the vertices in the view
        x = g.own_property(pos0).get_2d_array([0, 1]).T.astype("double")
        w = levels[0].w.a[vs]
        r = vs
        for k in range(l):
            r = bs[k][r]
        N = G.num_vertices(True)
        W = numpy.bincount(r, w, N)
        X = numpy.array([numpy.bincount(r, w * x[:, i], N) for i in range(2)])
        X[:, W > 0] /= W[W > 0]
        pos.set_2d_array(X)
        return pos

    # the per-level arguments, of which every level gets those of its own
    # level and above
    level_args = dict(gmu=per_level(largs["gmu"]), group_grav=group_grav,
                      group_grav_fac=_hfd_level_arrays(largs["group_grav_fac"],
                                                       bs, 1),
                      group_rep=_hfd_halving(largs["group_rep"], L),
                      group_conf_R=_hfd_level_arrays(largs["group_conf_R"],
                                                     bs, 0),
                      group_spring=per_level(largs["group_spring"]),
                      omu=per_level(largs["omu"]),
                      pmu=per_level(largs["pmu"]))
    # the distances of the groups are those between their disks
    odist, opair_dist = [None if largs[k] is None else per_level(largs[k])
                         for k in ["odist", "opair_dist"]]

    pos = None if pos0 is None else initial_pos(L)
    state = None
    for l in reversed(range(1, L + 1)):
        c = levels[l]
        eweight = c.g.new_ep("double")
        gmin_dist = c.g.new_vp("vector<double>")
        Rm = libgraph_tool_layout.hfd_level_params(c.g, c.props(), eweight,
                                                   gmin_dist, l)
        if pos is None:
            # the top level, inside a disk with the radius of the layout
            pos = c.g.new_vp("vector<double>")
            n = c.g.num_vertices()
            rho = L_top * numpy.sqrt(numpy.random.random(n))
            theta = 2 * numpy.pi * numpy.random.random(n)
            pos.set_2d_array(numpy.array([rho * numpy.cos(theta),
                                          rho * numpy.sin(theta)]))
        kw = dict(largs, g=c.g, pos=pos, vweight=c.w, eweight=eweight,
                  b=bs[l:], K=K_l[l],
                  # the minimum distance between the vertices applies only
                  # to the original graph
                  nmin_dist=None, rmin_dist=0,
                  gmin_dist=None if gmin is None else gmin_dist,
                  **{k: x[l:] for k, x in level_args.items()},
                  odist=None if odist is None else odist[l:] + 2 * Rm,
                  opair_dist=(None if opair_dist is None else
                              opair_dist[l:] + 2 * Rm),
                  barrier_r=None, rank=None, hierarchical=False,
                  # the coarse levels only place the groups, which are refined
                  # by the levels below, so that they need not be converged
                  # as tightly as the lowest level
                  epsilon=1e-4, patience=1,
                  rweight=rweight_of(l), return_state=True,
                  **grav_of(L_grav, l))
        # the state of the level above, whose vertices are the groups of
        # this level, is used for the attraction and repulsion of their
        # centers of mass (and those of the levels above it)
        kw["upper"] = state if grouped else None
        if l < L:
            kw.update(spring_args(l))
        if pos0 is not None:
            # the coarse levels are already given by the centers of mass, so
            # that their states are only needed as the upper states of the
            # levels below
            kw["max_iter"] = 0
        pos_c, state = yield from layout(l, **kw)
        if args["grav_c"] is not None and args["grav_L"] == "measured":
            L_grav = layout_radius(c, pos_c)

        if pos0 is not None:
            pos = initial_pos(l - 1)
            continue

        # the members of every group inside its disk
        f = levels[l - 1]
        pos = f.g.new_vp("vector<double>")
        libgraph_tool_layout.sample_group_positions(f.g, c.g, pos,
                                                    bs[l - 1], pos_c,
                                                    R_mf[l - 1],
                                                    _get_rng())

    # the lowest level, with the original partition (if any)
    kw = dict(args, pos=pos, K=K, hierarchical=False, **grav_of(L_grav, 0))
    kw["upper"] = state if grouped else None
    if L > 0:
        kw.update(spring_args(0))
    return (yield from layout(0, **kw))


def radial_tree_layout(g, root, rel_order=None, rel_order_leaf=False,
                       weighted=False, node_weight=None, r=1.):
    r"""Computes a radial layout of the graph according to the shortest-path
    tree centered at the ``root`` vertex.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    root : :class:`~graph_tool.Vertex` or ``int``
        The root of the radial tree.
    rel_order : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Relative order of the nodes at each respective branch.
    rel_order_leaf : ``bool`` (optional, default: ``False``)
        If ``True``, the relative order of the leaves will propagate to the
        root. Otherwise they will propagate in the opposite direction.
    weighted : ``bool`` (optional, default: ``False``)
        If true, the angle between the child branches will be computed according
        to the weight of the entire sub-branches.
    node_weight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        If given, the relative spacing between leaves will correspond to the node
        weights.
    r : ``float`` (optional, default: ``1.``)
        Layer spacing.

    Returns
    -------
    pos : :class:`~graph_tool.VertexPropertyMap`
        A vector-valued vertex property map with the coordinates of the
        vertices.

    Notes
    -----
    This algorithm has complexity :math:`O(V + E)`, or :math:`O(V\log V + E)` if
    ``rel_order`` is given.

    Examples
    --------
    .. testcode::
       :hide:

       np.random.seed(42)
       gt.seed_rng(42)

    >>> g = gt.price_network(1000)
    >>> pos = gt.radial_tree_layout(g, g.vertex(0))
    >>> gt.graph_draw(g, pos=pos, output="graph-draw-radial.pdf")
    <...>

    .. testcleanup::

       conv_png("graph-draw-radial.pdf")

    .. figure:: graph-draw-radial.png
       :align: center
       :width: 60%

       Radial tree layout of a Price network.

    """

    levels, pred_map = shortest_distance(GraphView(g, directed=False), root,
                                         pred_map=True)
    t = predecessor_tree(g, pred_map)
    pos = t.new_vertex_property("vector<double>")
    levels = t.own_property(levels)
    if rel_order is None:
        rel_order = g.vertex_index.copy("int")
    if node_weight is None:
        node_weight = g.new_vertex_property("double", 1)
    elif node_weight.value_type() != "double":
        node_weight = node_weight.copy("double")
    libgraph_tool_layout.get_radial(t._Graph__graph,
                                    _prop("v", t, pos),
                                    _prop("v", t, levels),
                                    _prop("v", g, rel_order),
                                    _prop("v", g, node_weight),
                                    int(root), weighted, r,
                                    rel_order_leaf)
    return g.own_property(pos)

def prop_to_size(prop, mi=0, ma=5, log=False, power=0.5):
    r"""Convert property map values to be more useful as a vertex size, or edge
    width. The new values are taken to be

    .. math::

        y_i = mi + (ma - mi) \left(\frac{x_i - \min(x)} {\max(x) - \min(x)}\right)^\text{power}

    If ``log=True``, :math:`x_i` is replaced with :math:`\ln(x_i)`.

    If :math:`\max(x) - \min(x)` is zero, :math:`y_i = mi`.

    Parameters
    ----------
    prop : :class:`~graph_tool.PropertyMap`
        Scalar property map with the values to be converted.
    mi : float (optional, default: ``0``)
        Minimum output value.
    ma : float (optional, default: ``5``)
        Maximum output value.
    log : bool (optional, default: ``False``)
        If ``True``, the logarithm of the values is used.
    power : float (optional, default: ``0.5``)
        Exponent applied to the normalized values.

    Returns
    -------
    prop : :class:`~graph_tool.PropertyMap`
        Property map of type ``double`` with the converted values.

    """
    prop = prop.copy(value_type="double")
    if len(prop.fa) == 0:
        return prop

    if log:
        vals = numpy.log(prop.fa)
    else:
        vals = prop.fa

    delta = vals.max() - vals.min()
    if delta == 0:
        delta = 1
    prop.fa = mi + (ma - mi) * ((vals - vals.min()) / delta) ** power
    return prop

try:
    from . cairo_draw import graph_draw, cairo_draw, \
        get_hierarchy_control_points, default_cm,  default_cm_bin, \
        default_clrs, draw_hierarchy
    __all__ += ["draw_hierarchy"]
except ImportError:
    pass

try:
    from . cairo_draw import GraphWidget, GraphWindow, \
        interactive_window, GraphArtist
    __all__ += ["interactive_window", "GraphWidget", "GraphWindow", "GraphArtist"]
except ImportError:
    pass
