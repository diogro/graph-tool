#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2023 Tiago de Paula Peixoto <tiago@skewed.de>
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

from .. import _prop, _get_rng, Vector_int32_t

from .. spectral import adjacency
from .. generation import complete_graph, generate_triadic_closure, \
    remove_parallel_edges
from .. dynamics import IsingGlauberState, NormalState, IsingBPState, \
    NormalBPState

from . uncertain_blockmodel import *
from . util import pmap

from abc import ABC
import inspect
import numpy as np
import numbers

class DynamicsBlockStateBase(UncertainBaseState):
    def __init__(self, g, s, t=[], x=1, x_range=(-np.inf, np.inf), theta=0,
                 theta_range=(-np.inf, np.inf), disable_xdist=False,
                 disable_tdist=False, nested=True, state_args={}, bstate=None,
                 self_loops=False, max_m=1 << 16, nmax_extend=8, **kwargs):
        r"""Base state for network reconstruction based on dynamical data, using the
        stochastic block model as a prior. This class is not meant to be
        instantiated directly, only indirectly via one of its
        subclasses. Nevertheless, its paramteres are inherited, and are
        documented as follows.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Initial graph state.
        s : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap`
            Collection of time-series with node states over time. Each entry in
            this list must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` or ``vector<double>``, depending on wether the model
            has discrete or continuous state values. If a single property map is
            given, then a single time series is assumed.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        t : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``[]``)
            If nonempty, this allows for a compressed representation of the
            time-series parameter ``s``, corresponding only to points in time
            where the state of each node changes. Each entry in this list
            must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the points in time where the state of
            each node change. The corresponding state of the nodes at these
            times are given by parameter ``s``. If a single property map is
            given, then a single time series is assumed.
        x : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
            Initial value of the edge weights. If a ``float`` is given, the edge
            weights will be assumed to be the same for all edges.
        x_bounds : pair of ``float``s (optional, default: ``None``)
            Determines the bounds on the values of the edge weights. If not
            provided, they will be determined from ``x``.
        x_discrete : ``bool`` (optional, default: ``True``)
            If ``True`` the, edge weights are assumed to be discrete.
        nested : ``boolean`` (optional, default: ``True``)
            If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
            will be used, otherwise
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        state_args : ``dict`` (optional, default: ``{}``)
            Arguments to be passed to
            :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState` (optional, default: ``None``)
            If passed, this will be used to initialize the block state
            directly.
        self_loops : bool (optional, default: ``False``)
            If ``True``, it is assumed that the inferred graph can contain
            self-loops.

        """

        UncertainBaseState.__init__(self, g, nested=nested,
                                    state_args=state_args, bstate=bstate,
                                    self_loops=self_loops)

        if isinstance(s, PropertyMap):
            s = [s]
        if isinstance(t, PropertyMap):
            t = [t]

        self.s = [self.u.own_property(x) for x in s]
        self.t = [self.u.own_property(x) for x in t]

        if x is None:
            x = self.u.new_ep("double")
        elif isinstance(x, EdgePropertyMap):
            x = self.u.copy_property(x, g=x.get_graph(), value_type="double")
        else:
            x = self.u.new_ep("double", val=x)

        self.x = x
        self.xmin_bound = x_range[0]
        self.xmax_bound = x_range[1]

        if theta is None:
            theta = self.u.new_vp("double")
        elif isinstance(theta, VertexPropertyMap):
            theta = self.u.copy_property(theta, g=theta.get_graph(),
                                         value_type="double")
        else:
            theta = self.u.new_vp("double", val=theta)

        self.theta = theta
        self.tmin_bound = theta_range[0]
        self.tmax_bound = theta_range[1]

        self.disable_xdist = disable_xdist
        self.disable_tdist = disable_tdist

        kwargs = kwargs.copy()
        for k in kwargs.keys():
            v = kwargs[k]
            if isinstance(v, PropertyMap):
                kwargs[k] = g.own_property(v)
            elif (isinstance(v, collections.abc.Iterable) and len(v) > 0 and
                  isinstance(v[0], PropertyMap)):
                kwargs[k] = [g.own_property(x) for x in v]

        self.rg = kwargs.pop("rg", None)
        self.elist = kwargs.pop("elist", None)

        self.params = kwargs
        self.os = [ns._get_any() for ns in self.s]
        self.ot = [nt._get_any() for nt in self.t]
        self.max_m = max_m
        self.nmax_extend = nmax_extend

        self._state = libinference.make_dynamics_state(self.bstate._state, self)
        self._dstate = self._make_state()
        self._state.set_dstate(self._dstate)

    def set_params(self, params):
        r"""Sets the model parameters via the dictionary ``params``."""
        self.params = dict(self.params, **params)
        self._state.set_params(self.params)

    def get_params(self, params):
        r"""Gets the model parameters via the dictionary ``params``."""
        return dict(self.params)

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, s=self.s, t=self.t, x=self.x,
                    x_range=(self.xmin_bound, self.xmax_bound), theta=self.theta,
                    theta_range=(self.tmin_bound, self.tmax_bound),
                    disable_xdist=self.disable_xdist, disable_tdist=self.disable_tdist,
                    rg=self.rg, elist=self.elist, max_m=self.max_m, **self.params)

    def __repr__(self):
        return "<%s object with %s, at 0x%x>" % \
            (self.__class__.__name__,
             self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _gen_eargs(self, args):
        uea = super()._gen_eargs(args)
        return libinference.dentropy_args(uea)

    @copy_state_wrap
    def entropy(self, latent_edges=True, density=False, aE=1, sbm=True,
                xdist=True, tdist=True, xdist_uniform=False,
                tdist_uniform=False, xl1=1, tl1=1, alpha=1, delta=1e-8,
                normal=False, mu=0, sigma=1, **kwargs):
        """Return the entropy, i.e. negative log-likelihood."""
        eargs = self._get_entropy_args(locals())
        S = self._state.entropy(eargs)
        if sbm:
            if self.nbstate is None:
                S += self.bstate.entropy(**kwargs)
            else:
                S += self.nbstate.entropy(**kwargs)
        return S

    def _get_elist(self, k, elist_args={}):
        elist_args = elist_args.copy()
        if self.elist is not None and self.elist[0] == k:
            return self.elist[1]
        elist, w = self.get_candidate_edges(k, **elist_args)
        self.elist = (k, elist, w)
        return elist

    def mcmc_sweep(self, beta=np.inf, niter=1, edge=True, edge_swap=True,
                   edge_multiflip=True, theta=True, theta_multiflip=True,
                   sbm=True, xvals=True, tvals=True, k=1, keep_elist=False,
                   verbose=False, elist_args={}, edge_mcmc_args={},
                   edge_swap_mcmc_args={}, edge_multiflip_mcmc_args={},
                   xvals_mcmc_args={}, theta_mcmc_args={},
                   theta_multiflip_mcmc_args={}, tvals_mcmc_args={},
                   sbm_mcmc_args={}, **kwargs):
        ret = (0, 0, 0)
        if self.tmax_bound != self.tmin_bound:
            if theta:
                if verbose:
                    print("theta_mcmc_sweep:")
                tret = self.theta_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                         **kwargs), **theta_mcmc_args))
                ret = (sum(x) for x in zip(ret, tret))
            if not self.disable_tdist:
                if theta_multiflip:
                    if verbose:
                        print("theta_multiflip_mcmc_sweep:")
                    tret = self.theta_multiflip_mcmc_sweep(**dict(dict(beta=beta,
                                                                       niter=10 * niter,
                                                                       **kwargs),
                                                                  **theta_multiflip_mcmc_args))
                    ret = (sum(x) for x in zip(ret, tret))
                if tvals:
                    if verbose:
                        print("tvals_sweep:")
                    eret = self.tvals_sweep(**dict(dict(beta=beta,
                                                        niter=10 * niter,
                                                        **kwargs),
                                                   **tvals_mcmc_args))
                    ret = (sum(x) for x in zip(ret, eret))
        if edge:
            if verbose:
                print("edge_mcmc_sweep:")
            edge_mcmc_args = dict(edge_mcmc_args, k=k, keep_elist=keep_elist,
                                  elist_args=elist_args)
            eret = self.edge_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                    **kwargs), **edge_mcmc_args))
            ret = (sum(x) for x in zip(ret, eret))
            keep_elist = True
        if edge_swap:
            if verbose:
                print("swap_mcmc_sweep:")
            edge_swap_mcmc_args = dict(edge_swap_mcmc_args, k=k,
                                       keep_elist=keep_elist,
                                       elist_args=elist_args)
            eret = self.swap_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                    **kwargs), **edge_swap_mcmc_args))
            ret = (sum(x) for x in zip(ret, eret))
            keep_elist = True
        if self.xmax_bound != self.xmin_bound and not self.disable_xdist:
            if edge_multiflip:
                if verbose:
                    print("edge_multiflip_sweep:")
                eret = self.edge_multiflip_mcmc_sweep(**dict(dict(beta=beta,
                                                                  niter=10 * niter,
                                                                  **kwargs),
                                                             **edge_multiflip_mcmc_args))
                ret = (sum(x) for x in zip(ret, eret))
            if xvals:
                if verbose:
                    print("xvals_sweep:")
                eret = self.xvals_sweep(**dict(dict(beta=beta, niter=10 * niter,
                                                    **kwargs),
                                               **xvals_mcmc_args))
                ret = (sum(x) for x in zip(ret, eret))
        if sbm:
            if verbose:
                print("sbm_mcmc_sweep:")
            kwargs = kwargs.copy()
            eargs = kwargs.get("entropy_args", {}).copy()
            for x in ["xdist", "tdist", "xdist_uniform", "tdist_uniform", "xl1",
                      "tl1", "sbm", "latent_edges", "density", "aE", "alpha"]:
                eargs.pop(x, None)
            kwargs["entropy_args"] = eargs
            sret = self.sbm_mcmc_sweep(**dict(dict(beta=beta, niter=niter,
                                                   **kwargs), **sbm_mcmc_args))
            ret = (sum(x) for x in zip(ret, sret))
        return tuple(ret)

    @mcmc_sweep_wrap
    def edge_mcmc_sweep(self, beta=np.inf, niter=1, k=1, elist_args={},
                        keep_elist=False, pold=1, pnew=1, pxu=.1, pm=1,
                        premove=1, maxiter=0, tol=1e-7, binary=False,
                        verbose=False, deterministic=None, sequential=True,
                        parallel=True, **kwargs):
        if deterministic is None:
            deterministic = np.isinf(beta)
        kwargs = dict(**kwargs)
        entropy_args = kwargs.pop("entropy_args", {})
        elist_args = dict(elist_args, entropy_args=entropy_args)
        entropy_args = self._get_entropy_args(entropy_args)
        if not keep_elist:
            self.elist = None
        elist = self._get_elist(k, elist_args)
        state = self._state
        if np.isinf(beta):
            pxu = 0
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_mcmc_sweep(mcmc_state, _get_rng())
        else:
            return self._state.mcmc_sweep(mcmc_state, _get_rng())


    @mcmc_sweep_wrap
    def swap_mcmc_sweep(self, beta=np.inf, niter=1, k=5, elist_args={},
                        keep_elist=False, pmove=1, ptmove=1, pswap=1,
                        verbose=False, deterministic=None, sequential=True,
                        parallel=True, **kwargs):
        if deterministic is None:
            deterministic = np.isinf(beta)
        kwargs = dict(**kwargs)
        entropy_args = kwargs.pop("entropy_args", {})
        elist_args = dict(elist_args, entropy_args=entropy_args)
        entropy_args = self._get_entropy_args(entropy_args)
        if not keep_elist:
            self.elist = None
        elist = self._get_elist(k, elist_args)
        state = self._state
        if parallel:
            sequential = True
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_swap_mcmc_sweep(mcmc_state, _get_rng())
        else:
            return self._state.swap_mcmc_sweep(mcmc_state, _get_rng())

    @mcmc_sweep_wrap
    def edge_multiflip_mcmc_sweep(self, beta=np.inf, pmerge=1, psplit=1,
                                  pmergesplit=1, pmovelabel=1, gibbs_sweeps=1,
                                  c=.1, niter=1, maxiter=0, tol=1e-7,
                                  entropy_args={}, accept_stats=None,
                                  verbose=False, **kwargs):
        E = self.u.num_edges()
        if E == 0:
            return (0, 0, 0)
        niter /= E
        gibbs_sweeps = max((gibbs_sweeps, 1))
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = self._get_entropy_args(entropy_args)
        mcmc_state.state = self._state

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        ret = self._state.multiflip_mcmc_sweep(mcmc_state, _get_rng())

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = np.zeros(len(nproposal),
                                                 dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return ret

    @mcmc_sweep_wrap
    def xvals_sweep(self, beta=np.inf, niter=100, maxiter=0, tol=1e-7,
                    min_size=1, entropy_args={}):
        xvals = self.get_xvals()
        r = min(niter/len(xvals), 1) if len(xvals) > 0 else 1
        niter = max(1, int(round(niter/len(xvals)))) if len(xvals) > 0 else 1
        ea = self._get_entropy_args(entropy_args)
        ret = (0, 0)
        for i in range(niter):
            eret = self._state.xvals_sweep(beta, r, maxiter, tol,
                                           min_size, ea, _get_rng())
            ret = tuple(sum(x) for x in zip(ret, eret))
        return ret

    def get_x(self):
        """Return latent edge weights."""
        return self.x.copy()

    def get_xvals(self):
        return self._state.get_xvals()

    @mcmc_sweep_wrap
    def theta_mcmc_sweep(self, beta=np.inf, niter=1, pold=1, pnew=1,
                         maxiter=0, tol=1e-7, verbose=False, deterministic=None,
                         sequential=True, parallel=True, **kwargs):
        if deterministic is None:
            deterministic = np.isinf(beta)
        entropy_args = kwargs.pop("entropy_args", {})
        entropy_args = self._get_entropy_args(entropy_args)
        state = self._state
        mcmc_state = DictState(dict(kwargs, **locals()))
        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        if parallel:
            return self._state.pseudo_mcmc_theta_sweep(mcmc_state, _get_rng())
        else:
            return self._state.mcmc_theta_sweep(mcmc_state, _get_rng())

    @mcmc_sweep_wrap
    def theta_multiflip_mcmc_sweep(self, beta=np.inf, pmerge=1, psplit=1, pmergesplit=1,
                                   pmovelabel=0, gibbs_sweeps=1, c=.1, niter=1,
                                   maxiter=0, tol=1e-7,
                                   entropy_args={}, accept_stats=None,
                                   verbose=False, **kwargs):

        niter /= self.u.num_vertices()
        gibbs_sweeps = max((gibbs_sweeps, 1))
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = self._get_entropy_args(entropy_args)
        mcmc_state.state = self._state

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        ret = self._state.multiflip_mcmc_theta_sweep(mcmc_state, _get_rng())

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = np.zeros(len(nproposal),
                                                 dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return ret

    @mcmc_sweep_wrap
    def tvals_sweep(self, beta=np.inf, niter=100, maxiter=0, min_size=1,
                    tol=1e-7, entropy_args={}):
        tvals = self.get_tvals()
        r = min(niter/len(tvals), 1) if len(tvals) > 0 else 1
        niter = max(1, int(round(niter/len(tvals)))) if len(tvals) > 0 else 1
        ea = self._get_entropy_args(entropy_args)
        ret = (0, 0)
        for i in range(niter):
            eret = self._state.tvals_sweep(beta, r, maxiter, tol,
                                           min_size, ea, _get_rng())
            ret = tuple(sum(x) for x in zip(ret, eret))
        return ret

    #@mcmc_sweep_wrap
    def xl1_mcmc_sweep(self, minval, maxval, beta=np.inf, niter=1, maxiter=0,
                       tol=1e-7, entropy_args={}):
        ea = self._get_entropy_args(entropy_args)
        x = ea.xl1
        for i in range(niter):
            nx, xcache = self.sample_xl1(minval, maxval, beta=beta,
                                         maxiter=maxiter, tol=tol,
                                         entropy_args=dict(entropy_args, xl1=x))
            dS = (self.entropy(**dict(entropy_args, xl1=nx)) -
                  self.entropy(**dict(entropy_args, xl1=x)))
            if np.isinf(beta):
                if dS < 0:
                    x = nx
            else:
                lf = self.sample_val_lprob(nx, xcache, beta=beta)
                lb = self.sample_val_lprob(x, xcache, beta=beta)
                if dS < 0 or np.random.random() < exp(-beta * dS + lb - lf):
                    x = nx
        return x

    #@mcmc_sweep_wrap
    def tl1_mcmc_sweep(self, minval, maxval, beta=np.inf, niter=1, maxiter=0,
                       tol=1e-7, entropy_args={}):
        ea = self._get_entropy_args(entropy_args)
        x = ea.tl1
        for i in range(niter):
            nx, xcache = self.sample_tl1(minval, maxval, beta=beta,
                                         maxiter=maxiter, tol=tol,
                                         entropy_args=dict(entropy_args, tl1=x))
            dS = (self.entropy(**dict(entropy_args, tl1=nx)) -
                  self.entropy(**dict(entropy_args, tl1=x)))
            if np.isinf(beta):
                if dS < 0:
                    n = nx
            else:
                lf = self.sample_val_lprob(nx, xcache, beta=beta)
                lb = self.sample_val_lprob(x, xcache, beta=beta)
                if dS < 0 or np.random.random() < exp(-beta * dS + lb - lf):
                    x = nx
        return x

    def delta_mcmc_sweep(self, minval, maxval, beta=np.inf, niter=1, maxiter=0,
                         tol=1e-7, entropy_args={}):
        ea = self._get_entropy_args(entropy_args)
        x = ea.tl1
        for i in range(niter):
            nx, xcache = self.sample_delta(minval, maxval, beta=beta,
                                           maxiter=maxiter, tol=tol,
                                           entropy_args=dict(entropy_args, tl1=x))
            dS = (self.entropy(**dict(entropy_args, tl1=nx)) -
                  self.entropy(**dict(entropy_args, tl1=x)))
            if np.isinf(beta):
                if dS < 0:
                    n = nx
            else:
                lf = self.sample_val_lprob(nx, xcache, beta=beta)
                lb = self.sample_val_lprob(x, xcache, beta=beta)
                if dS < 0 or np.random.random() < exp(-beta * dS + lb - lf):
                    x = nx
        return x

    def get_theta(self):
        """Return latent vertex thetas."""
        return self.theta.copy()

    def get_tvals(self):
        return self._state.get_tvals()

    def get_edge_prob(self, u, v, x, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of edge :math:`(u,v)`."""
        ea = self._get_entropy_args(entropy_args)
        return self._state.get_edge_prob(u, v, x, ea, epsilon)

    def get_edges_prob(self, elist, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of an edge list, with
        shape :math:`(E,2)`."""
        ea = self._get_entropy_args(entropy_args)
        elist = np.asarray(elist, dtype="double")
        probs = np.zeros(elist.shape[0])
        self._state.get_edges_prob(elist, probs, ea, epsilon)
        return probs

    def node_cov(self, u, v, toffset=True, pearson=False):
        r"""Return the covariance (or Pearson correlation if ``pearson ==
        True``) between nodes :math:`u` and :math:`v`, according to their
        time-series.
        """
        return self._state.node_cov(u, v, toffset, pearson)

    def node_TE(self, u, v):
        return self._state.node_TE(u, v)

    def node_MI(self, u, v):
        return self._state.node_MI(u, v)

    def get_candidate_edges(self, k=1, r=1, max_rk="k", epsilon=0.01,
                            c_stop=False, max_iter=0, knn=False, gradient=None,
                            h=1e-6, f_max_iter=10, tol=1e-6, allow_edges=False,
                            include_edges=True, nrandom=0, keep_all=False,
                            entropy_args={}, exact=False, return_graph=False,
                            keep_iter=False, verbose=False):
        g = Graph(self.g.num_vertices(), fast_edge_removal=True)
        w = g.new_ep("double")
        ei = g.new_ep("int32_t")
        N = g.num_vertices()
        if knn:
            M = k
        else:
            M = k * self.g.num_vertices()
        M = max((int(round(M)), 1))
        if max_rk is None:
            max_rk = g.num_vertices()
        elif max_rk == "k":
            max_rk = k
        ea = self._get_entropy_args(entropy_args)
        if gradient is None:
            gradient = len(self.get_xvals()) == 0
        n_tot = self._state.get_candidate_edges(g._Graph__graph, M, r, max_rk,
                                                epsilon, c_stop, max_iter,
                                                _prop("e", g, w),
                                                _prop("e", g, ei),
                                                keep_iter, ea, exact, knn,
                                                keep_all, gradient, h,
                                                f_max_iter, tol, allow_edges,
                                                include_edges, nrandom, verbose,
                                                _get_rng())
        if return_graph:
            if not knn and not self.u.is_directed():
                g.set_directed(False)
            if keep_iter:
                return g, w, ei, n_tot
            else:
                return g, w, n_tot
        elist = g.get_edges([w])
        idx = elist[:,2].argsort()
        a = elist[idx,2]
        elist = numpy.array((elist[idx,0], elist[idx,1]), dtype="int").T
        return elist, a

    def virtual_remove_edge(self, u, v, dm=1, entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.remove_edge_dS(int(u), int(v), dm, entropy_args)

    def virtual_add_edge(self, u, v, x, dm=1, entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.add_edge_dS(int(u), int(v), dm, x, entropy_args)

    def virtual_update_edge(self, u, v, nx, entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.update_edge_dS(int(u), int(v), nx, entropy_args)

    def remove_edge(self, u, v, dm=1):
        return self._state.remove_edge(int(u), int(v), dm)

    def add_edge(self, u, v, x, dm=1):
        return self._state.add_edge(int(u), int(v), dm, x)

    def update_edge(self, u, v, nx):
        return self._state.update_edge(int(u), int(v), nx)

    def bisect_x(self, u, v, maxiter=0, tol=1e-7, entropy_args={},
                 reversible=False, fb=False):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.bisect_x(int(u), int(v), maxiter, tol, entropy_args,
                                    reversible, fb, _get_rng())

    def bisect_t(self, v, maxiter=0, tol=1e-7, entropy_args={},
                 reversible=False, fb=False):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.bisect_t(int(v), maxiter, tol, entropy_args,
                                    reversible, fb, _get_rng())

    def sample_x(self, u, v, beta=np.inf, maxiter=0, tol=1e-7, entropy_args={}, fb=False):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_x(int(u), int(v), beta, maxiter, tol,
                                    entropy_args, fb, _get_rng())

    def sample_t(self, v, beta=np.inf, maxiter=0, tol=1e-7, entropy_args={}, fb=False):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_t(int(v), beta, maxiter, tol, entropy_args,
                                    fb, _get_rng())

    def sample_xl1(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                   entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_xl1(minval, maxval, beta, maxiter, tol,
                                      entropy_args, _get_rng())

    def sample_tl1(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                   entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_tl1(minval, maxval, beta, maxiter, tol,
                                      entropy_args, _get_rng())

    def sample_delta(self, minval, maxval, beta=np.inf, maxiter=0, tol=1e-7,
                     entropy_args={}):
        entropy_args = self._get_entropy_args(entropy_args)
        return self._state.sample_delta(minval, maxval, beta, maxiter, tol,
                                        entropy_args, _get_rng())

    def sample_val_lprob(self, x, xc, beta=np.inf):
        return self._state.sample_val_lprob(x, xc, beta)

    def quantize_x(self, delta):
        self._state.quantize_x(delta)

    def collect_marginal(self, g=None):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge :class:`~graph_tool.EdgePropertyMap`
            ``"eprob"``, containing the marginal probabilities for each edge.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.gp.count = g.new_gp("int", 0)
            g.ep.count = g.new_ep("int")
            g.ep.eprob = g.new_ep("double")

        if "x" not in g.ep:
            g.ep.xsum = g.new_ep("double")
            g.ep.x2sum = g.new_ep("double")
            g.ep.x = g.new_ep("double")
            g.ep.xdev = g.new_ep("double")

        u = self.get_graph()
        x = self.get_x()
        libinference.collect_xmarginal(g._Graph__graph,
                                       u._Graph__graph,
                                       _prop("e", u, x),
                                       _prop("e", g, g.ep.count),
                                       _prop("e", g, g.ep.xsum),
                                       _prop("e", g, g.ep.x2sum))
        g.gp.count += 1
        g.ep.eprob.fa = g.ep.count.fa / g.gp.count
        g.ep.x.fa = g.ep.xsum.fa / g.gp.count
        g.ep.xdev.fa = sqrt(g.ep.x2sum.fa / g.gp.count - g.ep.x.fa ** 2)
        return g

class BPBlockStateBase(DynamicsBlockStateBase, ABC):
    def __init__(self, *args, S_bp_0=None, S_bp_1=None, S_0=None, S_1=None,
                 **kwargs):
        DynamicsBlockStateBase.__init__(self, *args, **kwargs)
        self.S_bp_0 = S_bp_0
        self.S_0 = S_0
        self.S_bp_1 = S_bp_1
        self.S_1 = S_1
        if self.S_bp_0 is None or self.S_0 is None:
            if self.u.num_edges() == 0:
                state = self
            else:
                state = self.copy(S_bp_0=0, S_0=0)
                state.u.clear_edges()
                state = state.copy(S_bp_0=0, S_0=0)
            if self.S_bp_0 is None:
                self.S_bp_0 = state.get_S_bp()
            if self.S_0 is None:
                self.S_0 = state.get_S()
            if self.u.num_edges() == 0:
                self.update_alpha()

    def __getstate__(self):
        return dict(DynamicsBlockStateBase.__getstate__(self),
                    S_bp_0=self.S_bp_0, S_bp_1=self.S_bp_1, S_0=self.S_0,
                    S_1=self.S_1)

    def reset_alpha(self):
        self.S_bp_0 = self.get_S_bp()
        self.S_0 = self.get_S()
        self.S_bp_1 = self.S_1 = None

    def get_alpha(self):
        if self.S_bp_1 is None or self.S_1 is None:
            return 1
        if abs(self.S_1 - self.S_0) < 1e-6:
            return 1
        alpha = (self.S_bp_1 - self.S_bp_0) / (self.S_1 - self.S_0)
        if alpha < 0 or alpha > 1:
            return 1
        return alpha

    def get_beta(self):
        if self.S_bp_1 is None or self.S_1 is None:
            return 0
        alpha = self.get_alpha()
        beta = self.S_bp_1 - alpha * self.S_1
        return beta

    def update_alpha(self):
        self.S_bp_1 = self.get_S_bp()
        self.S_1 = self.get_S()

    def transform_s(self, s):
        return s

    def get_S_bp(self, **kwargs):
        bstate = self.get_bpstate(converge=False)
        bstate.converge(**kwargs)
        S = 0
        for s in self.s:
            s = self.transform_s(s)
            S -= bstate.log_prob(s)
        return S

    def get_S(self):
        return DynamicsBlockStateBase.entropy(self, sbm=False, tdist=False,
                                              xdist=False, density=False, xl1=0,
                                              tl1=0, alpha=-1, test=False)

    def mcmc_sweep(self, **kwargs):
        ret = DynamicsBlockStateBase.mcmc_sweep(self, **kwargs)
        self.update_alpha()
        return ret

    def _get_entropy_args(self, eargs, consume=False):
        if not consume:
            eargs = eargs.copy()
        if "alpha" in eargs and eargs["alpha"] < 0:
            eargs["alpha"] *= -1
        else:
            eargs["alpha"] = self.get_alpha()
        return DynamicsBlockStateBase._get_entropy_args(self, eargs,
                                                        consume=consume)

    @copy_state_wrap
    def entropy(self, **kwargs):
        alpha = kwargs.get("alpha", 1)
        if alpha < 0:
            return DynamicsBlockStateBase.entropy(self, **dict(kwargs,
                                                               alpha=-alpha,
                                                               test=False))
        S = DynamicsBlockStateBase.entropy(self, **dict(kwargs,
                                                        alpha=self.get_alpha(),
                                                        test=False))
        return S + self.get_beta()

    entropy.__signature__ = inspect.signature(DynamicsBlockStateBase.entropy)

    @abstractmethod
    def get_bpstate(**kwargs):
        pass

class EpidemicsBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, exposed=False, theta=-1, x_range=(-np.inf, 1e-8),
                 theta_range=(-np.inf, 1e-8), **kwargs):
        r"""Inference state for network reconstruction based on epidemic dynamics, using
        the stochastic block model as a prior.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Initial graph state.
        s : ``list`` of :class:`~graph_tool.VertexPropertyMap`
            Collection of time-series with node states over time. Each entry in
            this list must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the states of each node in each time
            step. A value of ``1`` means infected and ``0`` susceptible. Other
            values are allowed (e.g. for recovered), but their actual value is
            unimportant for reconstruction.

            If the parameter ``t`` is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        exposed : ``boolean`` (optional, default: ``False``)
            If ``True``, the data is supposed to come from a SEI, SEIR,
            etc. model, where a susceptible node (valued ``0``) first transits
            to an exposed state (valued ``-1``) upon transmission, before
            transiting to the infective state (valued ``1``).
        **kwargs : (optional)
            Remaining parameters to be passed to :class:`~graph_tool.inference.DynamicsBlockStateBase`

        References
        ----------
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`
        """

        super().__init__(g, s, exposed=exposed, theta=theta, x_range=x_range,
                         theta_range=theta_range, **kwargs)

    def _make_state(self):
        return libinference.make_epidemics_state(self._state, self.ot, self.os,
                                                 self.params)

    def get_t(self):
        """Return the latent edge transmission probabilities."""
        x = self.get_x()
        x.fa = 1-exp(x.fa)
        return x

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_epidemics_sweep(mcmc_state, self._state,
                                                 _get_rng())

class IsingBlockStateBase(ABC):
    @abstractmethod
    def __init__(self, g, s, has_zero=False, **kwargs):
        r"""Base state for network reconstruction based on the Ising model, using the
        stochastic block model as a prior. This class is not supposed to be
        instantiated directly. Instead one of its specialized subclasses must
        be used, which have the same signature: :class:`IsingGlauberBlockState`,
        :class:`PseudoIsingBlockState`, :class:`CIsingGlauberBlockState`,
        :class:`PseudoCIsingBlockState`.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Initial graph state.
        s : ``list`` of :class:`~graph_tool.VertexPropertyMap`
            Collection of time-series with node states over time. Each
            time-series must be a :class:`~graph_tool.VertexPropertyMap` with
            type ``vector<int>`` containing the Ising states (``-1`` or ``+1``)
            of each node in each time step.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        has_zero : bool (optional, default: ``False``)
            If ``True``, the three-state "Ising" model with values ``{-1,0,1}``
            is used.
        **kwargs : (optional)
            Remaining parameters to be passed to :class:`~graph_tool.inference.DynamicsBlockStateBase`

        References
        ----------
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`

        """
        pass

    def get_dyn_state(self, s=None):
        return IsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

    def transform_s(self, s):
        s = s.copy("vector<int>")
        for v in self.u.vertices():
            s[v].a = (s[v].a + 1)/2
        return s

class IsingGlauberBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, g, s, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the Ising
        model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details.
        """
        DynamicsBlockStateBase.__init__(self, g, s, has_zero=has_zero, **kwargs)

    def _make_state(self):
        return libinference.make_ising_glauber_state(self._state, self.ot,
                                                     self.os, self.params)

class CIsingGlauberBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, g, s, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the
        continuous Ising model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details. Note
        that in this case the ``s`` parameter should contain property maps of
        type ``vector<double>``, with values in the range :math:`[-1,1]`.
        """
        DynamicsBlockStateBase.__init__(self, g, s, has_zero=has_zero, **kwargs)

    def get_dyn_state(self, s=None):
        return CIsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

    def transform_s(self, s):
        return s

    def _make_state(self):
        return libinference.make_cising_glauber_state(self._state, self.ot,
                                                      self.os, self.params)

class PseudoIsingBlockState(IsingBlockStateBase, BPBlockStateBase):
    def __init__(self, g, s, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the equilibrium configurations of
        the Ising model, using the Pseudolikelihood approximation and the
        stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details. Note
        that in this model "time-series" should be interpreted as a set of
        uncorrelated samples, not a temporal sequence.
        """
        BPBlockStateBase.__init__(self, g, s, has_zero=has_zero, **kwargs)

    def get_bpstate(self, **kwargs):
        return IsingBPState(self.u, x=self.x,
                            theta=self.theta,
                            has_zero=self.params.get("has_zero"), **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_ising_state(self._state, self.ot,
                                                    self.os, self.params)

class PseudoCIsingBlockState(DynamicsBlockStateBase, IsingBlockStateBase):
    def __init__(self, g, s, has_zero=False, **kwargs):
        r"""State for network reconstruction based on the equilibrium configurations of
        the continuous Ising model, using the Pseudolikelihood approximation and
        the stochastic block model as a prior.

        See documentation for :class:`IsingBlockStateBase` for details. Note
        that in this model "time-series" should be interpreted as a set of
        uncorrelated samples, not a temporal sequence. Additionally, the ``s``
        parameter should contain property maps of type ``vector<double>``, with
        values in the range :math:`[-1,1]`.
        """
        DynamicsBlockStateBase.__init__(self, g, s, has_zero=has_zero, **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_cising_state(self._state, self.ot,
                                                     self.os, self.params)

    def get_dyn_state(self, s=None):
        return CIsingGlauberState(self.u, w=self.x, h=self.theta, s=s)

    def transform_s(self, s):
        return s

class VoterBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, x_range=(-np.inf, 1e-8), theta_range=(-np.inf, 1e-8),
                 **kwargs):
        super().__init__(g, s, x_range=x_range, theta_range=theta_range, **kwargs)

    def _make_state(self):
        return libinference.make_voter_state(self._state, self.ot, self.os,
                                             self.params)


class PseudoNormalBlockState(BPBlockStateBase):
    def __init__(self, g, s, fix_mean=True, positive=True, pslack=1e-6,
                 **kwargs):
        if fix_mean:
            s = [sx.copy("vector<double>") for sx in s]
            for sx in s:
                for v in g.vertices():
                    sx[v].a -= sx[v].a.mean()
        BPBlockStateBase.__init__(self, g, s, positive=positive, pslack=pslack,
                                  **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_normal_state(self._state, self.ot,
                                                     self.os, self.params)

    def get_dcov(self):
        N = self.g.num_vertices()
        S = np.zeros((N, N))
        M = 0
        for m in range(len(self.s)):
            M += len(self.s[m][0])
            for i in range(N):
                for j in range(N):
                    S[i,j] += np.dot(self.s[0][i].a,
                                     self.s[0][j].a)
        S /= M
        return S

    def get_precision(self):
        theta = self.theta.copy()
        w = self.x.copy()
        w.fa = abs(w.fa)
        pslack = self.params["pslack"]
        for v in self.u.vertices():
            k = v.out_degree(w)
            theta[v] =  min(theta[v], -log(k)/2-pslack) if k > 0 else theta[v]
        W = adjacency(self.u, self.x).tolil()
        for v in self.u.vertices():
            W[int(v), int(v)] = 1/exp(theta[v] * 2)
        return W

    def get_theta_shifted(self):
        h = self.theta.copy()
        w = self.x.copy()
        w.fa = np.abs(w.fa)
        k = self.u.degree_property_map("out", w)
        idx = k.a > 0
        pslack = self.params["pslack"]
        h.a[idx] = np.minimum(-np.log(k.a[idx])/2 - pslack, h.a[idx])
        return h

    def log_P_exact(self):
        S = self.get_dcov()
        W = self.get_precision().todense()
        N = W.shape[0]
        D = np.linalg.slogdet(W)[1]
        M = 0
        for m in range(len(self.s)):
            M += len(self.s[m][0])
        return float(-M*(S@W).trace()/2 + M/2*D - M*N*np.log(2*np.pi)/2)

    def log_Z_exact(self):
        W = self.get_precision().todense()
        N = W.shape[0]
        D = np.linalg.slogdet(W)[1]
        return -float(D/2 - N * np.log(2 * np.pi) /2)

    def get_dstate(self, s=None):
        h = self.get_theta_shifted()
        h.a = np.exp(h.a)
        return NormalState(self.u, w=self.x, h=h, s=s)

    def get_bpstate(self, **kwargs):
        theta = self.get_theta_shifted()
        theta.fa = np.exp(-2 * theta.fa)
        return NormalBPState(self.u, x=self.x, theta=theta, **kwargs)


class NormalGlauberBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, self_loops=False, **kwargs):
        super().__init__(g, s, self_loops=self_loops, **kwargs)

    def _make_state(self):
        return libinference.make_normal_glauber_state(self._state, self.ot,
                                                      self.os, self.params)

class LinearNormalBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, self_loops=True, **kwargs):
        super().__init__(g, s, self_loops=self_loops, **kwargs)

    def _make_state(self):
        return libinference.make_linear_normal_state(self._state, self.ot,
                                                     self.os, self.params)

class LVBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, self_loops=True, **kwargs):
        super().__init__(g, s, self_loops=self_loops, **kwargs)

    def _make_state(self):
        return libinference.make_lv_state(self._state, self.ot, self.os,
                                          self.params)
