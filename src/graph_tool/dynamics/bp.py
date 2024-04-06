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

from .. import _prop, Graph, GraphView, _get_rng, EdgePropertyMap, \
    VertexPropertyMap, group_vector_property, _check_prop_vector, \
    _check_prop_scalar

from collections.abc import Iterable
from abc import ABC, abstractmethod
import numpy
import numpy.random

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_dynamics as lib_dynamics")

class BPBaseState(ABC):
    """Base class for belief propagation (BP) states."""

    @abstractmethod
    def __init__(self):
        pass

    def copy(self):
        """Return a copy of the state."""
        return type(self)(**self.__getstate__())

    @abstractmethod
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        self.__init__(**state)

    def iterate(self, niter=1, parallel=True, update_marginals=True):
        if parallel:
            delta = self._state.iterate_parallel(self.g._Graph__graph, niter)
        else:
            delta = self._state.iterate(self.g._Graph__graph, niter)
        if update_marginals:
            self.update_marginals()
        return delta

    def converge(self, epsilon=1e-8, max_niter=1000, update_marginals=True,
                 **kwargs):
        delta = epsilon + 1
        niter = 0
        while delta > epsilon and niter < max_niter:
            delta = self.iterate(**kwargs)
            niter += kwargs.get("niter", 1)
        self.update_marginals()
        return niter, delta

    def update_marginals(self):
        return self._state.update_marginals(self.g._Graph__graph)

    def log_Z(self):
        return self._state.log_Z(self.g._Graph__graph)

    def energy(self, s):
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return self._state.energies(self.g._Graph__graph,
                                        _prop("v", self.g, s))
        else:
            _check_prop_scalar(s)
            return self._state.energy(self.g._Graph__graph,
                                      _prop("v", self.g, s))

    def log_prob(self, s):
        H = BPBaseState.energy(self, s)
        lZ = self.log_Z()
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return -H - lZ * len(s[next(self.g.vertices())])
        else:
            return -H - lZ

    def marginal_log_prob(self, s):
        if "vector" in s.value_type():
            _check_prop_vector(s, scalar=True)
            return self._state.marginal_lprobs(self.g._Graph__graph,
                                               _prop("v", self.g, s))
        else:
            _check_prop_scalar(s)
            return self._state.marginal_lprob(self.g._Graph__graph,
                                              _prop("v", self.g, s))

    def sample(self, update_marginals=True, val_type="int"):
        if update_marginals:
            self.update_marginals()
        s = self.g.new_vp(val_type)
        self._state.sample(self.g._Graph__graph, _prop("v", self.g, s),
                           _get_rng())
        return s

class GenPottsBPState(BPBaseState):
    def __init__(self, g, f, x=1, theta=0, em=None, vm=None, marginal_init=False,
                 converge=True):

        self.g = g
        self.f = numpy.asarray(f, dtype="float")
        if not isinstance(x, EdgePropertyMap):
            x = g.new_ep("double", val=x)
        elif x.value_type() != "double":
            x = x.copy("double")
        self.x = x
        if not isinstance(theta, VertexPropertyMap):
            if isinstance(theta, Iterable):
                theta = g.new_vp("double", val=theta)
            else:
                theta = g.new_vp("double", val=[theta] * self.f.shape[0])
        elif theta.value_type() != "vector<double>":
            theta = theta.copy("vector<double>")
        self.theta = theta
        if em is None:
            em = g.new_ep("vector<double>")
        self.em = em
        if vm is None:
            vm = g.new_vp("vector<double>")
        self.vm = vm
        self._state = lib_dynamics.make_potts_bp_state(self.g._Graph__graph,
                                                       self.f,
                                                       _prop("e", g, self.x),
                                                       _prop("v", g, self.theta),
                                                       _prop("e", g, self.em),
                                                       _prop("v", g, self.vm),
                                                       marginal_init, _get_rng())
        if converge:
            self.converge()

    def __getstate__(self):
        return dict(g=self.g, f=self.f, x=self.x, theta=self.theta, em=self.em,
                    vm=self.vm)

class IsingBPState(GenPottsBPState):
    def __init__(self, g, x=1, theta=0, em=None, vm=None, marginal_init=False,
                 has_zero=False, converge=True):
        if not has_zero:
            f = [[-1,  1],
                 [ 1, -1]]
        else:
            f = [[-1,  0,  1],
                 [ 0,  0,  0],
                 [ 1,  0, -1]]
        if not isinstance(theta, VertexPropertyMap):
            if not has_zero:
                theta = g.new_vp("vector<double>", val=[theta, -theta])
            else:
                theta = g.new_vp("vector<double>", val=[theta, 0, -theta])
        elif theta.value_type() == "double":
            ntheta = theta.copy()
            ntheta.a *= -1
            if not has_zero:
                theta = group_vector_property([theta, ntheta])
            else:
                zero = g.new_vp("double")
                theta = group_vector_property([ntheta, zero, theta])
        elif theta.value_type() != "vector<double>":
            theta = theta.copy("vector<double>")
        self.has_zero = has_zero
        super().__init__(g=g, f=f, x=x, theta=theta, em=em, vm=vm,
                         marginal_init=marginal_init, converge=converge)

    def __getstate__(self):
        return dict(g=self.g, x=self.x, theta=self.theta, em=self.em,
                    vm=self.vm, has_zero=self.has_zero, coverge=False)

    def from_spin(self, s):
        s = s.copy()
        f = 2 if not self.has_zero else 1
        if "vector" in s.value_type():
            for v in self.g.vertices():
                s[v].a = (s[v].a + 1)/f
        else:
            s.fa = (s.fa + 1)/f
        return s

    def to_spin(self, s):
        s = s.copy()
        f = 2 if not self.has_zero else 1
        if "vector" in s.value_type():
            for v in self.g.vertices():
                s[v].a = f * s[v].a - 1
        else:
            s.fa = f * s.fa - 1
        return s

    def energy(self, s):
        return GenPottsBPState.energy(self, self.from_spin(s))

    def log_prob(self, s):
        return GenPottsBPState.log_prob(self, self.from_spin(s))

    def marginal_log_prob(self, s):
        return GenPottsBPState.marginal_log_prob(self, self.from_spin(s))

    def sample(self, update_marginals=True, val_type="int"):
        s = GenPottsBPState.sample(self, update_marginals=update_marginals,
                                   val_type=val_type)
        return self.to_spin(s)

class NormalBPState(BPBaseState):
    def __init__(self, g, x=1, mu=0, theta=1, em_m=None, em_s=None, vm_m=None,
                 vm_s=None, marginal_init=False, converge=True):

        self.g = g
        if not isinstance(x, EdgePropertyMap):
            x = g.new_ep("double", val=x)
        elif x.value_type() != "double":
            x = x.copy("double")
        self.x = x
        if not isinstance(mu, VertexPropertyMap):
            mu = g.new_vp("double", val=mu)
        elif theta.value_type() != "double":
            mu = theta.copy("double")
        self.mu = mu
        if not isinstance(theta, VertexPropertyMap):
            theta = g.new_vp("double", val=theta)
        elif theta.value_type() != "double":
            theta = theta.copy("double")
        self.theta = theta
        if em_m is None:
            em_m = g.new_ep("vector<double>")
        if em_s is None:
            em_s = g.new_ep("vector<double>")
        self.em_m = em_m
        self.em_s = em_s
        if vm_m is None:
            vm_m = g.new_vp("double")
        if vm_s is None:
            vm_s = g.new_vp("double")
        self.vm_m = vm_m
        self.vm_s = vm_s
        self._state = lib_dynamics.make_normal_bp_state(self.g._Graph__graph,
                                                        _prop("e", g, self.x),
                                                        _prop("v", g, self.mu),
                                                        _prop("v", g, self.theta),
                                                        _prop("e", g, self.em_m),
                                                        _prop("e", g, self.em_s),
                                                        _prop("v", g, self.vm_m),
                                                        _prop("v", g, self.vm_s),
                                                        marginal_init, _get_rng())
        if converge:
            self.converge()

    def __getstate__(self):
        return dict(g=self.g, x=self.x, mu=self.mu, theta=self.theta,
                    em_m=self.em_m, em_s=self.em_s, vm_m=self.vm_m,
                    vm_s=self.vm_s)

    def sample(self, update_marginals=True):
        return BPBaseState.sample(self, update_marginals=update_marginals,
                                  val_type="double")
