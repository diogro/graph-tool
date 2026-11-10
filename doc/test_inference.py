#! /usr/bin/env python
# graph-tool -- a general graph modification and manipulation thingy
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
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Consistency tests for the SBM inference code.

All combinations of model variants (directed, overlapping, layered, weighted,
degree-corrected, ...) are exercised with all MCMC sweeps, nested states, and
the minimize_*() functions, also with vertex labels (``clabel`` and
``pclabel``, whose constraints are verified); the other partition models
(ranked, planted partition, modularity, normalized cut, Potts) with all sweeps
and minimize_blockmodel_dl(). The internal consistency checks are enabled
(``graph_tool.inference.set_test(True)``), which verify, among other things,
that the entropy differences reported by the sweeps match the actual ones, and
that copies of the states have the same entropy.

Each test is run as a separate job; jobs run in parallel, each with a
configurable number of OpenMP threads (since parallel code paths are part of
what is tested). The seed of each job is derived from its name, so that a job
is reproducible regardless of which other jobs are selected.

Usage: python test_inference.py [-j JOBS] [-t THREADS] [-k FILTER] [TESTS ...]
"""

import os
import sys
import zlib
import argparse
import itertools
import traceback
import dataclasses
from contextlib import contextmanager

import numpy as np

import test_runner
from test_runner import progress


# Per-worker data (see init())
gt = None
data = {}


def init(seed):
    """Per-worker initialization: graph and edge covariates."""
    global gt
    import graph_tool.all
    import graph_tool.inference
    gt = graph_tool.all
    graph_tool.inference.set_test(True)
    rng = np.random.default_rng(seed)
    g = gt.collection.data["football"].copy()
    data.update(g=g,
                ec=g.new_ep("int", rng.integers(0, 10, g.num_edges())),
                rec={"real-exponential":
                     g.new_ep("double", rng.random(g.num_edges())),
                     "real-normal":
                     g.new_ep("double", rng.normal(0, 10, g.num_edges()))})


def seed_job(seed):
    np.random.seed(seed)
    gt.seed_rng(seed)
    return np.random.default_rng(seed)


@dataclasses.dataclass(frozen=True)
class Model:
    """A model variant.

    ``layered`` is one of ``False``, ``"covariates"`` or ``"independent"``,
    and ``rec`` is ``None`` or an edge covariate type."""

    directed: bool
    overlap: bool = False
    layered: object = False
    rec: object = None
    deg_corr: bool = True

    @property
    def graph(self):
        return gt.GraphView(data["g"], directed=self.directed)

    @property
    def state_type(self):
        key = (self.overlap, bool(self.layered), self.rec is not None)
        return {(False, False, False): gt.BlockState,
                (False, False, True): gt.WeightedBlockState,
                (False, True, False): gt.LayeredBlockState,
                (False, True, True): gt.LayeredWeightedBlockState,
                (True, False, False): gt.OverlapBlockState,
                (True, False, True): gt.WeightedOverlapBlockState,
                (True, True, False): gt.LayeredOverlapBlockState,
                (True, True, True): gt.LayeredWeightedOverlapBlockState}[key]

    @property
    def state_args(self):
        args = dict(deg_corr=self.deg_corr)
        if self.layered:
            args.update(ec=data["ec"].copy(),
                        independent=self.layered == "independent")
        if self.rec is not None:
            args.update(rec=[data["rec"][self.rec].copy()],
                        rec_types=[self.rec])
        return args

    def partition(self, g, init, rng):
        """Initial partition: each vertex (or half-edge) in its own group, or
        random groups."""
        if self.overlap:
            if init == "singletons":
                s = g.edge_index.copy("int64_t")
                t = s.copy()
                t.fa += s.fa.max() + 1
                return gt.group_vector_property([s, t])
            b = g.new_ep("vector<int64_t>")
            for e in g.edges():
                b[e] = rng.integers(0, 10, 2)
            return b
        N = g.num_vertices()
        vals = np.arange(N) if init == "singletons" else rng.integers(0, 10, N)
        return g.new_vp("int64_t", vals=vals)

    def state(self, init="singletons", rng=None):
        g = self.graph
        return self.state_type(g, b=self.partition(g, init, rng),
                               **self.state_args)

    def nested_state(self):
        g = self.graph
        return gt.NestedBlockState(g, bs=[np.arange(g.num_vertices())] +
                                   [np.zeros(1)] * 6,
                                   base_state=self.state_type,
                                   base_state_args=self.state_args)


def product(where=None, **ranges):
    """All combinations of the given parameter ranges, as dicts, optionally
    filtered by ``where(params)``."""
    for vals in itertools.product(*ranges.values()):
        p = dict(zip(ranges.keys(), vals))
        if where is None or where(p):
            yield p


def model_of(p):
    fields = {f.name for f in dataclasses.fields(Model)}
    return Model(**{k: v for k, v in p.items() if k in fields})


class Steps:
    """Independent steps of a job: a failing step does not prevent the others
    from running, and the job fails at the end if any step failed."""

    def __init__(self):
        self.failed = []

    @contextmanager
    def step(self, name):
        try:
            yield
        except Exception as e:
            traceback.print_exc()
            self.failed.append(f"{name}: {type(e).__name__}: "
                               f"{str(e).splitlines()[0][:150]}")
        progress()

    def run(self, name, f):
        with self.step(name):
            print(f"\t{name}:", f(), flush=True)

    def check(self):
        if self.failed:
            raise AssertionError(f"{len(self.failed)} failed step(s): " +
                                 "; ".join(self.failed))


# Test registry: name -> (parameter generator, job function, number of steps)
tests = {}


def test(params, nsteps):
    def register(f):
        tests[f.__name__] = (params, f, nsteps)
        return f
    return register


# The degree DL kind is only relevant for degree-corrected models
dl_kind_ok = lambda p: p["deg_corr"] or p["degree_dl_kind"] == "uniform"

# parallel_thres=0 forces the parallel code paths regardless of graph size
par_args = lambda p: dict(parallel_thres=0 if p["parallel"] else sys.maxsize)


@test(lambda: product(directed=[False, True], overlap=[False, True],
                      layered=[False, "covariates", "independent"],
                      rec=[None, "real-exponential", "real-normal"],
                      deg_corr=[False, True], dl=[False, True],
                      degree_dl_kind=["uniform", "distributed"],
                      parallel=[False, True],
                      init=["singletons", "random"], where=dl_kind_ok),
      nsteps=lambda p: 4 if p["overlap"] else 8)
def sweeps(p, seed):
    """All sweeps on flat states, and on their (coupled) block states."""
    rng = seed_job(seed)
    model = model_of(p)
    eargs = dict(dl=p["dl"], degree_dl_kind=p["degree_dl_kind"],
                 beta_dl=0.95)
    pargs = par_args(p)
    sweeps = {"mcmc": lambda s: s.mcmc_sweep(beta=0, entropy_args=eargs),
              "multiflip":
              lambda s: s.multiflip_mcmc_sweep(beta=0, entropy_args=eargs,
                                               **pargs),
              "multilevel":
              lambda s: s.multilevel_mcmc_sweep(beta=0, M=3,
                                                entropy_args=eargs,
                                                parallel=p["parallel"],
                                                **pargs),
              "gibbs": lambda s: s.gibbs_sweep(beta=0, entropy_args=eargs)}

    steps = Steps()
    for name, sweep in sweeps.items():
        def run():
            state = model.state(p["init"], rng)
            return sweep(state), state.get_B()
        steps.run(name, run)

    if not model.overlap:
        state = bstate = None
        with steps.step("block state"):
            state = model.state(p["init"], rng)
            bstate = state._get_block_state(b=np.arange(state.get_B()))
        for name in ["mcmc", "multiflip", "gibbs"]:
            if bstate is None:
                progress()
                continue
            steps.run(f"block {name}",
                      lambda: (sweeps[name](bstate), bstate.get_B()))
    steps.check()


@test(lambda: product(directed=[False, True],
                      layered=[False, "covariates", "independent"],
                      rec=[None, "real-exponential", "real-normal"],
                      deg_corr=[True, False]),
      nsteps=lambda p: 2)
def nested(p, seed, niter=5):
    """Sweeps on nested states."""
    seed_job(seed)
    model = model_of(p)
    eargs = dict(beta_dl=0.95)

    def run(sweep):
        state = model.nested_state()
        for i in range(niter):
            ret = getattr(state, sweep)(beta=0, d=0.5, entropy_args=eargs)
            print("\t\t", ret, [s.get_B() for s in state.levels], flush=True)
        return state.entropy(**eargs)

    steps = Steps()
    for sweep in ["mcmc_sweep", "multiflip_mcmc_sweep"]:
        steps.run(f"nested {sweep}", lambda: run(sweep))
    steps.check()


@test(lambda: product(directed=[False, True], overlap=[False, True],
                      layered=[False, "covariates", "independent"],
                      rec=[None, "real-exponential", "real-normal"],
                      deg_corr=[False, True],
                      degree_dl_kind=["uniform", "distributed"],
                      parallel=[False, True], where=dl_kind_ok),
      nsteps=lambda p: 2)
def minimize(p, seed):
    """minimize_blockmodel_dl() and minimize_nested_blockmodel_dl()."""
    seed_job(seed)
    model = model_of(p)
    g, state_t, state_args = model.graph, model.state_type, model.state_args
    eargs = dict(degree_dl_kind=p["degree_dl_kind"], beta_dl=0.95)
    margs = dict(entropy_args=eargs, parallel=p["parallel"], **par_args(p))

    def flat():
        state = gt.minimize_blockmodel_dl(g, state=state_t,
                                          state_args=state_args,
                                          multilevel_mcmc_args=margs)
        return state.get_B(), state.entropy(**eargs)

    def nested():
        state = gt.minimize_nested_blockmodel_dl(g, base_state=state_t,
                                                 base_state_args=state_args,
                                                 multilevel_mcmc_args=margs)
        state.print_summary()
        return state.entropy(**eargs)

    steps = Steps()
    steps.run("minimize flat", flat)
    steps.run("minimize nested", nested)
    steps.check()


# Vertex labels (non-overlapping states only)

def label_args(p):
    """Random ``clabel`` and/or ``pclabel`` arguments, with 3 and 2 values."""
    rng = np.random.default_rng(zlib.crc32(p["labels"].encode()))
    g = data["g"]
    args = {}
    if p["labels"] in ["clabel", "both"]:
        args["clabel"] = g.new_vp("int64_t",
                                  vals=rng.integers(0, 3, g.num_vertices()))
    if p["labels"] in ["pclabel", "both"]:
        args["pclabel"] = g.new_vp("int64_t",
                                   vals=rng.integers(0, 2, g.num_vertices()))
    return args


def check_labels(state):
    """Check that no group mixes labels, at every level of a nested state
    (except the top one, which is unconstrained), and that the state can be
    copied, which validates the labels again."""
    levels = state.levels[:-1] if hasattr(state, "levels") else [state]
    for l, s in enumerate(levels):
        idx = s.vweight.fa > 0
        b = s.b.fa[idx]
        for c in [s.clabel, s.pclabel]:
            c = c.fa[idx]
            for r in np.unique(b):
                if len(np.unique(c[b == r])) > 1:
                    raise AssertionError(f"group {r} at level {l} "
                                         "mixes labels")
    state.copy()


@test(lambda: product(directed=[False, True],
                      layered=[False, "covariates", "independent"],
                      rec=[None, "real-normal"], deg_corr=[False, True],
                      labels=["clabel", "pclabel", "both"],
                      parallel=[False, True]),
      nsteps=lambda p: 11)
def labels(p, seed):
    """All sweeps (flat and nested) and the minimize_*() functions with
    vertex labels."""
    rng = seed_job(seed)
    model = model_of(p)
    g = model.graph
    state_args = dict(model.state_args, **label_args(p))
    pargs = par_args(p)
    margs = dict(parallel=p["parallel"], **pargs)

    def flat_state():
        # random groups, refined by the labels
        b = g.new_vp("int64_t", vals=rng.integers(0, 10, g.num_vertices()))
        for k, n in [("clabel", 3), ("pclabel", 2)]:
            if k in state_args:
                b.fa = b.fa * n + state_args[k].fa
        return model.state_type(g, b=b, **state_args)

    sweeps = {"mcmc": lambda s: s.mcmc_sweep(beta=0),
              "multiflip": lambda s: s.multiflip_mcmc_sweep(beta=0, **pargs),
              "multilevel M=3":
              lambda s: s.multilevel_mcmc_sweep(beta=0, M=3, **margs),
              "multilevel": lambda s: s.multilevel_mcmc_sweep(**margs),
              "gibbs": lambda s: s.gibbs_sweep(beta=0)}

    steps = Steps()
    for name, sweep in sweeps.items():
        def run():
            state = flat_state()
            ret = sweep(state)
            check_labels(state)
            return ret, state.get_B()
        steps.run(name, run)

    nested_sweeps = {"mcmc": lambda s: s.mcmc_sweep(beta=0, d=0.5),
                     "multiflip":
                     lambda s: s.multiflip_mcmc_sweep(beta=0, d=0.5),
                     "multilevel": lambda s: s.multilevel_mcmc_sweep(**margs)}
    for name, sweep in nested_sweeps.items():
        def run():
            state = gt.NestedBlockState(g, base_state=model.state_type,
                                        base_state_args=state_args)
            for i in range(3):
                ret = sweep(state)
                check_labels(state)
            return ret, [s.get_B() for s in state.levels]
        steps.run(f"nested {name}", run)

    def minimize_flat():
        state = gt.minimize_blockmodel_dl(g, state=model.state_type,
                                          state_args=state_args,
                                          multilevel_mcmc_args=margs)
        check_labels(state)
        return state.get_B(), state.entropy()

    def minimize_nested(**kwargs):
        state = gt.minimize_nested_blockmodel_dl(g,
                                                 base_state=model.state_type,
                                                 base_state_args=state_args,
                                                 multilevel_mcmc_args=margs,
                                                 **kwargs)
        check_labels(state)
        for i in range(3):
            state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
            check_labels(state)
        return state.entropy()

    steps.run("minimize flat", minimize_flat)
    steps.run("minimize nested", minimize_nested)
    steps.run("minimize nested simple_init",
              lambda: minimize_nested(simple_init=True))
    steps.check()


# Other partition models (flat states only)

other_models = ["ranked", "ranked-distance", "ranked-unique", "pp",
                "pp-uniform", "modularity", "normcut", "potts"]


def other_state_args(name, p, rng):
    """Graph, state type, state arguments and initial partition of one of the
    ``other_models``."""
    kind = name.split("-")[0]
    g = gt.GraphView(data["g"], directed=kind == "ranked")
    N = g.num_vertices()
    q = 10                                  # Potts states
    if p["init"] == "singletons":
        b = np.arange(N)
    else:
        b = rng.integers(0, q, N)
    b = g.new_vp("int64_t", vals=b)
    if kind == "ranked":
        args = dict(deg_corr=p["deg_corr"], distance=name == "ranked-distance",
                    unique=name == "ranked-unique")
        return g, gt.RankedBlockState, args, b
    if kind == "pp":
        args = dict(deg_corr=p["deg_corr"], uniform=name == "pp-uniform")
        return g, gt.PPBlockState, args, b
    if kind == "potts":
        f = rng.normal(0, 1, (q, q))
        return g, gt.PottsState, dict(f=(f + f.T) / 2), b
    return g, {"modularity": gt.ModularityState,
               "normcut": gt.NormCutState}[kind], {}, b


def other_ok(p):
    kind = p["model"].split("-")[0]
    # degree correction only exists for the ranked and planted partition
    # models, and Potts states have a fixed number of labels
    return ((p["deg_corr"] or kind in ["ranked", "pp"]) and
            (p["init"] == "random" or kind != "potts"))


@test(lambda: product(model=other_models, deg_corr=[True, False],
                      parallel=[False, True], init=["singletons", "random"],
                      where=other_ok),
      nsteps=lambda p: 5)
def other(p, seed):
    """All sweeps and minimize_blockmodel_dl() on the other partition
    models."""
    rng = seed_job(seed)
    g, state_t, state_args, b = other_state_args(p["model"], p, rng)
    pargs = par_args(p)
    sweeps = {"mcmc": lambda s: s.mcmc_sweep(beta=0),
              "multiflip": lambda s: s.multiflip_mcmc_sweep(beta=0, **pargs),
              "multilevel":
              lambda s: s.multilevel_mcmc_sweep(beta=0, M=3,
                                                parallel=p["parallel"],
                                                **pargs),
              "gibbs": lambda s: s.gibbs_sweep(beta=0)}

    steps = Steps()
    for name, sweep in sweeps.items():
        def run():
            state = state_t(g, b=b.copy(), **state_args)
            return sweep(state), len(np.unique(state.b.fa))
        steps.run(name, run)

    def minimize():
        state = gt.minimize_blockmodel_dl(g, state=state_t,
                                          state_args=state_args,
                                          multilevel_mcmc_args=dict(
                                              parallel=p["parallel"],
                                              **pargs))
        return len(np.unique(state.b.fa)), state.entropy()
    steps.run("minimize", minimize)
    steps.check()


def job_name(test, p):
    return f"{test}: " + " ".join(f"{k}={v}" for k, v in p.items())


def job_seed(seed, name):
    return (seed + zlib.crc32(name.encode())) % 2**31


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, os.cpu_count() // 2),
                        help="number of parallel jobs (default: %(default)s)")
    parser.add_argument("-t", "--threads", type=int, default=2,
                        help="OpenMP threads per job (default: %(default)s)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="random seed (default: %(default)s)")
    parser.add_argument("-k", "--filter", action="append", default=[],
                        help="only run jobs whose name contains all the "
                        "given substrings (can be repeated)")
    parser.add_argument("-l", "--log-dir", default="test_inference_logs",
                        help="directory for the per-job logs "
                        "(default: %(default)s)")
    parser.add_argument("--list", action="store_true",
                        help="only list the selected jobs")
    parser.add_argument("tests", nargs="*", default=list(tests),
                        choices=list(tests),
                        help="which tests to run (default: all)")
    args = parser.parse_args()

    jobs = []
    for test in args.tests:
        params, func, nsteps = tests[test]
        for p in params():
            name = job_name(test, p)
            if all(k in name for k in args.filter):
                jobs.append(test_runner.Job(name, func,
                                            (p, job_seed(args.seed, name)),
                                            nsteps(p)))

    if args.list:
        for job in jobs:
            print(job.name)
        return 0
    if not jobs:
        print("no jobs selected")
        return 0

    print(f"running {len(jobs)} jobs, {args.jobs} in parallel, with "
          f"{args.threads} OpenMP threads each...")
    failed = test_runner.run_jobs(jobs, workers=args.jobs,
                                  omp_threads=args.threads,
                                  log_dir=args.log_dir, init=init,
                                  init_args=(args.seed,),
                                  title="test_inference")
    return test_runner.summary(jobs, failed, args.log_dir)


if __name__ == "__main__":
    sys.exit(main())
