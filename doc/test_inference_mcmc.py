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

"""Statistical tests for the MCMC algorithms of the inference module.

- reverse: the reverse move probabilities computed with
  ``get_move_prob(..., reverse=True)`` match the forward ones after the move.
- moveprob: the moves sampled with ``sample_vertex_move()`` follow the
  distribution computed by ``get_move_prob()`` (KS test, per vertex).
- edges: the edge sampler of the SBM is normalized, and its samples follow
  the computed probabilities (chi-squared test).
- hists: the different MCMC algorithms sample from the same equilibrium
  distribution. A long chain is run for each algorithm (cached in
  ``OUTPUT/cache``), and the distributions of the entropy and of the effective
  number of groups are compared, via KS statistics over time, and histogram and
  KDE residues, in figures written to ``OUTPUT``. Chains whose final KS
  statistic exceeds the critical value are reported.

Jobs run in parallel; the comparisons are made by separate jobs, after the
chains are done. The seed of each job is derived from its name.

Usage: python test_inference_mcmc.py [-j JOBS] [-o OUTPUT] [TESTS ...]
"""

import os
import sys
import zlib
import argparse
import dataclasses

import numpy as np
import scipy.stats
import scipy.special

import test_runner
from test_runner import progress, set_total

# Per-worker modules (see init())
gt = plt = None


def init(test):
    global gt, plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    import graph_tool.all
    import graph_tool.inference
    plt = matplotlib.pyplot
    gt = graph_tool.all
    graph_tool.inference.set_test(test)


def seed_job(seed):
    np.random.seed(seed)
    gt.seed_rng(seed)


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)


def new_group(state, v):
    """Returns the label of an empty group, which is allocated if needed (the
    labels of groups that are not allocated are invalid)."""
    return state.sample_vertex_move(v, d=1.)


def ordered_state(directed, seed, B=10):
    """Non-degree-corrected SBM fit of the football network with B groups,
    with ordered group labels. Depends only on the seed."""
    seed_job(seed)
    g = gt.collection.data["football"].copy()
    g.set_directed(directed)
    state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False),
                                      multilevel_mcmc_args=dict(B_min=B,
                                                                B_max=B))
    return state.copy(b=gt.order_partition_labels(state.b.a))


# Test 1: reverse move probabilities

def reverse(directed, seed, output, c=0.01, d=0.1, tol=1e-8):
    state = ordered_state(directed, seed)
    B = state.get_B()
    set_total(state.g.num_vertices())
    fig = plt.figure()
    errors = []
    for v in state.g.vertices():
        r = state.b[v]
        res = []
        for s in list(range(B)) + [new_group(state, v)]:
            pf = state.get_move_prob(v, s, c=c, d=d, reverse=True)
            state.move_vertex(v, s)
            pb = state.get_move_prob(v, r, c=c, d=d)
            state.move_vertex(v, r)
            res.append(pf - pb)
            if abs(pf - pb) > tol:
                errors.append((int(v), r, s, pf, pb))
                print("wrong reverse probability:", errors[-1])
        plt.plot(res)
        progress()
    plt.ylim(-.1, .1)
    savefig(fig, f"{output}/test_mcmc_reverse_prob_directed{directed}.pdf")
    if errors:
        raise AssertionError(f"{len(errors)} wrong reverse probabilities, "
                             f"e.g. (v, r, s, pf, pb) = {errors[0]}")


# Test 2: sampled moves vs. computed move probabilities

def moveprob(directed, vs, seed, output, c=0.1, d=0.1, pmin=1e-3,
             max_samples=1000000):
    state = ordered_state(directed, seed)
    seed_job(seed + vs[0] + 1)
    B = state.get_B()
    set_total(len(vs))
    failed = []
    for v in vs:
        # computed probabilities; the last entry is a new group
        mp = np.exp([state.get_move_prob(v, s, c=c, d=d)
                     for s in list(range(B)) + [new_group(state, v)]])
        if abs(mp.sum() - 1) > 1e-8:
            failed.append((v, f"total probability {mp.sum()}"))
        n = min(int(200 / mp[mp > 0].min()), max_samples)

        # sampled moves (labels of empty groups are new groups)
        samples = np.array([state.sample_vertex_move(v, c=c, d=d)
                            for i in range(n)])
        samples[samples >= B] = B   # all B groups are occupied
        expected = np.repeat(np.arange(B + 1),
                             np.random.multinomial(n, mp / mp.sum()))

        p = scipy.stats.ks_2samp(samples, expected)[1]
        print("vertex:", v, "samples:", n, "p-value:", p, flush=True)
        if p < pmin:
            failed.append((v, f"p-value {p:g}"))
            fig = plt.figure()
            plt.plot(mp - np.bincount(samples, minlength=B + 1) / n)
            plt.ylim(-.1, .1)
            savefig(fig, f"{output}/test_mcmc_move_prob_directed{directed}"
                    f"_v{v}.pdf")
        progress()
    if failed:
        raise AssertionError("move probabilities do not match the computed "
                             "distribution for (vertex, reason): " +
                             str(failed))


# Test 3: edge sampler

def edges(directed, deg_corr, seed, output, n_samples, tol=1e-6, pmin=1e-3):
    import graph_tool
    seed_job(seed)
    g = gt.collection.data["football"].copy()
    g.set_directed(directed)
    # lstate must be kept alive: the edge sampler refers to its block state
    lstate = gt.LatentMultigraphBlockState(g,
                                           state_args=dict(deg_corr=deg_corr))
    for i in range(1000):
        lstate.mcmc_sweep(niter=10)
    state = lstate.bstate
    esampler = state._state.get_edge_sampler(False)
    rng = graph_tool._get_rng()

    N = g.num_vertices()
    count = np.zeros((N, N), dtype="int")
    set_total(n_samples)
    for i in range(n_samples):
        u, v = esampler.sample(rng)
        if not directed and v < u:
            u, v = v, u
        count[u, v] += 1
        if (i + 1) % 100000 == 0:
            progress(100000)

    pairs = [(u, v) for u in range(N) for v in range(N)
             if directed or u <= v]
    Lp = np.zeros(len(pairs))
    for i, (u, v) in enumerate(pairs):
        e = state.g.edge(u, v)
        Lp[i] = esampler.log_prob(u, v, 0 if e is None else state.eweight[e])
    observed = np.array([count[u, v] for u, v in pairs], dtype="float")

    norm = scipy.special.logsumexp(Lp)
    print("log-normalization of the computed probabilities:", norm)

    # chi-squared test, merging the pairs with small expected counts
    expected = np.exp(Lp) * n_samples
    idx = expected >= 5
    o, x = observed[idx], expected[idx]
    if not idx.all():
        o = np.append(o, observed[~idx].sum())
        x = np.append(x, expected[~idx].sum())
    chi2, p = scipy.stats.chisquare(o, x * o.sum() / x.sum())
    print("chi2:", chi2, "p-value:", p)

    fig = plt.figure()
    with np.errstate(divide="ignore"):
        Ls = np.log(observed / n_samples)
    plt.plot(Ls, Lp, "o")
    plt.plot(Lp, Lp, "-")
    plt.xlabel("measured log-probability")
    plt.ylabel("computed log-probability")
    savefig(fig, f"{output}/test_mcmc_edge_sample_directed{directed}"
            f"_deg_corr{deg_corr}.pdf")

    errors = []
    if abs(norm) > tol:
        errors.append(f"edge sampler not normalized: {norm:g}")
    if p < pmin:
        errors.append("samples do not match the computed probabilities "
                      f"(chi2 p-value {p:g})")
    if errors:
        raise AssertionError("; ".join(errors))
    return norm, p


# Test 4: equilibrium distributions of the different algorithms

# name: (sweep method, arguments); each recorded step performs ``istep``
# iterations of the sweep. Multilevel MCMC is a pseudo-MCMC (detailed balance
# is not preserved), so it is not expected to sample from the same
# distribution, and is only compared if requested.
algorithms = {"metropolis-c0.1": ("mcmc_sweep", dict(c=0.1)),
              "metropolis-cinf": ("mcmc_sweep", dict(c=np.inf)),
              "gibbs": ("gibbs_sweep", dict()),
              "multiflip-c0.1": ("multiflip_mcmc_sweep",
                                 dict(c=0.1, gibbs_sweeps=3)),
              "multiflip-cinf": ("multiflip_mcmc_sweep",
                                 dict(c=np.inf, gibbs_sweeps=3)),
              "multilevel": ("multilevel_mcmc_sweep",
                             dict(c=0.1, mh_sweeps=3, M=2, gibbs=False))}

default_algorithms = [a for a in algorithms if a != "multilevel"]

models = ["sbm", "nsbm", "pp", "modularity"]
graphs = ["small", "lesmis", "karate"]


def make_small_graph(path, seed):
    """Two cliques and a bipartite-like random part."""
    import graph_tool.all as gt
    rng = np.random.default_rng(seed)
    g = gt.graph_union(gt.complete_graph(4), gt.complete_graph(4))
    g.add_edge(3, 4)
    vs = list(g.add_vertex(8))
    for i in range(3 * 8):
        g.add_edge(vs[rng.integers(4)], vs[rng.integers(4) + 4])
    g.save(path)


@dataclasses.dataclass(frozen=True)
class Comparison:
    """The chains of all algorithms for one graph and model."""
    graph: str
    directed: bool
    model: str
    wait: int

    @property
    def name(self):
        return (f"{self.graph} directed={self.directed} model={self.model} "
                f"wait={self.wait}")

    @property
    def prefix(self):
        return (f"{self.graph}_model{self.model}_directed{self.directed}"
                f"_wait{self.wait}")

    def chains(self, algos):
        return [Chain(self, algo) for algo in algos]


@dataclasses.dataclass(frozen=True)
class Chain:
    comp: Comparison
    algo: str

    @property
    def name(self):
        return f"{self.comp.name} {self.algo}"

    def path(self, output):
        return f"{output}/cache/{self.comp.prefix}_{self.algo}.npz"

    def load(self, output):
        with np.load(self.path(output)) as h:
            return {"S": h["S"], "Be": h["Be"]}

    def state(self, output):
        c = self.comp
        if c.graph == "small":
            g = gt.load_graph(f"{output}/g_small.gt")
        else:
            g = gt.collection.data[c.graph].copy()
        g.set_directed(c.directed)
        b = np.arange(g.num_vertices())   # start from singletons
        if c.model == "nsbm":
            state = gt.NestedBlockState(g, bs=[b] + [np.zeros(1)] * 6)
            return state, lambda: state.levels[0].get_Be()
        state_t = {"sbm": gt.BlockState, "pp": gt.PPBlockState,
                   "modularity": gt.ModularityState}[c.model]
        state = state_t(g, b=b)
        return state, state.get_Be


def run_chain(chain, seed, output, istep=5):
    seed_job(seed)
    state, get_Be = chain.state(output)
    method, args = algorithms[chain.algo]
    beta = .3 if chain.comp.graph == "small" else .9
    sweep = getattr(state, method)
    wait = chain.comp.wait
    S, Be = np.zeros(wait), np.zeros(wait)
    set_total(wait)
    for t in range(wait):
        sweep(beta=beta, niter=istep, **args)
        S[t], Be[t] = state.entropy(), get_Be()
        progress()
    path = chain.path(output)
    np.savez(path + ".tmp.npz", S=S, Be=Be)
    os.replace(path + ".tmp.npz", path)


def ks_critical(n, m, alpha_coeff=1.63):
    """Critical value of the two-sample KS statistic (alpha ~ 0.01)."""
    return alpha_coeff * np.sqrt((n + m) / (n * m))


def compare(comp, algos, output, replot):
    hists = {}
    for chain in comp.chains(algos):
        if os.path.exists(chain.path(output)):
            hists[chain.algo] = chain.load(output)
    if len(hists) < 2:
        raise ValueError(f"only {len(hists)} chain(s) available")

    set_total(8)
    deviating = []
    for var, label in [("Be", r"$B_e$"), ("S", "Entropy")]:
        vals = {a: h[var] for a, h in hists.items()}
        allv = np.concatenate(list(vals.values()))
        base = f"{output}/test_mcmc_{comp.prefix}_{var}"

        # KS statistic of each chain against all others, over time
        ts = np.unique(np.logspace(1, np.log10(comp.wait), 100, dtype="int"))
        fig = plt.figure(figsize=(10 * 4/3, 10))
        for a, x in vals.items():
            others = [y for b, y in vals.items() if b != a]
            Ds = [scipy.stats.ks_2samp(x[:t], np.concatenate([y[:t] for y
                                                              in others]))[0]
                  for t in ts]
            Dc = [ks_critical(t, t * len(others)) for t in ts]
            plt.loglog(ts, Ds, label=a)
            print(f"{var} {a}: final KS D={Ds[-1]:g}, D_c={Dc[-1]:g}")
            if Ds[-1] > Dc[-1]:
                deviating.append((var, a, Ds[-1], Dc[-1]))
        plt.plot(ts, Dc, "--", label="D_c")
        plt.legend(loc="best")
        plt.xlabel("Time")
        plt.ylabel("KS statistics")
        if replot or not os.path.exists(base + "-KS.pdf"):
            savefig(fig, base + "-KS.pdf")
        else:
            plt.close(fig)
        progress()

        # histogram residues
        bins = np.linspace(allv.min(), allv.max(), 60)
        hs = {a: np.histogram(x, bins=bins, density=True)[0]
              for a, x in vals.items()}
        plot_residues(base + "-res.pdf", replot, bins[:-1], hs, label,
                      plt.step)
        progress()

        # KDE and KDE residues
        x = np.linspace(allv.min(), allv.max(), 1000)
        ys = {}
        for a, v in vals.items():
            try:
                ys[a] = scipy.stats.gaussian_kde(v)(x)
            except np.linalg.LinAlgError:   # constant values
                pass
        if replot or not os.path.exists(base + "-kde.pdf"):
            fig = plt.figure(figsize=(10 * 4/3, 10))
            for a, y in ys.items():
                plt.plot(x, y, "-", linewidth=.8, label=a)
            plt.legend(loc="best")
            plt.xlabel(label)
            plt.ylabel("Prob. density")
            savefig(fig, base + "-kde.pdf")
        progress()
        plot_residues(base + "-kde-res.pdf", replot, x, ys, label, plt.plot)
        progress()
    return deviating


def plot_residues(path, replot, x, ys, xlabel, plot):
    """Plot the deviations of each curve from the mean of all curves."""
    if not ys or (os.path.exists(path) and not replot):
        return
    fig = plt.figure(figsize=(10 * 4/3, 10))
    ymean = np.mean(list(ys.values()), axis=0)
    ymax = max(y.max() for y in ys.values())
    for a, y in ys.items():
        plot(x, y - ymean, label=a)
    plt.ylim(-ymax/10, ymax/10)
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel("Prob. density residue")
    savefig(fig, path)


tests = ["reverse", "moveprob", "edges", "hists"]


def job_seed(seed, name):
    return (seed + zlib.crc32(name.encode())) % 2**31


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, os.cpu_count() // 2),
                        help="number of parallel jobs (default: %(default)s)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="OpenMP threads per job (default: %(default)s)")
    parser.add_argument("-s", "--seed", type=int, default=43,
                        help="random seed (default: %(default)s)")
    parser.add_argument("-o", "--output", default="test_mcmc",
                        help="output directory for figures, cached chains "
                        "and logs (default: %(default)s)")
    parser.add_argument("-k", "--filter", action="append", default=[],
                        help="only run jobs whose name contains all the "
                        "given substrings (can be repeated)")
    parser.add_argument("--wait", type=int, nargs="+", default=[50000],
                        help="chain lengths for the 'hists' test "
                        "(default: %(default)s)")
    parser.add_argument("--graphs", nargs="+", default=graphs,
                        help="graphs for the 'hists' test "
                        "(default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=models,
                        choices=models,
                        help="models for the 'hists' test "
                        "(default: %(default)s)")
    parser.add_argument("--algorithms", nargs="+",
                        default=default_algorithms, choices=list(algorithms),
                        help="algorithms for the 'hists' test (default: all "
                        "but multilevel, which is a pseudo-MCMC)")
    parser.add_argument("--edge-samples", type=int, default=10000000,
                        help="number of samples for the 'edges' test "
                        "(default: %(default)s)")
    parser.add_argument("--rerun", action="store_true",
                        help="rerun cached chains of the 'hists' test")
    parser.add_argument("--replot", action="store_true",
                        help="overwrite existing figures")
    parser.add_argument("--test", action="store_true",
                        help="enable the internal consistency checks of the "
                        "inference module (slow)")
    parser.add_argument("--list", action="store_true",
                        help="only list the selected jobs")
    parser.add_argument("tests", nargs="*", default=tests, choices=tests,
                        help="which tests to run (default: all)")
    args = parser.parse_args()

    output = args.output
    os.makedirs(f"{output}/cache", exist_ok=True)
    log_dir = f"{output}/logs"
    run_args = dict(workers=args.jobs, omp_threads=args.threads,
                    init=init, init_args=(args.test,))

    def job(name, func, make_args):
        """A job if selected; ``make_args(seed)`` gives its arguments."""
        if not all(k in name for k in args.filter):
            return []
        return [test_runner.Job(name, func,
                                make_args(job_seed(args.seed, name)))]

    jobs = []
    # the state used by these tests depends only on the base seed
    if "reverse" in args.tests:
        for directed in [True, False]:
            jobs += job(f"reverse: directed={directed}", reverse,
                        lambda s: (directed, args.seed, output))
    if "moveprob" in args.tests:
        N, chunk = 115, 10     # football
        for directed in [False, True]:
            for i in range(0, N, chunk):
                vs = list(range(i, min(i + chunk, N)))
                jobs += job(f"moveprob: directed={directed} "
                            f"v={vs[0]}-{vs[-1]}", moveprob,
                            lambda s: (directed, vs, args.seed, output))
    if "edges" in args.tests:
        for directed in [True, False]:
            for deg_corr in [True, False]:
                jobs += job(f"edges: directed={directed} "
                            f"deg_corr={deg_corr}", edges,
                            lambda s: (directed, deg_corr, s, output,
                                       args.edge_samples))

    comps = []
    if "hists" in args.tests:
        if not os.path.exists(f"{output}/g_small.gt"):
            make_small_graph(f"{output}/g_small.gt", args.seed)
        comps = [Comparison(g, d, m, w) for w in args.wait
                 for g in args.graphs for d in [False, True]
                 for m in args.models]
        for comp in comps:
            for chain in comp.chains(args.algorithms):
                if os.path.exists(chain.path(output)) and not args.rerun:
                    continue
                jobs += job(f"hists: {chain.name}", run_chain,
                            lambda s: (chain, s, output))

    if args.list:
        for j in jobs:
            print(j.name)
        for comp in comps:
            print(f"compare: {comp.name}")
        return 0

    failed = []
    if jobs:
        print(f"running {len(jobs)} jobs, {args.jobs} in parallel...")
        failed += test_runner.run_jobs(jobs, title="test_inference_mcmc",
                                       log_dir=log_dir, **run_args)

    # comparisons of the chains, after all of them are done
    cjobs = []
    for comp in comps:
        cjobs += job(f"compare: {comp.name}", compare,
                     lambda s: (comp, args.algorithms, output, args.replot))
    if cjobs:
        print(f"comparing the chains of {len(cjobs)} graph/model pairs...")
        failed += test_runner.run_jobs(cjobs, title="comparisons",
                                       log_dir=f"{log_dir}/compare",
                                       **run_args)
        print("\nchains whose distribution deviates from the others "
              "(final KS statistic D > D_c):")
        deviating = [(j.name[9:], *d) for j in cjobs for d in j.result or []]
        for name, var, a, D, Dc in deviating:
            print(f"  {name} {var} {a}: D={D:.3g} D_c={Dc:.3g}")
        if not deviating:
            print("  (none)")

    all_jobs = jobs + cjobs
    if not all_jobs:
        print("no jobs selected")
        return 0
    return test_runner.summary(all_jobs, failed, log_dir)


if __name__ == "__main__":
    sys.exit(main())
