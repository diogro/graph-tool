(reconstruction_dynamics)=

# Network reconstruction from dynamics and behavior

In some cases, direct measurements of the edges of a network are either
impossible, or can be done only at significant experimental cost. In such
situations, we need to infer the network of interactions from the observed
functional behavior {cite}`rec-peixoto_network_2025`. In `graph_tool` this can be done
for a variety of static and dynamical models, where the data $\boldsymbol
X$ is a $N\times M$ matrix containing $M$ samples (or "features")
for each node.

For static models we have that the likelihood is given by

$$
P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) =
\prod_m P(\boldsymbol X_{m}|\boldsymbol W, \boldsymbol\theta),
$$

where each sample is assumed to be i.i.d., whereas for dynamic models
$\boldsymbol X$ is assumed to contain the states of a Markov chain with
likelihood

$$
P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) =
\prod_m P(\boldsymbol X_{m}|X_{m-1},\boldsymbol W, \boldsymbol\theta).
$$

In both cases above, $\boldsymbol W$ represents the weighted adjacency
matrix and $\boldsymbol \theta$ are additional node parameters.

The inference procedure consists in the maximum *a posteriori* (MAP) estimation:

$$
\begin{aligned}
\widehat{\boldsymbol W}, \widehat{\boldsymbol\theta} &=
\underset{\boldsymbol W, \boldsymbol\theta}{\operatorname{arg max}}\; P(\boldsymbol W,\boldsymbol\theta | \boldsymbol X) \\
 &= \underset{\boldsymbol W, \boldsymbol\theta}{\operatorname{arg max}}\; \log P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) + \log P(\boldsymbol W,\boldsymbol\theta),
\end{aligned}
$$

where the prior $P(\boldsymbol W,\boldsymbol\theta)$ implements regularization.

In `graph-tool`, a variety of models are implemented, as given by the following table:

```{csv-table}
:header: Name | State | Data domain | Kind
:widths: 10, 5, 5, 5
:delim: "|"
:align: center

[Multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | {class}`~graph_tool.inference.PseudoNormalBlockState` | $\mathbb{R}$ | Static
[Dynamic multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | {class}`~graph_tool.inference.NormalGlauberBlockState` | $\mathbb{R}$ | Dynamic
[Generalized Lotka-Volterra](https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation) | {class}`~graph_tool.inference.LVBlockState` | $\mathbb{R}^+$ | Dynamic
[Compartmental epidemics](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) | {class}`~graph_tool.inference.EpidemicsBlockState` | $\{0, 1\}$ | Dynamic
[Ising model](https://en.wikipedia.org/wiki/Ising_model) | {class}`~graph_tool.inference.PseudoIsingBlockState` | $\{-1, +1\}$ | Static
[Kinetic Ising model](https://en.wikipedia.org/wiki/Glauber_dynamics) | {class}`~graph_tool.inference.IsingGlauberBlockState` | $\{-1, +1\}$ | Dynamic
Continuous Ising model | {class}`~graph_tool.inference.PseudoCIsingBlockState` | $[-1,1]$ | Static
Continuous kinetic Ising model | {class}`~graph_tool.inference.CIsingGlauberBlockState` | $[-1,1]$ | Dynamic
[Generalized Potts model](https://en.wikipedia.org/wiki/Potts_model) | {class}`~graph_tool.inference.PseudoPottsBlockState` | $\{0,\dots,q-1\}$ | Static
Kinetic generalized Potts model | {class}`~graph_tool.inference.PottsGlauberBlockState` | $\{0,\dots,q-1\}$ | Dynamic
```

## Reconstruction with synthetic data

Here we demonstrate how reconstruction is performed with a simple example, based
on the simulation of the [Kinetic Ising model](https://en.wikipedia.org/wiki/Glauber_dynamics) on the {ns}`dolphins` network (we can use the
{class}`~graph_tool.dynamics.IsingGlauberState` class for this purpose):

```{testsetup} dynamics
mkchdir(DOC_DIR + "/demos/reconstruction_indirect/output")
np.random.seed(42)
gt.seed_rng(42)
```

```{testcode} dynamics
:hide:

g = gt.collection.data["dolphins"]
gt.graph_draw(g, g.vp.pos, output="dolphin.svg")
```

::::{note}
:class: margin

:::{figure} output/dolphin.svg
This is the {ns}`dolphins` network we will attempt to reconstruct from a
the kinetic Ising model.
:::
::::

```{testcode} dynamics
import numpy as np

g = gt.collection.data["dolphins"]
E = g.num_edges()
N = g.num_vertices()
w = g.new_ep("double", vals=np.random.normal(N/(2*E), .05, E))  # True edge weights
istate = gt.IsingGlauberState(g, w=w)

M = 1000
X = []
for m in range(M):
    istate.iterate_sync()
    X.append(istate.get_state().a.copy())
X = np.array(X).T
```

At this point the array `X` contains our simulated data. We will now
reconstruct the original network from it using a nonparametric approach, which
consists of initializing an {class}`~graph_tool.inference.IsingGlauberBlockState`
object, and iterating over
{meth}`~graph_tool.inference.IsingGlauberBlockState.mcmc_sweep` until sufficient
convergence is reached:

:::{note}
:class: margin

When doing reconstruction, the default for
{meth}`~graph_tool.inference.IsingGlauberBlockState.mcmc_sweep` is the
inverse "temperature" `beta=1`, meaning that this function performs
sweeps of a MCMC. Here we are interested in a greedy optimization instead, so
we need to pass `beta=numpy.inf`.
:::

```{testcode} dynamics
state = gt.IsingGlauberBlockState(X, directed=False, self_loops=False)

for i in range(10):
    delta, *_ = state.mcmc_sweep(beta=np.inf, niter=10)
    print(delta)
```

This outputs the following deltas, indicating that convergence was achieved.

```{testoutput} dynamics
 -4686.196037213296
 -3286.049921543921
 -818.8563094596482
 -115.3160422147071
 -19.18297219710969
 -2.427082044189543
 -0.002041309916862133
 -0.0029370158383912326
 -0.0002670763882761662
 -2.5721647034515627e-12
```

We can compare the reconstructed network using the {func}`~graph_tool.topology.similarity` function, as follows

```{testcode} dynamics
u = state.get_graph()      # reconstructed network
w_r = state.get_x()        # reconstructed weights
t_r = state.get_theta()    # reconstructed thetas

print(gt.similarity(g, u, w, w_r))
```

which gives:

```{testoutput} dynamics
0.908651...
```

We can visualize the reconstruction with:

```{testcode} dynamics
cnorm = matplotlib.colors.Normalize(vmin=-abs(w.fa).max(), vmax=abs(w.fa).max())
# Original graph
gt.graph_draw(g, g.vp.pos, edge_pen_width=gt.prop_to_size(w.t(abs), 2, 8, power=1), edge_color=w,
              ecmap=matplotlib.cm.coolwarm_r, ecnorm=cnorm, output="dolphins-orig.svg")

# Inferred graph
gt.graph_draw(u, g.vp.pos, edge_pen_width=gt.prop_to_size(w_r.t(abs), 2, 8, power=1), edge_color=w_r,
              ecmap=matplotlib.cm.coolwarm_r, ecnorm=cnorm, output="dolphins-inf.svg")

# Inferred graph with SBM partition
state.bstate.draw(pos=g.vp.pos, edge_gradient=[],
                  edge_pen_width=gt.prop_to_size(w_r.t(abs), 2, 8, power=1),
                  edge_color=w_r, ecmap=matplotlib.cm.coolwarm_r, ecnorm=cnorm,
                  output="dolphins-inf-sbm.svg")
```

which outputs the networks below:

```{list-table}
:class: borderless
:align: center
:widths: 1 1 1

* - True network
  - Reconstructed
  - Reconstructed with SBM
* - ![](output/dolphins-orig.svg)
  - ![](output/dolphins-inf.svg)
  - ![](output/dolphins-inf-sbm.svg)

```

:::{note}
The algorithm implemented is the one described in {cite}`rec-peixoto_scalable_2024`
combined with the MDL regularization based on weight quantization from
{cite}`rec-peixoto_network_2025` {cite}`rec-peixoto_uncertainty_2025`, which includes also a
SBM as part of the prior for the latent edges.

The algorithm runs in time $O(N\log^2 N)$ in the typical case, where
$N$ is the number of nodes, and if enabled during compilation will run
in parallel using [OpenMP](https://de.wikipedia.org/wiki/OpenMP), and
therefore can be employed to reconstruct large networks.
:::

### $L_1$ regularization

The default reconstruction methods employs a nonparametric regularization scheme
{cite}`rec-peixoto_network_2025`, but optionally $L_1$ can be used instead.
This is done by initializing the reconstruction state as

```{testcode} dynamics
lstate = gt.IsingGlauberBlockState(X, disable_xdist=True, disable_tdist=True, directed=False, self_loops=False)
```

This method requires a sparsity parameter $\lambda$ to be set, e.g.
the following sets $\lambda=200$,

```{testcode} dynamics
lstate.update_entropy_args(xl1=200)
```

and then we can proceed in the same way,

```{testcode} dynamics
for i in range(10):
   delta, *_ = lstate.mcmc_sweep(beta=np.inf, niter=10)
```

which produces the network:

```{testcode} dynamics
u = lstate.get_graph()
w_r = lstate.get_x()
gt.graph_draw(u, g.vp.pos, edge_pen_width=gt.prop_to_size(w_r.t(abs), 2, 8, power=1), edge_color=w_r,
              ecmap=matplotlib.cm.coolwarm_r, ecnorm=cnorm, output="dolphins-inf-l1.svg")
```

:::{figure} output/dolphins-inf-l1.svg
:scale: 50 %

Network reconstructed using $L_1$ regularization with $\lambda=200$.
:::

:::{caution}
The $L_1$ regularization method is faster than MDL, but the most
appropriate value of $\lambda$ is tricky to find, usually requiring
cross-validation, which in the end makes it slower than MDL. It also
introduces a shrinkage bias and tends to overfit when compared to MDL, which
in general should be preferred. See {cite}`rec-peixoto_network_2025` for details.
:::

## Reconstruction with empirical data

As an example of how to perform reconstructions on empirical data, we will
consider the log-returns of the stock prices of the [S&P 500 companies](https://en.wikipedia.org/wiki/S%26P_500) containing the 500 largest US
companies.

```{testsetup} rec-empirical
mkchdir(DOC_DIR + "/demos/reconstruction_indirect/output")
np.random.seed(44)
gt.seed_rng(44)
nt = gt.openmp_get_num_threads()
gt.openmp_set_num_threads(16)
```

We will begin by fetching the data and computing the log-returns:

```{testcode} rec-empirical
import requests
from collections import defaultdict
import csv

r = requests.get('https://downloads.skewed.de/data/all_stocks_5yr.csv')
lines = r.iter_lines(decode_unicode=True)
next(lines)                                     # remove header
prices = defaultdict(list)
for vals in csv.reader(lines, delimiter=','):
    prices[vals[-1]].append(float(vals[4]))     # use closing price
stocks, s = zip(*[(stock, vals) for stock, vals in prices.items() if len(vals) == 1259])

# compute log-returns
s = [[np.log(p[i+1]) - np.log(p[i]) for i in range(len(p)-1)] for p in s]

s = array(s)
```

We will use a multivariate normal model via {class}`~graph_tool.inference.PseudoNormalBlockState`:

```{testcode} rec-empirical
state = gt.PseudoNormalBlockState(s)

delta = np.inf
while delta > 20:
    ret = state.mcmc_sweep(beta=np.inf, niter=1)
    delta = abs(ret[0])
```

We can now visualize the reconstructed network:

```{testcode} rec-empirical
bstate = state.get_block_state().levels[0]
pos = gt.sfdp_layout(state.get_graph(), groups=bstate.get_blocks(), gamma=.02)
bstate.draw(pos=pos, vertex_color="white", edge_gradient=[],
            edge_pen_width=gt.prop_to_size(state.get_x().t(abs), .1, 3, power=1),
            output="SP500.svg")
```

```{testcleanup} rec-empirical
gt.openmp_set_num_threads(nt)
```

:::{figure} output/SP500.svg
:scale: 70 %

Reconstructed network of interactions between the S&P500 companies
according to their daily log-returns and a multivariate normal model. The edge
thickness corresponds to the inferred negative inverse covariance (a.k.a.
precision matrix entries).
:::

## References

```{bibliography}
:keyprefix: rec-
```
