(weights)=

# Edge weights and covariates

Very often networks cannot be completely represented by simple graphs,
but instead have arbitrary "weights" $x_{ij}$ on the edges. Edge
weights can be continuous or discrete numbers, and either strictly
positive or positive or negative, depending on context. The SBM can be
extended to cover these cases by treating edge weights as covariates
that are sampled from some distribution conditioned on the node
partition {cite}`inf-aicher_learning_2014` {cite}`inf-peixoto_nonparametric_2018`, i.e.

$$
P(\boldsymbol x,\boldsymbol A|\boldsymbol b) =
P(\boldsymbol x|\boldsymbol A,\boldsymbol b) P(\boldsymbol A|\boldsymbol b),
$$

where $P(\boldsymbol A|\boldsymbol b)$ is the likelihood of the
unweighted SBM described previously, and $P(\boldsymbol
x|\boldsymbol A,\boldsymbol b)$ is the integrated likelihood of the edge
weights

$$
P(\boldsymbol x|\boldsymbol A,\boldsymbol b) =
\prod_{r\le s}\int P({\boldsymbol x}_{rs}|\gamma)P(\gamma)\,\mathrm{d}\gamma,
$$

where $P({\boldsymbol x}_{rs}|\gamma)$ is some model for the weights
${\boldsymbol x}_{rs}$ between groups $(r,s)$, conditioned on
some parameter $\gamma$, sampled from its prior
$P(\gamma)$. A hierarchical version of the model can also be
implemented by replacing this prior by a nested sequence of priors and
hyperpriors, as described in {cite}`inf-peixoto_nonparametric_2018`. The posterior
partition distribution is then simply

$$
P(\boldsymbol b | \boldsymbol A,\boldsymbol x) =
\frac{P(\boldsymbol x|\boldsymbol A,\boldsymbol b) P(\boldsymbol A|\boldsymbol b)
      P(\boldsymbol b)}{P(\boldsymbol A,\boldsymbol x)},
$$

which can be sampled from, or maximized, just like with the unweighted
case, but will use the information on the weights to guide the partitions.

A variety of weight models is supported, reflecting different kinds of
edge covariates:

```{csv-table} Edge covariate kernel distributions
:header: Name | Domain | Bounds | Shape
:widths: 10, 5, 5, 5
:delim: "|"

`"real-exponential"`   | Real    $(\mathbb{R})$ | $[0,\infty]$       | [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
`"real-normal"`        | Real    $(\mathbb{R})$ | $[-\infty,\infty]$ | [Normal](https://en.wikipedia.org/wiki/Normal_distribution)
`"discrete-geometric"` | Natural $(\mathbb{N})$ | $[0,\infty]$       | [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution)
`"discrete-binomial"`  | Natural $(\mathbb{N})$ | $[0,M]$            | [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
`"discrete-poisson"`   | Natural $(\mathbb{N})$ | $[0,\infty]$       | [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)

```

In fact, the actual model implements [microcanonical](https://en.wikipedia.org/wiki/Microcanonical_ensemble) versions of
these distributions that are asymptotically equivalent, as described in
{cite}`inf-peixoto_nonparametric_2018`. These can be combined with arbitrary weight
transformations to achieve a large family of associated
distributions. For example, to use a [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) weight model
for positive real weights $\boldsymbol x$, we can use the
transformation $y_{ij} = \ln x_{ij}$ together with the
`"real-normal"` model for $\boldsymbol y$. To model weights that
are positive or negative integers in $\mathbb{Z}$, we could either
subtract the minimum value, $y_{ij} = x_{ij} - x^*$, with
$x^*=\operatorname{min}_{ij}x_{ij}$, and use any of the above
models for non-negative integers in $\mathbb{N}$, or
alternatively, consider the sign as an additional covariate,
i.e. $s_{ij} = [\operatorname{sign}(x_{ij})+1]/2 \in \{0,1\}$,
using the Binomial distribution with $M=1$ (a.k.a. the [Bernoulli
distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)),
and any of the other discrete distributions for the magnitude,
$y_{ij} = \operatorname{abs}(x_{ij})$.

The support for weighted networks is activated by passing the parameters
`rec` and `rec_types` to
{class}`~graph_tool.inference.WeightedBlockState`,
that specify the edge covariates (an edge
{class}`~graph_tool.PropertyMap`) and their types (a string from the
table above), respectively. Note that these parameters expect *lists*,
so that multiple edge weights can be used simultaneously.

For example, let us consider a network of suspected terrorists involved
in the train bombing of Madrid on March 11, 2004
{cite}`inf-hayes_computing_2006`. An edge indicates that a connection between
the two persons have been identified, and the weight of the edge (an
integer in the range $[0,3]$) indicates the "strength" of the
connection. We can apply the weighted SBM, using a Binomial model for
the weights, as follows:

```{testsetup} weighted-model
import os
mkchdir(DOC_DIR + "/demos/inference/output")
gt.seed_rng(46)
```

```{testcode} weighted-model
g = gt.collection.ns["train_terrorists"]

# This network contains an internal edge property map with name
# "weight" that contains the strength of interactions. The values
# integers in the range [0, 3].

state = gt.minimize_nested_blockmodel_dl(g, base_state=gt.WeightedBlockState, 
                                         base_state_args=dict(rec=[g.ep.weight],
                                                              rec_types=["discrete-binomial"]))

# improve solution with merge-split

for i in range(100):
    ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

state.draw(edge_color=g.ep.weight.copy("double"), ecmap=matplotlib.cm.plasma,
           eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 2, 8, power=1),
           edge_gradient=[], output="moreno-train-wsbm.svg")
```

:::{figure} output/moreno-train-wsbm.svg
:align: center
:width: 450px

Best fit of the Binomial-weighted degree-corrected SBM for a network
of terror suspects, using the strength of connection as edge
covariates. The edge colors and widths correspond to the strengths.
:::

## Model selection

In order to select the best weighted model, we proceed in the same
manner as described in Sec. {ref}`model_selection`. However, when using
transformations on continuous weights, we must include the associated
scaling of the probability density, as described in
{cite}`inf-peixoto_nonparametric_2018`.

For example, consider a [food web](https://en.wikipedia.org/wiki/Food_web) between species in south
Florida {cite}`inf-ulanowicz_network_2005`. A directed link exists from species
$i$ to $j$ if a energy flow exists between them, and a
weight $x_{ij}$ on this edge indicates the magnitude of the energy
flow (a positive real value, i.e. $x_{ij}\in [0,\infty]$). One
possibility, therefore, is to use the `"real-exponential"` model, as
follows:

```{testsetup} food-web
mkchdir(DOC_DIR + "/demos/inference/output")
gt.seed_rng(44)
```

```{testcode} food-web
g = gt.collection.ns["foodweb_baywet"]

# This network contains an internal edge property map with name
# "weight" that contains the energy flow between species. The values
# are continuous in the range [0, infinity].

state = gt.minimize_nested_blockmodel_dl(g, base_state=gt.WeightedBlockState, 
                                         base_state_args=dict(rec=[g.ep.weight],
                                                              rec_types=["real-exponential"]))

# improve solution with merge-split

for i in range(100):
    ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

state.draw(edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
           edge_gradient=[], output="foodweb-wsbm.svg")
```

:::{figure} output/foodweb-wsbm.svg
:align: center
:width: 450px

Best fit of the exponential-weighted degree-corrected SBM for a food
web, using the energy flow as edge covariates (indicated by the edge
colors and widths).
:::

Alternatively, we may consider a transformation of the type

$$
y_{ij} = \ln x_{ij}
$$ (log_transform)

so that $y_{ij} \in [-\infty,\infty]$. If we use a model
`"real-normal"` for $\boldsymbol y$, it amounts to a [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) model for
$\boldsymbol x$. This can be a better choice if the weights are
distributed across many orders of magnitude, or show multi-modality. We
can fit this alternative model simply by using the transformed weights:

```{testcode} food-web
# Apply the weight transformation
y = g.ep.weight.copy()
y.a = log(y.a)

state_ln = gt.minimize_nested_blockmodel_dl(g,
                                            base_state=gt.WeightedBlockState, 
                                            base_state_args=dict(rec=[y],
                                                                 rec_types=["real-normal"]))

# improve solution with merge-split

for i in range(100):
    ret = state_ln.multiflip_mcmc_sweep(niter=10, beta=np.inf)

state_ln.draw(edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
              eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
              edge_gradient=[], output="foodweb-wsbm-lognormal.svg")
```

:::{figure} output/foodweb-wsbm-lognormal.svg
:align: center
:width: 450px

Best fit of the log-normal-weighted degree-corrected SBM for a food
web, using the energy flow as edge covariates (indicated by the edge
colors and widths).
:::

At this point, we ask ourselves which of the above models yields the
best fit of the data. This is answered by performing model selection via
posterior odds ratios just like in Sec. {ref}`model_selection`. However,
here we need to take into account the scaling of the probability density
incurred by the variable transformation, i.e.

$$
P(\boldsymbol x | \boldsymbol A, \boldsymbol b) =
P(\boldsymbol y(\boldsymbol x) | \boldsymbol A, \boldsymbol b)
\prod_{ij}\left[\frac{\mathrm{d}y_{ij}}{\mathrm{d}x_{ij}}(x_{ij})\right]^{A_{ij}}.
$$

In the particular case of Eq. {eq}`log_transform`, we have

$$
\prod_{ij}\left[\frac{\mathrm{d}y_{ij}}{\mathrm{d}x_{ij}}(x_{ij})\right]^{A_{ij}}
= \prod_{ij}\frac{1}{x_{ij}^{A_{ij}}}.
$$

Therefore, we can compute the posterior odds ratio between both models as:

```{testcode} food-web
L1 = -state.entropy()
L2 = -state_ln.entropy() - log(g.ep.weight.a).sum()

print(u"ln \u039b: ", L2 - L1)
```

```{testoutput} food-web
:options: +NORMALIZE_WHITESPACE

ln Λ: -96.153069...
```

A value of $\Lambda \approx \mathrm{e}^{138} \approx 10^{60}$ in
favor the exponential model indicates that the log-normal model does not
provide a better fit for this particular data.

## Posterior sampling

The procedure to sample from the posterior distribution is identical to
what is described in Sec. {ref}`sampling`, but with the appropriate
initialization, e.g..

```{testcode} weighted-model
g = gt.collection.ns["foodweb_baywet"]

state = gt.NestedBlockState(g, base_state=gt.WeightedBlockState,
                            base_state_args=dict(rec=[g.ep.weight],
                                                 rec_types=["real-exponential"]))

gt.mcmc_equilibrate(state, force_niter=100, mcmc_args=dict(niter=10))
```
