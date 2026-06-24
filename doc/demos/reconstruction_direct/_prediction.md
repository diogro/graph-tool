# Edge prediction as binary classification

A more traditional approach to the prediction of missing and spurious edges
formulates it as a supervised [binary classification
task](https://en.wikipedia.org/wiki/Binary_classification), where the
edge/non-edge scores are computed by fitting a generative model to the observed
data, and computing their probabilities under that model
{cite}`pred-clauset_hierarchical_2008` {cite}`pred-guimera_missing_2009`. In
this setting, one typically omits any explicit model of the measurement process
(hence intrinsically assuming it to be uniform), and as a consequence of the
overall setup, only *relative probabilities* between individual missing and
spurious edges can be produced, instead of the full posterior distribution
considered in the last section. Since this limits the overall network
reconstruction, and does not yield confidence intervals, it is a less powerful
approach. Nevertheless, it is a popular procedure, which can also be performed
with graph-tool, as we describe in the following.

We set up the classification task by dividing the edges/non-edges into two sets
$\boldsymbol A$ and $\delta \boldsymbol A$, where the former corresponds to the
observed network and the latter either to the missing or spurious edges. We may
compute the posterior of $\delta \boldsymbol A$ as
{cite}`pred-valles-catala_consistencies_2018`

$$
P(\delta \boldsymbol A | \boldsymbol A) \propto
\sum_{\boldsymbol b}\frac{P(\boldsymbol A \cup \delta\boldsymbol A| \boldsymbol b)}{P(\boldsymbol A| \boldsymbol b)}P(\boldsymbol b | \boldsymbol A)
$$ (posterior-missing)

up to a normalization constant [^prediction_posterior]. Although the
normalization constant is difficult to obtain in general (since we need
to perform a sum over all possible spurious/missing edges), the
numerator of Eq. {eq}`posterior-missing` can be computed by sampling
partitions from the posterior, and then inserting or deleting edges from
the graph and computing the new likelihood. This means that we can
easily compare alternative predictive hypotheses $\{\delta
\boldsymbol A_i\}$ via their likelihood ratios

$$
\lambda_i = \frac{P(\delta \boldsymbol A_i | \boldsymbol A)}{\sum_j P(\delta \boldsymbol A_j | \boldsymbol A)}
$$

which do not depend on the normalization constant.

The values $P(\delta \boldsymbol A | \boldsymbol A, \boldsymbol b)$
can be computed with
{meth}`~graph_tool.inference.BlockState.get_edges_prob`. Hence, we can
compute spurious/missing edge probabilities just as if we were
collecting marginal distributions when doing model averaging.

Below is an example for predicting the two following edges in the
football network, using the nested model (for which we need to replace
$\boldsymbol b$ by $\{\boldsymbol b_l\}$ in the equations
above).

```{testcode} missing-edges
:hide:

mkchdir(DOC_DIR + "/demos/reconstruction_direct/output")

g = gt.collection.data["football"].copy()
color = g.new_vp("string", val="#cccccc")
ecolor = g.new_ep("string", val="#cccccc")
ewidth = g.new_ep("double", 1)
e = g.add_edge(0, 101)
ecolor[e] = "#a40000"
ewidth[e] = 5
e = g.add_edge(17, 56)
ecolor[e] = "#a40000"
ewidth[e] = 5
eorder = g.edge_index.copy("int")

gt.graph_draw(g, pos=g.vp.pos, vertex_color=color,
              vertex_fill_color=color, edge_color=ecolor,
              eorder=eorder, edge_pen_width=ewidth,
              output="football_missing.svg")
```

:::{figure} output/football_missing.*
:align: center
:width: 350px

Two non-existing edges in the football network (in red):
$(0,101)$ in the middle, and $(17,56)$ in the upper
right region of the figure.
:::

```{testsetup} missing-edges
gt.seed_rng(7)
```

```{testcode} missing-edges
g = gt.collection.data["football"]

missing_edges = [(0, 101), (17, 56)]

L = 10

state = gt.minimize_nested_blockmodel_dl(g)

probs = ([], [])

def collect_edge_probs(s):
    p1 = -s.levels[0].modify_edge_dS(*missing_edges[0], dm=1)
    p2 = -s.levels[0].modify_edge_dS(*missing_edges[1], dm=1)
    probs[0].append(p1)
    probs[1].append(p2)

# Now we collect the probabilities for exactly 100,000 sweeps
gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                    callback=collect_edge_probs)

from scipy.special import logsumexp

p1 = logsumexp(probs[0])
p2 = logsumexp(probs[1])

Z = logsumexp([p1, p2])

l1 = p1 - Z
l2 = p2 - Z

print("likelihood-ratio for %s: %g" % (missing_edges[0], exp(l1)))
print("likelihood-ratio for %s: %g" % (missing_edges[1], exp(l2)))
```

```{testoutput} missing-edges
likelihood-ratio for (0, 101): 0.273606
likelihood-ratio for (17, 56): 0.726394
```

From which we can conclude that edge $(17, 56)$ is more likely
than $(0, 101)$ to be a missing edge.

The prediction using the non-nested model can be performed in an
entirely analogous fashion.


[^prediction_posterior]: Note that the posterior of Eq. {eq}`posterior-missing`
    cannot be used to sample the reconstruction $\delta \boldsymbol
    G$, as it is not informative of the overall network density
    (i.e. absolute number of missing and spurious edges). It can,
    however, be used to compare different reconstructions with the same
    density.
