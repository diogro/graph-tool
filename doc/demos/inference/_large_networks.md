(large_networks)=

# Dealing with large networks

The functions {func}`~graph_tool.inference.minimize_blockmodel_dl` and
{func}`~graph_tool.inference.minimize_nested_blockmodel_dl` are implemented in
{ref}`parallel <parallel_algorithms>`, and hence can take advantage of multiple
cores to increase performance on large networks.

In addition to this, there are special provisions that can be taken to speed up
the performance of these functions even more, which can become particularly
useful for large networks, allowing these algorithms to be employed on graphs
with hundreds of millions to billions of edges.

## Strictly agglomerative search

The standard algorithms implemented by
{func}`~graph_tool.inference.minimize_blockmodel_dl` and
{func}`~graph_tool.inference.minimize_nested_blockmodel_dl` employ a random
[bisection search](https://en.wikipedia.org/wiki/Golden-section_search) to
determine the most appropriate number of groups. This has the benefit of being
less susceptible to getting trapped in local minima. However, in typical cases,
a strictly agglomerative search reaches an answer faster, where one always
tentatively moves from a partition with a larger number of groups to one with a
lower number (allowing for single-node moves in between), and often the results
obtained in this way are of similar quality. This version of the algorithm can
be used by calling the same functions with ``bisection=False`` passed as
``multilevel_mcmc_args``.

As an example, consider the following:

```{testcode} no_bisect

from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timer():
    start = perf_counter()
    try:
        yield
    finally:
        print(f"Elapsed time: {perf_counter() - start:.2f} seconds")


g = gt.collection.data["pgp-strong-2009"]

# We time the bisection-based approach first

with timer():
    state = gt.minimize_nested_blockmodel_dl(g)
    print("Description length:", state.entropy())

# Now the agglomerative version

with timer():
    state = gt.minimize_nested_blockmodel_dl(g, multilevel_mcmc_args=dict(bisection=False))
    print("Description length:", state.entropy())
```

which gives us:

```{testoutput} no_bisect
Description length: 1275016.342277...
Elapsed time: 46.54 seconds
Description length: 1268788.861484...
Elapsed time: 28.49 seconds
```

In this case the agglomerative algorithm was not only faster, but actually
resulted in a shorter description length! This is not always what happens; in
most cases the bisection search will actually yield better answers, but the
agglomerative version is at least competitive, and usually faster. In the
example above we only won around 18 seconds, but this is a speedup of
$1.6\times$, which becomes relevant for much larger networks.

## Initialization of the hierarchy shape

The nested SBM can be significantly slower than the non-nested version, since
the optimal clustering of a lower level depends on the clustering at a higher
level, and vice versa. The algorithm implemented in the function
{func}`~graph_tool.inference.minimize_nested_blockmodel_dl` moves upward in the
hierarchy, updating a level above whenever the level below improves, and comes
back to a lower level whenever this is not the case. As a result, the same
level, including the bottom one which is the most time consuming, needs to be
clustered repeatedly from scratch.

An alternative is possible, which combines the strictly agglomerative algorithm
considered before, consisting of a heuristic that estimates an upper bound on
the number of groups at each level. In this way we can proceed simultaneously
for every level of the hierarchy, without having to restart the search at any
level. This can significantly reduce the runtime, while giving answers that are
very competitive with the standard approach. This can be achieved by passing the
option `simple_init=True` to
{func}`~graph_tool.inference.minimize_nested_blockmodel_dl`, which subsumes
`bisection=False`.

Consider the following comparison to the above:
```{testcode} no_bisect
with timer():
    state = gt.minimize_nested_blockmodel_dl(g, simple_init=True)
    print("Description length:", state.entropy())
```

which gives us:

```{testoutput} no_bisect
Description length: 1275331.333979...
Elapsed time: 7.11 seconds
```

This gives us a $4\times$ speedup when compared to the `bisection=False` case,
and $6\times$ to the standard bisection-based algorithm. The description length
obtained was worse than the former, but this can change with repeated runs.




