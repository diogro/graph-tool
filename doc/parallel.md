(parallel_algorithms)=

# Parallel algorithms

`graph-tool` has support for shared memory parallelization via
[OpenMP](https://en.wikipedia.org/wiki/OpenMP) for many algorithms, as is
indicated in their docstring.

OpenMP support is optional, and must be enabled during compilation. To check
whether it is available, the function {func}`~graph_tool.openmp_enabled` can be
used.

```{eval-rst}
.. autofunction:: graph_tool.openmp_enabled
   :no-index:
```

:::{note}
By default, `graph-tool` will try to configure the [OpenMP wait
policy](https://www.openmp.org/spec-html/5.0/openmpse55.html) to "passive",
since this usually results in improved performance for the majority of
algorithms.

In order to change this behavior, the following environment variable should
be set before the Python interpreted is evoked:

```bash
export OMP_WAIT_POLICY=active
```

or alternatively from Python before graph-tool is imported:

```python
import os
os.environ["OMP_WAIT_POLICY"] = "active"
```

Due to an OpenMP API limitation, this can no longer be changed after
graph-tool has been imported.
:::

:::{warning}
If another library that uses OpenMP is imported before `graph-tool`, the
wait policy will be set to the default value of "active", which will no
longer be able to be changed when `graph-tool` is first imported, or any
time later. The only way to ensure that this policy is chosen is to set the
environment variable before the Python interpreter is first evoked:

```bash
export OMP_WAIT_POLICY=passive
```

It is recommended for user to set this value to guarantee the best
performance with `graph-tool` in every circumstance.
:::

Several parallelization parameters can be controlled at runtime, including the
number of threads, the work sharing schedule, and the minimum number of nodes
required for parallelization to be enabled.

```{eval-rst}
.. autofunction:: graph_tool.openmp_get_num_threads
   :no-index:
```

```{eval-rst}
.. autofunction:: graph_tool.openmp_set_num_threads
   :no-index:
```

```{eval-rst}
.. autofunction:: graph_tool.openmp_get_schedule
   :no-index:
```

```{eval-rst}
.. autofunction:: graph_tool.openmp_set_schedule
   :no-index:
```

```{eval-rst}
.. autofunction:: graph_tool.openmp_get_thresh
   :no-index:
```

```{eval-rst}
.. autofunction:: graph_tool.openmp_set_thresh
   :no-index:
```

It's possible to set these parameter temporarily using a context manager:

```{eval-rst}
.. autofunction:: graph_tool.openmp_context
   :no-index:
```

For example, to constrain temporarily the number of threads to 3 and use a
"guided" scheduling one could do:

```{doctest} parallel
>>> g = gt.collection.data["polblogs"]
>>> with gt.openmp_context(nthreads=3, schedule="guided"):
...     ret = gt.pagerank(g)
```

Parallelization can be disabled altogether in the same way, but using `nthreads=1`.

## The global interpreter lock (GIL)

`graph-tool` releases Python's
[GIL](https://wiki.python.org/moin/GlobalInterpreterLock) as soon as the C++
implementations are reached, even for algorithms that are not implemented in
parallel with OpenMP. This means that Python's {mod}`threading` functionally can
be used with many functions to achieve parallelism. For example, the following
code will run several calls of {func}`~graph_tool.topology.subgraph_isomorphism`
in parallel using 16 threads, which each individual call running sequentially:

```{doctest} parallel
>>> from concurrent.futures import ThreadPoolExecutor
>>> g = gt.collection.data["netscience"]
>>> def find_sub():
...     u = gt.random_graph(11, lambda: 4, directed=False, model="erdos")
...     gt.subgraph_isomorphism(u, g, max_n=100)
>>> with ThreadPoolExecutor(max_workers=16) as executor:
...     futures = [executor.submit(find_sub) for i in range(16)]
...     for future in futures:
...         future.result()
```

The same kind of functionality can be achieved with {mod}`multiprocessing` with
a nearly identical interface, but the above offers smaller overhead, since no
inter-process communication is needed.
