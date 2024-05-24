.. _parallel_algorithms:

Parallel algorithms
-------------------

``graph-tool`` has support for shared memory parallelization via `OpenMP
<https://en.wikipedia.org/wiki/OpenMP>`_ for many algorithms, as is indicated in
their docstring.

OpenMP support is optional, and must be enabled during compilation. To check
whether it is available, the function :func:`~graph_tool.openmp_enabled` can be
used.

.. autofunction:: graph_tool.openmp_enabled
   :no-index:

Several parallelization parameters can be controlled at runtime, including the
number of threads, the work sharing schedule, and the minimum number of nodes
required for parallelization to be enabled.

.. autofunction:: graph_tool.openmp_get_num_threads
   :no-index:

.. autofunction:: graph_tool.openmp_set_num_threads
   :no-index:

.. autofunction:: graph_tool.openmp_get_schedule
   :no-index:

.. autofunction:: graph_tool.openmp_set_schedule
   :no-index:

.. autofunction:: graph_tool.openmp_get_thresh
   :no-index:

.. autofunction:: graph_tool.openmp_set_thresh
   :no-index:

It's possible to set these parameter temporarily using a context manager:

.. autofunction:: graph_tool.openmp_context
   :no-index:

For example, to constrain temporarily the number of threads to 3 and use a
"guided" scheduling one could do:

>>> g = gt.collection.data["polblogs"]
>>> with gt.openmp_context(nthreads=3, schedule="guided"):
...     ret = gt.pagerank(g)

Parallelization can be disabled altogether in the same way, but using ``nthreads=1``.

.. warning::

   By default, ``graph-tool`` will configure the OpenMP wait policy to
   "passive", since this usually results in improved performance for the
   majority of algorithms.

   In order for change this behavor, the following environment variable should
   be set before the Python interpreted is evoked:

   .. code-block:: bash

       export OMP_WAIT_POLICY=active

   or alternatively from Python before graph-tool is imported:

   .. code-block:: python

       import os
       os.environ["OMP_WAIT_POLICY"] = "active"

   Due to an OpenMP API limitation, this can no longer be changed after graph-tool has been imported.

The global interpreter lock (GIL)
=================================

``graph-tool`` releases Python's `GIL
<https://wiki.python.org/moin/GlobalInterpreterLock>`_ as soon as the C++
implementations are reached, even for algorithms that are not implemented in
parallel with OpenMP. This means that Python's :mod:`threading` functionally can
be used with many functions to achieve parallelism. For example, the following
code will run several calls of :func:`~graph_tool.topology.subgraph_isomorphism`
in parallel using 16 threads, which each individual call running sequentially:

.. doctest:: parallel

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> g = gt.collection.data["netscience"]
    >>> def find_sub():
    ...     u = gt.random_graph(11, lambda: 4, directed=False, model="erdos")
    ...     gt.subgraph_isomorphism(u, g, max_n=100)
    >>> with ThreadPoolExecutor(max_workers=16) as executor:
    ...     futures = [executor.submit(find_sub) for i in range(16)]
    ...     for future in futures:
    ...         future.result()

The same kind of functionality can be achieved with :mod:`multiprocessing` with
a nearly identical interface, but the above offers smaller overhead, since no
inter-process communication is needed.
