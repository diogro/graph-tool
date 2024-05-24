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

Several parallelization parameters can be controlled at runtime, including the
number of threads, the work sharing schedule, and the minimum number of nodes
required for parallelization to be enabled.

.. autofunction:: graph_tool.openmp_get_num_threads

.. autofunction:: graph_tool.openmp_set_num_threads

.. autofunction:: graph_tool.openmp_get_schedule

.. autofunction:: graph_tool.openmp_set_schedule

.. autofunction:: graph_tool.openmp_get_thresh

.. autofunction:: graph_tool.openmp_set_thresh

It's possible to set these parameter temporarily using a context manager:

.. autofunction:: graph_tool.openmp_context

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

   In order for change this behavor, the following environmental variable should
   be set before the Python interpreted is evoked:

   .. code-block:: bash

       export OMP_WAIT_POLICY=active

   or alternatively from Python before graph-tool is imported:

   .. code-block:: python

       import os
       os.environ["OMP_WAIT_POLICY"] = "active"

   Due to an OpenMP API limitation, this can no longer be changed after graph-tool has been imported.
