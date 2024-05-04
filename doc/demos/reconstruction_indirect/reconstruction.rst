.. _reconstruction_dynamics:

Network reconstruction from dynamics and behavior
+++++++++++++++++++++++++++++++++++++++++++++++++

In some cases, direct measurements of the edges of a network are either
impossible, or can be done only at significant experimental cost. In such
situations, we need to infer the network of interactions from the observed
functional behavior [peixoto-network-2024]_. In ``graph-tool`` this can be done
for a variety of static and dynamical models, where the data :math:`\boldsymbol
X` is a :math:`N\times M` matrix containing :math:`M` samples (or "features")
for each node.

For static models we have that the likelihood is given by

.. math::

   P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) =
   \prod_m P(\boldsymbol X_{m}|\boldsymbol W, \boldsymbol\theta),

where each sample is assumed to be i.i.d., whereas for dynamic models
:math:`\boldsymbol X` is assumed to contain the states of a Markov chain with
likelihood

.. math::

   P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) =
   \prod_m P(\boldsymbol X_{m}|X_{m-1},\boldsymbol W, \boldsymbol\theta).

In both cases above, :math:`\boldsymbol W` represents the weighted adjacency
matrix and :math:`\boldsymbol \theta` are additional node parameters.

The inference procedure consists in the maximum `a posteriori` (MAP) estimation:

.. math::

   \begin{aligned}
   \widehat{\boldsymbol W}, \widehat{\boldsymbol\theta} &=
   \underset{\boldsymbol W, \boldsymbol\theta}{\operatorname{arg max}}\; P(\boldsymbol W,\boldsymbol\theta | \boldsymbol X) \\
    &= \underset{\boldsymbol W, \boldsymbol\theta}{\operatorname{arg max}}\; \log P(\boldsymbol X | \boldsymbol W,\boldsymbol\theta) + \log P(\boldsymbol W,\boldsymbol\theta),
   \end{aligned}

where the prior :math:`P(\boldsymbol W,\boldsymbol\theta)` implements regularization.

In ``graph-tool``, a variety of models are implemented, as given by the following table:

.. csv-table::
   :header: Name | State | Data domain | Kind
   :widths: 10, 5, 5, 5
   :delim: |
   :align: center

   `Multivariate normal <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_ | :class:`~graph_tool.inference.PseudoNormalBlockState` | :math:`\mathbb{R}` | Static
   `Dynamic multivariate normal <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_ | :class:`~graph_tool.inference.NormalGlauberBlockState` | :math:`\mathbb{R}` | Dynamic
   `Generalized Lotka-Volterra <https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation>`_ | :class:`~graph_tool.inference.LVBlockState` | :math:`\mathbb{R}^+` | Dynamic
   `Compartmental epidemics <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_ | :class:`~graph_tool.inference.EpidemicsBlockState` | :math:`\{0, 1\}` | Dynamic
   `Ising model <https://en.wikipedia.org/wiki/Ising_model>`_ | :class:`~graph_tool.inference.PseudoIsingBlockState` | :math:`\{-1, +1\}` | Static
   `Kinetic Ising model <https://en.wikipedia.org/wiki/Glauber_dynamics>`_ | :class:`~graph_tool.inference.IsingGlauberBlockState` | :math:`\{-1, +1\}` | Dynamic
   Continuous Ising model | :class:`~graph_tool.inference.PseudoCIsingBlockState` | :math:`[-1,1]` | Static
   Continuous kinetic Ising model | :class:`~graph_tool.inference.CIsingGlauberBlockState` | :math:`[-1,1]` | Dynamic


.. warning::

   For optimal parallel performance, the following environmental variable should
   be set **before the Python interpreted is evoked**:

   .. code:: bash

      export OMP_WAIT_POLICY=passive

   or alternatively from Python **before** ``graph-tool`` **is imported**:

   .. code::

      import os
      os.environ["OMP_WAIT_POLICY"] = "passive"

   Due to an OpenMP API limitation, this can no longer be changed after ``graph-tool``
   has been imported.

   Without passing this option, the parallel performance might be severely degraded.


Reconstruction with synthetic data
----------------------------------

Here we demonstrate how reconstruction is performed with a simple example, based
on the simulation of the `Kinetic Ising model
<https://en.wikipedia.org/wiki/Glauber_dynamics>`_ on the :ns:`dolphins` network (we can use the
:class:`~graph_tool.dynamics.IsingGlauberState` class for this purpose):

.. testsetup:: dynamics

   import os
   try:
       os.chdir("demos/reconstruction_indirect")
   except FileNotFoundError:
       pass
   np.random.seed(42)
   gt.seed_rng(42)

.. testcode:: dynamics
   :hide:

   g = gt.collection.data["dolphins"]
   gt.graph_draw(g, g.vp.pos, output="dolphin.svg")

.. note::
   :class: margin

   .. figure:: dolphin.svg

      This is the :ns:`dolphins` network we will attempt to reconstruct from a
      the kinetic Ising model.

.. testcode:: dynamics

   g = gt.collection.data["dolphins"]
   E = g.num_edges()
   N = g.num_vertices()
   w = g.new_ep("double", vals=np.random.normal(N/(2**E), .05, E))  # True edge weights
   istate = gt.IsingGlauberState(g, w=w)

   M = 1000
   X = []
   for m in range(M):
       istate.iterate_sync()
       X.append(istate.get_state().a.copy())
   X = np.array(X).T

At this point the array ``X`` contains our simulated data. We will now
reconstruct the original network from it using a nonparametric approach, which
consists of initializing an :class:`~graph_tool.inference.IsingGlauberBlockState`
object, and iterating over
:meth:`~graph_tool.inference.IsingGlauberBlockState.mcmc_sweep` until sufficient
convergence is reached:

.. note::
   :class: margin

   When doing reconstruction, the default for
   :meth:`~graph_tool.inference.IsingGlauberBlockState.mcmc_sweep` is
   ``beta=numpy.inf``, meaning that this function performs a greedy optimization,
   not actual samples from a reversible Markov chain.

.. testcode:: dynamics

   state = gt.IsingGlauberBlockState(X)

   for i in range(10):
       delta, *_ = state.mcmc_sweep(niter=10)
       print(delta)

This outputs the following deltas, indicating that convergence was achieved.

.. testoutput:: dynamics

    -4364.74776006487
    -3432.2361244413296
    -948.5777205382129
    -153.03149056130377
    -22.481765866866525
    -1.9345513708453002
    -2.098691104058476
    -0.0022805857834384824
    -2.4582982405263465e-07
    -1.0942358130705543e-11

We can compare the reconstructed network using the  :func:`~graph_tool.topology.similarity` function, as follows

.. testcode:: dynamics

   u = state.get_graph()      # reconstructed network
   w_r = state.get_x()        # reconstructed weights
   t_r = state.get_theta()    # reconstructed thetas

   print(gt.similarity(g, u, w, w_r))

which gives:

.. testoutput:: dynamics

   0.911081...

We can visualize the reconstruction with:

.. testcode:: dynamics

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

which outputs the networks below:

.. list-table::
   :class: borderless
   :align: center
   :widths: 1 1 1

   * - True network
     - Reconstructed
     - Reconstructed with SBM
   * - .. image:: dolphins-orig.svg
     - .. image:: dolphins-inf.svg
     - .. image:: dolphins-inf-sbm.svg


.. note::

   The algorithm implemented is the one described in [peixoto-scalable-2024]_
   combined with the MDL regularization based on weight quantization from
   [peixoto-network-2024]_, which includes also a SBM as part of the prior
   for the latent edges.

   The algorithm runs in time :math:`O(N\log^2 N)` in the typical case, where
   :math:`N` is the number of nodes, and if enabled during compilation will run
   in parallel using `OpenMP <https://de.wikipedia.org/wiki/OpenMP>`_, and
   therefore can be employed to reconstruct large networks.


:math:`L_1` regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^

The default reconstruction methods employs a nonparametric regularization scheme
[peixoto-network-2024]_, but optionally :math:`L_1` can be used instead.
This is done by initializing the reconstruction state as


.. testcode:: dynamics

   lstate = gt.IsingGlauberBlockState(X, disable_xdist=True, disable_tdist=True)

This method requires a sparsity parameter :math:`\lambda` to be set, e.g.
the following sets :math:`\lambda=200`,

.. testcode:: dynamics

    lstate.update_entropy_args(xl1=200)

and then we can proceed in the same way,

.. testcode:: dynamics

    for i in range(10):
       delta, *_ = lstate.mcmc_sweep(niter=10)

which produces the network:

.. testcode:: dynamics

    u = lstate.get_graph()
    w_r = lstate.get_x()
    gt.graph_draw(u, g.vp.pos, edge_pen_width=gt.prop_to_size(w_r.t(abs), 2, 8, power=1), edge_color=w_r,
                 ecmap=matplotlib.cm.coolwarm_r, ecnorm=cnorm, output="dolphins-inf-l1.svg")

.. figure:: dolphins-inf-l1.svg
   :scale: 50 %

   Network reconstructed using :math:`L_1` regularization with :math:`\lambda=200`.

.. caution::

   The :math:`L_1` regularization method is faster than MDL, but the most
   appropriate value of :math:`\lambda` is tricky to find, usually requiring
   cross-validation, which in the end makes it slower than MDL. It also
   introduces a shrinkage bias and tends to overfit when compared to MDL, which
   in general should be preferred. See [peixoto-network-2024]_ for details.


Reconstruction with empirical data
----------------------------------

As an example of how to perform reconstructions on empirical data, we will
consider the log-returns of the stock prices of the `S&P 500 companies
<https://en.wikipedia.org/wiki/S%26P_500>`_ containing the 500 largest US
companies.

.. testsetup:: rec-empirical

   import os
   try:
       os.chdir("demos/reconstruction_indirect")
   except FileNotFoundError:
       pass
   np.random.seed(43)
   gt.seed_rng(43)
   nt = gt.openmp_get_num_threads()
   gt.openmp_set_num_threads(16)

We will begin by fetching the data and computing the log-returns:

.. testcode:: rec-empirical

   import requests
   from collections import defaultdict
   import csv

   r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv')
   lines = r.iter_lines(decode_unicode=True)
   next(lines)                                     # remove header
   prices = defaultdict(list)
   for vals in csv.reader(lines, delimiter=','):
       prices[vals[-1]].append(float(vals[4]))     # use closing price
   stocks, s = zip(*[(stock, vals) for stock, vals in prices.items() if len(vals) == 1259])

   # compute log-returns
   s = [[log(p[i+1]) - log(p[i]) for i in range(len(p)-1)] for p in s]

   s = array(s)

We will use a multivariate normal model via :class:`~graph_tool.inference.PseudoNormalBlockState`:

.. tip::
   :class: margin

   For some models/data it may be better to first do a sweep over the edges
   before any other latent variable. This happens to be true for this case!

.. testcode:: rec-empirical

   state = gt.PseudoNormalBlockState(s)

   # Do an initial sweep over the edges before any other latent variable.
   ret = state.edge_mcmc_sweep(niter=10)

   delta = abs(ret[0])
   while delta > 20:
       ret = state.mcmc_sweep(niter=10)
       delta = abs(ret[0])

We can now visualize the reconstructed network:

.. testcode:: rec-empirical

   bstate = state.get_block_state().levels[0]
   pos = gt.sfdp_layout(state.get_graph(), groups=bstate.get_blocks(), gamma=.02)
   bstate.draw(pos=pos, vertex_color="white", edge_gradient=[],
               edge_pen_width=gt.prop_to_size(state.get_x().t(abs), .1, 3, power=1),
               output="SP500.svg")

.. testcleanup:: rec-empirical

   gt.openmp_set_num_threads(nt)

.. figure:: SP500.svg
   :scale: 70 %

   Reconstructed network of interactions between the S&P500 companies
   according to their daily log-returns and a multivariate normal model. The edge
   thickness corresponds to the inferred negative inverse covariance (a.k.a.
   precision matrix entries).


References
----------

.. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
   community detection from dynamics", Phys. Rev. Lett. 123 128301
   (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`
.. [peixoto-scalable-2024] Tiago P. Peixoto, "Scalable network reconstruction in
   subquadratic time", :arxiv:`2401.01404`
.. [peixoto-network-2024] Tiago P. Peixoto, "Network reconstruction via the
   minimum description length principle", :arxiv:`2405.01015`
