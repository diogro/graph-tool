.. _inference-howto:

Inferring modular network structure
===================================

``graph-tool`` includes algorithms to identify the large-scale structure
of networks via statistical inference in the
:mod:`~graph_tool.inference` submodule. Here we explain the basic
functionality with self-contained examples. For a more thorough
theoretical introduction to the methods described here, the reader is
referred to [peixoto-bayesian-2019]_.

See also [peixoto-descriptive-2023]_ and the corresponding `blog post
<https://skewed.de/tiago/blog/descriptive-inferential>`_ for an overall
discussion about inferential approaches to structure identification in
networks, and how they contrast with descriptive approaches.

.. include:: _background.rst
.. include:: _minimization.rst
.. include:: _model_selection.rst
.. include:: _sampling.rst
.. include:: _model_class_selection.rst
.. include:: _edge_weights.rst
.. include:: _layers.rst
.. include:: _assortative.rst
.. include:: _ranked.rst

References
----------

.. [peixoto-bayesian-2019] Tiago P. Peixoto, "Bayesian stochastic blockmodeling",
   Advances in Network Clustering and Blockmodeling, edited
   by P. Doreian, V. Batagelj, A. Ferligoj, (Wiley, New York, 2019)
   :doi:`10.1002/9781119483298.ch11`, :arxiv:`1705.10225`

.. [peixoto-descriptive-2023] Tiago P. Peixoto, “Descriptive vs. inferential
   community detection in networks: pitfalls, myths and half-truths”, Elements
   in the Structure and Dynamics of Complex Networks, Cambridge University Press
   (2023), :doi:`10.1017/9781009118897` :arxiv:`2112.00183`

.. [holland-stochastic-1983] Paul W. Holland, Kathryn Blackmond Laskey,
   Samuel Leinhardt, "Stochastic blockmodels: First steps", Social Networks
   Volume 5, Issue 2, Pages 109-137 (1983). :doi:`10.1016/0378-8733(83)90021-7`

.. [karrer-stochastic-2011] Brian Karrer, M. E. J. Newman "Stochastic
   blockmodels and community structure in networks", Phys. Rev. E 83,
   016107 (2011). :doi:`10.1103/PhysRevE.83.016107`, :arxiv:`1008.3926`
   
.. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
   Bayesian inference of the microcanonical stochastic block model",
   Phys. Rev. E 95 012317 (2017). :doi:`10.1103/PhysRevE.95.012317`,
   :arxiv:`1610.02703`

.. [peixoto-parsimonious-2013] Tiago P. Peixoto, "Parsimonious module
   inference in large networks", Phys. Rev. Lett. 110, 148701 (2013).
   :doi:`10.1103/PhysRevLett.110.148701`, :arxiv:`1212.4794`.

.. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
   structures and high-resolution model selection in large networks",
   Phys. Rev. X 4, 011047 (2014). :doi:`10.1103/PhysRevX.4.011047`,
   :arxiv:`1310.4377`.

.. [peixoto-model-2016] Tiago P. Peixoto, "Model selection and hypothesis
   testing for large-scale network models with overlapping groups",
   Phys. Rev. X 5, 011033 (2016). :doi:`10.1103/PhysRevX.5.011033`,
   :arxiv:`1409.3059`.

.. [peixoto-inferring-2015] Tiago P. Peixoto, "Inferring the mesoscale
   structure of layered, edge-valued and time-varying networks",
   Phys. Rev. E 92, 042807 (2015). :doi:`10.1103/PhysRevE.92.042807`,
   :arxiv:`1504.02381`

.. [aicher-learning-2015] Christopher Aicher, Abigail Z. Jacobs, and
   Aaron Clauset, "Learning Latent Block Structure in Weighted
   Networks", Journal of Complex Networks 3(2). 221-248
   (2015). :doi:`10.1093/comnet/cnu026`, :arxiv:`1404.0431`

.. [peixoto-weighted-2017] Tiago P. Peixoto, "Nonparametric weighted
   stochastic block models", Phys. Rev. E 97, 012306 (2018),
   :doi:`10.1103/PhysRevE.97.012306`, :arxiv:`1708.01432`
          
.. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
   greedy heuristic for the inference of stochastic block models", Phys.
   Rev. E 89, 012804 (2014). :doi:`10.1103/PhysRevE.89.012804`,
   :arxiv:`1310.4378`

.. [peixoto-merge-split-2020] Tiago P. Peixoto, "Merge-split Markov
   chain Monte Carlo for community detection", Phys. Rev. E 102, 012305
   (2020), :doi:`10.1103/PhysRevE.102.012305`, :arxiv:`2003.07070`

.. [peixoto-revealing-2021] Tiago P. Peixoto, "Revealing consensus and
   dissensus between network partitions", Phys. Rev. X 11 021003 (2021)
   :doi:`10.1103/PhysRevX.11.021003`, :arxiv:`2005.13977`

.. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
   inference of assortative community structures", Phys. Rev. Research 2
   043271 (2020), :doi:`10.1103/PhysRevResearch.2.043271`, :arxiv:`2006.14493`

.. [peixoto-ordered-2022] Tiago P. Peixoto, "Ordered community detection
   in directed networks", Phys. Rev. E 106, 024305 (2022),
   :doi:`10.1103/PhysRevE.106.024305`, :arxiv:`2203.16460`

