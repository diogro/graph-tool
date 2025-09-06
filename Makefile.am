## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = $(CPPFLAGS_) $(MOD_CPPFLAGS) $(EXTRA_CPPFLAGS)
AM_CXXFLAGS = $(CXXFLAGS_) $(MOD_CXXFLAGS) $(EXTRA_CXXFLAGS)
AM_CFLAGS = $(AM_CXXFLAGS)

CXXFLAGS = $(AM_CXXFLAGS)
CPPFLAGS =

graphtooldocdir = ${PACKAGE_DOC_DIR}
nobase_dist_graphtooldoc_DATA = \
	README.md \
	COPYING \
	COPYING.LESSER \
	AUTHORS \
	INSTALL \
	doc/Makefile \
	doc/_static/graph-tool-logo-circle.svg \
	doc/_static/graph-tool-logo.svg \
	doc/_static/custom.css \
	doc/_static/fonts.css \
	doc/_static/fonts/6NUO8FuJNQ2MbkrZ5-J8lKFrp7pRef2r.woff2 \
	doc/_static/fonts/tDbK2oqRg1oM3QBjjcaDkOr4nAfcHg.woff2 \
	doc/_static/fonts/SlGVmQWMvZQIdix7AFxXkHNSbQ.woff2 \
	doc/_static/fonts/tDbV2oqRg1oM3QBjjcaDkOJGiRD7OwE.woff2 \
	doc/_static/fonts/tDbI2oqRg1oM3QBjjcaDkOr9rAU.woff2 \
	doc/_static/fonts/tDbX2oqRg1oM3QBjjcaDkOr4lLz5CwOnSA.woff2 \
	doc/centrality.rst \
	doc/clustering.rst \
	doc/conf.py \
	doc/correlations.rst \
	doc/demos/animation/animation.md \
	doc/demos/cppextensions/Makefile \
	doc/demos/cppextensions/cppextensions.md \
	doc/demos/cppextensions/kcore.cc \
	doc/demos/cppextensions/kcore.hh \
	doc/demos/cppextensions/kcore.py \
	doc/demos/index.md \
	doc/demos/inference/inference.md \
	doc/demos/inference/_assortative.md \
	doc/demos/inference/_background.md \
	doc/demos/inference/_edge_weights.md \
	doc/demos/inference/_layers.md \
	doc/demos/inference/_minimization.md \
	doc/demos/inference/_model_class_selection.md \
	doc/demos/inference/_model_selection.md \
	doc/demos/inference/_ranked.md \
	doc/demos/inference/_sampling.md \
	doc/demos/matplotlib/matplotlib.md \
    doc/demos/reconstruction_direct/reconstruction.md \
    doc/demos/reconstruction_direct/_prediction.md \
    doc/demos/reconstruction_indirect/reconstruction.md \
	doc/draw.rst \
	doc/flow.rst \
	doc/generation.rst \
	doc/graph_tool.rst \
    doc/gt_format.md \
	doc/index.md \
	doc/inference.rst  \
	doc/modules.md \
	doc/parallel.md \
	doc/price.py \
	doc/pyenv.py \
	doc/quickstart.md \
	doc/search_example.xml \
	doc/search_module.rst \
	doc/spectral.rst \
	doc/sphinxext/comment_eater.py \
	doc/sphinxext/compiler_unparse.py \
	doc/sphinxext/phantom_import.py \
	doc/sphinxext/traitsdoc.py \
	doc/stats.rst \
	doc/topology.rst \
	doc/util.rst \
	src/boost-workaround/LICENSE_1_0.txt

EXTRA_DIST = autogen.sh

pkgconfigdir = @pkgconfigdir@
pkgconfig_DATA = graph-tool-py${PYTHON_VERSION}.pc

# Copy all the spec files. Of course, only one is actually used.
dist-hook:
	(cd ${top_srcdir} && git log --stat --name-only --date=short --abbrev-commit) > ${distdir}/ChangeLog
	for specfile in *.spec; do \
		if test -f $$specfile; then \
			cp -p $$specfile $(distdir); \
		fi \
	done

install-data-hook:
	cd $(DESTDIR)$(pkgconfigdir) && $(LN_S) -f graph-tool-py${PYTHON_VERSION}.pc graph-tool-py.pc

uninstall-hook:
	cd $(DESTDIR)$(pkgconfigdir) && rm -f graph-tool-py.pc

##############
# src/graph/ #
##############

libgraph_tool_coredir = $(MOD_DIR)
libgraph_tool_core_LTLIBRARIES = libgraph_tool_core.la
libgraph_tool_core_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_core_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_core_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_core_la_SOURCES = \
	src/graph/base64.cc \
	src/graph/demangle.cc \
	src/graph/graph.cc \
	src/graph/graph_exceptions.cc \
	src/graph/graph_bind.cc \
	src/graph/graph_copy.cc \
	src/graph/graph_filtering.cc \
	src/graph/graph_io.cc \
	src/graph/graph_openmp.cc \
	src/graph/graph_properties.cc \
	src/graph/graph_properties_imp1.cc \
	src/graph/graph_properties_imp2.cc \
	src/graph/graph_properties_imp3.cc \
	src/graph/graph_properties_imp4.cc \
	src/graph/graph_properties_copy.cc \
	src/graph/graph_properties_copy_imp1.cc \
	src/graph/graph_properties_group.cc \
	src/graph/graph_properties_ungroup.cc \
	src/graph/graph_properties_map_values.cc \
	src/graph/graph_properties_map_values_imp1.cc \
	src/graph/graph_python_interface.cc \
	src/graph/graph_python_interface_imp1.cc \
	src/graph/graph_python_interface_export.cc \
	src/graph/graph_selectors.cc \
	src/graph/graphml.cpp \
	src/graph/openmp.cc \
	src/graph/random.cc \
	src/graph/read_graphviz_new.cpp
libgraph_tool_core_la_includedir = $(MOD_DIR)/include
nobase_libgraph_tool_core_la_include_HEADERS = \
	config.h \
	src/graph/base64.hh \
	src/graph/coroutine.hh \
	src/graph/demangle.hh \
	src/graph/dispatch.hh \
	src/graph/dynamic_sampler.hh \
	src/graph/fast_vector_property_map.hh \
	src/graph/gil_release.hh \
	src/graph/gml.hh \
	src/graph/graph.hh \
	src/graph/graph_adaptor.hh \
	src/graph/graph_adjacency.hh \
	src/graph/graph_exceptions.hh \
	src/graph/graph_filtered.hh \
	src/graph/graph_filtering.hh \
	src/graph/graph_io_binary.hh \
	src/graph/graph_properties.hh \
	src/graph/graph_properties_copy.hh \
	src/graph/graph_properties_group.hh \
	src/graph/graph_properties_map_values.hh \
	src/graph/graph_python_interface.hh \
	src/graph/graph_reverse.hh \
	src/graph/graph_selectors.hh \
	src/graph/graph_tool.hh \
	src/graph/graph_util.hh \
	src/graph/hash_map_wrap.hh \
	src/graph/histogram.hh \
	src/graph/idx_map.hh \
	src/graph/lock_free_stack.hh \
	src/graph/module_registry.hh \
	src/graph/npy_2_compat.h \
	src/graph/numpy_bind.hh \
	src/graph/openmp.hh \
	src/graph/openmp_lock.hh \
	src/graph/parallel_rng.hh \
	src/graph/parallel_util.hh \
	src/graph/random.hh \
	src/graph/samplers.hh \
	src/graph/scoped_hash_map.hh \
	src/graph/shared_heap.hh \
	src/graph/shared_map.hh \
	src/graph/str_repr.hh \
	src/graph/transform_iterator.hh \
	src/graph/urn_sampler.hh \
	src/graph/value_convert.hh

libgraph_tool_core_la_workarounddir = $(MOD_DIR)/include/boost-workaround/boost/graph/
nobase_libgraph_tool_core_la_workaround_HEADERS = \
	src/boost-workaround/boost/container/small_vector_old.hpp \
	src/boost-workaround/boost/container/vector_old.hpp \
	src/boost-workaround/boost/graph/adjacency_iterator.hpp \
	src/boost-workaround/boost/graph/betweenness_centrality.hpp \
	src/boost-workaround/boost/graph/detail/read_graphviz_new.hpp \
	src/boost-workaround/boost/graph/graphml.hpp \
	src/boost-workaround/boost/graph/graphviz.hpp \
	src/boost-workaround/boost/graph/hawick_circuits_local.hpp \
	src/boost-workaround/boost/graph/kamada_kawai_spring_layout.hpp \
	src/boost-workaround/boost/graph/maximum_weighted_matching_new.hpp \
	src/boost-workaround/boost/graph/maximum_weighted_matching_brute.hpp \
	src/boost-workaround/boost/graph/metric_tsp_approx.hpp \
	src/boost-workaround/boost/graph/named_function_params-alt.hpp \
	src/boost-workaround/boost/graph/overloading.hpp \
	src/boost-workaround/boost/graph/push_relabel_max_flow.hpp \
	src/boost-workaround/boost/graph/copy_alt.hpp \
	src/boost-workaround/boost/graph/stoer_wagner_min_cut.hpp \
	src/boost-workaround/boost/graph/vf2_sub_graph_iso.hpp \
	src/boost-workaround/boost/math/special_functions/relative_difference.hpp
libgraph_tool_core_la_pcg_cppdir = $(MOD_DIR)/include/pcg-cpp
nobase_libgraph_tool_core_la_pcg_cpp_HEADERS = \
	src/pcg-cpp/include/pcg_extras.hpp \
	src/pcg-cpp/include/pcg_random.hpp \
	src/pcg-cpp/include/pcg_uint128.hpp

#########################
# src/graph/centrality/ #
#########################

libgraph_tool_centralitydir = $(MOD_DIR)/centrality
libgraph_tool_centrality_LTLIBRARIES = libgraph_tool_centrality.la
libgraph_tool_centrality_la_includedir = $(MOD_DIR)/include
libgraph_tool_centrality_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_centrality_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_centrality_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_centrality_la_SOURCES = \
	src/graph/centrality/graph_betweenness.cc \
	src/graph/centrality/graph_centrality_bind.cc \
	src/graph/centrality/graph_closeness.cc \
	src/graph/centrality/graph_eigentrust.cc \
	src/graph/centrality/graph_eigenvector.cc \
	src/graph/centrality/graph_hits.cc \
	src/graph/centrality/graph_katz.cc \
	src/graph/centrality/graph_pagerank.cc \
	src/graph/centrality/graph_trust_transitivity.cc
nobase_libgraph_tool_centrality_la_include_HEADERS = \
	src/graph/centrality/graph_closeness.hh \
	src/graph/centrality/graph_eigentrust.hh \
	src/graph/centrality/graph_eigenvector.hh \
	src/graph/centrality/graph_pagerank.hh \
	src/graph/centrality/graph_hits.hh \
	src/graph/centrality/graph_katz.hh \
	src/graph/centrality/graph_trust_transitivity.hh \
	src/graph/centrality/minmax.hh

#########################
# src/graph/clustering/ #
#########################

libgraph_tool_clusteringdir = $(MOD_DIR)/clustering
libgraph_tool_clustering_LTLIBRARIES = libgraph_tool_clustering.la
libgraph_tool_clustering_la_includedir = $(MOD_DIR)/include
libgraph_tool_clustering_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_clustering_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_clustering_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_clustering_la_SOURCES = \
	src/graph/clustering/graph_clustering.cc \
	src/graph/clustering/graph_extended_clustering.cc \
	src/graph/clustering/graph_motifs.cc
nobase_libgraph_tool_clustering_la_include_HEADERS = \
	src/graph/clustering/graph_clustering.hh \
	src/graph/clustering/graph_extended_clustering.hh \
	src/graph/clustering/graph_motifs.hh

###########################
# src/graph/correlations/ #
###########################

libgraph_tool_correlationsdir = $(MOD_DIR)/correlations
libgraph_tool_correlations_LTLIBRARIES = libgraph_tool_correlations.la
libgraph_tool_correlations_la_includedir = $(MOD_DIR)/include
libgraph_tool_correlations_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_correlations_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_correlations_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_correlations_la_SOURCES = \
	src/graph/correlations/graph_assortativity.cc \
	src/graph/correlations/graph_correlations.cc \
	src/graph/correlations/graph_avg_correlations.cc \
	src/graph/correlations/graph_avg_correlations_combined.cc \
	src/graph/correlations/graph_correlations_combined.cc \
	src/graph/correlations/graph_correlations_bind.cc
nobase_libgraph_tool_correlations_la_include_HEADERS = \
	src/graph/correlations/graph_assortativity.hh \
	src/graph/correlations/graph_correlations.hh \
	src/graph/correlations/graph_corr_hist.hh \
	src/graph/correlations/graph_avg_correlations.hh

###################
# src/graph/draw/ #
###################

libgraph_tool_drawdir = $(MOD_DIR)/draw
libgraph_tool_draw_LTLIBRARIES = libgraph_tool_draw.la
libgraph_tool_draw_la_includedir = $(MOD_DIR)/include
libgraph_tool_draw_la_CPPFLAGS = $(AM_CPPFLAGS) $(CAIROMM_CFLAGS)
libgraph_tool_draw_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_draw_la_LDFLAGS = $(MOD_LDFLAGS) $(CAIROMM_LIBS)
libgraph_tool_draw_la_SOURCES = \
	src/graph/draw/graph_cairo_draw.cc \
	src/graph/draw/graph_tree_cts.cc
nobase_libgraph_tool_draw_la_include_HEADERS =

libgt_pycairo_auxdir = $(MOD_DIR)/draw
libgt_pycairo_aux_LTLIBRARIES = libgt_pycairo_aux.la
libgt_pycairo_aux_la_includedir = $(MOD_DIR)/include
libgt_pycairo_aux_la_CPPFLAGS = $(AM_CPPFLAGS) $(CAIROMM_CFLAGS)
libgt_pycairo_aux_la_LIBADD = $(MOD_LIBADD)
libgt_pycairo_aux_la_LDFLAGS = $(MOD_LDFLAGS) $(CAIROMM_LIBS)
libgt_pycairo_aux_la_SOURCES = \
	src/graph/draw/pycairo_aux.cc
nobase_libgt_pycairo_aux_la_include_HEADERS =

#######################
# src/graph/dynamics/ #
#######################

libgraph_tool_dynamicsdir = $(MOD_DIR)/dynamics
libgraph_tool_dynamics_LTLIBRARIES = libgraph_tool_dynamics.la
libgraph_tool_dynamics_la_includedir = $(MOD_DIR)/include
libgraph_tool_dynamics_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_dynamics_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_dynamics_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_dynamics_la_SOURCES = \
	src/graph/dynamics/graph_continuous.cc \
	src/graph/dynamics/graph_discrete.cc \
	src/graph/dynamics/graph_dynamics.cc \
	src/graph/dynamics/graph_normal_bp.cc \
	src/graph/dynamics/graph_potts_bp.cc
nobase_libgraph_tool_dynamics_la_include_HEADERS = \
	src/graph/dynamics/graph_continuous.hh \
	src/graph/dynamics/graph_discrete.hh \
	src/graph/dynamics/graph_normal_bp.hh \
	src/graph/dynamics/graph_potts_bp.hh

###################
# src/graph/flow/ #
###################

libgraph_tool_flowdir = $(MOD_DIR)/flow
libgraph_tool_flow_LTLIBRARIES = libgraph_tool_flow.la
libgraph_tool_flow_la_includedir = $(MOD_DIR)/include
libgraph_tool_flow_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_flow_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_flow_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_flow_la_SOURCES = \
	src/graph/flow/graph_edmonds_karp.cc \
	src/graph/flow/graph_push_relabel.cc \
	src/graph/flow/graph_kolmogorov.cc \
	src/graph/flow/graph_minimum_cut.cc \
	src/graph/flow/graph_flow_bind.cc
nobase_libgraph_tool_flow_la_include_HEADERS = \
	src/graph/flow/graph_augment.hh

#########################
# src/graph/generation/ #
#########################

libgraph_tool_generationdir = $(MOD_DIR)/generation
libgraph_tool_generation_LTLIBRARIES = libgraph_tool_generation.la
libgraph_tool_generation_la_includedir = $(MOD_DIR)/include
libgraph_tool_generation_la_CPPFLAGS = $(AM_CPPFLAGS) $(CGAL_CPPFLAGS)
libgraph_tool_generation_la_LIBADD = $(MOD_LIBADD) $(CGAL_LIBADD) $(CGAL_LDFLAGS)
libgraph_tool_generation_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_generation_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_generation_la_SOURCES = \
	src/graph/generation/graph_bip_projection.cc \
	src/graph/generation/graph_complete.cc \
	src/graph/generation/graph_contract_edges.cc \
	src/graph/generation/graph_generation.cc \
	src/graph/generation/graph_geometric.cc \
	src/graph/generation/graph_knn.cc \
	src/graph/generation/graph_lattice.cc \
	src/graph/generation/graph_line_graph.cc \
	src/graph/generation/graph_maxent_sbm.cc \
	src/graph/generation/graph_merge.cc \
	src/graph/generation/graph_merge_imp.cc \
	src/graph/generation/graph_merge_eprop.cc \
	src/graph/generation/graph_merge_vprop.cc \
	src/graph/generation/graph_parallel.cc \
	src/graph/generation/graph_predecessor.cc \
	src/graph/generation/graph_price.cc \
	src/graph/generation/graph_random_edges.cc \
	src/graph/generation/graph_rewiring.cc \
	src/graph/generation/graph_sbm.cc \
	src/graph/generation/graph_triadic_closure.cc \
	src/graph/generation/graph_triangulation.cc
nobase_libgraph_tool_generation_la_include_HEADERS = \
	src/graph/generation/graph_bip_projection.hh \
	src/graph/generation/graph_complete.hh \
	src/graph/generation/graph_contract_edges.hh \
	src/graph/generation/graph_generation.hh \
	src/graph/generation/graph_geometric.hh \
	src/graph/generation/graph_knn.hh \
	src/graph/generation/graph_lattice.hh \
	src/graph/generation/graph_maxent_sbm.hh \
	src/graph/generation/graph_merge.hh \
	src/graph/generation/graph_parallel.hh \
	src/graph/generation/graph_predecessor.hh \
	src/graph/generation/graph_price.hh \
	src/graph/generation/graph_random_edges.hh \
	src/graph/generation/graph_rewiring.hh \
	src/graph/generation/graph_sbm.hh \
	src/graph/generation/graph_triadic_closure.hh \
	src/graph/generation/graph_triangulation.hh

########################
# src/graph/inference/ #
########################

libgraph_tool_inference_ladir = $(MOD_DIR)/inference
libgraph_tool_inference_la_LTLIBRARIES = libgraph_tool_inference.la
libgraph_tool_inference_la_includedir = $(MOD_DIR)/include
libgraph_tool_inference_la_LIBADD = $(MOD_LIBADD) libgraph_tool_inference_fast.la
libgraph_tool_inference_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_inference_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_inference_la_SOURCES = \
	src/graph/inference/blockmodel/state.cc \
	src/graph/inference/blockmodel/em.cc \
	src/graph/inference/blockmodel/exhaustive.cc \
	src/graph/inference/blockmodel/gibbs.cc \
	src/graph/inference/blockmodel/imp.cc \
	src/graph/inference/blockmodel/marginals.cc \
	src/graph/inference/blockmodel/mcmc.cc \
	src/graph/inference/blockmodel/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/layered/spec.cc \
	src/graph/inference/blockmodel/layered/exhaustive.cc \
	src/graph/inference/blockmodel/layered/gibbs.cc \
	src/graph/inference/blockmodel/layered/mcmc.cc \
	src/graph/inference/blockmodel/layered/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/layered/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/layered_overlap/spec.cc \
	src/graph/inference/blockmodel/layered_overlap/exhaustive.cc \
	src/graph/inference/blockmodel/layered_overlap/gibbs.cc \
	src/graph/inference/blockmodel/layered_overlap/mcmc.cc \
	src/graph/inference/blockmodel/layered_overlap/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/layered_overlap/multilevel_mcmc.cc \
    src/graph/inference/blockmodel/overlap/spec.cc \
	src/graph/inference/blockmodel/overlap/exhaustive.cc \
	src/graph/inference/blockmodel/overlap/gibbs.cc \
	src/graph/inference/blockmodel/overlap/mcmc.cc \
	src/graph/inference/blockmodel/overlap/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/overlap/multilevel_mcmc.cc \
    src/graph/inference/blockmodel/ranked/spec.cc \
	src/graph/inference/blockmodel/ranked/exhaustive.cc \
	src/graph/inference/blockmodel/ranked/gibbs.cc \
	src/graph/inference/blockmodel/ranked/label_mcmc.cc \
	src/graph/inference/blockmodel/ranked/mcmc.cc \
	src/graph/inference/blockmodel/ranked/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/ranked/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/weighted_overlap/spec.cc \
	src/graph/inference/blockmodel/weighted_overlap/exhaustive.cc \
	src/graph/inference/blockmodel/weighted_overlap/gibbs.cc \
	src/graph/inference/blockmodel/weighted_overlap/mcmc.cc \
	src/graph/inference/blockmodel/weighted_overlap/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/weighted_overlap/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/weighted/spec.cc \
	src/graph/inference/blockmodel/weighted/exhaustive.cc \
	src/graph/inference/blockmodel/weighted/gibbs.cc \
	src/graph/inference/blockmodel/weighted/mcmc.cc \
	src/graph/inference/blockmodel/weighted/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/weighted/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted/spec.cc \
	src/graph/inference/blockmodel/layered_weighted/exhaustive.cc \
	src/graph/inference/blockmodel/layered_weighted/gibbs.cc \
	src/graph/inference/blockmodel/layered_weighted/mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted/multilevel_mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/spec.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/exhaustive.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/gibbs.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/multiflip_mcmc.cc \
	src/graph/inference/blockmodel/layered_weighted_overlap/multilevel_mcmc.cc \
	src/graph/inference/cliques/graph_clique_decomposition.cc \
	src/graph/inference/histogram/graph_histogram.cc \
	src/graph/inference/histogram/graph_histogram_mcmc.cc \
	src/graph/inference/support/bisection_sampler.cc \
	src/graph/inference/support/cache.cc \
	src/graph/inference/support/int_part.cc \
	src/graph/inference/support/spence.cc \
	src/graph/inference/graph_inference.cc \
	src/graph/inference/graph_latent_multigraph.cc \
	src/graph/inference/graph_modularity.cc \
	src/graph/inference/uncertain/dynamics/dynamics.cc \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc.cc \
	src/graph/inference/uncertain/dynamics/dynamics_swap_mcmc.cc \
	src/graph/inference/uncertain/dynamics/dynamics_multiflip_mcmc.cc \
	src/graph/inference/uncertain/dynamics/dynamics_multiflip_mcmc_theta.cc \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_theta.cc \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_tdelta.cc \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_xdelta.cc \
	src/graph/inference/uncertain/dynamics/dynamics_parallel_mcmc.cc \
	src/graph/inference/uncertain/dynamics/dynamics_parallel_mcmc_theta.cc \
	src/graph/inference/uncertain/dynamics/dynamics_parallel_swap_mcmc.cc \
	src/graph/inference/uncertain/dynamics/cising_glauber/state.cc \
	src/graph/inference/uncertain/dynamics/epidemics/state.cc \
	src/graph/inference/uncertain/dynamics/ising_glauber/state.cc \
	src/graph/inference/uncertain/dynamics/linear_normal/state.cc \
	src/graph/inference/uncertain/dynamics/lotka_volterra/state.cc \
	src/graph/inference/uncertain/dynamics/normal_glauber/state.cc \
	src/graph/inference/uncertain/dynamics/potts_glauber/state.cc \
	src/graph/inference/uncertain/dynamics/pseudo_cising/state.cc \
	src/graph/inference/uncertain/dynamics/pseudo_ising/state.cc \
	src/graph/inference/uncertain/dynamics/pseudo_normal/state.cc \
	src/graph/inference/uncertain/dynamics/pseudo_potts/state.cc \
	src/graph/inference/uncertain/dynamics/test/state.cc \
	src/graph/inference/uncertain/latent_closure.cc \
	src/graph/inference/uncertain/latent_closure_mcmc.cc \
	src/graph/inference/uncertain/latent_mask.cc \
	src/graph/inference/uncertain/latent_mask_mcmc.cc \
	src/graph/inference/uncertain/measured.cc \
	src/graph/inference/uncertain/measured_mcmc.cc \
	src/graph/inference/uncertain/uncertain.cc \
	src/graph/inference/uncertain/uncertain_marginal.cc \
	src/graph/inference/uncertain/uncertain_marginal_imp1.cc \
	src/graph/inference/uncertain/uncertain_mcmc.cc \
	src/graph/inference/modularity/state.cc \
	src/graph/inference/modularity/mcmc.cc \
	src/graph/inference/modularity/multilevel_mcmc.cc \
	src/graph/inference/modularity/multiflip_mcmc.cc \
	src/graph/inference/modularity/gibbs.cc \
	src/graph/inference/norm_cut/state.cc \
	src/graph/inference/norm_cut/mcmc.cc \
	src/graph/inference/norm_cut/multilevel_mcmc.cc \
	src/graph/inference/norm_cut/multiflip_mcmc.cc \
	src/graph/inference/norm_cut/gibbs.cc \
	src/graph/inference/potts/state.cc \
	src/graph/inference/potts/mcmc.cc \
	src/graph/inference/potts/multilevel_mcmc.cc \
	src/graph/inference/potts/multiflip_mcmc.cc \
	src/graph/inference/potts/gibbs.cc \
	src/graph/inference/planted_partition/state.cc \
	src/graph/inference/planted_partition/mcmc.cc \
	src/graph/inference/planted_partition/multilevel_mcmc.cc \
	src/graph/inference/planted_partition/multiflip_mcmc.cc \
	src/graph/inference/planted_partition/gibbs.cc \
	src/graph/inference/partition_centroid/state.cc \
	src/graph/inference/partition_centroid/mcmc.cc \
	src/graph/inference/partition_centroid/multilevel_mcmc.cc \
	src/graph/inference/partition_centroid/multiflip_mcmc.cc \
	src/graph/inference/partition_centroid/gibbs.cc \
	src/graph/inference/partition_modes/partition_mode.cc \
	src/graph/inference/partition_modes/state.cc \
	src/graph/inference/partition_modes/mcmc.cc \
	src/graph/inference/partition_modes/multilevel_mcmc.cc \
	src/graph/inference/partition_modes/multiflip_mcmc.cc \
	src/graph/inference/partition_modes/gibbs.cc

noinst_LTLIBRARIES = libgraph_tool_inference_fast.la
libgraph_tool_inference_fast_la_includedir = $(MOD_DIR)/include
libgraph_tool_inference_fast_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_inference_fast_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_inference_fast_la_CXXFLAGS = $(MOD_CXXFLAGS) -ffast-math -funswitch-loops -funroll-loops -fno-lto
libgraph_tool_inference_fast_la_SOURCES = \
	src/graph/inference/uncertain/dynamics/epidemics/base_vec.cc \
	src/graph/inference/uncertain/dynamics/cising_glauber/base_vec.cc \
	src/graph/inference/uncertain/dynamics/ising_glauber/base_vec.cc \
	src/graph/inference/uncertain/dynamics/potts_glauber/base_vec.cc \
	src/graph/inference/uncertain/dynamics/pseudo_cising/base_vec.cc \
	src/graph/inference/uncertain/dynamics/pseudo_ising/base_vec.cc \
	src/graph/inference/uncertain/dynamics/pseudo_normal/base_vec.cc \
	src/graph/inference/uncertain/dynamics/pseudo_potts/base_vec.cc \
	src/graph/inference/uncertain/dynamics/linear_normal/base_vec.cc \
	src/graph/inference/uncertain/dynamics/lotka_volterra/base_vec.cc \
	src/graph/inference/uncertain/dynamics/normal_glauber/base_vec.cc

nobase_libgraph_tool_inference_la_HEADERS = \
	src/graph/inference/base/loop_template.hh \
	src/graph/inference/base/mcmc.hh \
	src/graph/inference/base/multilevel_mcmc.hh \
	src/graph/inference/base/multiflip_mcmc.hh \
	src/graph/inference/base/gibbs.hh \
	src/graph/inference/base/graph_spec_base.hh \
	src/graph/inference/base/spec_base.hh \
	src/graph/inference/base/state.hh \
	src/graph/inference/blockmodel/state.hh \
	src/graph/inference/blockmodel/elist.hh \
	src/graph/inference/blockmodel/em.hh \
	src/graph/inference/blockmodel/emat.hh \
	src/graph/inference/blockmodel/entries.hh \
	src/graph/inference/blockmodel/entropy.hh \
	src/graph/inference/blockmodel/exhaustive.hh \
	src/graph/inference/blockmodel/gibbs.hh \
	src/graph/inference/blockmodel/layered/deg_stats.hh \
	src/graph/inference/blockmodel/layered/entropy.hh \
	src/graph/inference/blockmodel/layered/spec.hh \
	src/graph/inference/blockmodel/layered/types.hh \
	src/graph/inference/blockmodel/layered/vec_ref.hh \
	src/graph/inference/blockmodel/layered_overlap/entropy.hh \
	src/graph/inference/blockmodel/layered_overlap/spec.hh \
	src/graph/inference/blockmodel/layered_overlap/types.hh \
	src/graph/inference/blockmodel/mcmc.hh \
	src/graph/inference/blockmodel/multiflip_mcmc.hh \
	src/graph/inference/blockmodel/multilevel_mcmc.hh \
	src/graph/inference/blockmodel/overlap/entropy.hh \
	src/graph/inference/blockmodel/overlap/half_edge_adaptor.hh \
	src/graph/inference/blockmodel/overlap/partition.hh \
	src/graph/inference/blockmodel/overlap/partition_simple.hh \
	src/graph/inference/blockmodel/overlap/spec.hh \
	src/graph/inference/blockmodel/overlap/types.hh \
	src/graph/inference/blockmodel/weighted_overlap/entropy.hh \
	src/graph/inference/blockmodel/weighted_overlap/spec.hh \
	src/graph/inference/blockmodel/weighted_overlap/types.hh \
	src/graph/inference/blockmodel/deg_partition.hh \
	src/graph/inference/blockmodel/partition.hh \
	src/graph/inference/blockmodel/spec.hh \
	src/graph/inference/blockmodel/ranked/entropy.hh \
	src/graph/inference/blockmodel/ranked/label_mcmc.hh \
	src/graph/inference/blockmodel/ranked/spec.hh \
	src/graph/inference/blockmodel/ranked/types.hh \
	src/graph/inference/blockmodel/weighted/entropy.hh \
	src/graph/inference/blockmodel/weighted/spec.hh \
	src/graph/inference/blockmodel/weighted/types.hh \
	src/graph/inference/blockmodel/layered_weighted/entropy.hh \
	src/graph/inference/blockmodel/layered_weighted/spec.hh \
	src/graph/inference/blockmodel/layered_weighted/types.hh \
	src/graph/inference/blockmodel/layered_weighted_overlap/entropy.hh \
	src/graph/inference/blockmodel/layered_weighted_overlap/spec.hh \
	src/graph/inference/blockmodel/layered_weighted_overlap/types.hh \
	src/graph/inference/cliques/graph_clique_decomposition.hh \
	src/graph/inference/graph_latent_multigraph.hh \
	src/graph/inference/graph_modularity.hh \
	src/graph/inference/histogram/graph_histogram.hh \
	src/graph/inference/histogram/graph_histogram_mcmc.hh \
	src/graph/inference/loops/bundled_vacate_loop.hh \
	src/graph/inference/loops/exhaustive_loop.hh \
	src/graph/inference/loops/gibbs_loop.hh \
	src/graph/inference/loops/mcmc_loop.hh \
	src/graph/inference/loops/merge_split.hh \
	src/graph/inference/loops/multilevel.hh \
	src/graph/inference/loops/parallel_pseudo_mcmc.hh \
	src/graph/inference/modularity/state.hh \
	src/graph/inference/modularity/spec.hh \
	src/graph/inference/norm_cut/state.hh \
	src/graph/inference/norm_cut/spec.hh \
	src/graph/inference/potts/state.hh \
	src/graph/inference/potts/spec.hh \
	src/graph/inference/planted_partition/state.hh \
	src/graph/inference/planted_partition/spec.hh \
	src/graph/inference/partition_centroid/state.hh \
	src/graph/inference/partition_centroid/spec.hh \
	src/graph/inference/partition_modes/partition_mode.hh \
	src/graph/inference/partition_modes/state.hh \
	src/graph/inference/partition_modes/spec.hh \
	src/graph/inference/support/bisection_sampler.hh \
	src/graph/inference/support/cache.hh \
	src/graph/inference/support/contingency.hh \
	src/graph/inference/support/fibonacci_search.hh \
	src/graph/inference/support/graph_state.hh \
	src/graph/inference/support/int_part.hh \
	src/graph/inference/support/sample_util.hh \
	src/graph/inference/support/segment_sampler.hh \
	src/graph/inference/support/util.hh \
	src/graph/inference/uncertain/dynamics/cising_glauber/setup.hh \
	src/graph/inference/uncertain/dynamics/dynamics.hh \
	src/graph/inference/uncertain/dynamics/dynamics_base.hh \
	src/graph/inference/uncertain/dynamics/dynamics_base_imp.hh \
	src/graph/inference/uncertain/dynamics/dynamics_continuous.hh \
	src/graph/inference/uncertain/dynamics/dynamics_discrete.hh \
	src/graph/inference/uncertain/dynamics/dynamics_discrete_potts.hh \
	src/graph/inference/uncertain/dynamics/dynamics_elist_state.hh \
	src/graph/inference/uncertain/dynamics/dynamics_imp.hh \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc.hh \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_tdelta.hh \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_theta.hh \
	src/graph/inference/uncertain/dynamics/dynamics_mcmc_xdelta.hh \
	src/graph/inference/uncertain/dynamics/dynamics_multiflip_mcmc.hh \
	src/graph/inference/uncertain/dynamics/dynamics_multiflip_mcmc_theta.hh \
	src/graph/inference/uncertain/dynamics/dynamics_swap_mcmc.hh \
	src/graph/inference/uncertain/dynamics/dynamics_util.hh \
	src/graph/inference/uncertain/dynamics/epidemics/setup.hh \
	src/graph/inference/uncertain/dynamics/ising_glauber/setup.hh \
	src/graph/inference/uncertain/dynamics/linear_normal/setup.hh \
	src/graph/inference/uncertain/dynamics/lotka_volterra/setup.hh \
	src/graph/inference/uncertain/dynamics/normal_glauber/setup.hh \
	src/graph/inference/uncertain/dynamics/potts_glauber/setup.hh \
	src/graph/inference/uncertain/dynamics/pseudo_cising/setup.hh \
	src/graph/inference/uncertain/dynamics/pseudo_ising/setup.hh \
	src/graph/inference/uncertain/dynamics/pseudo_normal/setup.hh \
	src/graph/inference/uncertain/dynamics/pseudo_potts/setup.hh \
	src/graph/inference/uncertain/dynamics/test/test.hh \
	src/graph/inference/uncertain/latent_closure.hh \
	src/graph/inference/uncertain/latent_layers.hh \
	src/graph/inference/uncertain/latent_layers_mcmc.hh \
	src/graph/inference/uncertain/latent_mask.hh \
	src/graph/inference/uncertain/measured.hh \
	src/graph/inference/uncertain/sample_edge_sbm.hh \
	src/graph/inference/uncertain/uncertain.hh \
	src/graph/inference/uncertain/uncertain_mcmc.hh \
	src/graph/inference/uncertain/uncertain_util.hh

#####################
# src/graph/layout/ #
#####################

libgraph_tool_layoutdir = $(MOD_DIR)/draw
libgraph_tool_layout_LTLIBRARIES = libgraph_tool_layout.la
libgraph_tool_layout_la_includedir = $(MOD_DIR)/include
libgraph_tool_layout_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_layout_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_layout_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_layout_la_SOURCES = \
	src/graph/layout/graph_arf.cc \
	src/graph/layout/graph_planar_layout.cc \
	src/graph/layout/graph_fruchterman_reingold.cc \
	src/graph/layout/graph_sfdp.cc \
	src/graph/layout/graph_radial.cc \
	src/graph/layout/graph_bind_layout.cc
nobase_libgraph_tool_layout_la_include_HEADERS = \
	src/graph/layout/graph_arf.hh \
	src/graph/layout/graph_sfdp.hh \
	src/graph/layout/quad_tree.hh

#####################
# src/graph/search/ #
#####################

libgraph_tool_searchdir = $(MOD_DIR)/search
libgraph_tool_search_LTLIBRARIES = libgraph_tool_search.la
libgraph_tool_search_la_includedir = $(MOD_DIR)/include
libgraph_tool_search_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_search_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_search_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_search_la_SOURCES = \
	src/graph/search/graph_bfs.cc \
	src/graph/search/graph_dfs.cc \
	src/graph/search/graph_dijkstra.cc \
	src/graph/search/graph_bellman_ford.cc \
	src/graph/search/graph_astar.hh \
	src/graph/search/graph_astar.cc \
	src/graph/search/graph_astar_implicit.cc \
	src/graph/search/graph_search_bind.cc
nobase_libgraph_tool_search_la_include_HEADERS =

#######################
# src/graph/spectral/ #
#######################

libgraph_tool_spectraldir = $(MOD_DIR)/spectral
libgraph_tool_spectral_LTLIBRARIES = libgraph_tool_spectral.la
libgraph_tool_spectral_la_includedir = $(MOD_DIR)/include
libgraph_tool_spectral_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_spectral_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_spectral_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_spectral_la_SOURCES = \
	src/graph/spectral/graph_adjacency.cc \
	src/graph/spectral/graph_incidence.cc \
	src/graph/spectral/graph_laplacian.cc \
	src/graph/spectral/graph_norm_laplacian.cc \
	src/graph/spectral/graph_matrix.cc \
	src/graph/spectral/graph_transition.cc \
	src/graph/spectral/graph_nonbacktracking.cc
nobase_libgraph_tool_spectral_la_include_HEADERS = \
	src/graph/spectral/graph_adjacency.hh \
	src/graph/spectral/graph_incidence.hh \
	src/graph/spectral/graph_laplacian.hh \
	src/graph/spectral/graph_transition.hh \
	src/graph/spectral/graph_nonbacktracking.hh

####################
# src/graph/stats/ #
####################

libgraph_tool_statsdir = $(MOD_DIR)/stats
libgraph_tool_stats_LTLIBRARIES = libgraph_tool_stats.la
libgraph_tool_stats_la_includedir = $(MOD_DIR)/include
libgraph_tool_stats_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_stats_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_stats_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_stats_la_SOURCES = \
	src/graph/stats/graph_histograms.cc \
	src/graph/stats/graph_average.cc \
	src/graph/stats/graph_distance.cc \
	src/graph/stats/graph_distance_sampled.cc \
	src/graph/stats/graph_stats_bind.cc
nobase_libgraph_tool_stats_la_include_HEADERS = \
	src/graph/stats/graph_histograms.hh \
	src/graph/stats/graph_average.hh \
	src/graph/stats/graph_distance_sampled.hh \
	src/graph/stats/graph_distance.hh

#######################
# src/graph/topology/ #
#######################

libgraph_tool_topologydir = $(MOD_DIR)/topology
libgraph_tool_topology_LTLIBRARIES = libgraph_tool_topology.la
libgraph_tool_topology_la_includedir = $(MOD_DIR)/include
libgraph_tool_topology_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_topology_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_topology_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_topology_la_SOURCES = \
	src/graph/topology/graph_all_circuits.cc \
	src/graph/topology/graph_all_distances.cc \
	src/graph/topology/graph_bipartite.cc \
	src/graph/topology/graph_components.cc \
	src/graph/topology/graph_distance.cc \
	src/graph/topology/graph_diameter.cc \
	src/graph/topology/graph_dominator_tree.cc \
	src/graph/topology/graph_isomorphism.cc \
	src/graph/topology/graph_kcore.cc \
	src/graph/topology/graph_matching.cc \
	src/graph/topology/graph_maximal_cliques.cc \
	src/graph/topology/graph_maximal_planar.cc \
	src/graph/topology/graph_maximal_vertex_set.cc \
	src/graph/topology/graph_minimum_spanning_tree.cc \
	src/graph/topology/graph_percolation.cc \
	src/graph/topology/graph_planar.cc \
	src/graph/topology/graph_random_matching.cc \
	src/graph/topology/graph_random_spanning_tree.cc \
	src/graph/topology/graph_reciprocity.cc \
	src/graph/topology/graph_sequential_color.cc \
	src/graph/topology/graph_similarity.cc \
	src/graph/topology/graph_similarity_imp.cc \
	src/graph/topology/graph_subgraph_isomorphism.cc \
	src/graph/topology/graph_topological_sort.cc \
	src/graph/topology/graph_topology.cc \
	src/graph/topology/graph_tsp.cc \
	src/graph/topology/graph_transitive_closure.cc \
	src/graph/topology/graph_vertex_similarity.cc
nobase_libgraph_tool_topology_la_include_HEADERS = \
	src/graph/topology/graph_bipartite_weighted_matching.hh \
	src/graph/topology/graph_components.hh \
	src/graph/topology/graph_kcore.hh \
	src/graph/topology/graph_maximal_cliques.hh \
	src/graph/topology/graph_percolation.hh \
	src/graph/topology/graph_similarity.hh \
	src/graph/topology/graph_vertex_similarity.hh

###################
# src/graph/util/ #
###################

libgraph_tool_utildir = $(MOD_DIR)/util
libgraph_tool_util_LTLIBRARIES = libgraph_tool_util.la
libgraph_tool_util_la_includedir = $(MOD_DIR)/include
libgraph_tool_util_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_util_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_util_la_CXXFLAGS = $(MOD_CXXFLAGS)
libgraph_tool_util_la_SOURCES = \
	src/graph/util/graph_search.cc \
	src/graph/util/graph_util_bind.cc
nobase_libgraph_tool_util_la_include_HEADERS = \
	src/graph/util/graph_search.hh

###################
# src/graph_tool/ #
###################

graph_tool_PYTHON = \
	src/graph_tool/__init__.py \
	src/graph_tool/all.py \
	src/graph_tool/dl_import.py \
	src/graph_tool/decorators.py \
	src/graph_tool/gt_io.py \
	src/graph_tool/openmp.py
graph_tooldir = $(MOD_DIR)

graph_tool_generation_PYTHON = \
	src/graph_tool/generation/__init__.py
graph_tool_generationdir = $(MOD_DIR)/generation

graph_tool_collection_PYTHON = \
	src/graph_tool/collection/__init__.py \
	src/graph_tool/collection/netzschleuder.py \
	src/graph_tool/collection/small.py
dist_graph_tool_collection_DATA = \
	src/graph_tool/collection/atlas.dat.gz \
	src/graph_tool/collection/adjnoun.gt.gz \
	src/graph_tool/collection/as-22july06.gt.gz \
	src/graph_tool/collection/astro-ph.gt.gz \
	src/graph_tool/collection/celegansneural.gt.gz \
	src/graph_tool/collection/cond-mat-2003.gt.gz \
	src/graph_tool/collection/cond-mat-2005.gt.gz \
	src/graph_tool/collection/cond-mat.gt.gz \
	src/graph_tool/collection/dolphins.gt.gz \
	src/graph_tool/collection/football.gt.gz \
	src/graph_tool/collection/hep-th.gt.gz \
	src/graph_tool/collection/karate.gt.gz \
	src/graph_tool/collection/lesmis.gt.gz \
	src/graph_tool/collection/netscience.gt.gz \
	src/graph_tool/collection/polblogs.gt.gz \
	src/graph_tool/collection/polbooks.gt.gz \
	src/graph_tool/collection/power.gt.gz \
	src/graph_tool/collection/pgp-strong-2009.gt.gz \
	src/graph_tool/collection/serengeti-foodweb.gt.gz \
	src/graph_tool/collection/email-Enron.gt.gz
graph_tool_collectiondir = $(MOD_DIR)/collection

graph_tool_correlations_PYTHON = \
	src/graph_tool/correlations/__init__.py
graph_tool_correlationsdir = $(MOD_DIR)/correlations

graph_tool_stats_PYTHON = \
	src/graph_tool/stats/__init__.py
graph_tool_statsdir = $(MOD_DIR)/stats

graph_tool_clustering_PYTHON = \
	src/graph_tool/clustering/__init__.py
graph_tool_clusteringdir = $(MOD_DIR)/clustering

graph_tool_centrality_PYTHON = \
	src/graph_tool/centrality/__init__.py
graph_tool_centralitydir = $(MOD_DIR)/centrality

graph_tool_draw_PYTHON = \
	src/graph_tool/draw/__init__.py \
	src/graph_tool/draw/cairo_draw.py \
	src/graph_tool/draw/gtk_draw.py \
	src/graph_tool/draw/graphviz_draw.py
dist_graph_tool_draw_DATA = \
	src/graph_tool/draw/graph-tool-logo.svg
graph_tool_drawdir = $(MOD_DIR)/draw

graph_tool_inference_PYTHON = \
	src/graph_tool/inference/__init__.py \
	src/graph_tool/inference/base_states.py \
	src/graph_tool/inference/blockmodel.py \
	src/graph_tool/inference/blockmodel_em.py \
	src/graph_tool/inference/clique_decomposition.py \
	src/graph_tool/inference/histogram.py \
	src/graph_tool/inference/layered_blockmodel.py \
	src/graph_tool/inference/nested_blockmodel.py \
	src/graph_tool/inference/overlap_blockmodel.py \
	src/graph_tool/inference/uncertain_blockmodel.py \
	src/graph_tool/inference/reconstruction.py \
	src/graph_tool/inference/mcmc.py \
	src/graph_tool/inference/minimize.py \
	src/graph_tool/inference/modularity.py \
	src/graph_tool/inference/norm_cut.py \
	src/graph_tool/inference/partition_centroid.py \
	src/graph_tool/inference/partition_modes.py \
	src/graph_tool/inference/planted_partition.py \
	src/graph_tool/inference/potts.py \
	src/graph_tool/inference/ranked.py \
	src/graph_tool/inference/latent_multigraph.py \
	src/graph_tool/inference/latent_layers.py \
	src/graph_tool/inference/util.py
graph_tool_inferencedir = $(MOD_DIR)/inference

graph_tool_util_PYTHON = \
	src/graph_tool/util/__init__.py
graph_tool_utildir = $(MOD_DIR)/util

graph_tool_topology_PYTHON = \
	src/graph_tool/topology/__init__.py
graph_tool_topologydir = $(MOD_DIR)/topology

graph_tool_flow_PYTHON = \
	src/graph_tool/flow/__init__.py
graph_tool_flowdir = $(MOD_DIR)/flow

graph_tool_spectral_PYTHON = \
	src/graph_tool/spectral/__init__.py
graph_tool_spectraldir = $(MOD_DIR)/spectral

graph_tool_dynamics_PYTHON = \
	src/graph_tool/dynamics/__init__.py \
	src/graph_tool/dynamics/bp.py
graph_tool_dynamicsdir = $(MOD_DIR)/dynamics

graph_tool_search_PYTHON = \
	src/graph_tool/search/__init__.py
graph_tool_searchdir = $(MOD_DIR)/search
