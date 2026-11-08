(ranked)=

# Ordered community structure

The modular structure of directed networks might possess an inherent
ordering of the groups, such that most edges flow either “downstream” or
“upstream” according to that ordering. The directed version of the SBM
will inherently capture this ordering, but it will not be visible from
the model parameters — in particular the group labels — since the model is
invariant to group permutations. This ordering can be obtained from a
modified version of the model {cite}`inf-peixoto_ordered_2022`, which can be
inferred with graph-tool using
{class}`~graph_tool.inference.RankedBlockState`. This class behaves just
like {class}`~graph_tool.inference.BlockState`, therefore all algorithms
described in this documentation work in the same way (including when
{class}`~graph_tool.inference.NestedBlockState` is used).

Below we show how this model can be inferred for a {ns}`faculty_hiring` network.

```{testsetup} ordered
mkchdir(DOC_DIR + "/demos/inference/output")
gt.seed_rng(42)
```

```{testcode} ordered
g = gt.collection.ns["faculty_hiring/computer_science"].copy()

# For visualization purposes, it will be more useful to work with a
# weighted graph than with a multigraph, but the results are
# insensitive to this.

ew = gt.contract_parallel_edges(g)

# We will use a nested SBM, with the base state being the ordered SBM.

state = gt.NestedBlockState(g, base_state=gt.RankedBlockState, base_state_args=dict(eweight=ew))

# The number of iterations below is sufficient for a good estimate of
# the ground state for this network.

for i in range(100):
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

# We can use hfd_layout() with the nested partition to obtain a ranked
# visualization, where the centers of mass of the groups are ordered
# vertically according to their ranks (which are negated so that the lowest
# ranks appear at the top), and kept apart horizontally.

rank = state.levels[0].get_vertex_order().t(lambda x: -x, value_type="double")
pos = gt.hfd_layout(g, b=state.get_bs(), rank=rank, rmu=0, rcm_dist=.4,
                    rcm_xdist=.4)

state.levels[0].draw(pos=pos, edge_pen_width=gt.prop_to_size(ew, 1, 5),
                     output="hiring.svg")
```

:::{figure} output/hiring.svg
:align: center
:width: 450px

Best fit of the ordered degree-corrected SBM to a faculty hiring
network. The vertical position indicates the rank, and the edge color
the edge direction: upstream (blue), downstream (red), lateral
(grey).
:::

It is possible to perform model comparison with other model variations
in the same manner as described in Sec. {ref}`model_selection` above.
