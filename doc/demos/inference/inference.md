(inference-howto)=

# Inferring modular network structure

`graph-tool` includes algorithms to identify the large-scale structure
of networks via statistical inference in the
{mod}`~graph_tool.inference` submodule. Here we explain the basic
functionality with self-contained examples. For a more thorough
theoretical introduction to the methods described here, the reader is
referred to {cite}`inf-peixoto_bayesian_2019`.

See also {cite}`inf-peixoto_descriptive_2023` and the corresponding [blog
post](https://skewed.de/tiago/blog/descriptive-inferential) for an overall
discussion about inferential approaches to structure identification in networks,
and how they contrast with descriptive approaches.

```{include} _background.md
```

```{include} _minimization.md
```

```{include} _model_selection.md
```

```{include} _sampling.md
```

```{include} _model_class_selection.md
```

```{include} _edge_weights.md
```

```{include} _layers.md
```

```{include} _assortative.md
```

```{include} _ranked.md
```

# References

```{bibliography}
:keyprefix: inf-
```
