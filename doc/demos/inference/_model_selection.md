(model_selection)=

# Model selection

As mentioned above, one can select the best model according to the choice that
yields the smallest description length {cite}`inf-peixoto_model_2015`. For instance,
in case of the {ns}`political blogs <polblogs>` network we have

```{testsetup} model-selection
gt.seed_rng(42)
```

```{testcode} model-selection
g = gt.collection.ns["polblogs"]

state_ndc = gt.minimize_nested_blockmodel_dl(g, base_state_args=dict(deg_corr=False))
state_dc  = gt.minimize_nested_blockmodel_dl(g, base_state_args=dict(deg_corr=True))

print("Non-degree-corrected DL:\t", state_ndc.entropy())
print("Degree-corrected DL:\t", state_dc.entropy())
```

```{testoutput} model-selection
:options: +NORMALIZE_WHITESPACE

Non-degree-corrected DL:    63185.4565554...
Degree-corrected DL:        61411.6708563...
```

Since it yields the smallest description length, the degree-corrected
fit should be preferred. The statistical significance of the choice can
be assessed by inspecting the posterior odds ratio
{cite}`inf-peixoto_nonparametric_2017`

$$
\Lambda &= \frac{P(\boldsymbol b, \mathcal{H}_\text{NDC} | \boldsymbol A)}{P(\boldsymbol b, \mathcal{H}_\text{DC} | \boldsymbol A)} \\
        &= \frac{P(\boldsymbol A, \boldsymbol b | \mathcal{H}_\text{NDC})}{P(\boldsymbol A, \boldsymbol b | \mathcal{H}_\text{DC})}\times\frac{P(\mathcal{H}_\text{NDC})}{P(\mathcal{H}_\text{DC})} \\
        &= \exp(-\Delta\Sigma)
$$

where $\mathcal{H}_\text{NDC}$ and $\mathcal{H}_\text{DC}$
correspond to the non-degree-corrected and degree-corrected model
hypotheses (assumed to be equally likely *a priori*), respectively, and
$\Delta\Sigma$ is the difference of the description length of both
fits. In our particular case, we have

```{testcode} model-selection
print(u"ln \u039b: ", state_dc.entropy() - state_ndc.entropy())
```

```{testoutput} model-selection
:options: +NORMALIZE_WHITESPACE

ln Λ:  -1773.785699...
```

The precise threshold that should be used to decide when to [reject a
hypothesis](https://en.wikipedia.org/wiki/Hypothesis_testing) is subjective and
context-dependent, but the value above implies that the particular
degree-corrected fit is around $\mathrm{e}^{1774} \approx 10^{770}$ times more
likely than the non-degree corrected one, and hence it can be safely concluded
that it provides a substantially better fit.

Although it is often true that the degree-corrected model provides a
better fit for many empirical networks, there are also exceptions. For
example, for the American football network above, we have:

```{testsetup} model-selection-football
gt.seed_rng(43)
```

```{testcode} model-selection-football
g = gt.collection.data["football"]

state_ndc = gt.minimize_nested_blockmodel_dl(g, base_state_args=dict(deg_corr=False))
state_dc  = gt.minimize_nested_blockmodel_dl(g, base_state_args=dict(deg_corr=True))

print("Non-degree-corrected DL:\t", state_ndc.entropy())
print("Degree-corrected DL:\t", state_dc.entropy())
print(u"ln \u039b:\t\t\t", state_dc.entropy() - state_ndc.entropy())
```

```{testoutput} model-selection-football
:options: +NORMALIZE_WHITESPACE

Non-degree-corrected DL:    1733.525685...
Degree-corrected DL:        1782.195884...
ln Λ:                       48.670199...
```

Hence, with a posterior odds ratio of $\Lambda \approx
\mathrm{e}^{48} \approx 10^{21}$ in favor of the non-degree-corrected
model, we conclude that the degree-corrected variant is an unnecessarily
complex description for this network.
