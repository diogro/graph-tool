(matplotlib_sec)=

# Integration with matplotlib

The drawing capabilities of `graph-tool` (see {mod}`~graph_tool.draw`
module) can be integrated with [matplotlib](https://matplotlib.org),
as we demonstrate in the following.

:::{note}
:class: margin

Integration with {mod}`matplotlib` works with every backend, but [vector
drawing](https://en.wikipedia.org/wiki/Vector_graphics) only works with a
[cairo](https://cairographics.org)-based backend (e.g. `cairo` or
`GTK3Cairo`). The backend can be changed by calling
{func}`matplotlib.pyplot.switch_backend`:

```
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
```

When using a backend not based on cairo,
[rasterization](https://en.wikipedia.org/wiki/Raster_graphics) will be used
instead. In this case, the resolution can be controlled via the `dpi` parameter
of {class}`matplotlib.figure.Figure`.
:::

Drawing with matplotlib is done by calling {func}`~graph_tool.draw.graph_draw`
and passing a container (e.g. {class}`matplotlib.axes.Axes`) as the `mplfig`
parameter. When this option is passed, the function will return a
{class}`~graph_tool.draw.GraphArtist` object that has been added to the figure.

:::{tip}
:class: margin

[Axis autoscaling](https://matplotlib.org/stable/users/explain/axes/autoscale.html) will work
as expected with {class}`~graph_tool.draw.GraphArtist`, but the aspect ratio
will be set by the figure axis shape, rather than the node positions themselves.

In order to restore the aspect ratio independently of the axis shape,
{class}`~graph_tool.draw.GraphArtist` offers a
{meth}`~graph_tool.draw.GraphArtist.fit_view` method that does this automatically.

Furthermore, when calling {func}`~graph_tool.draw.graph_draw` without
integrating with matplotlib, the node positions correspond to cairo
coordinates, which have an origin in the upper left corner, and with the y
axis increasing from top to bottom.

In order for the visualization to be the same when matplotlib is being used,
the y axis needs to be flipped by inverting the limits with
{meth}`matplotlib.axes.Axes.set_ylim`.

Alternatively, the option `yflip` can be passed to
{meth}`~graph_tool.draw.GraphArtist.fit_view` for this to be done
automatically.
:::

The example below shows how to plot several graphs in different subplots of the
same figure.

```{testsetup} matplotlib
mkchdir(DOC_DIR + "/demos/matplotlib/output")
import matplotlib
backend = matplotlib.get_backend()
gt.seed_rng(44)
```

```{testcode} matplotlib
import graph_tool.all as gt
import matplotlib.pyplot as plt

# plt.switch_backend("cairo")   # to enable vector drawing

fig, ax = plt.subplots(2, 2, figsize=(12, 11.5))

g = gt.collection.data["polbooks"]

a = gt.graph_draw(g, g.vp.pos, vertex_size=1.5, mplfig=ax[0,0])
a.fit_view(yflip=True)

ax[0,0].set_xlabel("$x$ coordinate")
ax[0,0].set_ylabel("$y$ coordinate")

state = gt.minimize_nested_blockmodel_dl(g)

a = state.draw(mplfig=ax[0,1])[0]
a.fit_view(yflip=True)

ax[0,1].set_xlabel("$x$ coordinate")
ax[0,1].set_ylabel("$y$ coordinate")

g = gt.collection.data["lesmis"]
a = gt.graph_draw(g, g.vp.pos, vertex_size=1.5, mplfig=ax[1,0])
a.fit_view(yflip=True)

ax[1,0].set_xlabel("$x$ coordinate")
ax[1,0].set_ylabel("$y$ coordinate")

state = gt.minimize_nested_blockmodel_dl(g)

a = state.draw(mplfig=ax[1,1])[0]
a.fit_view(yflip=True)

ax[1,1].set_xlabel("$x$ coordinate")
ax[1,1].set_ylabel("$y$ coordinate")

plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.06)
fig.savefig("gt-mpl.svg")
```

```{testcleanup} matplotlib
plt.switch_backend(backend)
```

:::{figure} output/gt-mpl.svg
:width: 90%

Four subplots showing networks drawn using graph-tool.
:::

## Integration with {mod}`~mpl_toolkits.basemap`

As a slightly more elaborate example, below we show how we can draw the
{ns}`European airline <eu_airlines>` graph on a map using
{mod}`mpl_toolkits.basemap`.

```{testcode} matplotlib

import numpy as np
from itertools import chain
from mpl_toolkits.basemap import Basemap

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = gt.collection.ns["eu_airlines"]
pos = gt.group_vector_property([g.vp.nodeLong, g.vp.nodeLat])

m = Basemap(projection='ortho', resolution=None,
            lat_0=g.vp.nodeLat.fa.mean(), lon_0=g.vp.nodeLong.fa.mean())

m.shadedrelief(scale=.2)

lats = m.drawparallels(np.linspace(-90, 90, 13))
lons = m.drawmeridians(np.linspace(-180, 180, 13))
lat_lines = chain(*(tup[1][0] for tup in lats.items()))
lon_lines = chain(*(tup[1][0] for tup in lons.items()))
all_lines = chain(lat_lines, lon_lines)
for line in all_lines:
   line.set(linestyle='-', alpha=0.3, color='w')

a = gt.graph_draw(g, pos=pos.t(lambda x: m(*x)),          # project positions
                  edge_color=(.1,.1,.1,.1), mplfig=ax)

a.fit_view()
a.set_zorder(10)
plt.tight_layout()
fig.savefig("gt-map.svg")
```

:::{figure} output/gt-map.svg
:width: 90%

{ns}`Network of European flights <eu_airlines>` drawn on the globe with an
[orthographic projection](https://en.wikipedia.org/wiki/Orthographic_projection).
:::
