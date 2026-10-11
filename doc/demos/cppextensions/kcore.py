import graph_tool

# We import the C++ module (called libkcore.so)
import libkcore

# The function below is what will be used from Python, and dispatch to the
# C++ module.

def kcore_decomposition(g, core=None):
    if core is None:
        core = g.new_vertex_property("int")

    # For graph objects we need to pass the internal GraphInterface which is
    # accessed via g._Graph__graph.

    # Property maps need to be passed as std::any objects, which is done via
    # the _get_any() method.

    libkcore.kcore(g._Graph__graph, core._get_any())
    return core