from modulefinder import Module
import warnings

from ._version import get_versions
from .chain import MarkovChain
from .xx_graph import *
from .partition.partition import Partition
from .updaters import Election

from .tree import (
    epsilon_tree_bipartition,
    bipartition_tree,
    bipartition_tree_random,
    _bipartition_tree_random_all,
    uniform_spanning_tree,
    find_balanced_edge_cuts_memoization,
    ReselectException,
)


# Will need to change this to a logging option later
# It might be good to see how often this happens
warnings.simplefilter("once")

try:
    import geopandas

    # warn about https://github.com/geopandas/geopandas/issues/2199
    if geopandas.options.use_pygeos:
        warnings.warn(
            "GerryChain cannot use GeoPandas when PyGeos is enabled. Disable or "
            "uninstall PyGeos. You can disable PyGeos in GeoPandas by setting "
            "`geopandas.options.use_pygeos = False` before importing your shapefile."
        )
except ModuleNotFoundError:
    pass

__version__ = get_versions()["version"]
del get_versions

from . import _version

__version__ = _version.get_versions()["version"]
