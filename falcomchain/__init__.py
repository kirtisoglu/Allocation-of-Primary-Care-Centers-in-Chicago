from modulefinder import Module
import warnings

from .markovchain.chain import MarkovChain
from .graph import *
from .partition.partition import Partition
from meta import *
from .helper import *
from .travel_time import *
from prepared_data import *



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
