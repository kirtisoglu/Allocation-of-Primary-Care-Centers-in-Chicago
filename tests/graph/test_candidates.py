
import geopandas as gpd
import matplotlib as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from falcomchain.candidates.candidates import random_candidates_from_blocks


def test_grid_cells():
    
    assert

def select_candidates_geographically(Chicago, k, grid_size=0.01):

    return result




def test_random_candidate_blocks(data):  
      
    blocks = random_candidates_from_blocks

    return phc_df, selected_blocks



def add_candidates(data, graph, collection, attr):
    """
    For a given collection of nodes, assigns attribute `attr` to all nodes in the graph.
    Sets value to 1 for nodes in `collection`, and 0 otherwise.
    Also adds a column to the `data` (e.g., a pandas DataFrame or Series) with the same values.
    """
    # Create a binary mask: 1 if in collection, else 0
    binary_mask = data.index.to_series().apply(lambda node: int(node in collection))

    # Assign to data
    data[attr] = binary_mask

    # Assign to graph
    for node, value in binary_mask.items():
        graph.nodes[node][attr] = value