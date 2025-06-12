import pytest
from unittest.mock import patch

import random
from typing import Optional
import networkx as nx

from falcomchain.graph import Graph
from falcomchain.tree import SpanningTree
from falcomchain.partition import Partition


random.seed(2025)


@pytest.fixture
def tree_with_twenty_nodes():
    """Returns a tree that looks like this:
    
    1 - 2 - 3 - 4
        |       |
    5 - 6 - 7   8 - 9 - 10
    |       |       |
    11-12   13      14
    |
    15-16-17
    |
    18-19
    |
    20
    
    """
    graph = Graph()
    graph.add_edges_from(
        [
            (1, 2),(2, 3),(3, 4),
            (2, 6),(4, 8),
            (5, 6),(6, 7),(8, 9),(9, 10),
            (5, 11),(7, 13),(9, 14),
            (11, 12),
            (11, 15),
            (15, 16),(16, 17),
            (15, 18),
            (18, 19),
            (18, 20),
        ]
    )
    return graph


@pytest.fixture
def tree_with_attributes(tree_with_twenty_nodes):
    
    candidates = {3, 7, 8, 10, 12, 17, 18}  # for pop target = 40 and n_teams = 5, where total population = 200.
    for node in tree_with_twenty_nodes:
        tree_with_twenty_nodes.nodes[node]["population"] = 10
        tree_with_twenty_nodes.nodes[node]["area"] = 10
        tree_with_twenty_nodes.nodes[node]["density"] = 1
        if node in candidates:
            tree_with_twenty_nodes.nodes[node]["candidate"] = True
        else:
            tree_with_twenty_nodes.nodes[node]["candidate"] = False
    return tree_with_twenty_nodes



@pytest.fixture
def spanningtree_with_forced_root(tree_with_attributes):
    n_teams = 5
    epsilon = 0.3  # population of 30 will be excepted since pop_target = 40 and (1-0.3)*40 < 30
    pop_target = 40
    capacity_level = 1
    column_names = ['population', 'area', 'candidate', 'density']
    two_sided = True
    supergraph = False
    
    with patch("random.choice", return_value=2):
        tree = SpanningTree(graph=tree_with_attributes, ideal_pop=pop_target, epsilon=epsilon, n_teams=n_teams, 
                         capacity_level=capacity_level, column_names=column_names, two_sided=two_sided, supergraph=supergraph) 
    return tree


@pytest.fixture
def partition():
    
   
    
    return partition


