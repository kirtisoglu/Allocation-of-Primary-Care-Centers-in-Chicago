import networkx as nx
import random
import copy
import collections

import random
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Hashable,
    Sequence,
    Tuple,
)

# Used in bipartition of a subtree when a spanning tree cannot be found. Kruskal is the fastest algorithm for finding a spanning tree.
def random_spanning_tree(
    graph: nx.Graph, weight_dict: Optional[Dict] = None
) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph: The input graph to build the spanning tree from. Should be a Networkx Graph.
    :type graph: nx.Graph
    :param weight_dict: Dictionary of weights to add to the random weights used in region-aware
        variants.
    :type weight_dict: Optional[Dict], optional

    :returns: The maximal spanning tree represented as a Networkx Graph.
    :rtype: nx.Graph
    """
    if weight_dict is None:
        weight_dict = dict()

    for edge in graph.edges():
        weight = random.random()
        for key, value in weight_dict.items():
            if (
                graph.nodes[edge[0]][key] == graph.nodes[edge[1]][key]
                and graph.nodes[edge[0]][key] is not None
            ):
                weight += value

        graph.edges[edge]["random_weight"] = weight

    spanning_tree = nx.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
    return spanning_tree




def uniform_spanning_tree(graph: nx.Graph, choice: Callable = random.choice) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph. Uses Wilson's algorithm.

    :param graph: Networkx Graph
    :type graph: nx.Graph
    :param choice: :func:`random.choice`. Defaults to :func:`random.choice`.
    :type choice: Callable, optional

    :returns: A spanning tree of the graph chosen uniformly at random.
    :rtype: nx.Graph
    """

    new_graph = graph.copy(as_view=False)

    # remove the edges between stops before sampling.
    """    for edge in new_graph.edges:
        endpoint1, endpoint2 = edge
        if new_graph.nodes[endpoint1]["id"] == 1 and new_graph.nodes[endpoint2]["id"] == 1:
            new_graph.remove_edge(endpoint1, endpoint2)"""

    root = choice(list(new_graph.nodes))
    tree_nodes = set([root])
    next_node = {root: None}

    for node in new_graph.nodes:
        u = node
        while u not in tree_nodes:
            next_node[u] = choice(list(new_graph[u].keys()))
            u = next_node[u]
            
        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    # re-assign the attributes of the nodes.
    for node in G.nodes:
        G.nodes[node].update(graph.nodes[node])

    return G