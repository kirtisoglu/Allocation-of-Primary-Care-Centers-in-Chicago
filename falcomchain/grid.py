"""
This module provides a Grid class used for creating and manipulating grid partitions.
It's part of the GerryChain suite, designed to facilitate experiments with redistricting
plans without the need for extensive data processing. This module relies on NetworkX for
graph operations and integrates with GerryChain's Partition class.

Dependencies:

- math: For math.floor() function.
- networkx: For graph operations with using the graph structure in
    :class:`~gerrychain.graph.Graph`.
- typing: Used for type hints.
"""

import math
import networkx
from partition import Partition
from graph import Graph

from markovchain import polsby_popper
from typing import Callable, Dict, Optional, Tuple, Any
import random


class Grid:
    """
    The :class:`Grid` class creates a grid with pre-specified attributes and attribute values.
    It is useful for running little experiments with Falcomchain without needing to do any data 
    processing or cleaning to get started.

    Example usage::

        grid = Grid((10,10))

    The nodes of ``grid.graph`` are labelled by numbers from 1 to m x n where m, n are dimensions. 
    Node attributes: area, C_X, C_Y, population, candidate  
    Edge attributes: shared_perimeter
    """


    def __init__(
        self,
        dimensions: Tuple[int, int],
        num_candidates: int,
        density: str,
        threshold: Optional[tuple] = None,
        candidate_ignore: Optional[int] = None,
    ) -> None:
        """
        :param dimensions: The grid dimensions (rows, columns), defaults to None.
        :type dimensions: Tuple[int, int], optional
        :param num_candidates:
        :type num_candidates:
        :param density: receives one of 'uniform', 'opposite', 'corners'.
        :type density: string
        :param candidate_ignore: a value of (x_0,y_0). Any node (x, y) with x < x_0 or y < y_0 will not be a candidate.
        :type candidate_ignore: tuple
        
        :raises Exception: If neither dimensions nor parent is provided.
        """
        if len(dimensions)!=2:
            raise Exception("Dimension must be 2.")

        self.dimensions = dimensions
        self.density = density
        self.graph = self.create_grid_graph()
        self.num_candidates = num_candidates
        self.threshold = threshold
        self.candidate_ignore = candidate_ignore
        
        self.assign_coordinates()
        self.assign_candidates()
        self.tag_boundary_nodes()
        
        if self.density != 'uniform':
            self.assign_population()

        # final step
        self.graph = Graph.from_networkx(self.graph)  # convert graph into Graph object


    # Main function which creates a grid graph with required node and edge attributes
    def create_grid_graph(self) -> Graph:
        """
        Creates a grid graph with the specified dimensions.
        Optionally includes diagonal connections between nodes.

        :param dimensions: The grid dimensions (rows, columns).
        :type dimensions: Tuple[int, int]
        :param with_diagonals: If True, includes diagonal connections.
        :type with_diagonals: bool

        :returns: A grid graph.
        :rtype: Graph

        :raises ValueError: If the dimensions are not a tuple of length 2.
        """
        m, n = self.dimensions
        graph = networkx.generators.lattice.grid_2d_graph(m, n)

        networkx.set_edge_attributes(graph, 1, "shared_perim")

        networkx.set_node_attributes(graph, 50, "population")
        networkx.set_node_attributes(graph, 1, "C_X")
        networkx.set_node_attributes(graph, 1, "C_Y")
        
        networkx.set_node_attributes(graph, 1, "area")
        networkx.set_node_attributes(graph, False, "candidate")

        return graph


    def assign_coordinates(self) -> None:
        """
        Sets the specified attribute to the specified value for all nodes in the graph.

        :param graph: The graph to modify.
        :type graph: Graph
        :param attribute: The attribute to set.
        :type attribute: Any
        :param value: The value to set the attribute to.
        :type value: Any

        :returns: None
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]['C_X'] = node[0]
            self.graph.nodes[node]['C_Y'] = node[1]
            

    def assign_candidates(self) -> None:
        "Sets self.num_candidates many nodes as candidates uniformly random on permitted region"
        nodes = set(list(self.graph.nodes))
        
        if self.candidate_ignore != None:
            x_0, y_0 = node
            ignore = {node for node in set if node[0] < x_0 or node[1] < y_0}
            nodes = nodes - ignore
        
        candidates = random.choices(population = list(nodes) , k = self.num_candidates)
        
        for node in candidates:
            self.graph.nodes[node]['candidate'] = True
        
        

    def tag_boundary_nodes(self) -> None:
        """
        Adds the boolean attribute ``boundary_node`` to each node in the graph.
        If the node is on the boundary of the grid, that node also gets the attribute
        ``boundary_perim`` which is determined by the function :func:`get_boundary_perim`.

        :param graph: The graph to modify.
        :type graph: Graph
        :param dimensions: The dimensions of the grid.
        :type dimensions: Tuple[int, int]

        :returns: None
        """
        m, n = self.dimensions
        for node in self.graph.nodes:
            if node[0] in [0, m - 1] or node[1] in [0, n - 1]:
                self.graph.nodes[node]["boundary_node"] = True
                self.graph.nodes[node]["boundary_perim"] = self.get_boundary_perim(node)
            else:
                self.graph.nodes[node]["boundary_node"] = False


    def get_boundary_perim(self, node: Tuple[int, int]) -> int:
        """
        Determines the boundary perimeter of a node on the grid.
        The boundary perimeter is the number of sides of the node that
        are on the boundary of the grid.

        :param node: The node to check.
        :type node: Tuple[int, int]
        :param dimensions: The dimensions of the grid.
        :type dimensions: Tuple[int, int]

        :returns: The boundary perimeter of the node.
        :rtype: int
        """
        m, n = self.dimensions
        if node in [(0, 0), (m - 1, 0), (0, n - 1), (m - 1, n - 1)]:
            return 2
        elif node[0] in [0, m - 1] or node[1] in [0, n - 1]:
            return 1
        else:
            return 0


    def assign_population(self) -> int:
        """
        Assigns a color (as an integer) to a node based on its x-coordinate.

        This function is used to partition the grid into two parts based on a given threshold.
        Nodes with an x-coordinate less than or equal to the threshold are assigned one color,
        and nodes with an x-coordinate greater than the threshold are assigned another.

        :param node: The node to color, represented as a tuple of coordinates (x, y).
        :type node: Tuple[int, int]
        :param threshold: The x-coordinate value that determines the color assignment.
        :type threshold: int

        :returns: An integer representing the color of the node. Returns 0 for nodes with
            x-coordinate less than or equal to the threshold, and 1 otherwise.
        :rtype: int
        """
            
        if self.density == "opposite":  
            for node in self.graph.nodes:
                x, y = node
                if x >= self.threshold[0] and y>= self.threshold[1]:
                    self.graph.nodes[node]['population'] = 70
                elif x < self.threshold[0] and y < self.threshold[1]:
                    self.graph.nodes[node]['population'] = 70
                else:
                    self.graph.nodes[node]['population'] = 30
                  
        if self.density == "corners":
            k_1, k_2 = self.threshold
            m, n = self.dimensions
            for node in self.graph.nodes:
                x, y = node
                if k_1 <= x < m - k_1 or k_2 <= y < n - k_2:
                    self.graph.nodes[node]['population'] = 30
                else:
                    self.graph.nodes[node]['population'] = 70
            
 


