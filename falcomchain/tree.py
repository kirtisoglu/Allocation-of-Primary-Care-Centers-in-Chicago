"""
This module provides tools and algorithms for manipulating and analyzing graphs,
particularly focused on partitioning graphs based on population data. It leverages the
NetworkX library to handle graph structures and implements various algorithms for graph
partitioning and tree traversal.

Key functionalities include:

- Predecessor and successor functions for graph traversal using breadth-first search.
- Implementation of random and uniform spanning trees for graph partitioning.
- The `PopulatedGraph` class, which represents a graph with additional population data,
  and methods for assessing and modifying this data.
- Functions for finding balanced edge cuts in a populated graph, either through
  contraction or memoization techniques.
- A suite of functions (`bipartition_tree`, `recursive_tree_part`, `get_seed_chunks`, etc.)
  for partitioning graphs into balanced subsets based on population targets and tolerances.
- Utility functions like `get_max_prime_factor_less_than` and `recursive_seed_part_inner`
  to assist in complex partitioning tasks.

Dependencies:

- networkx: Used for graph data structure and algorithms.
- random: Provides random number generation for probabilistic approaches.
- typing: Used for type hints.

Last Updated: 8 October 2024
"""

import networkx as nx
from networkx.algorithms import tree

from functools import partial
from inspect import signature
import random
from collections import deque, namedtuple
import itertools
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
import warnings
import time



class SpanningTree:
    """
    A class representing a spanning tree with population and density information of its subtrees.

    :ivar graph: The underlying graph structure.
    :type graph: nx.Graph
    :ivar subsets: A dictionary mapping nodes to their subsets.
    :type subsets: Dict
    :ivar population: A dictionary mapping nodes to their populations.
    :type population: Dict
    :ivar tot_pop: The total population of the graph.
    :type tot_pop: Union[int, float]
    :ivar ideal_pop: The ideal population for each district.
    :type ideal_pop: float
    :ivar epsilon: The tolerance for population deviation from the ideal population within each
        district.
    :type epsilon: float
    :ivar preccessor: The predecessor
    :type preccessor: Dict
    :ivar successor: 
    :type successor: Dict
    :ivar
    :type
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        column_names: tuple[str],
        ideal_pop: Union[float, int],
        epsilon: float,
        n_teams: int,
        capacity_level: int,
        density: Optional[float] = None,
        add_root: Optional[bool] = False,
    ) -> None:
        """ 
        :param graph: The underlying graph structure.
        :type graph: nx.Graph
        :param populations: A dictionary mapping nodes to their populations.
        :type populations: Dict
        :param ideal_pop: The ideal population for each district.
        :type ideal_pop: Union[float, int]
        :param epsilon: The tolerance for population deviation as a percentage of
            the ideal population within each district.
        :type epsilon: float
        """
        self.graph = graph
        self.density = density
        
        root = random.choice([x for x in self.graph.nodes if self.graph.degree(x) > 1])
        if add_root:  # For considering that all nodes might be in the same district
            artifical_node = -1
            self.graph.add_node(artifical_node)
            self.graph.add_edge(root, artifical_node)
            self.root = artifical_node
        else:
            self.root = root
        
        self.pop_col, self.area_col, self.facility_col = column_names[0], column_names[1], column_names[2]
        self.ideal_pop = ideal_pop
        self.n_teams = n_teams
        self.capacity_level = capacity_level
        self.epsilon = epsilon
        self.successors = self.successors()  # do we need to save this?
        self.subtree_info = self.calc_subtree() 
        self.tot_area, self.tot_pop = sum(self.subtree_info[c][0] for c in self.successors[self.root]), sum(self.subtree_info[c][1] for c in self.successors[self.root])
    
        


    def __iter__(self):
        return iter(self.graph)
        
    def successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self.graph, self.root)}

    def calc_subtree(self):
        """
        Calculates population and density of the subtree under each node
        in the graph by traversing the graph using a depth-first search.
        return: dictionary {node: (population, area)}
        """
    
        subtree_info = {}
        stack = deque(n for n in self.successors[self.root])
        while stack:
            next_node = stack.pop() 
            if next_node not in subtree_info:
                if next_node in self.successors:
                    children = self.successors[next_node]
                    if all(c in subtree_info for c in children):
                        subtree_info[next_node] = (sum(subtree_info[c][0] for c in children) + self.graph.nodes[next_node][self.pop_col], 
                                                   sum(subtree_info[c][1] for c in children) + self.graph.nodes[next_node][self.area_col])
                        
                    else:
                        stack.append(next_node)
                        for c in children:
                            if c not in subtree_info:
                                stack.append(c)
                else:
                    subtree_info[next_node] = (self.graph.nodes[next_node][self.pop_col], self.graph.nodes[next_node][self.area_col])

        return subtree_info

    def has_center(self, node):
        return self.graph.nodes[node].get(self.facility_col, False) 
 
    
    def has_ideal_pop(self):
        "Checks if subtree beneath a node has an ideal population within the spanning tree up to epsilon."
        return
        
    def has_ideal_density(self, node):
        " Checks if the subtree beneath a node has an ideal density up to tolerance 'density'."
        d = self.tot_area / self.tot_pop
        d_node = self.subtree_info[node][1] / self.subtree_info[node][0]
        return ((1- self.density)*d <= d_node <=(1 + self.density)*d)
    
    def __repr__(self) -> str:
        graph_info = (
            f"Graph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"
        )
        return (
            f"{self.__class__.__name__}("
            f"graph={graph_info}, "
            f"total_population={self.tot_pop}, "
            f"ideal_population={self.ideal_pop}, "
            f"epsilon={self.epsilon})"
        )

# Tuple that is used in the find_balanced_edge_cuts function
Cut = namedtuple("Cut", "subset assigned_teams")
Cut.__new__.__defaults__ = (None, None, None)
Cut.__doc__ = "Represents a cut in a graph."
#Cut.edge.__doc__ = "The edge where the cut is made. Defaults to None."
Cut.subset.__doc__ = ("The (frozen) subset of nodes on one side of the cut. Defaults to None.")
Cut.assigned_teams.__doc__ = "The number of doctor-nurse teams for the subtree beneath the cut edge."

class BipartitionWarning(UserWarning):
    """
    Generally raised when it is proving difficult to find a balanced cut.
    """

    pass

class ReselectException(Exception):
    """
    Raised when the tree-splitting algorithm is unable to find a
    balanced cut after some maximum number of attempts, but the
    user has allowed the algorithm to reselect the pair of
    districts from parent graph to try and recombine.
    """

    pass

class BalanceError(Exception):
    """Raised when a balanced cut cannot be found."""

class PopulationBalanceError(Exception):
    """Raised when the population of a district is outside the acceptable epsilon range."""




def random_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph: The input graph to build the spanning tree from. Should be a Networkx Graph.
    :type graph: nx.Graph

    :returns: The maximal spanning tree represented as a Networkx Graph.
    :rtype: nx.Graph
    """
    for edge in graph.edges():
        weight = random.random()
        graph.edges[edge]["random_weight"] = weight
        
    spanning_tree = tree.minimum_spanning_tree(graph, algorithm="kruskal", weight="random_weight")
    return spanning_tree

def uniform_spanning_tree(graph: nx.Graph) -> nx.Graph:
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
    root = random.choice(list(graph.node_indices))
    tree_nodes = set([root])
    next_node = {root: None}

    for node in graph.node_indices:
        u = node
        while u not in tree_nodes:
            next_node[u] = random.choice(list(graph.neighbors(u)))
            u = next_node[u]

        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    G.add_nodes_from(graph.nodes(data=True))
    
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    return G


"""------------------------------------------------------------------------------------------------------------------------"""



def _part_nodes(start, succ):
    """
    Partitions the nodes of a graph into two sets.
    based on the start node and the successors of the graph.

    :param start: The start node.
    :type start: Any
    :param succ: The successors of the graph.
    :type succ: Dict

    :returns: A set of nodes for a particular district (only one side of the cut).
    :rtype: Set
    """
    nodes = set()
    queue = deque([start])
    while queue:
        next_node = queue.pop()
        if next_node not in nodes:
            nodes.add(next_node)
            if next_node in succ:
                for c in succ[next_node]:
                    if c not in nodes:
                        queue.append(c)
    
    return nodes




"""  ------------------------------ Main Functions ------------------------------  """

 

def find_edge_cuts(h: SpanningTree, add_root: Optional[bool] = False) -> List[Cut]:
    """
    This function takes a SpanningTree object as input and returns a list of balanced edge cuts. 
    A balanced edge cut is defined as a cut that divides the graph into two subsets, such that 
    the population of each subset is close to the ideal population defined by the SpanningTree object.

    :param h: The SpanningTree object representing the graph.
    :type h: SpanningTree
    :param add_root: If set to True, an artifical node is connected to root and edge is considered as a possible cut.
    :type add_root: bool, optional

    :returns: A list of balanced edge cuts.
    :rtype: List[Cut]
    """

    cuts = []

    for node, tree_info in h.subtree_info.items():
        part_nodes = _part_nodes(node, h.successors)
        assign_team = 1    
        
        if h.density != None:  # skips nodes that does not have ideal density. We can eliminate these nodes in subtree_info calculation to fasten the algorithm 
            if h.has_ideal_density(node)==False:
                continue
            
        if add_root==False:  
            while assign_team < h.capacity_level + 1 and assign_team < h.n_teams:
                if abs(tree_info[0] - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:    
                    if any(h.has_center(node) for node in part_nodes):   # h can keep all centers. we can check that if intersection is not empty
                        cuts.append(Cut(subset=frozenset(part_nodes), assigned_teams = assign_team))     
                elif abs((h.tot_pop - tree_info[0]) - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:    
                    if any(h.has_center(node) for node in set(h.graph.nodes) - part_nodes):
                            cuts.append(Cut(subset=frozenset(set(h.graph.nodes) - part_nodes), assigned_teams = assign_team))
                assign_team += 1
                
        else:
            while assign_team < h.capacity_level + 1 and assign_team < h.n_teams:
                if (abs(tree_info[0] - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon) and (   
                    abs((h.tot_pop - tree_info[0]) - (h.n_teams - assign_team) * h.ideal_pop) <= h.ideal_pop * (h.n_teams - assign_team) * h.epsilon):
                    if any(h.has_center(node) for node in set(h.graph.nodes) - part_nodes) and any(h.has_center(node) for node in part_nodes):  # check if there is a center in part_nodes
                        cuts.append(Cut(subset=frozenset(part_nodes), assigned_teams = assign_team))
                assign_team += 1
            
    return cuts


def bipartition_tree(
    graph: nx.Graph,
    column_names: tuple[str],
    pop_target: Union[int, float],
    epsilon: float,
    capacity_level: int,
    n_teams: int,
    initial_solution: bool = False,
    density: float = None,
    add_root: bool = False,
    max_attempts: Optional[int] = 1000,
    warn_attempts: int = 100,
    allow_pair_reselection: bool = False, # do we need this?
) -> Set:
    """
    This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, a new tree is drawn.

    Builds up a connected subgraph with a connected complement whose population
    is ``epsilon * pop_target`` away from ``pop_target``.

    :param graph: The graph to partition.
    :type graph: nx.Graph
    :param column_names: 
    :type column_names: Tuple[str]
    :param pop_target: The target population for the returned subset of nodes.
    :type pop_target: Union[int, float]
    :param epsilon: The allowable deviation from ``pop_target`` (as a percentage of
        ``pop_target``) for the subgraph's population.
    :type epsilon: float
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, 
        If it is 1, n_teams many districts are created.
    :type capacity_level: int
    :param n_teams: Number of doctor-nurse teams for the facilities in the subgraph.
    :type n_teams: int
    :param add_root: 
    :type add_root: bool, optional

    :param max_attempts: The maximum number of attempts that should be made to bipartition.
        Defaults to 10000.
    :type max_attempts: Optional[int], optional
    :param warn_attempts: The number of attempts after which a warning is issued if a balanced
        cut cannot be found. Defaults to 1000.
    :type warn_attempts: int, optional
    :param allow_pair_reselection: Whether we would like to return an error to the calling
        function to ask it to reselect the pair of nodes to try and recombine. Defaults to False.
    :type allow_pair_reselection: bool, optional

    :returns: A subset of nodes of ``graph`` (whose induced subgraph is connected). The other
        part of the partition is the complement of this subset.
    :rtype: Set

    :raises BipartitionWarning: If a possible cut cannot be found after 1000 attempts.
    :raises RuntimeError: If a possible cut cannot be found after the maximum number of attempts
        given by ``max_attempts``.
    """
    attempts = 0 

    while max_attempts is None or attempts < max_attempts: 
        if initial_solution:
            spanning_tree = uniform_spanning_tree(graph)
        else:
            spanning_tree = random_spanning_tree(graph) 
        
        #start_time = time.perf_counter()
        h = SpanningTree(graph=spanning_tree, ideal_pop=pop_target, epsilon=epsilon, n_teams=n_teams, 
                         capacity_level=capacity_level, density=density, add_root=add_root, column_names=column_names) 
        #end_time = time.perf_counter()
        #elapsed_time = end_time - start_time
        #print(f"Elapsed time for Populated Graph: {elapsed_time:.6f} seconds") 
        # This returns a list of Cut objects with attributes edge, subset, team.
        possible_cuts = find_edge_cuts(h, add_root = add_root)
        
        if len(possible_cuts) != 0:
            print("cut edge is nonempty")
            return random.choice(possible_cuts)
        print("empty")
        attempts += 1

        if attempts == warn_attempts:
            warnings.warn(
                f"\nFailed to find a balanced cut after {warn_attempts} attempts.\n"
                "If possible, consider enabling pair reselection within your\n"
                "MarkovChain proposal method to allow the algorithm to select\n"
                "a different pair of districts for recombination.",
                BipartitionWarning,
            )

    if allow_pair_reselection:
        raise ReselectException(
            f"Failed to find a balanced cut after {max_attempts} attempts.\n"
            f"Selecting a new district pair."
        )

    raise RuntimeError(f"Could not find a possible cut after {max_attempts} attempts.")


def capacitated_recursive_tree(
    graph: nx.Graph,
    column_names: tuple[str],
    n_teams: int,
    pop_target: int,  # think about this. union of two districts may get far from average in population
    epsilon: float,
    capacity_level: int,
    initial_solution: bool=False,
    density: float = None,
    max_attempts: Optional[int] = 1000,
) -> Dict:
    """
     Recursively partitions a graph into balanced districts using bipartition_tree.

    :param graph: The graph to partition into ``len(parts)`` :math:`\varepsilon`-balanced parts.
    :type graph: nx.Graph
    :param n_teams: Total number of doctor-nurse teams for all facilities.
    :type n_teams: int
    :param pop_target: Target population for each part of the partition.
    :type pop_target: Union[float, int]
    :param column_names: 
    :type column_names: Tuple[str]
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts of the partition can be.
    :type epsilon: float
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, If it is 1, n_teams many districts are created.
    :type capacity_level: int
    :param initial_solution: States if the function is running for an initial solution. Default is `False`.
    :type initial_solution: bool, optional
    :param density: Defaluts to None.
    :type density: float, optional

    :returns: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    flips = {}  # maps nodes to their districts
    teams = {}  # maps districts to their number of teams
    remaining_nodes = set(graph.nodes())
    remaining_teams = n_teams
    debt = 0
    district = 1
    
    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    #"Change  this later"
    # Capacity level update: We multiply min_pop and max_pop by capacity level of a 
    # district to set its population target correctly. This enlarges error bounds
    # for districts with high population densities. 
        
        
    lb_pop, ub_pop = pop_target *(1- epsilon), pop_target * (1 + epsilon) 
    print(f"fixed unit interval: {[lb_pop, ub_pop]}")
    check_workload = lambda x: lb_pop <= x <= ub_pop

    while remaining_nodes: 
            
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt) 
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt) 
        pop_target = (min_pop + max_pop) / 2
        print(f"new unit interval: {[min_pop, max_pop]}")
        
        if remaining_teams <= capacity_level:  
            add_root = True
            
        try:
            cut_object = bipartition_tree(
                graph.subgraph(remaining_nodes),
                column_names=column_names,
                pop_target=pop_target, 
                capacity_level=capacity_level,
                n_teams=remaining_teams,
                epsilon=(max_pop - min_pop) / (2 * pop_target),
                initial_solution=initial_solution,
                density=density,
                add_root=add_root,
                max_attempts=max_attempts,
            )
        except Exception:
            raise
        
        nodes = cut_object.subset 
        
        if nodes is None: 
            raise BalanceError()
        
        hired_teams = cut_object.assigned_teams # number of teams hired for 'nodes'
        teams[district] = hired_teams

        part_pop = 0  
        for node in nodes:
            flips[node] = district
            part_pop += graph.nodes[node][column_names[0]]
        print(f"part_pop {part_pop}")

        if not check_workload(part_pop / hired_teams):  # part_pop / hired_teams is the workload of each D-N team
            raise PopulationBalanceError()

        debt += part_pop / hired_teams - pop_target  # unit debt
        remaining_nodes -= nodes
        remaining_teams -= hired_teams
        teams[district] = hired_teams
        print(f"created district {district}")
        district += 1


    return flips, teams




