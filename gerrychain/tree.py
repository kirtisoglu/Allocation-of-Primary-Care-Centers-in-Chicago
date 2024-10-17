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



class PopulatedGraph:
    """
    A class representing a graph with population information.

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
        populations: Dict,
        ideal_pop: Union[float, int],
        epsilon: float,
        n_teams: int,
        hierarchy: int,
        add_root: bool,
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
        self.subsets = {node: {node} for node in graph.nodes} # ????????
        self.population = populations.copy()
        self.tot_pop = sum(self.population.values())
        self.ideal_pop = ideal_pop
        self.n_teams = n_teams
        self.hierarchy = hierarchy
        self.epsilon = epsilon
        self.root = random.choice([x for x in self.graph.nodes if self.graph.degree(x) > 1])
        self.predecessors = self.predecessors()
        self.successors = self.successors()
        self.subtree_pops = self.calc_pops()
        
        if add_root:  # For considering the case that all nodes might be in the same district
            self.subtree_pops[self.root] = self.tot_pop  
            artifical_node = -1
            self.graph.add_node(artifical_node)
            self.graph.add_edge(self.root, artifical_node)
            self.predecessors[self.root] = artifical_node  

    def __iter__(self):
        return iter(self.graph)

    def predecessors(self):
        return {a: b for a, b in nx.bfs_predecessors(self, self.root)}
        
    def successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self, self.root)}

    def calc_pops(self):
        """
        Calculates the population of each subtree in the graph
        by traversing the graph using a depth-first search."""
        subtree_pops = {}
        stack = deque(n for n in self.successors[self.root])
        while stack:
            next_node = stack.pop() 
            if next_node not in subtree_pops:
                if next_node in self.successors:
                    children = self.successors[next_node]
                    if all(c in subtree_pops for c in children):
                        subtree_pops[next_node] = sum(subtree_pops[c] for c in children) + self.population[next_node]
                        
                    else:
                        stack.append(next_node)
                        for c in children:
                            if c not in subtree_pops:
                                stack.append(c)
                else:
                    subtree_pops[next_node] = self.population[next_node]

        return subtree_pops 

    def has_center(self, node):
        return self.graph.nodes[node].get('real_phc', False) 
    
    def has_ideal_population(self, node, one_sided_cut: bool = False) -> bool:
        """
        Checks if a node has an ideal population within the graph up to epsilon.

        :param node: The node to check.
        :type node: Any
        :param one_sided_cut: Whether or not we are cutting off a single district. When
            set to False, we check if the node we are cutting and the remaining graph
            are both within epsilon of the ideal population. When set to True, we only
            check if the node we are cutting is within epsilon of the ideal population.
            Defaults to False.
        :type one_sided_cut: bool, optional

        :returns: True if the node has an ideal population within the graph up to epsilon.
        :rtype: bool
        """
        if one_sided_cut:
            return (
                abs(self.population[node] - self.ideal_pop)
                < self.epsilon * self.ideal_pop
            )

        return (
            abs(self.population[node] - self.ideal_pop) <= self.epsilon * self.ideal_pop
            and abs((self.tot_pop - self.population[node]) - self.ideal_pop)
            <= self.epsilon * self.ideal_pop
        )

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
Cut.edge.__doc__ = "The edge where the cut is made. Defaults to None."
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




def random_spanning_tree(
    graph: nx.Graph, region_surcharge: Optional[Dict] = None
) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph: The input graph to build the spanning tree from. Should be a Networkx Graph.
    :type graph: nx.Graph
    :param region_surcharge: Dictionary of surcharges to add to the random
        weights used in region-aware variants.
    :type region_surcharge: Optional[Dict], optional

    :returns: The maximal spanning tree represented as a Networkx Graph.
    :rtype: nx.Graph
    """
    if region_surcharge is None:
        region_surcharge = dict()

    for edge in graph.edges():
        weight = random.random()
        for key, value in region_surcharge.items():
            # We surcharge edges that cross regions and those that are not in any region
            if (
                graph.nodes[edge[0]][key] != graph.nodes[edge[1]][key]
                or graph.nodes[edge[0]][key] is None
                or graph.nodes[edge[1]][key] is None
            ):
                weight += value

        graph.edges[edge]["random_weight"] = weight

    spanning_tree = tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
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




"""  ------------------------------ Initial Solution Functions ------------------------------  """

 

def find_balanced_edge_cuts_memoization(h: PopulatedGraph, add_root: bool,
) -> List[Cut]:
    """
    This function takes a PopulatedGraph object as input and returns a list of balanced edge cuts. 
    A balanced edge cut is defined as a cut that divides the graph into two subsets, such that 
    the population of each subset is close to the ideal population defined by the PopulatedGraph object.

    :param h: The PopulatedGraph object representing the graph.
    :type h: PopulatedGraph
    :param add_root: If set to True, an artifical node is connected to root and edge is considered as a possible cut.
    :type add_root: bool, optional

    :returns: A list of balanced edge cuts.
    :rtype: List[Cut]
    """

    cuts = []

    for node, tree_pop in h.subtree_pops.items():
        part_nodes = _part_nodes(node, h.successors)
        assign_team = 1
            
        if add_root==False:  
            while assign_team < h.hierarchy + 1 and h.n_teams:
                if abs(tree_pop - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:    
                    if any(h.has_center(node) for node in part_nodes):   # h can keep all centers. we can check that if intersection is not empty
                        e = (node, h.preccessor[node])
                        cuts.append(Cut(edge=e, subset=frozenset(part_nodes), assigned_teams = assign_team))     
                elif abs((h.tot_pop - tree_pop) - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:    
                    if any(h.has_center(node) for node in set(h.graph.nodes) - part_nodes):
                            e = (node, h.predecessors[node])
                            cuts.append(Cut(edge=e, subset=frozenset(set(h.graph.nodes) - part_nodes), assigned_teams = assign_team))
                assign_team += 1
            return cuts

        while assign_team < h.hierarchy + 1 and assign_team < h.n_teams:
            if (abs(tree_pop - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon) and (   
                abs((h.tot_pop - tree_pop) - (h.n_teams - assign_team) * h.ideal_pop) <= h.ideal_pop * (h.n_teams - assign_team) * h.epsilon):
                if any(h.has_center(node) for node in set(h.graph.nodes) - part_nodes) and any(h.has_center(node) for node in part_nodes):  # check if there is a center in part_nodes
                    e = (node, h.predecessors[node])
                    cuts.append(Cut(edge=e, subset=frozenset(part_nodes), assigned_teams = assign_team))
            assign_team += 1
            
    return cuts


def bipartition_tree(
    graph: nx.Graph,
    pop_col: str,
    pop_target: Union[int, float],
    epsilon: float,
    hierarchy: int,
    n_teams: int,
    node_repeats: int = 1,
    spanning_tree: Optional[nx.Graph] = None,
    spanning_tree_fn: Callable = random_spanning_tree,
    balance_edge_fn: Callable = find_balanced_edge_cuts_memoization,
    add_root: bool = False,
    max_attempts: Optional[int] = 100000,
    warn_attempts: int = 1000,
    allow_pair_reselection: bool = False
) -> Set:
    """
    This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, new roots
    are tried until node_repeats in which case a new tree is drawn.

    Builds up a connected subgraph with a connected complement whose population
    is ``epsilon * pop_target`` away from ``pop_target``.

    :param graph: The graph to partition.
    :type graph: nx.Graph
    :param pop_col: The node attribute holding the population of each node.
    :type pop_col: str
    :param pop_target: The target population for the returned subset of nodes.
    :type pop_target: Union[int, float]
    :param epsilon: The allowable deviation from ``pop_target`` (as a percentage of
        ``pop_target``) for the subgraph's population.
    :type epsilon: float
    :param hierarchy: The maximum number of doctor-nurse teams in a facility, 
        If it is 1, n_teams many districts are created.
    :type hierarchy: int
    :param n_teams: Number of doctor-nurse teams for the facilities in the subgraph.
    :type n_teams: int
    :param node_repeats: A parameter for the algorithm: how many different choices
        of root to use before drawing a new spanning tree. Defaults to 1.
    :type node_repeats: int, optional
    :param spanning_tree: The spanning tree for the algorithm to use (used when the
        algorithm chooses a new root and for testing).
    :type spanning_tree: Optional[nx.Graph], optional
    :param spanning_tree_fn: The random spanning tree algorithm to use if a spanning
        tree is not provided. Defaults to :func:`random_spanning_tree`.
    :type spanning_tree_fn: Callable, optional
    :param balance_edge_fn: The function to find balanced edge cuts. Defaults to
        :func:`find_balanced_edge_cuts_memoization`.
    :type balance_edge_fn: Callable, optional
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

    populations = {node: graph.nodes[node][pop_col] for node in graph.node_indices}
    restarts = attempts = 0
    
    if spanning_tree is None:
        spanning_tree = spanning_tree_fn(graph)  # does this make sense for the initial solution?

    while max_attempts is None or attempts < max_attempts:
        if restarts == node_repeats:
            spanning_tree = spanning_tree_fn(graph) 
            restarts = 0
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon, n_teams, hierarchy) 

        # This returns a list of Cut objects with attributes edge, subset, team.
        possible_cuts = balance_edge_fn(h, add_root = add_root)



        if len(possible_cuts) != 0:
            return random.choice(possible_cuts)

        restarts += 1
        attempts += 1

        # Don't forget to change the documentation if you change this number
        if attempts == warn_attempts and not allow_pair_reselection:
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

 def recursive_tree_part(
    graph: nx.Graph,
    pop_col=pop_col,
    pop_target,
    n_teams: int,
    epsilon,
    hierarchy: int,
    node_repeats: int = 1,
    method: Callable = partial(bipartition_tree, max_attempts=10000),
) -> Dict:
    """
     Recursively partitions a graph into balanced districts using bipartition_tree.

    :param graph: The graph to partition into ``len(parts)`` :math:`\varepsilon`-balanced parts.
    :type graph: nx.Graph
    :param n_teams: Total number of doctor-nurse teams for all facilities.
    :type n_teams: int
    :param pop_target: Target population for each part of the partition.
    :type pop_target: Union[float, int]
    :param pop_col: Node attribute key holding population data.
    :type pop_col: str
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts of the partition can be.
    :type epsilon: float
    :param hierarchy: The maximum number of doctor-nurse teams in a facility, If it is 1, n_teams many districts are created.
    :type hierarchy: int
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use. Defaluts to 1.
    :type node_repeats: int, optional
    :param balance_final_districts:
    :type balance_final_districts: bool, optional
    :param method: The partition method to use. Defaults to
        `partial(bipartition_tree, max_attempts=10000)`.
    :type method: Callable, optional

    :returns: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    flips = {}  # maps nodes to their districts
    remaining_teams = n_teams  
    remaining_nodes = set(graph.nodes())
    debt = 0
    district = 1
    average_pop = pop_target
    
    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    #"Change  this later"
    # Hierarchy Update: We multiply min_pop and max_pop by hierarchy_level of a 
    # district to set its population target correctly. This enlarges error bounds
    # for districts with high population densities. 
    

    lb_pop, ub_pop = pop_target *(1- epsilon), pop_target * (1 + epsilon) 
    check_workload = lambda x: lb_pop <= x <= ub_pop

    while remaining_teams > hierarchy: # to make sure that last district is balanced as well
            
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt) 
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt) 
        pop_target = (min_pop + max_pop) / 2

        try:
            cut_object = method(
                graph.subgraph(remaining_nodes),
                pop_col=pop_col,
                pop_target=pop_target, 
                hierarchy=hierarchy,
                n_teams=remaining_teams,
                epsilon=(max_pop - min_pop) / (2 * pop_target),
                node_repeats=node_repeats,
                add_root=False,
                one_sided_cut=True,  
            )
        except Exception:
            raise
        
        nodes = cut_object.subset 
        
        if nodes is None:
            raise BalanceError()
        
        hired_teams = cut_object.assigned_teams # number teams hired for 'nodes'

        part_pop = 0
        for node in nodes:
            flips[node] = district
            part_pop += graph.nodes[node][pop_col]

        if not check_workload(part_pop / hired_teams):  # part_pop / hired_teams is the workload of each D-N team
            raise PopulationBalanceError()

        debt += part_pop / hired_teams - pop_target  # unit debt
        remaining_nodes -= nodes
        remaining_teams -= hired_teams
        district += 1

    # remaining_teams <= hierarchy. We ...
    while remaining_teams > 0:
        
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt) 
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt) 
        pop_target = (min_pop + max_pop) / 2

        try:
            cut_object = method(
                graph.subgraph(remaining_nodes),
                pop_col=pop_col,
                pop_target=pop_target, 
                hierarchy=hierarchy,
                n_teams=remaining_teams,
                epsilon=(max_pop - min_pop) / (2 * pop_target),
                node_repeats=node_repeats,
                add_root=True,   # this is the key point of that loop
                one_sided_cut=False,
            )
        except Exception:
            raise
        

        edge = cut_object.edge
        nodes = cut_object.subset 
        
        if nodes is None:
            raise BalanceError()
            
        
        if edge[0] or edge[1] == -1:
            hired_teams = remaining_teams
        else:
            hired_teams = cut_object.assigned_teams
        
        
        part_pop = 0
        for node in nodes:
            flips[node] = district
            part_pop += graph.nodes[node][pop_col]

        if not check_workload(part_pop / hired_teams):   
            raise PopulationBalanceError()

        remaining_teams -= hired_teams  # if root is choosen, remaining_teams is zero, loop stops.
        remaining_nodes -= nodes
        debt += part_pop / hired_teams - pop_target
    
        district += 1


    return flips


def epsilon_tree_bipartition(
    graph: nx.Graph,
    parts: Sequence,
    pop_target: Union[float, int],
    pop_col: str,
    epsilon: float,
    node_repeats: int = 1,
    method: Callable = partial(bipartition_tree, max_attempts=10000),
) -> Dict:
    """
    Uses :func:`~gerrychain.tree.bipartition_tree` to partition a tree into
    two parts of population ``pop_target`` (within ``epsilon``).

    :param graph: The graph to partition into two :math:`\varepsilon`-balanced parts.
    :type graph: nx.Graph
    :param parts: Iterable of part (district) labels (like ``[0,1,2]`` or ``range(4)``).
    :type parts: Sequence
    :param pop_target: Target population for each part of the partition.
    :type pop_target: Union[float, int]
    :param pop_col: Node attribute key holding population data.
    :type pop_col: str
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts
        of the partition can be.
    :type epsilon: float
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use.
        Defaults to 1.
    :type node_repeats: int, optional
    :param method: The partition method to use. Defaults to
        `partial(bipartition_tree, max_attempts=10000)`.
    :type method: Callable, optional

    :returns: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    if len(parts) != 2:
        raise ValueError(
            "This function only supports bipartitioning. Please ensure that there"
            + " are exactly 2 parts in the parts list."
        )

    flips = {}
    remaining_nodes = graph.node_indices

    lb_pop = pop_target * (1 - epsilon)
    ub_pop = pop_target * (1 + epsilon)
    check_pop = lambda x: lb_pop <= x <= ub_pop

    nodes = method(
        graph.subgraph(remaining_nodes),
        pop_col=pop_col,
        pop_target=pop_target,
        epsilon=epsilon,
        node_repeats=node_repeats,
        one_sided_cut=False,
    )

    if nodes is None:
        raise BalanceError()

    part_pop = 0
    for node in nodes:
        flips[node] = parts[-2]
        part_pop += graph.nodes[node][pop_col]

    if not check_pop(part_pop):
        raise PopulationBalanceError()

    remaining_nodes -= nodes

    # All of the remaining nodes go in the last part
    part_pop = 0
    for node in remaining_nodes:
        flips[node] = parts[-1]
        part_pop += graph.nodes[node][pop_col]

    if not check_pop(part_pop):
        raise PopulationBalanceError()

    return flips


