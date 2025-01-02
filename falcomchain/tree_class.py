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
import folium
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
import time
from helper import DataHandler

def plot_map(assignment, attr):
    import folium
    import matplotlib
    import mapclassify

    handler = DataHandler()
    chicago = handler.load_chicago()
    geo_centers = handler.load_geo_centers()  ## Define a function for that

    chicago[attr] = [assignment[node] for node in chicago.index]
    regions = chicago.dissolve(by=attr, as_index=False)

    # m = folium.Map([41.85, -87.68], zoom_start=10)
    m = regions.explore(
        column=attr,  # make choropleth based on "district" column
        tooltip=attr,  # show "district" value in tooltip (on hover)
        popup=True,  # show all values in popup (on click)
        tiles="OpenStreetMap",  # use "CartoDB positron" or "OpenStreetMap" tiles
        cmap="Set1",  # use "Set1" matplotlib colormap
        style_kwds=dict(color="black"),  # use black outline
        legend_kwds=dict(colorbar=False),
        #tooltip_kwds=dict(labels=False),  # do not show column label in the tooltip
        #smooth_factor=2,
        #fill_opacity=0.3,  #  transparency of fill colors
        #line_opacity=0.1,  # to de-emphasize border lines
        #fill_color="RdYlGn_r",  # or "YlGn"
        #nan_fill_color="white", # Also see nan_fill_opacity=0.4,
        highlight=True,
        name = "chicago"
    )

    #Adds a button to enable/disable zoom scrolling
    folium.plugins.ScrollZoomToggler().add_to(m)

    # To make the map full screen
    folium.plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)


    geo_centers.explore(
        m=m,  # pass the map object
        color="black",  # use red color on all points
        marker_kwds=dict(radius=3, fill=True),  # make marker radius 10px with fill
        name="Candidates",  # name of the layer in the map
    )
    #folium.TileLayer("CartoDB positron", show=False).add_to(m)  
    # use folium to add alternative tiles
    folium.LayerControl().add_to(m)  # use folium to add layer control

    return m, regions, chicago, geo_centers 



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
    :ivar epsilon: The tolerance for population deviation from the ideal population within each district.
    :type epsilon: float
    :ivar preccessor: The predecessor
    :type preccessor: Dict
    :ivar successor: 
    :type successor: Dict
    :ivar
    :type
    """

    __slots__ = ('graph', 'root', 'pop_col', 'area_col', 'facility_col', 'density_col', 'ideal_pop', 'n_teams', 
                 'capacity_level', 'epsilon', 'successors', 'tot_pop', 'subtree_info')

    def __init__(self, graph, column_names, ideal_pop, epsilon, n_teams, capacity_level, two_sided) -> None:

        # Do we really need to define the following lines?
        self.graph = graph
        self.pop_col, self.area_col, self.facility_col, self.density_col = column_names  # direk isimlerini gecebiliriz. Sonradan dene.
        self.ideal_pop =ideal_pop
        self.root = random.choice(list(node for node in self.graph.nodes if self.graph.degree(node) > 1))
        self.n_teams, self.capacity_level, self.epsilon = n_teams, capacity_level, epsilon
        self.successors = self.find_successors()
        
        if two_sided:
            attr = nx.get_node_attributes(self.graph, self.facility_col, default=None)
            nx.set_node_attributes(self.graph, attr, "facility_copy")
        
        self.accumulate_tree()
        self.tot_pop = self.graph.nodes[self.root][self.pop_col]

            
    
    def find_successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self.graph, self.root)}

    def accumulate_tree(self):
        """
        Accumulates population, area and facility attributes for the subtree under 
        each node by traversing the graph using a depth-first search.
        return: None
        """
        accumulated = set()
        stack = deque([(self.root)]) 

        while stack:
            node = stack.pop()
            children = self.successors.get(node, [])
            if all(c in accumulated for c in children): # all children are processed, accumulate attributes from children to node
                self.graph.nodes[node][self.facility_col] |= any(self.graph.nodes[c][self.facility_col] for c in children)
                self.graph.nodes[node][self.pop_col] += sum(self.graph.nodes[c][self.pop_col] for c in children)
                self.graph.nodes[node][self.area_col] += sum(self.graph.nodes[c][self.area_col] for c in children)
                accumulated.add(node)  
            else:
                stack.append(node)
                for c in children:
                    if c not in accumulated: 
                        stack.append(c)
        return 


    def has_ideal_pop(self):
        "Checks if subtree beneath a node has an ideal population within the spanning tree up to epsilon."
        return
 
    def has_ideal_density(self, node):
        " Checks if the subtree beneath a node has an ideal density up to tolerance 'density'."
        return 

    def has_facility(self, node):
        return self.graph.nodes[node][self.facility_col]
    
    def complement_has_facility(self, part_nodes):
        complement_nodes = set(self.graph.nodes) - part_nodes
        return any(self.graph.nodes[node]["facility_copy"] for node in complement_nodes)
    #def __repr__(self) -> str:
    #    graph_info = (
    #        f"Graph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"
    #    )
    #    return (
    #        f"{self.__class__.__name__}("
    #        f"graph={graph_info}, "
    #        f"total_population={self.tot_pop}, "
    #        f"ideal_population={self.ideal_pop}, "
    #        f"epsilon={self.epsilon})"
    #    )
 
 
# Tuple that is used in the find_balanced_edge_cuts function
Cut = namedtuple("Cut", "subset assigned_teams pop")
Cut.__doc__ = "Represents a cut in a graph."
#Cut.edge.__doc__ = "The edge where the cut is made. Defaults to None."
Cut.subset.__doc__ = ("The (frozen) subset of nodes on one side of the cut. Defaults to None.")
Cut.assigned_teams.__doc__ = "The number of doctor-nurse teams for the subtree beneath the cut edge."
Cut.pop.__doc__ = "Total population of nodes in Cut.subset"

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

    :returns: The minimum spanning tree represented as a Networkx Graph.
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

def _part_nodes(successors, start):
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
                if next_node in successors:
                    for c in successors[next_node]:
                        if c not in nodes:
                            queue.append(c)
        
        return nodes


"""  ------------------------------ Main Functions ------------------------------  """

 

def find_edge_cuts(h: SpanningTree, two_sided: Optional[bool] = False, density_check = None) -> List[Cut]:
    """
    This function takes a SpanningTree object as input and returns a list of balanced edge cuts. 
    A balanced edge cut is defined as a cut that divides the graph into two subsets, such that 
    the population of each subset is close to the ideal population defined by the SpanningTree object.

    :param h: The SpanningTree object representing the graph.
    :param add_root: If set to True, an artifical node is connected to root and edge is considered as a possible cut.

    :returns: A list of balanced edge cuts.
    """
    cuts = []
    nodes = h.graph.nodes
    facility_valid = 0
    plus_pop_valid = 0
    plus_complement_pop_valid_or_root = 0
    complement_facility_valid_or_root = 0
    
    only_pop_valid = 0
    print("--------------iteration starts")
    print("remaining pop", h.tot_pop)
    for node in nodes:
        for assign_team in range(1, min(h.capacity_level + 1, h.n_teams + 1)):
            pop = nodes[node][h.pop_col]
            if nodes[node][h.facility_col]:  # if there is a facility in the cumulative subtree
                facility_valid += 1
                part_nodes = _part_nodes(h.successors, node)
                #pop = nodes[node][h.pop_col]
                #if two_sided:
                #    print("does the complement have a facility", h.complement_has_facility(part_nodes))
                #for assign_team in range(1, min(h.capacity_level + 1, h.n_teams)):
                if two_sided:
                    if node == h.root or h.complement_has_facility(part_nodes):
                        complement_facility_valid_or_root += 1
                        if abs(pop - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:
                            plus_pop_valid += 1
                            if node == h.root or abs((h.tot_pop - pop) - (h.n_teams - assign_team) * h.ideal_pop) <= h.ideal_pop * (h.n_teams - assign_team) * h.epsilon:
                                plus_complement_pop_valid_or_root += 1
                                cuts.append(Cut(subset=frozenset(part_nodes), assigned_teams=assign_team, pop=pop))
                if not two_sided:         
                    if abs(pop - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:
                        plus_pop_valid += 1
                        cuts.append(Cut(subset=frozenset(part_nodes), assigned_teams=assign_team, pop=pop))
                    elif abs((h.tot_pop - pop) - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:  # for same node, both can be true and we add it twice!!!
                        plus_pop_valid += 1 
                        cuts.append(Cut(subset=frozenset(set(nodes) - part_nodes), assigned_teams=assign_team, pop=(h.tot_pop - pop)))
            else:
                if abs(pop - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:
                    only_pop_valid += 1
                elif abs((h.tot_pop - pop) - assign_team * h.ideal_pop) <= h.ideal_pop * assign_team * h.epsilon:
                    only_pop_valid += 1
    print("number of facility valid nodes", facility_valid)
    print("number of pop and facility valid nodes", plus_pop_valid)
    print("only pop valid", only_pop_valid)
    print("ideal pop", h.ideal_pop)
    print("epsilon", h.epsilon)
    print("root pop", nodes[h.root][h.pop_col])
    print("root facility", nodes[h.root][h.facility_col])
    print("complement facility valid or root", complement_facility_valid_or_root)
    print("plus complement pop vali or root", plus_complement_pop_valid_or_root)
    print("--------------iteration ends")    
    
    return cuts



def bipartition_tree(
    graph: nx.Graph,
    column_names: tuple[str],
    pop_target: Union[int, float],
    epsilon: float,
    capacity_level: int,
    n_teams: int,
    two_sided: bool,
    density: Optional[float] = None,
    max_attempts = 5000,
    allow_pair_reselection: bool = False # do we need this?
) -> Set:
    """
    This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, a new tree is drawn.

    :param graph: The graph to partition.
    :param column_names: 
    :param pop_target: The target population for the returned subset of nodes.
    :param epsilon: The allowable deviation from ``pop_target`` (as a percentage of
        ``pop_target``) for the subgraph's population.
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, 
        If it is 1, n_teams many districts are created.
    :param n_teams: Number of doctor-nurse teams for the facilities in the subgraph.
    :param add_root: 
    :param max_attempts: The maximum number of attempts that should be made to bipartition.
        Defaults to 10000.
    :param allow_pair_reselection: Whether we would like to return an error to the calling
        function to ask it to reselect the pair of nodes to try and recombine. Defaults to False.

    :returns: A tuple of set of nodes and number of doctor-nurse teams for the nodes.
    :raises RuntimeError: If a possible cut cannot be found after the maximum number of attempts
        given by ``max_attempts``.
    """
    for _ in range(max_attempts):
        
        spanning_tree = random_spanning_tree(graph) 

        h = SpanningTree(graph=spanning_tree, ideal_pop=pop_target, epsilon=epsilon, n_teams=n_teams, 
                         capacity_level=capacity_level, column_names=column_names, two_sided=two_sided) 
        
        possible_cuts = find_edge_cuts(h, two_sided=two_sided)
        
        if possible_cuts:
            return random.choice(possible_cuts)
        print("cut is empty")

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
    density: float = None,
) -> Dict:
    """
     Recursively partitions a graph into balanced districts using bipartition_tree.

    :param graph: The graph to partition into ``len(parts)`` :math:`\varepsilon`-balanced parts.
    :param n_teams: Total number of doctor-nurse teams for all facilities.
    :param pop_target: Target population for each part of the partition.
    :param column_names: 
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts of the partition can be.
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, If it is 1, n_teams many districts are created.
    :param initial_solution: States if the function is running for an initial solution. Default is `False`.
    :param density: Defaluts to None.

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
        

    while remaining_nodes: 
            
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt) 
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt) 
        new_pop_target = (min_pop + max_pop) / 2
        
        two_sided = remaining_teams <= 2 * capacity_level  # If True ....
        
        try:
            cut_object = bipartition_tree(
                graph.subgraph(remaining_nodes),
                column_names=column_names,
                pop_target=new_pop_target, 
                capacity_level=capacity_level,
                n_teams=remaining_teams,
                epsilon=(max_pop - min_pop) / (2 * new_pop_target),
                density=density,
                two_sided=two_sided,
            )
        except Exception:
            for node in remaining_nodes:
                flips[node] = 0
            m, regions, chicago, geo_centers = plot_map(assignment=flips, attr= "district")
            folium.display(m)
            m.save("map.html")
            import webbrowser
            webbrowser.open("map.html")
            raise
        
        if cut_object.subset is None:
            raise BalanceError()
        
        hired_teams = cut_object.assigned_teams # number of teams hired for 'nodes'
        teams[district] = hired_teams
        
    
        debt += cut_object.pop / hired_teams - pop_target  # unit debt
        remaining_teams -= hired_teams
        
        if remaining_teams == 0:
            flips.update({node: district for node in remaining_nodes})
            remaining_nodes = set()
        else:
            remaining_nodes -= cut_object.subset
            flips.update({node: district for node in cut_object.subset})
            
        print(f"Created district {district}")
        district += 1


    return flips, teams




#start_time = time.perf_counter()
        #end_time = time.perf_counter()
        #elapsed_time = end_time - start_time
        #print(f"part_nodes spends: {elapsed_time:.6f} seconds")