"""
This module provides tools and algorithms for manipulating and analyzing graphs,
particularly focused on partitioning graphs based on population data. It leverages the
NetworkX library to handle graph structures and implements various algorithms for graph
partitioning and tree traversal.

Key functionalities include:

- A spanning tree class that keeps accumulated weights for each node. Accumulation starts from leaves
  and each node keeps total weights of the subtrees beneath the node for searching a cut edge using breadth-first search.
- Random and uniform spanning tree generation for graph partitioning.
- Search functions for finding cut edges in a tree and supertree.
- Functions for finding balanced edge cuts in a populated graph

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
from collections import deque, namedtuple, Counter
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


def plot_map(chicago, geo_centers, assignment, attr):
    import folium
    import matplotlib
    import mapclassify


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
                 'capacity_level', 'epsilon', 'successors', 'tot_pop', 'supertree', 'two_sided', 'tot_candidates')

    def __init__(self,
                 graph, 
                 column_names, 
                 ideal_pop, 
                 epsilon,
                 capacity_level, 
                 n_teams: int,
                 two_sided: Optional[bool]=False, 
                 supergraph: Optional[bool]=False
                 ) -> None:            
        

        self.supertree = supergraph
        self.graph = graph
        self.pop_col, self.area_col, self.facility_col, self.density_col = column_names  # direk isimlerini gecebiliriz. Sonradan dene.
        self.ideal_pop =ideal_pop
        self.root = random.choice(list(node for node in self.graph.nodes if self.graph.degree(node) > 1))
        self.n_teams, self.capacity_level, self.epsilon = n_teams, capacity_level, epsilon
        self.two_sided = two_sided
        
        # if we do this before, can we speed up the algorithm significantly?
        self.tot_candidates = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node][self.facility_col]==True:
                self.graph.nodes[node][self.facility_col] = 1
                self.tot_candidates += 1
            else:
                self.graph.nodes[node][self.facility_col] = 0
        
        if self.supertree == True:
            accumulation_columns = {self.pop_col, self.area_col, self.team_col} 
        else:
            accumulation_columns = {self.pop_col, self.area_col, self.facility_col}

        self.successors = self.find_successors()
        accumulate_tree(self, accumulation_columns)  
        self.tot_pop = self.graph.nodes[self.root][self.pop_col]

    
    def find_successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self.graph, self.root)}

    def has_ideal_pop(self, node, assign_team):
        return abs(self.graph.nodes[node][self.pop_col] - assign_team * self.ideal_pop) <= self.ideal_pop * assign_team * self.epsilon
    
    
    def complement_has_the_ideal_pop(self, node, assign_team):
        return abs((self.tot_pop - self.graph.nodes[node][self.pop_col]) - assign_team * self.ideal_pop) <= self.ideal_pop * assign_team * self.epsilon
    
    def complement_has_ideal_pop_too(self, node, assign_team):
        return abs((self.tot_pop - self.graph.nodes[node][self.pop_col]) - (self.n_teams - assign_team) * self.ideal_pop) <= self.ideal_pop * (self.n_teams - assign_team) * self.epsilon
    
    def has_ideal_density(self, node):
        " Checks if the subtree beneath a node has an ideal density up to tolerance 'density'."
        return 

    def has_facility(self, node):
        return self.graph.nodes[node][self.facility_col] > 0
    
    def complement_has_facility(self, node):
        return self.graph.nodes[node][self.facility_col] < self.tot_candidates

    def remarkable_nodes(self):
        return {node: data for node, data in self.graph.nodes(data=True) if  data[self.pop_col] > 2*self.ideal_pop/3}
 
# Tuple that is used in the find_balanced_edge_cuts function
Cut = namedtuple("Cut", "subset assigned_teams pop")
Cut.__doc__ = "Represents a cut in a graph."
#Cut.edge.__doc__ = "The edge where the cut is made. Defaults to None."
Cut.subset.__doc__ = ("The (frozen) subset of nodes on one side of the cut. Defaults to None.")
Cut.assigned_teams.__doc__ = "The number of doctor-nurse teams for the subtree beneath the cut edge."
Cut.pop.__doc__ = "Total population of nodes in Cut.subset"


def accumulate_tree(tree: SpanningTree, accumulation_columns):
        """
        Accumulates population, area and facility attributes for the subtree under 
        each node by traversing the graph using a depth-first search.
        return: None
        """
        accumulated = set()
        stack = deque([(tree.root)])

        while stack:
            node = stack.pop()
            children = tree.successors.get(node, [])
            if all(c in accumulated for c in children): # all children are processed, accumulate attributes from children to node
                for column in accumulation_columns:
                    tree.graph.nodes[node][column] += sum(tree.graph.nodes[c][column] for c in children)
                accumulated.add(node)      
            else:
                stack.append(node)
                for c in children:
                    if c not in accumulated: 
                        stack.append(c)




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



def compute_subtree_nodes(tree, succ, root) -> Dict:
    """
    Precompute subtree nodes for all nodes.
    Returns a dict: node -> set of nodes in the subtree rooted at node.
    """
    subtree_nodes = {}

    def dfs(node):
        nodes_set = {node}
        for child in succ.get(node, []):
            nodes_set.update(dfs(child))
        subtree_nodes[node] = nodes_set
        return nodes_set

    dfs(root)
    return subtree_nodes


"""  ------------------------------ Main Functions ------------------------------  """

 
def find_edge_cuts(h: SpanningTree, density_check = None) -> List[Cut]:
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
    
    print("--------------iteration starts")
    print("remaining pop", h.tot_pop)
    for node in nodes:
        pop = nodes[node][h.pop_col]
        print(f"pop of selected node:{pop}")
        
        if nodes[node][h.facility_col]>0:  # if there is a facility in the cumulative subtree
            
            if h.two_sided:
                if node == h.root or h.complement_has_facility(node):
                    for assign_team in range(1, min(h.capacity_level + 1, h.n_teams + 1)): 
                        if h.has_ideal_pop(node, assign_team):
                            if node == h.root or h.complement_has_ideal_pop_too(node, assign_team):
                                cuts.append(Cut(subset=frozenset(_part_nodes(h.successors, node)), 
                                                assigned_teams=assign_team, pop=pop))                
            else:
                for assign_team in range(1, min(h.capacity_level + 1, h.n_teams + 1)):  
                    if h.has_ideal_pop(node, assign_team):
                        cuts.append(Cut(subset=frozenset(_part_nodes(h.successors, node)), assigned_teams=assign_team, pop=pop))
                    elif h.complement_has_the_ideal_pop(node, assign_team):  # for same node, both can be true and we add it twice!!!
                        if h.complement_has_facility(node):
                            cuts.append(Cut(subset=frozenset(set(nodes) - _part_nodes(h.successors, node)), 
                                            assigned_teams=assign_team, pop=(h.tot_pop - pop)))

    print("ideal pop", h.ideal_pop)
    print("epsilon", h.epsilon)
    print("root pop", nodes[h.root][h.pop_col])
    print("root facility", nodes[h.root][h.facility_col])
    print("--------------iteration ends")    
    return cuts


def find_superedge_cuts(h: SpanningTree, density_check = None) -> List[Cut]: # always two-sided
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

    for node in nodes:
        teams = nodes[node][h.team_col]
        if  teams <= 2*h.capacity_level:
            pop = nodes[node][h.pop_col]

            if abs(pop - teams * h.ideal_pop) <= h.ideal_pop * teams * h.epsilon:
                if node == h.root or abs((h.tot_pop - pop) - (h.n_teams - teams) * h.ideal_pop) <= h.ideal_pop * (h.n_teams - teams) * h.epsilon:
                    parts_to_merge = _part_nodes(h.successors, node)
                    cuts.append(Cut(subset=frozenset(parts_to_merge), assigned_teams=teams, pop=pop))
    
    return cuts


def bipartition_tree(
    graph: nx.Graph,
    column_names: tuple[str],
    pop_target: Union[int, float],
    epsilon: float,
    capacity_level: int,
    n_teams: int,
    two_sided: bool,
    supergraph: bool,
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
                         capacity_level=capacity_level, column_names=column_names, two_sided=two_sided, supergraph=supergraph) 
       
        if h.supertree == False:
            possible_cuts = find_edge_cuts(h)
        else:
            possible_cuts = find_superedge_cuts(h)
        
        if possible_cuts:
            return random.choice(possible_cuts)
        print("cut is empty")

    if allow_pair_reselection:
        raise ReselectException(
            f"Failed to find a balanced cut after {max_attempts} attempts.\n"
            f"Selecting a new district pair."
        )
    raise RuntimeError(f"Could not find a possible cut after {max_attempts} attempts.")




def determine_district_id(ids, max_id, assignments, district_nodes):
    """
     Assigns district id for a set of newly selected nodes in the intermidate step.
     We first consider assigning the district id that was the district id of the 
     most of the nodes in district_nodes. If this id is already choosen, we go for 

    Args:
        ids (list): a set of numbers from 1 to n, where n is big enough to assign an id to each newly created district in intial partition
        max_id (int): 
        assignments (dict): _description_
        district_nodes (set): _description_
    """

    best_id = None
    max_count = -1 # Initialize with a value lower than any possible count

    assignment_counts = Counter([assignments[node] for node in district_nodes])
    for district, count in assignment_counts.items():
        # Check if the current district id is used before
        if district in ids: # id is not used
            # Check if its count is greater than the current max_count
            if count > max_count:
                best_id = district
                max_count = count
                    
    if best_id == None: # all ids in merged_parts are used. Happens when # of merged districts < # of splitted districts
        best_id = max_id + 1 # return something that has not been used before. 
        max_id = best_id
    else:
        ids.remove(best_id)
    
    return best_id, ids, max_id



def capacitated_recursive_tree(
    graph: nx.Graph,
    column_names: tuple[str],
    n_teams: int,
    pop_target: int,  # think about this. union of two districts may get far from average in population
    epsilon: float,
    capacity_level: int,
    supergraph: Optional[bool] = False,
    density: float = None,
    assignments: Optional[dict] = None,
    merged_ids: Optional[set] = None,
    max_id: Optional[int] = 0) -> Dict:
    """
     Recursively partitions a graph into balanced districts using bipartition_tree.

    :param graph: The graph to partition into ``len(parts)`` :math:`epsilon`-balanced parts.
    :param filtered_parts:
    :param n_parts:
    :param n_teams: Total number of doctor-nurse teams for all facilities.
    :param pop_target: Target population for each part of the partition.
    :param column_names: 
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts of the partition can be.
    :param capacity_level: The maximum number of doctor-nurse teams in a facility, If it is 1, n_teams many districts are created.
    :param density: Defaluts to None.
    :param ids: set of ids whose districts are merged
    :param assignments: Old assignments for the nodes of ``graph``.
    :param max_id: maximum district id that has been used before
    :returns: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    
    flips = {}  # maps nodes to their districts
    team_flips = {}  # maps districts to their number of teams
    new_ids = set()
    remaining_ids = merged_ids
    remaining_nodes = set(graph.nodes())
    remaining_teams = n_teams
    debt = 0
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
                two_sided=two_sided,
                supergraph=supergraph,
                density=density)
        except Exception:
            raise
        
        hired_teams = cut_object.assigned_teams # number of r\
        district_nodes = cut_object.subset
        
        # check if we are still in an epsilon range of pop_target
        #if district_nodes:
        #    raise BalanceError()
        
        
        # determine district id
        if assignments == None: # initial partitioning
            district = max_id + 1
            max_id += 1
        else:     
            district, remaining_ids, max_id = determine_district_id(remaining_ids, max_id, assignments, district_nodes)
        
        # assign number of hired teams to the district        
        team_flips[district] = hired_teams
        
        # updates for the next iteration
        debt += cut_object.pop / hired_teams - pop_target  # unit debt
        remaining_teams -= hired_teams
        
        # I don't like this. Think about it. If we check epsilon range of pop_target here, we will not need this.
        if remaining_teams == 0:
            flips.update({node: district for node in remaining_nodes})
            remaining_nodes = set()
        
        else:
            remaining_nodes -= cut_object.subset
            flips.update({node: district for node in cut_object.subset})
        
        new_ids.add(district) 
        print(f"Created district {district}")

    return flips, team_flips, new_ids

