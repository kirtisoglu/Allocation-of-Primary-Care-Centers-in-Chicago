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

import random
from collections import Counter, deque, namedtuple
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import networkx as nx
from networkx.algorithms import tree

from falcomchain.helper import save_pickle
from falcomchain.tree import export_district_frame, export_tree
from falcomchain.tree.errors import (
    BalanceError,
    BipartitionWarning,
    PopulationBalanceError,
    ReselectException,
)


# Not: write a spanning tree update function and use when generating a new spanning tree in bipartition_tree
#       we are passing the same parameters everytime
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

    __slots__ = (
        "graph",
        "root",
        "ideal_pop",
        "n_teams",
        "capacity_level",
        "epsilon",
        "successors",
        "tot_pop",
        "supertree",
        "two_sided",
        "tot_candidates",
    )

    def __init__(
        self,
        graph,
        ideal_pop,
        epsilon,
        capacity_level,
        n_teams: int,
        two_sided: Optional[bool] = False,
        supergraph: Optional[bool] = False,
    ) -> None:

        self.supertree = supergraph  # remove and use supergraph
        self.graph = graph

        self.ideal_pop = ideal_pop
        self.root = random.choice(
            list(node for node in self.graph.nodes if self.graph.degree(node) > 1)
        )
        self.n_teams, self.capacity_level, self.epsilon = (
            n_teams,
            capacity_level,
            epsilon,
        )
        self.two_sided = two_sided

        if self.supertree == True:
            accumulation_columns = {"population", "area", "n_teams"}
        else:
            self.tot_candidates = sum(
                1
                for node in self.graph.nodes
                if self.graph.nodes[node]["candidate"] == 1
            )
            accumulation_columns = {"population", "area", "candidate"}

        self.successors = self.find_successors()
        accumulate_tree(self, accumulation_columns)
        self.tot_pop = self.graph.nodes[self.root]["population"]

    def find_successors(self) -> Dict:
        return {a: b for a, b in nx.bfs_successors(self.graph, self.root)}

    def has_ideal_pop(self, assign_team, pop):
        return abs(pop - assign_team * self.ideal_pop) <= self.ideal_pop * self.epsilon

    def complement_has_the_ideal_pop(self, assign_team, pop):
        return (
            abs((self.tot_pop - pop) - assign_team * self.ideal_pop)
            <= self.ideal_pop * self.epsilon
        )

    def complement_has_ideal_pop_too(self, assign_team, pop):
        return (
            abs((self.tot_pop - pop) - (self.n_teams - assign_team) * self.ideal_pop)
            <= self.ideal_pop * self.epsilon
        )

    def has_ideal_density(self, node):
        "Checks if the subtree beneath a node has an ideal density up to tolerance 'density'."
        return

    def has_facility(self, node):
        return self.graph.nodes[node]["candidate"] > 0

    def complement_has_facility(self, node):
        return (
            self.graph.nodes[node]["candidate"] < self.tot_candidates
            or node == self.root
        )

    def pop_remarkable_nodes(self):
        return {
            node
            for node, data in self.graph.nodes(data=True)
            if data["population"] > 2 * self.ideal_pop / 3
        }

    def facility_remarkable_nodes(self):
        nodes = {
            n
            for n, attr in self.graph.nodes(data=True)
            if 0
            < attr["candidate"]
            < self.tot_candidates  # note: used for two_sided. < tot_candidates guarantees a candidate in the complement
        }
        nodes.add(self.root)
        return nodes

    def team_remarkable_nodes(self):
        nodes = {
            n
            for n, attr in self.graph.nodes(data=True)
            if 0
            < attr["n_teams"]
            < self.capacity_level  # note: used for two_sided. < tot_candidates guarantees a candidate in the complement
        }
        return

    def facility_remarkable_nodes_one_sided(self):
        nodes = {
            node for node, attr in self.graph.nodes(data=True) if attr["candidate"] > 0
        }
        return nodes


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
        if all(
            c in accumulated for c in children
        ):  # all children are processed, accumulate attributes from children to node
            for column in accumulation_columns:
                tree.graph.nodes[node][column] += sum(
                    tree.graph.nodes[c][column] for c in children
                )
            accumulated.add(node)
        else:
            stack.append(node)
            for c in children:
                if c not in accumulated:
                    stack.append(c)


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
# Tuple that is used in the find_balanced_edge_cuts function
Cut = namedtuple("Cut", "node subnodes assigned_teams pop")
Cut.__doc__ = "Represents a cut in a graph."
Cut.node.__doc__ = ""
Cut.subnodes.__doc__ = (
    "The (frozen) subset of nodes on one side of the cut. Defaults to None."
)
Cut.assigned_teams.__doc__ = (
    "The number of doctor-nurse teams for the subtree beneath the cut edge."
)
Cut.pop.__doc__ = "Total population of nodes in Cut.subset"


@dataclass(frozen=True)
class Flip:
    flips: Dict[Any, Any] = field(default_factory=dict)
    team_flips: Dict[Any, Any] = field(default_factory=dict)
    new_ids: frozenset = field(default_factory=frozenset)
    merged_ids: frozenset = field(default_factory=frozenset)
    super_cut_object: Optional[Cut] = None

    def add_merged_ids(self, new: FrozenSet) -> "Flip":
        return replace(self, merged_ids=new)


def two_sided_cut(h: SpanningTree, density_check) -> List[Cut]:
    cuts = []
    nodes = h.facility_remarkable_nodes()

    for node in nodes:
        pop = h.graph.nodes[node]["population"]
        for assign_team in range(1, min(h.capacity_level + 1, h.n_teams + 1)):

            if h.has_ideal_pop(assign_team, pop):  # 3. workload
                if node == h.root or (
                    h.complement_has_ideal_pop_too(  # 4. compelement's workload too
                        assign_team, pop
                    )
                    and h.n_teams - assign_team > 0
                ):
                    cuts.append(  # 5. ACCEPTED
                        Cut(
                            node=node,
                            subnodes=frozenset(_part_nodes(h.successors, node)),
                            assigned_teams=assign_team,
                            pop=pop,
                        )
                    )
    return cuts


def one_sided_cut(h: SpanningTree, density_check):
    cuts = []
    nodes = h.graph.nodes

    for node in nodes:
        pop = h.graph.nodes[node]["population"]

        for assign_team in range(1, min(h.capacity_level + 1, h.n_teams + 1)):
            if h.has_ideal_pop(assign_team, pop) and h.has_facility(node):
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(_part_nodes(h.successors, node)),
                        assigned_teams=assign_team,
                        pop=pop,
                    )
                )
            elif h.complement_has_the_ideal_pop(
                assign_team, pop
            ) and h.complement_has_facility(node):
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(
                            set(nodes) - _part_nodes(h.successors, node)
                        ),
                        assigned_teams=assign_team,
                        pop=(h.tot_pop - pop),
                    )
                )
    return cuts


def find_edge_cuts(h: SpanningTree, density_check: Optional[float] = None) -> List[Cut]:
    """
    This function takes a SpanningTree object as input and returns a list of balanced edge cuts.
    A balanced edge cut is defined as a cut that divides the graph into two subsets, such that
    the population of each subset is close to the ideal population defined by the SpanningTree object.

    :param h: The SpanningTree object representing the graph.
    :param add_root: If set to True, an artifical node is connected to root and edge is considered as a possible cut.

    :returns: A list of balanced edge cuts.
    """
    # print("--------------iteration starts")
    # print("remaining pop", h.tot_pop)

    if h.two_sided == True:
        cuts = two_sided_cut(h, density_check)
    else:
        cuts = one_sided_cut(h, density_check)

    # print("length of cuts", len(cuts))
    # print("ideal pop", h.ideal_pop)
    # print("epsilon", h.epsilon)
    # print("root pop", h.graph.nodes[h.root][h.pop_col])
    # print("root facility", h.graph.nodes[h.root][h.facility_col])
    # print("--------------iteration ends")

    return cuts


def find_superedge_cuts(
    h: SpanningTree,
    density_check=None,
) -> List[Cut]:  # always two-sided
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
        teams = nodes[node]["n_teams"]
        pop = nodes[node]["population"]

        # print("-------------------------")
        # print("supernode", node)
        # print("number of teams in the super subtree", teams)
        # print("pop of the super subtree", pop)
        # print("total pop of the subtree", h.tot_pop)
        # print("capacity level", h.capacity_level)
        # print("ideal pop", h.ideal_pop)
        # print("epsilon", h.epsilon)

        if h.two_sided:
            if (
                teams >= 2
                and abs(pop - teams * h.ideal_pop) <= h.ideal_pop * teams * h.epsilon
            ):
                if (
                    node == h.root
                    or abs((h.tot_pop - pop) - (h.n_teams - teams) * h.ideal_pop)
                    <= h.ideal_pop * (h.n_teams - teams) * h.epsilon
                ):
                    cuts.append(
                        Cut(
                            node=node,
                            subnodes=frozenset(_part_nodes(h.successors, node)),
                            assigned_teams=teams,
                            pop=pop,
                        )
                    )
        else:  # one sided
            if (2 <= teams <= h.capacity_level) and abs(
                pop - teams * h.ideal_pop
            ) <= h.ideal_pop * teams * h.epsilon:
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(_part_nodes(h.successors, node)),
                        assigned_teams=teams,
                        pop=pop,
                    )
                )
            elif (2 <= h.n_teams - teams <= h.capacity_level) and abs(
                (h.tot_pop - pop) - teams * h.ideal_pop
            ) <= h.ideal_pop * h.epsilon:
                cuts.append(
                    Cut(
                        node=node,
                        subnodes=frozenset(
                            set(nodes) - _part_nodes(h.successors, node)
                        ),
                        assigned_teams=teams,
                        pop=(h.tot_pop - pop),
                    )
                )
    return cuts


def bipartition_tree(
    graph: nx.Graph,
    pop_target: Union[int, float],
    epsilon: float,
    capacity_level: int,
    n_teams: int,
    two_sided: bool,
    supergraph: bool,
    iteration: int = 0,
    density: Optional[float] = None,
    max_attempts=5000,
    allow_pair_reselection: bool = False,  # do we need this?
    snapshot=False,
    initial=False,
) -> Cut:
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

        h = SpanningTree(
            graph=spanning_tree,
            ideal_pop=pop_target,
            epsilon=epsilon,
            n_teams=n_teams,
            capacity_level=capacity_level,
            two_sided=two_sided,
            supergraph=supergraph,
        )

        if h.supertree == False:
            possible_cuts = find_edge_cuts(h)
        else:
            possible_cuts = find_superedge_cuts(h)

        if possible_cuts:
            if snapshot == True:
                export_tree(h, iteration, initial=initial)
            return random.choice(possible_cuts)

    if allow_pair_reselection:
        raise ReselectException(
            f"Failed to find a balanced cut after {max_attempts} attempts.\n"
            f"Selecting a new district pair."
        )

    raise RuntimeError(
        f"Could not find a possible cut after {max_attempts} attempts. Supergraph = {h.supertree}."
    )


def determine_district_id(ids, max_id, assignments, district_nodes):
    """
     Assigns district id for a set of newly selected nodes in the intermidate step.
     We first consider assigning the district id that was the district id of the
     most of the nodes in district_nodes. If this id is already choosen, and ids still
     have an id to use, we select a random id from ids. Otherwise, we use max_id.

    Args:
        ids (list): a set of numbers from 1 to n, where n is big enough to assign an id to each newly created district in intial partition
        max_id (int):
        assignments (dict): _description_
        district_nodes (set): _description_
    """
    if len(ids) > 0:
        remarkable_district_nodes = {
            node for node in district_nodes if assignments[node] in ids
        }

        if len(remarkable_district_nodes) > 0:
            assignment_counts = Counter(
                [assignments[node] for node in remarkable_district_nodes]
            )
            district = max(assignment_counts, key=assignment_counts.get)
            ids.remove(district)
        else:
            district = random.choice(list(ids))
            ids.remove(district)
    else:
        district = max_id + 1
        max_id = max_id + 1

    return district, ids, max_id


def capacitated_recursive_tree(
    graph: nx.Graph,
    n_teams: int,
    pop_target: int,  # think about this. union of two districts may get far from average in population
    epsilon: float,
    capacity_level: int,
    density=None,
    snapshot=False,
    supergraph=False,
    assignments=None,
    merged_ids=None,
    max_id=0,
    debt=None,
    iteration=0
) -> Flip:
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

    current_flips = {}  # maps nodes to their districts
    current_team_flips = {}  # maps districts to their number of teams
    current_new_ids = set()
    ids = merged_ids
    remaining_nodes = set(graph.nodes())
    remaining_teams = n_teams
    debt = 0
    hired_teams = 1

    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    # "Change  this later"
    # Capacity level update: We multiply min_pop and max_pop by capacity level of a
    # district to set its population target correctly. This enlarges error bounds
    # for districts with high population densities.

    # print(f"------ recursive function starts.")
    # print(f"num of teams {n_teams}.")
    # print(print(f"total pop {sum(graph.nodes[node]["population"] for node in remaining_nodes)}."))
    min_pop = pop_target * (1 - epsilon)
    max_pop = pop_target * (1 + epsilon)
    check_pop = lambda x: min_pop <= x <= max_pop

    # new_epsilon = epsilon
    # new_pop_target = pop_target
    while remaining_teams > 0:  # better to take len(remaining_nodes) > 0

        two_sided = remaining_teams <= capacity_level  # If True ...

        # if two_sided==False:
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt)
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt)

        new_pop_target = (min_pop + max_pop) / 2
        new_epsilon = (max_pop - min_pop) / (2 * new_pop_target)

        # else:
        #    new_pop_target=1500
        #    new_epsilon=0.1
        # print("min pop, max pop:", min_pop, max_pop)
        # print("new pop target:", new_pop_target)
        # print("new epsilon:", new_epsilon)

        try:
            cut_object = bipartition_tree(
                graph.subgraph(remaining_nodes),
                pop_target=new_pop_target,
                capacity_level=capacity_level,
                n_teams=remaining_teams,
                epsilon=new_epsilon,
                two_sided=two_sided,
                supergraph=supergraph,
                density=density,
                snapshot=snapshot,
                iteration=iteration,
            )

        except Exception:
            raise

        hired_teams = cut_object.assigned_teams  # number of r\
        # print("hired teams", hired_teams)
        # print("district pop:", cut_object.pop)
        district_nodes = cut_object.subnodes
        pop = cut_object.pop

        if not check_pop(pop / hired_teams):
            raise PopulationBalanceError()

        # determine district id
        if assignments == None:  # initial partitioning
            district_id = max_id + 1
            max_id += 1
        else:
            district_id, ids, max_id = determine_district_id(
                ids,
                max_id,
                assignments,
                district_nodes,  # we need a better function for this. (look also for remaining nodes)
            )
            assignments = {
                key: value
                for key, value in assignments.items()
                if key not in district_nodes
            }

        # assign number of hired teams to the district
        current_team_flips[district_id] = hired_teams

        # updates for the next iteration
        debt += pop - pop_target * hired_teams
        remaining_teams -= hired_teams

        remaining_nodes -= district_nodes
        current_flips.update({node: district_id for node in district_nodes})

        current_new_ids.add(district_id)

        if snapshot == True:
            export_district_frame(
                cut_object.node,
                iteration,
                district_nodes,
                hired_teams,
                cut_object.pop,
                district_id,
                debt,
                merged_ids,
                initial=(assignments == None),
                superdistrict=False
            )

        iteration += 1

    # print("------ recursive function ends sucessfully.")
    
    return Flip(
        flips=current_flips,
        team_flips=current_team_flips,
        new_ids=frozenset(current_new_ids),
    )
