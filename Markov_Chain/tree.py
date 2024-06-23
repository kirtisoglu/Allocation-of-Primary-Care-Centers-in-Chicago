

import networkx as nx
from networkx import tree
from functools import partial
import random
from collections import deque, namedtuple
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



# Iki fonksiyonu disari al.
class PopulatedGraph:
    """
    A class representing a graph with population information.
    """
    
    __slots__ = (
        "graph",
        "num_centers",
        "total_pop",
        "pop_target",
        "epsilon",
        "root",
        "population",
    )

    def __init__(
        self,
        graph: nx.Graph,
        num_centers: int,
        total_pop: int,
        pop_target: float,
        epsilon: float,
    ) -> None:
        self.graph = graph
        self.num_centers = num_centers  
        self.population = {node: graph.nodes[node]['pop'] for node in graph.nodes}
        self.total_pop = total_pop
        self.pop_target = pop_target
        self.epsilon = epsilon
        self.root = self.choose_root()
    
    # This is not a good way of defining center attributes. Change graph attribute in saved files.    
    def is_center(self, node):
        return self.graph.nodes[node].get('is_initial_center', False) 
    
    def choose_root(self):
        return random.choice(list(self.graph.nodes))

    def __iter__(self):
        return iter(self.graph)

    """def contract_node(self, node, parent, subtree) -> None:

        Contracts a node into its parent, merging their populations and updating subsets and degrees.

        self.population[parent] += self.population[node]
        subtree.add(parent)
        self.degree[parent] -= 1"""

    def __repr__(self) -> str:
        graph_info = (
            f"Graph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"
        )
        return (
            f"{self.__class__.__name__}("
            f"graph={graph_info}, "
            f"num_centers={self.num_centers}, "
            f"total_population={self.total_pop}, "
            f"pop_target={self.pop_target}, "
            f"epsilon={self.epsilon})"
        )

    # Calculates deepest center, predecessors and successors in a tree.
    def tree_traversal(self):
        
        queue = deque([(self.root, 0)])  # (node, depth)
        depth_dict = {self.root: 0}
        predecessors = {self.root: None}
        successors = {node: set() for node in self.graph}
        
        deepest_center = None
        max_depth = -1

        while queue:
            current_node, current_depth = queue.popleft()

            # Check if it's a center and deeper than the current deepest
            if self.is_center(current_node) == True and current_depth > max_depth:
                deepest_center = current_node
                max_depth = current_depth

            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in depth_dict:  # Ensure each node is visited only once
                    predecessors[neighbor] = current_node
                    successors[current_node].add(neighbor)
                    depth_dict[neighbor] = current_depth + 1
                    queue.append((neighbor, current_depth + 1))
        
        if deepest_center is None:
            print("No deepest center found.")
            return None, predecessors, successors
        return deepest_center, predecessors, successors
    
    
    def subtree_nodes(self, node, successors, exclude_tree=None, exclude_pop=None):
        """Finds nodes and total population of subtree under 'node'. If this function has been called
            for one of the children of 'node', we exclude total population and subtree of that child 
            from the search, and add them manually to avoid redundant calculations. 
        
        Args:
            center (Tuple): graph node that is a center
            successors (Dict): key: node, value: set of children of node. Obtained by :func:`tree_traversal`.
            exclude_tree (Dict, optional): key:center, value: set of nodes in subtree under center. Defaults to None.
            exclude_pop (int, optional): Total population of exclude_tree. Defaults to None.

        Returns:
            Tuple(Dict, int): Subtree nodes under 'node' and total population of subtree.
        """
        if exclude_tree is None:
            exclude_tree = {}
            
        subtree_nodes = {node: {node}}
        subtree_pop = self.population[node]
        node_list = deque([])
        
        for child in successors[node]:
            if child not in exclude_tree:
                node_list.append(child)
                while node_list:
                    subnode = node_list.popleft()
                    subtree_nodes[node].add(subnode)
                    subtree_pop += self.population[subnode]  
                    children = successors[subnode]
                    for grandchild in children:
                        node_list.append(grandchild)  
            else: 
                subtree_nodes[node].update(exclude_tree[child])
                subtree_pop += exclude_pop
                      
        return subtree_nodes, subtree_pop
       

    def center_info(self, node, subtree_nodes):
        return sum(1 for subnode in subtree_nodes[node] if self.is_center(subnode))


                      

class BipartitionWarning(UserWarning):
    """
    Raised when it is proving difficult to find a balanced cut.
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



"""Functions for Initial Solution"""


def uniform_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph. Uses Wilson's algorithm.

    :param graph: Networkx Graph
    :returns (Networkx graph): A spanning tree of the graph chosen uniformly at random.
    """
    root = random.choice(list(graph.node_indices)) 
    tree_nodes = set([root])
    next_node = {root: None}

    for node in graph.node_indices: 
        u = node
        while u not in tree_nodes:  # iterates until we find a vertex u in tree_nodes
            next_node[u] = random.choice(list(graph.neighbors(u)))  # randomly select a neighbor of u
            u = next_node[u]  # replace u with its neighbor, because u is not in tree_nodes  
        # when the loop stops, next_node[u] is in tree_nodes.

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


# Should we change centers after a certain number of iterations? We only have max_attempts condition. 
# A new root chosen, and a new spanning tree is drawn. But we defined root in the class?
def find_balanced_district(h: PopulatedGraph) -> Dict:
    """
    This function takes a PopulatedGraph object and returns a district with total population of 
    at least 'h.epsilon * h.ideal_pop'. District is used in :func 'recursive_partition' to 
    construct an initial solution. 
    Returns {center: set of nodes}
    """
    deepest_center, pred, succ = h.tree_traversal()
    subtree_nodes, subtree_pop = h.subtree_nodes(deepest_center, succ)
    node = deepest_center
    parent = pred[node] 
    print(f"-----------------------------------BALANCED_DISTRICT: Started with subtree_pop {subtree_pop}-----------------------------------------------")
    while abs(subtree_pop - h.pop_target) > h.epsilon* h.pop_target:
        print(f"Entered while. subtree pop, pop difference and upper bound: {subtree_pop, abs(subtree_pop - h.pop_target), h.epsilon*h.pop_target}")
        
        if h.is_center(parent) == True: 
            print("parent is a center, returning none")
            return None  # Since parent is a center, subtree cannot be extended. subtree_pop is lower than expected. 
        
        for sibling in succ[parent]: 
            print(f"Parent is not a center. Sibling is choosen: {sibling}. Original node was {node}")
            if sibling != node:
                sibling_subtree_nodes, sibling_subtree_pop = h.subtree_nodes(sibling, succ)
                num_centers = h.center_info(sibling, sibling_subtree_nodes)
                print(f"number of centers in sibling's subtree {num_centers}. 0: continue, 1: check bounds, >1: return none")
                
                if num_centers >1:
                    return None   # we cannot search for sub subtrees. We do not want a district with two centers.
                
                if num_centers==1:
                    if abs(sibling_subtree_pop - h.pop_target) > h.epsilon* h.pop_target:
                        return None  # we have to add parent to either subtree_nodes or sibling_subtree_nodes. One of them will be out of the population range.
                    else:
                        print("Sibling satisfied the bound with one center. Returning its subtree.")  
                        return sibling_subtree_nodes, sibling_subtree_pop  # sibling_subtree_nodes defines a balanced district.
                else: 
                    continue
            else:
                continue             

                       
        # None of siblings is a center. parent is not a center either.
        # REDUNDANT CALCULATIONS: we can get the following info from siblings above. 
        subtree_nodes, subtree_pop = h.subtree_nodes(parent, succ, exclude_tree=subtree_nodes, exclude_pop=subtree_pop) # Exclusion is just for avoiding redundant calculations. 
        print(f"final part: none of siblings, neither parent, is a center. Parent's subtree pop: {subtree_pop}")
        
        if subtree_pop > (1 + h.epsilon) * h.pop_target:
            return None

        node = parent
        parent = pred[parent]
        
    return subtree_nodes, subtree_pop
 
 

def split_district(graph: nx.Graph, 
          num_centers: int, 
          total_pop: int, 
          pop_target: float, 
          epsilon: float, 
          max_attempts: int) ->Tuple[Dict, int]:


    attempts = 0
    
    while attempts < max_attempts: 

        print(f"------------------------------------- SPLIT: ATTEMPT {attempts} & SUCCESS {num_centers} -------------------------------------")
        spanning_tree = uniform_spanning_tree(graph)
        h = PopulatedGraph(spanning_tree, num_centers, total_pop, pop_target, epsilon)
        result = find_balanced_district(h)

        if result != None: 
            #print(f"splitting is done at attempt {attempts}")
            #print("target population", pop_target)
            #print(f"lower bound in splitting {h.epsilon*pop_target}")
            #print("epsilon attribute", epsilon)
            #print("realized population", result[1])
            success = 0
            print(f"SPLIT: Result is succesful. Created districts {100-num_centers}")
            return result
        
        attempts += 1
        print(f"-------------------------------------- SPLIT: FAILED. & SUCCESS {num_centers}. ---------------------------------------------")
        if attempts == max_attempts:
            warnings.warn(
                "\nFailed to find a balanced cut after 50 attempts.\n"
                "If possible, consider enabling pair reselection within your\n"
                "MarkovChain proposal method to allow the algorithm to select\n"
                "a different pair of districts to try and recombine.",
                BipartitionWarning,
            )
            return True, True

    raise RuntimeError(f"Could not find a possible cut after {max_attempts} attempts.")
        
   
                
def recursive_partition(
    graph: nx.Graph,
    num_centers: int,
    total_pop: int,
    epsilon: float,
    max_attempts: int,
) -> Dict:
    """
    Uses `find_balanced_district` to partition a tree recursively into ``len(centers)`` components of 
    population ``pop_target`` (within ``epsilon``) around centers. Used to generate initial redistricting plan.
    
    :param graph: The graph
    :param centers: List of center nodes.
    :param total_pop: total population of districts
    :param epsilon: The allowable deviation from ``pop_target`` in percentage. 
    :param max_attempts: The maximum number of attempts that should be made to bipartition.

    :returns (Dict): Partition of nodes of ``graph``, districts. key: center, value: set of nodes.
    
    :raises BipartitionWarning: If a possible cut cannot be found after 50 attempts.
    :raises RuntimeError: If a possible cut cannot be found after the maximum number of attempts
        given by ``max_attempts``.
    """
    
    districts = {}
    remaining_nodes = graph.node_indices
    pop_target = total_pop / num_centers
    num_cent = num_centers
    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    debt: Union[int, float] = 0
    
    
    lb_pop = pop_target * (1 - 2*epsilon)
    ub_pop = pop_target * (1 + 2*epsilon)
    check_pop = lambda x: lb_pop <= x <= ub_pop

    
    for i in range(num_centers - 2):
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt)
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt)
        new_pop_target = (min_pop + max_pop) / 2
        #epsilon=(max_pop - min_pop) / (2 * new_pop_target)

        try:
            subtree_nodes, subtree_pop = split_district(graph.subgraph(remaining_nodes), num_cent, 
                                                        total_pop, new_pop_target, epsilon, max_attempts)
            
        except Exception:
            raise 

        if subtree_nodes is None:
            raise BalanceError()
        
        if  subtree_nodes == True:
            return districts
            

        for center, node_set in subtree_nodes.items():  # subtree_nodes has only one key. Its value is a set of nodes.
            districts[center] = node_set
            remaining_nodes -= node_set
            num_cent -= 1
            
        print("------------------------------------------------------RECURSIVE PARTITION: Results------------------------------------------------------------------")   
        print(f"{i+1}.th district is set. Num of nodes: {len(node_set)} pop: {subtree_pop}, lb_pop:{lb_pop} ub_pop: {ub_pop}")
        print(f" epsilon: {epsilon}, min_pop: {min_pop}, max_pop:{max_pop}, new_pop_target: {new_pop_target}")
        print(f"number of remaining nodes: {len(remaining_nodes)}. Their total population: {sum(graph.nodes[node]['pop'] for node in remaining_nodes)}")  

        if not check_pop(subtree_pop):
            raise PopulationBalanceError()
            
        debt += subtree_pop - pop_target
        print(f" Check_pop is valid. Debt in {i+1}.th iteration: {debt}.")
        print("----------------------------------------------RECURSIVE PARTITION: Going to new iteration. ----------------------------------------------------")
    # After making n-2 districts, we need to make sure that the last two districts are both balanced.
    subtree_nodes, subtree_pop = split_district(graph.subgraph(remaining_nodes), num_centers, 
                                                total_pop, new_pop_target, epsilon, max_attempts)
    
    print(f"n-1.th subtree nodes {subtree_nodes} and their total pop {subtree_pop}")

    if subtree_nodes is None:
        raise BalanceError()

    for center, node_set in subtree_nodes.items():
        districts[center] = node_set
        remaining_nodes -= node_set
    if not check_pop(subtree_pop):
        raise PopulationBalanceError()
    
    debt += subtree_pop - pop_target

    # All of the remaining nodes go in the last part
    part_pop = 0
    centers = set()
    for node in remaining_nodes:
        part_pop += graph.nodes[node]['pop']
        if graph.nodes[node].get('is_initial_center', False)==True:
            centers.add(node)
    
    if len(centers) != 1:
        raise (f"Last district has {len(centers)} centers in the initial solution process.")

    for center in centers:
        districts[center] = set(remaining_nodes)

    if not check_pop(part_pop):
        raise PopulationBalanceError()

    return districts


"""Functions for Intermadiate Steps"""



# Used in bipartition of a subtree when a spanning tree cannot be found. Kruskal is the fastest algorithm for finding a spanning tree.
# if edges are not weighted, random weights are assigned. Randomness comes from that operation.
def random_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph (Networkx): The input graph to build the spanning tree from.
    :returns: The maximal spanning tree represented as a Networkx Graph."""
    
    """if weight_dict is None:
        weight_dict = dict()

    # for not having the same weights for any two edges
    for edge in graph.edges():
        weight = random.random()
        for key, value in weight_dict.items():
            if (
                graph.nodes[edge[0]][key] == graph.nodes[edge[1]][key]
                and graph.nodes[edge[0]][key] is not None
            ):
                weight += value
        graph.edges[edge]["random_weight"] = weight"""

    spanning_tree = tree.maximum_spanning_tree(graph, algorithm="kruskal")
    return spanning_tree





def balanced_resplit():
    return


def bipartition_tree(
    graph: nx.Graph,
    pop_target: Union[int, float],
    epsilon: float,
    max_attempts: int,
    choice: Callable = random.choice,
    allow_pair_reselection: bool = False,
) -> Set:
    """
    Bipartition of Tree
    This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, new roots
    are tried until node_repeats in which case a new tree is drawn.
    Builds up a connected subgraph with a connected complement whose population
    is ``epsilon * pop_target`` away from ``pop_target``.

    :param spanning_tree: The spanning tree for the algorithm to use (used when the
        algorithm chooses a new root and for testing).
    """
    
    # Try to add the region-aware in if the spanning_tree_fn accepts a weight dictionary
    if "weight_dict" in signature(spanning_tree_fn).parameters:
        spanning_tree_fn = partial(spanning_tree_fn, weight_dict=weight_dict)

    populations = {node: graph.nodes[node][pop_col] for node in graph.node_indices}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = spanning_tree_fn(graph)

    restarts = 0
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        if restarts == node_repeats:
            spanning_tree = spanning_tree_fn(graph)
            restarts = 0
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = balance_edge_fn(h, choice=choice)

        if len(possible_cuts) != 0:
            return choice(possible_cuts).subset

        restarts += 1
        attempts += 1

        # Don't forget to change the documentation if you change this number
        if attempts == 50 and not allow_pair_reselection:
            warnings.warn(
                "\nFailed to find a balanced cut after 50 attempts.\n"
                "If possible, consider enabling pair reselection within your\n"
                "MarkovChain proposal method to allow the algorithm to select\n"
                "a different pair of districts to try and recombine.",
                BipartitionWarning,
            )

    if allow_pair_reselection:
        raise ReselectException(
            f"Failed to find a balanced cut after {max_attempts} attempts.\n"
            f"Selecting a new district pair."
        )

    raise RuntimeError(f"Could not find a possible cut after {max_attempts} attempts.")


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
    Uses :func:`bipartition_tree` to partition a tree into
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
        part_pop += graph.nodes[node]["pop"]

    if not check_pop(part_pop):
        raise PopulationBalanceError()

    remaining_nodes -= nodes

    # All of the remaining nodes go in the last part
    part_pop = 0
    for node in remaining_nodes:
        flips[node] = parts[-1]
        part_pop += graph.nodes[node]["pop"]

    if not check_pop(part_pop):
        raise PopulationBalanceError()

    return flips