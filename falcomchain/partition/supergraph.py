

from typing import List, Any, Tuple

from graph import Graph  
from .partition import Partition # graph import ediyoruz
from .cut_edges import cut_edges, put_edges_into_parts
import networkx as nx


# At the end, investigate frozen, cache, __slot__ strategies.


class Supergraph(Graph):
    """
    Generates a supergraph of parts in a partition to pick a random set of parts from which total 
    number of teams does not exceed the capacity level. In each state, we update merged and re-splitted 
    parts in the metagraph locally. All possible sets of possible neighboring parts are produced 
    and one of them is uniformly selected.
    """
    
    
    def __init__(self, partition: Partition):
        
        self.nodes = set(partition.parts.keys())
        self.neighborhood = {node: {} for node in self.nodes} 
        
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3)])
        G.add_edges_from([(3, 4), (4, 5)], color='red')
        adjacency_dict = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        H = nx.Graph(adjacency_dict)  # create a Graph dict mapping nodes to nbrs
        G = nx.Graph([(1, 2, {"color": "yellow"})])
        G[1]  # same as G.adj[1]
            #AtlasView({2: {'color': 'yellow'}})
        
        G.adjacency(), or G.adj.items()
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
        for (u, v, wt) in FG.edges.data('weight')
        
        G.nodes.data()
        
        
        self.node_subsets = self.parts_to_merge()
        
        #sum = sum(partition.radius[part] for part in partition.radius.keys())
        
        def __repr__(self):
            pass
        
        
        def parts_to_merge(self):
            
            return
        
        
        def node_ant(self):
            
            return
        
        
        def update_supergraph():
            return




    
    


from typing import List, Any, Tuple
from graph import Graph


class Supergraph:
    """
    A view for accessing supergraphs of :class:`Graph` objects.

    This class makes use of a supergraph cache to avoid recomputing supergraphs
    which can speed up computations when working with district assignments
    within a partition class.

    :ivar graph: The parent graph from which supergraphs are derived.
    :type graph: Graph
    :ivar parts: A list-of-lists dictionary (so a dict with key values indicated by
        the list index) mapping keys to subsets of nodes in the graph.
    :type parts: List[List[Any]]
    :ivar supergraphs_cache: Cache to store supergraph views for quick access.
    :type supergraphs_cache: Dict
    """

    __slots__ = ["graph", "parts", "supergraphs_cache"]

    def __init__(self, graph: Graph, parts: List[List[Any]]) -> None:
        """
        :param graph: The parent graph from which supergraphs are derived.
        :type graph: Graph
        :param parts: A list of lists of nodes corresponding the different
            parts of the partition of the graph.
        :type parts: List[List[Any]]

        :returns: None
        """
        self.graph = graph
        self.parts = parts
        self.supergraphs_cache = {}

    def __getitem__(self, part: int) -> Graph:
        """
        :param part: The id of the partition to return the supergraph for.
        :type part: int

        :returns: The supergraph of the parent graph corresponding to the
            partition with id `part`.
        :rtype: Graph
        """
        if part not in self.subgraphs_cache:
            self.subgraphs_cache[part] = self.graph.subgraph(self.parts[part])
        return self.subgraphs_cache[part]

    def __iter__(self) -> Graph:
        for part in self.parts:
            yield self[part]

    def items(self) -> Tuple[int, Graph]:
        for part in self.parts:
            yield part, self[part]

    def __repr__(self) -> str:
        return (
            f"<SubgraphView with {len(self.parts)}"
            f" and {len(self.subgraphs_cache)} cached graphs>"
        )

    def update_supergraph(self):
    
    
    
    
    
        return


def create_supergraph(G, partition):
    supergraph = nx.Graph()
    
    # Add supernodes
    for part_id, nodes in enumerate(partition.parts):
        supergraph.add_node(part_id, nodes=nodes)
    
    # Add superedges
    for i, part1 in enumerate(partition.parts):
        for j, part2 in enumerate(partition.parts[i+1:], start=i+1):
            if any(G.has_edge(n1, n2) for n1 in part1 for n2 in part2):
                supergraph.add_edge(i, j)
    
    return supergraph

