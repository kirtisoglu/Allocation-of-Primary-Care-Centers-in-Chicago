from collections import namedtuple
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import networkx as nx

from falcomchain.graph import FrozenGraph, Graph
from falcomchain.helper import load_pickle, save_pickle
from falcomchain.tree.tree import Flip, capacitated_recursive_tree

from .assignment import Assignment, get_assignment
from .flows import (
    compute_candidate_flows,
    compute_node_flows,
    compute_part_flows,
    neighbor_flips,
)
from .subgraphs import SubgraphView


class Partition:
    """
    Partition represents a partition of the nodes of the graph. It will perform
    the first layer of computations at each step in the Markov chain - basic
    aggregations and calculations that we want to optimize.

    :ivar graph: The underlying graph.
    :type graph: :class:`~gerrychain.Graph`
    :ivar assignment: Maps node IDs to district IDs.
    :type assignment: :class:`~gerrychain.assignment.Assignment`
    :ivar parts: Maps district IDs to the set of nodes in that district.
    :type parts: Dict
    :ivar subgraphs: Maps district IDs to the induced subgraph of that district.
    :type subgraphs: Dict
    """

    __slots__ = (
        "graph",
        "capacity_level",
        "subgraphs",
        "supergraph",
        "assignment",
        "parent",
        "superflip",
        "flip",
        "node_flows",
        "part_flows",
        "candidate_flows",
        "step",
    )


    def __init__(
        self,
        capacity_level: Optional[int] = None, 
        graph=None, 
        flip=None,
        superflip=None,
        parent=None,
        assignment=None, # ?
    ):
        """
        :param graph: Underlying graph.
        :param assignment: Dictionary assigning nodes to districts.
        :param updaters: Dictionary of functions to track data about the partition.
            The keys are stored as attributes on the partition class,
            which the functions compute.
        :param use_default_updaters: If `False`, do not include default updaters.
        """
    
        
        if parent is None:
            self._first_time(
                graph,
                assignment,  
                #updaters,
                #use_default_updaters,
                capacity_level,
                flip
            )
        else:
            self._from_parent(parent, flip, superflip)

        # define here if it is not needed to be defined before _from_parent()
        #self._cache = dict()
        self.subgraphs = SubgraphView(self.graph, self.parts)

    @classmethod
    def from_random_assignment(
        cls,
        graph: Graph,
        epsilon: float,
        pop_target: int,
        assignment_class: Assignment,
        #updaters: Optional[Dict[str, Callable]] = None,
        #use_default_updaters: bool = True,
        capacity_level=1,
        density: Optional[float] = None,
        snapshot=False,
    ) -> "Partition":
        """
        Create a Partition with a random assignment of nodes to districts.

        :param graph: The graph to create the Partition from.
        :type graph: :class:`~gerrychain.Graph`
        :param teams:The total of number of doctor-nurse teams to hire at centers
        :type teams: int
        :param capacity_level: The maximum number of doctor nurse teams at a facility
        :type capacity_level: int
        :param epsilon: The maximum relative population deviation from the ideal
        :type epsilon: float
            population. Should be in [0,1].
        :param pop_col: The column of the graph's node data that holds the population data.
        :type pop_col: str
        :param updaters: Dictionary of updaters
        :type updaters: Optional[Dict[str, Callable]], optional
        :param use_default_updaters: If `False`, do not include default updaters.
        :type use_default_updaters: bool, optional
        :param method: The function to use to partition the graph into ``n_parts``. Defaults to
            :func:`~gerrychain.tree.recursive_tree_part`.
        :type method: Callable, optional

        :returns: The partition created with a random assignment
        :rtype: Partition
        """
        total_pop = sum(graph.nodes[n]["population"] for n in graph)
        n_teams = int(total_pop // pop_target)
        # if capacity_level is 1, n_teams becomes number of districts.

        flip = capacitated_recursive_tree(
            graph=graph,
            n_teams=n_teams,
            pop_target=pop_target,
            epsilon=epsilon,
            capacity_level=capacity_level,
            density=density,
            snapshot=snapshot,
        )

        return cls(
            capacity_level=capacity_level,
            assignment=flip.flips,
            #updaters=updaters,
            #use_default_updaters=use_default_updaters,
            graph=graph,
            flip= flip
        )

    def _first_time(
        self,
        graph,
        assignment,  
        #updaters,
        #use_default_updaters,
        capacity_level,
        flip,
    ):
        if isinstance(graph, Graph):
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, networkx.Graph):
            graph = Graph.from_networkx(graph)
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, FrozenGraph):
            self.graph = graph
        else:
            raise TypeError(f"Unsupported Graph object with type {type(graph)}")


        self.step = 1
        self.parent = None
        self.capacity_level = capacity_level
        
        self.flip = flip
        self.superflip = None
        
        self.node_flows = None
        self.candidate_flows = None
        self.part_flows = {"in": set(flip.new_ids), "out": set()}
        self.assignment = get_assignment(assignment, graph, flip.team_flips)

        #if updaters is None:
        #    updaters = {}

        #if use_default_updaters:
        #    self.updaters = self.default_updaters.copy()  # copy
        #else:
        #    self.updaters = {}
        #self.updaters.update(updaters)
        
        #self.cut_edges = cut_edges(self)
        self.supergraph = supergraph(self)


    def _from_parent(
        self,
        parent: "Partition",
        flip: Flip,
        superflip: Flip,
    ) -> None:

        self.step = parent.step + 1
        self.parent = parent
        self.graph = parent.graph
        self.capacity_level = parent.capacity_level
        #self.updaters = parent.updaters.copy()
        self.flip = flip
        self.superflip = superflip
        
        #define a dataclass for these three functions
        self.part_flows = compute_part_flows(superflip.merged_ids, flip.new_ids)
        self.node_flows = compute_node_flows(parent, self)
        self.candidate_flows = compute_candidate_flows(self)
        self.assignment = parent.assignment.copy()
        self.assignment.update_flows(self.node_flows, self.part_flows, self.flip.team_flips, self.candidate_flows)
        
        #self.cut_edges = cut_edges(self) # done
        self.supergraph = supergraph(self) # done
        

    def __repr__(self):
        number_of_parts = len(self)
        s = "s" if number_of_parts > 1 else ""
        return "<{} [{} part{}]>".format(self.__class__.__name__, number_of_parts, s)

    def __len__(self):
        return len(self.parts)


    def perform_flip(self, flipp: Flip, superflipp: Flip) -> "Partition":
        """
        Returns the new partition obtained by performing the given `flips` and new_teams.
        on this partition.
        :param flip: 
        :param superflip: 
        :returns: the new :class:`Partition`
        """
        return self.__class__(
            parent=self,
            flip = flipp,
            superflip=superflipp,
        )


    def crosses_parts(self, edge: Tuple) -> bool:
        """
        :param edge: tuple of node IDs
        :type edge: Tuple

        :returns: True if the edge crosses from one part of the partition to another
        :rtype: bool
        """
        return self.assignment.mapping[edge[0]] != self.assignment.mapping[edge[1]]

    def part_pop(self, part):
        return sum(self.graph.nodes[node]["population"] for node in self.parts[part])
    
    
    def part_area(self, part):
        return sum(self.graph.nodes[node]["area"] for node in self.parts[part])


    @property
    def parts(self):
        return self.assignment.parts

    @property
    def teams(self):
        return self.assignment.teams

    @property
    def candidates(self):
        return self.assignment.candidates

    @property
    def centers(self):
        return self.assignment.centers

    @property
    def radius(self):
        return self.assignment.radius


    def plot(self, geometries=None, **kwargs):
        """
        Plot the partition, using the provided geometries.

        :param geometries: A :class:`geopandas.GeoDataFrame` or :class:`geopandas.GeoSeries`
            holding the geometries to use for plotting. Its :class:`~pandas.Index` should match
            the node labels of the partition's underlying :class:`~gerrychain.Graph`.
        :type geometries: geopandas.GeoDataFrame or geopandas.GeoSeries
        :param `**kwargs`: Additional arguments to pass to :meth:`geopandas.GeoDataFrame.plot`
            to adjust the plot.

        :returns: The matplotlib axes object. Which plots the Partition.
        :rtype: matplotlib.axes.Axes
        """
        import geopandas

        if geometries is None:
            geometries = self.graph.geometry

        if set(geometries.index) != set(self.graph.nodes):
            raise TypeError(
                "The provided geometries do not match the nodes of the graph."
            )
        assignment_series = self.assignment.to_series()
        if isinstance(geometries, geopandas.GeoDataFrame):
            geometries = geometries.geometry
        df = geopandas.GeoDataFrame(
            {"assignment": assignment_series}, geometry=geometries
        )
        return df.plot(column="assignment", **kwargs)


    def save(self):
        flips = self.assignment.mapping
        teams = self.teams
        metadata = {"capacity_level": self.capacity_level}
        data = {"flips": flips, "team_flips": teams, "metadata": metadata}
        path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/initial.pkl"
        
        #with open(path, "w") as f:
        #    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        save_pickle(data, path)
        
    
    @classmethod
    def load_partition(cls, assignment_class: Assignment):
        
        graph_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/graphhh.pkl"
        partition_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/initial.pkl"
        column_names = ['population', 'area', 'candidate', 'density']
        
        my_graph = load_pickle(graph_path)
        partition = load_pickle(partition_path)
        
        my_graph.nodes[5158]["population"] = 100
        my_graph.nodes[17159]["population"] = 100

        return cls(capacity_level=partition["metadata"]["capacity_level"],
                    assignment=partition["flips"],
                    flip=Flip(flips=partition["flips"], team_flips=partition["team_flips"], new_ids=set(partition["team_flips"].keys())),
                    graph=my_graph,)
    
    
    
    def plot_supergraph(self, gdf):
        for part in self.parts:
            district_nodes = list(self.parts[part])
            district_gdf = gdf.loc[district_nodes]
        return
    


class SupergraphError(Exception):
    """Raised when supergraph constructed wrong."""

    
def supergraph(partition:Partition):
    # Later, you can define this over superflips. 
    
    new_ones = set(partition.flip.new_ids.copy())
    
    if new_ones != set(partition.flip.flips.values()):
        raise SupergraphError(f"new ids do not match with flip values.\n"
                              f"new ids: {new_ids}\n"
                              f"flips values: {set(partition.flip.flips.values())}")
        
    
    # if flips are correct, then new ones are correct.
    
    # starts here
    if partition.parent==None:  # initial partition
        graph = nx.Graph()
        merged = set()
        
    else:
        merged =set(partition.superflip.merged_ids.copy())
        graph = partition.parent.supergraph.copy()
        graph.remove_nodes_from(list(merged))  # Edges has gone too.

    leaving = merged - new_ones
    if leaving != partition.part_flows["out"]:
        raise SupergraphError(f"leaving parts {leaving} is not same as part out flow {part_flows["out"]}")
    
    for node in leaving:
        if node in graph.nodes:
            raise SupergraphError(f"leaving node {node} is still here.")
            
            
    for new in new_ones:
        if new not in partition.parts.keys():
            raise SupergraphError(f"new_ones has an id that is not in partition.parts {new}")
        

    for part in leaving:
        if part in partition.parts:
            raise SupergraphError(f"part {part} is not in the partition parts")
        
        
    # ---- add nodes
    nodes = [(node, {"population":partition.part_pop(node),
                     "area": partition.part_area(node), 
                     "n_teams":partition.flip.team_flips[node],
                     "n_candidates":len(partition.candidates[node])
                    }
              ) for node in new_ones]
    
    try:
        graph.add_nodes_from(nodes)
        
    except Exception:
        raise SupergraphError("couldn't add nodes to supergraph")


    # add edges
    add = {(node, neighbor) for node in partition.flip.flips 
            for neighbor in partition.graph.neighbors(node)
            #if partition.flip.flips[neighbor] not in leaving
            }

    for edge in add:
        u,v = edge
        uu, vv = partition.assignment.mapping[u], partition.assignment.mapping[v]
        
        if uu not in graph.nodes or vv not in graph.nodes:
            raise SupergraphError(f"one of endpoints {u,v} not in supergraph.\n"
                                  f"endpoints are {uu, vv}\n"
                                  f"leaving is {leaving}")

    try:
        add_edges = {
            (node, neighbor)
            for (node, neighbor) in add
            if partition.crosses_parts((node, neighbor))}
    
    except Exception:
        raise print(f"add edges {add_edges}")
    
    for edge in add_edges:
        uu = partition.assignment.mapping[edge[0]]
        vv = partition.assignment.mapping[edge[1]]
        graph.add_edge(uu,vv)

    return graph


