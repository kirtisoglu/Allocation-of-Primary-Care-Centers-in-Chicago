import json
import networkx
import folium
import folium.plugins

from .cut_edges import cut_edges 
from .flows import compute_edge_flows, flows_from_changes, id_flows
from .tally import Tally

from .assignment import get_assignment, Assignment
from .subgraphs import SubgraphView

from falcomchain.graph import Graph, FrozenGraph
from falcomchain.tree import capacitated_recursive_tree

from typing import Any, Callable, Dict, Optional, Tuple
from collections import namedtuple

from .compactness import boundary_nodes, exterior_boundaries, interior_boundaries, perimeter
from .cut_edges import cut_edges, cut_edges_by_part



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
        "subgraphs",
        "supergraph",
        "assignment",
        "parent",
        "flips",
        "flows", 
        "id_flow", 
        "edge_flows", 
        "_cache",
        "team_flips",
        "capacity_level",
        "merged_parts",
        "updaters",
        "new_ids",
        "superflip",
        "column_names"
    )
        
    default_updaters = {"population": Tally('population', alias="population"),
                        "cut_edges": cut_edges,
                        #"perimeter": perimeter,
                        "area": Tally("area", alias="area")}
    
    

    
    def __init__(
        self,
        column_names,
        capacity_level: Optional[int] = None,
        team_flips = None,
        graph=None,
        assignment=None,
        parent=None,
        flips=None,
        updaters=None, # must contain "cut_edges" if it is not none
        use_default_updaters=True,
        merged_ids: Optional[set] = None,
        new_ids: Optional[set] = None,
        super_flip: Optional[namedtuple] = None
    ):
        """
        :param graph: Underlying graph.
        :param assignment: Dictionary assigning nodes to districts.
        :param updaters: Dictionary of functions to track data about the partition.
            The keys are stored as attributes on the partition class,
            which the functions compute.
        :param use_default_updaters: If `False`, do not include default updaters.
        """
        
        self.column_names = column_names
        self._cache = dict()

        if parent is None:
            self._first_time(graph, assignment, updaters, use_default_updaters, team_flips, capacity_level)
        else:
            self._from_parent(parent, flips, team_flips, merged_ids, new_ids, super_flip)

        self.subgraphs = SubgraphView(self.graph, self.parts)
        

    @classmethod
    def from_random_assignment(
        cls,
        graph: Graph,
        epsilon: float,
        pop_target: int,
        column_names,
        assignment_class: Assignment,
        updaters: Optional[Dict[str, Callable]] = None,
        use_default_updaters: bool = True,
        capacity_level: Optional[int] = 1,
        density: Optional[float]=None,
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
        total_pop = sum(graph.nodes[n]['population'] for n in graph)
        n_teams = (total_pop // pop_target) # if capacity_level is 1, n_teams becomes number of districts.
        
        flips, team_flips, new_ids = capacitated_recursive_tree(
            graph=graph,
            n_teams=n_teams,
            pop_target=pop_target,  
            epsilon=epsilon,
            capacity_level=capacity_level,
            column_names=column_names,
            density=density
        )

        return cls(
            capacity_level=capacity_level,
            assignment = flips,
            team_flips=team_flips,
            updaters=updaters,
            use_default_updaters=use_default_updaters,
            graph=graph,
            column_names=column_names
        )

    def _first_time(self, graph, assignment, updaters, use_default_updaters, teams, capacity_level):
        
        if isinstance(graph, Graph):
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, networkx.Graph):
            graph = Graph.from_networkx(graph)
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, FrozenGraph):
            self.graph = graph
        else:
            raise TypeError(f"Unsupported Graph object with type {type(graph)}")
        
        self.capacity_level = capacity_level
        self.assignment = get_assignment(assignment, graph, self.column_names, teams)
        
        if updaters is None:
            updaters = dict()

        if use_default_updaters:
            self.updaters = self.default_updaters
        else:
            self.updaters = {}

        self.updaters.update(updaters)
        # force to calculate updaters during initialization because we will use them in update_supergraph
        # make them attributes if we use them every time for sure
        _ = self["area"]
        _ = self["cut_edges"]
        _ = self["population"]
        self.supergraph = Graph()
        update_supergraph(self, incoming_edges=self["cut_edges"], outgoing_edges=set())
        
        self.parent = None
        self.flips = None
        self.team_flips = None
        self.flows = None
        self.id_flow = None
        self.edge_flows = None
        
        self.merged_parts = None
        self.new_ids = None
        self.superflip = None

            
    def _from_parent(self, parent: "Partition", flips: Dict, team_flips: Dict, merged_ids, new_ids, super_flip) -> None:
        self.parent = parent
        self.flips = flips
        self.team_flips = team_flips

        self.graph = parent.graph
        self.capacity_level = parent.capacity_level

        self.merged_parts = merged_ids
        self.new_ids = new_ids
        self.superflip = super_flip

        self.updaters = parent.updaters
        self.flows = flows_from_changes(parent, self) 
        self.id_flow = id_flows(self.merged_parts, self.new_ids) 
        
        self.assignment = parent.assignment.copy()
        self.assignment.update_flows(self.flows, self.id_flow, self.team_flips) 

        for part in self.id_flow["in"]:
            self.edge_flows[part] = set()
        self.edge_flows = compute_edge_flows(self)
        for part in self.id_flow["out"]:
            self.edge_flows.pop(part, None)
        
        self.supergraph = parent.supergraph.copy()
        update_supergraph(self, incoming_edges=self.edge_flows["in"], outgoing_edges=self.edge_flows["out"])
        
        

    def __repr__(self):
        number_of_parts = len(self)
        s = "s" if number_of_parts > 1 else ""
        return "<{} [{} part{}]>".format(self.__class__.__name__, number_of_parts, s)

    def __len__(self):
        return len(self.parts)

    def flip(self, flips: Dict, new_teams: Dict, merged_ids, new_ids, super_flip: namedtuple) -> "Partition":
        """
        Returns the new partition obtained by performing the given `flips` and new_teams.
        on this partition.

        :param flips: dictionary assigning nodes of the graph to their new districts
        :param new_teams: dictionary assigning resplitted districts to their new numbers of teams
        
        :returns: the new :class:`Partition`
        """
        return self.__class__(parent=self, flips=flips, team_flips = new_teams, merged_ids = merged_ids, new_ids = new_ids, super_flip = super_flip)

    def crosses_parts(self, edge: Tuple) -> bool:
        """
        :param edge: tuple of node IDs
        :type edge: Tuple

        :returns: True if the edge crosses from one part of the partition to another
        :rtype: bool
        """
        return self.assignment.mapping[edge[0]] != self.assignment.mapping[edge[1]]

    def __getitem__(self, key: str) -> Any:
        """
        Allows accessing the values of updaters computed for this
        Partition instance.

        :param key: Property to access.
        :type key: str

        :returns: The value of the updater.
        :rtype: Any
        """
        print(f"Accessing updater key: {key}")
        if key not in self._cache:
            print(f"Computing updater for: {key}")
            self._cache[key] = self.updaters[key](self)
        return self._cache[key]
    

    def __getattr__(self, key):
        return self[key]

    def keys(self):
        return self.updaters.keys()

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
    
    
     
    
def update_supergraph(partition: Partition, incoming_edges, outgoing_edges):
    
    supergraph = partition.supergraph
    
    if not partition.parent is None: # if not initial partition
        
        # remove the districts that do not exist anymore (incident edges will be removed as well)
        for district in partition.merged_parts - partition.new_ids:
            supergraph.remove_node(district)
        
        # add new districts    
        for district in partition.new_ids - partition.merged_parts:
            supergraph.add_node(superedge, 
                                population=partition["population"][district],
                                area=partition["area"][district],
                                n_teams=partition.teams[district],
                                n_candidates=len(partition.candidates[district]))
            

    # remove old cut edges whose both endpoints still exist. If initial partition, it will be empty set
    supergraph.remove_edges_from(outgoing_edges)
    
    # add new cut edges. If parent is none, all edges in cut_edges are new. 
    for edge in incoming_edges:
        superedge = tuple(sorted((partition.assignment.mapping[edge[0]], partition.assignment.mapping[edge[1]])))
        
        if superedge in supergraph.edges:
            supergraph.edges[edge]['edge_power'] += 1
        else: 
            supergraph.add_edge(superedge, edge_power=1)
            
    return supergraph
    
    
