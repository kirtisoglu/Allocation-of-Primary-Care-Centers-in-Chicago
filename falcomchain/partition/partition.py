import json
import networkx
import folium
import folium.plugins

from .cut_edges import cut_edges 
from .flows import compute_edge_flows, flows_from_changes

from .assignment import get_assignment
from .subgraphs import SubgraphView
from graph import Graph, FrozenGraph
from tree import capacitated_recursive_tree
from helper import DataHandler
from typing import Any, Callable, Dict, Optional, Tuple

from .tally import Tally
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
        "updaters", 
        "parent",
        "flips",
        "flows",  
        "edge_flows", 
        "_cache",
        "travel_times",
        "teams",
        "capacity_level",
    )


    default_updaters = {"cut_edges": cut_edges}

    def __init__(
        self,
        teams = Dict,
        travel_times=None,
        capacity_level=None,
        graph=None,
        assignment=None,
        updaters=None,
        parent=None,
        flips=None,
        use_default_updaters=True,
        column_names: Tuple[str] = None,
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
            self._first_time(graph, assignment, updaters, use_default_updaters, travel_times, teams, capacity_level, column_names)
        else:
            self._from_parent(parent, flips, teams)

        self._cache = dict()
        self.subgraphs = SubgraphView(self.graph, self.parts)
        

    @classmethod
    def from_random_assignment(
        cls,
        travel_times: Dict,
        graph: Graph,
        capacity_level: int,
        epsilon: float,
        pop_target: int,
        column_names: tuple[str],
        density: Optional[float]=None,
        updaters: Optional[Dict[str, Callable]] = None,
        use_default_updaters: bool = True,
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
        total_pop = sum(graph.nodes[n][column_names[0]] for n in graph)
        n_teams = (total_pop // pop_target) # if capacity_level is 1, n_teams becomes number of districts.
        assignment, teams = capacitated_recursive_tree(
            graph=graph,
            column_names=column_names,
            n_teams=n_teams,
            pop_target=pop_target,  
            epsilon=epsilon,
            capacity_level=capacity_level,
            density=density
        )

        return cls(
            capacity_level=capacity_level,
            travel_times=travel_times,
            teams=teams,
            graph=graph,
            assignment=assignment,
            updaters=updaters,
            use_default_updaters=use_default_updaters,
            column_names=column_names
        )

    def _first_time(self, graph, assignment, updaters, use_default_updaters, travel_times, teams, capacity_level, column_names):
        if isinstance(graph, Graph):
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, networkx.Graph):
            graph = Graph.from_networkx(graph)
            self.graph = FrozenGraph(graph)
        elif isinstance(graph, FrozenGraph):
            self.graph = graph
        else:
            raise TypeError(f"Unsupported Graph object with type {type(graph)}")

        self.assignment = get_assignment(assignment, graph, column_names, travel_times)

        if updaters is None:
            updaters = dict()

        if use_default_updaters:
            self.updaters = self.default_updaters
        else:
            self.updaters = {}

        self.updaters.update(updaters)

        self.travel_times = travel_times # do we need this as an attribute?
        self.teams = teams
        self.capacity_level = capacity_level # create a facility class?
        
        
        self.parent = None
        self.flips = None
        self.flows = None
        self.edge_flows = None

      
            
    def _from_parent(self, parent: "Partition", flips: Dict, new_teams: Dict) -> None:
        self.parent = parent
        self.flips = flips

        self.graph = parent.graph
        self.updaters = parent.updaters
        self.travel_times = parent.travel_times
        self.capacity_level = parent.capacity_level

        self.flows = flows_from_changes(parent, self)  

        self.assignment = parent.assignment.copy(self.travel_times)
        self.assignment.update_flows(self.flows, self.travel_times) 
        
        self.teams = parent.teams.copy()
        for part in new_teams:
            self.teams[part] = new_teams[part]
        
        if "cut_edges" in self.updaters:
            self.edge_flows = compute_edge_flows(self)
        
        
        

    def __repr__(self):
        number_of_parts = len(self)
        s = "s" if number_of_parts > 1 else ""
        return "<{} [{} part{}]>".format(self.__class__.__name__, number_of_parts, s)

    def __len__(self):
        return len(self.parts)

    def flip(self, flips: Dict, new_teams: Dict) -> "Partition":
        """
        Returns the new partition obtained by performing the given `flips` and new_teams.
        on this partition.

        :param flips: dictionary assigning nodes of the graph to their new districts
        :param new_teams: dictionary assigning resplitted districts to their new numbers of teams
        
        :returns: the new :class:`Partition`
        """
        return self.__class__(parent=self, travel_times=self.travel_times, flips=flips, teams = new_teams)

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
        if key not in self._cache:
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
    
    


class GeographicPartition(Partition):
    """
    A :class:`Partition` with areas, perimeters, and boundary information included.
    These additional data allow you to compute compactness scores like
    `Polsby-Popper <https://en.wikipedia.org/wiki/Polsby-Popper_Test>`_.
    """

    default_updaters = {
        "perimeter": perimeter,
        "exterior_boundaries": exterior_boundaries,
        "interior_boundaries": interior_boundaries,
        "boundary_nodes": boundary_nodes,
        "cut_edges": cut_edges,
        "area": Tally("area", alias="area"),
        "cut_edges_by_part": cut_edges_by_part,
    }


