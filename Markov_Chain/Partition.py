

"""
This module defines a `Partition` class for managing and visualizing partitions of a networkx graph.
The `Partition` class supports the creation of partitions, flipping nodes between districts, calculating
various properties of the partition, and visualizing the current state of the partition using Plotly.

Class:
    Partition: A class to represent and manipulate graph partitions.

Functions:
    __init__(self, graph: nx.Graph, districts: Dict[str, set], travel_times: Dict[Tuple[int, str], float], flips: Optional[Dict[int, str]] = None, parent: Optional["Partition"] = None):
        Initializes a Partition instance.

    get_induced_subgraph(self, district: str) -> nx.Graph:
        Returns the induced subgraph for a given district.

    get_district_population(self, district: str) -> int:
        Returns the population of a given district.

    calculate_total_populations(self, districts: Dict[str, set]) -> Dict[str, int]:
        Calculates and returns the total populations for all districts.

    calculate_total_travel_time(self, districts: Dict[str, set], travel_times: Dict[Tuple[int, str], float]) -> Dict[str, float]:
        Calculates and returns the total travel time for all districts.

    generate_assignment(self, districts: Dict[str, set]) -> Dict[int, str]:
        Generates and returns the assignment of nodes to districts.

    determine_cut_edges(self) -> list:
        Determines and returns the cut edges between districts.

    crosses_parts(self, edge: Tuple[int, int]) -> bool:
        Checks if an edge crosses different districts.

    _from_parent(self, parent: "Partition", flips: Dict[int, str]) -> None:
        Initializes a new partition from a parent partition with specified node flips.

    flip(self, flips: Dict[int, str]) -> "Partition":
        Creates a new partition by flipping nodes between districts.

    __repr__(self) -> str:
        Returns a string representation of the partition, including total travel time.

    plot(self, data: pd.DataFrame, attribute: str) -> None:
        Plots the current state of the partition using Plotly.

Note:
    - The centers of the districts are the keys of the `districts` dictionary.
    - This class is designed to handle large graphs efficiently by using the `__slots__` directive.
"""



import networkx as nx
import pandas as pd
import plotly.express as px
from shapely.geometry import Point
from typing import Any, Dict, Optional, Tuple



class Partition:
    
    __slots__ = (
        "graph",
        "districts",
        "flips",
        "cut_edges",
        "populations",
        "total_travel_times",
        "assignment",
        "parent",
    )


    def __init__(self, graph: nx.Graph, districts: Dict[str, set], travel_times: Dict[Tuple[int, str], float], flips: Optional[Dict[int, str]] = None, parent: Optional["Partition"] = None):
        self.graph = graph
        self.districts = districts
        self.flips = flips
        self.assignment = self.generate_assignment(districts)
        self.cut_edges = self.determine_cut_edges()
        self.populations = self.calculate_total_populations(districts)
        self.total_travel_times = self.calculate_total_travel_time(districts, travel_times)

        if parent is None:
            self.parent = self
        else:
            self._from_parent(parent, flips)


    def get_induced_subgraph(self, district: str) -> nx.Graph:
        if district not in self.districts:
            raise ValueError(f"District {district} is not in the partition.")
        nodes_in_district = list(self.districts[district])
        return self.graph.subgraph(nodes_in_district)


    def get_population(self, district: str) -> int:
        return sum(self.graph.nodes[node]["pop"] for node in self.districts[district])


    def calculate_total_populations(self, districts: Dict[str, set]) -> Dict[str, int]:
        return {district: self.get_population(district) for district in districts.keys()}


    def calculate_total_travel_time(self, districts: Dict[str, set], travel_times: Dict[Tuple[int, str], float]) -> Dict[str, float]:
        return {district: sum(travel_times[(node, district)] for node in nodes) for district, nodes in districts.items()}


    def generate_assignment(self, districts: Dict[str, set]) -> Dict[int, str]:
        return {node: district for district, nodes in districts.items() for node in nodes}


    def determine_cut_edges(self) -> list:
        cut_edges = []
        for u, v in self.graph.edges():
            if self.assignment[u] != self.assignment[v]:
                cut_edges.append((u, v))
        return cut_edges


    def crosses_parts(self, edge: Tuple[int, int]) -> bool:
        """
        Check if an edge crosses different districts.
        """
        return self.assignment[edge[0]] != self.assignment[edge[1]]


    def _from_parent(self, parent: "Partition", flips: Dict[int, str]) -> None:
        self.parent = parent
        self.flips = flips
        self.graph = parent.graph
        self.districts = {district: nodes.copy() for district, nodes in parent.districts.items()}

        # Apply the flips to the districts
        for node, new_district in flips.items():
            old_district = parent.assignment[node]
            self.districts[old_district].remove(node)
            self.districts[new_district].add(node)

        # Generate the new assignment based on the updated districts
        self.assignment = self.generate_assignment(self.districts)
        self.cut_edges = self.determine_cut_edges()
        self.populations = self.calculate_total_populations(self.districts)
        self.total_travel_times = self.calculate_total_travel_time(self.districts, parent.total_travel_times)


    def flip(self, flips: Dict[int, str]) -> "Partition":
        return self.__class__(self.graph, self.districts, self.total_travel_times, flips=flips, parent=self)


    def __repr__(self) -> str:
        total_travel_time = sum(self.total_travel_times.values())
        return f"<{self.__class__.__name__} [Total Travel Time: {total_travel_time}]>"


    def plot(self, data: pd.DataFrame, attribute: str) -> None:
        """
        Plot the current partition state.
        
        :param data: DataFrame with geometry and population data.
        :param attribute: Name of the attribute column to color by.
        """
        # Add assignment to the data
        data[attribute] = data.index.map(self.assignment)
        data[attribute] = pd.Categorical(data[attribute])
        data['color'] = data[attribute].codes

        fig = px.choropleth_mapbox(
            data,
            geojson=data.geometry.__geo_interface__,
            locations=data.index,
            color='color',
            mapbox_style="open-street-map",
            center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
            height=800,
            zoom=10,
            opacity=0.5,
            color_continuous_scale="Viridis",  # Use a continuous color scale
            hover_data=['pop']  # Show population data on hover
        )

        # Add cluster centers as markers
        for center in self.districts.keys():
            center_point = Point(data.loc[data[attribute] == center].geometry.centroid.x.mean(),
                                 data.loc[data[attribute] == center].geometry.centroid.y.mean())
            fig.add_scattermapbox(
                lat=[center_point.y],
                lon=[center_point.x],
                mode='markers',
                marker=dict(size=10, color='black'),  # Black markers for centers
                name=f'District={center}'
            )

        fig.show()


""" Example usage of the functions

# Initialize a Partition instance
partition = Partition(graph=graph, districts=initial_districts, travel_times=travel_times)

# Example usage of get_induced_subgraph
subgraph = partition.get_induced_subgraph('district1')
print("Induced Subgraph Nodes:", list(subgraph.nodes))

# Example usage of get_district_population
population = partition.get_district_population('district1')
print("District Population:", population)

# Example usage of calculate_total_populations
total_populations = partition.calculate_total_populations(partition.districts)
print("Total Populations:", total_populations)

# Example usage of calculate_total_travel_time
total_travel_time = partition.calculate_total_travel_time(partition.districts, travel_times)
print("Total Travel Time:", total_travel_time)

# Example usage of generate_assignment
assignment = partition.generate_assignment(partition.districts)
print("Assignment:", assignment)

# Example usage of determine_cut_edges
cut_edges = partition.determine_cut_edges()
print("Cut Edges:", cut_edges)

# Example usage of crosses_parts
edge_crosses = partition.crosses_parts((1, 2))
print("Edge Crosses Parts:", edge_crosses)

# Example usage of flip
flips = {1: 'district2', 3: 'district1'}
new_partition = partition.flip(flips)
print("New Partition:", new_partition)

# Example usage of __repr__
print("Partition Representation:", partition)

# Example usage of plot
data = pd.DataFrame({
    'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
    'pop': [100, 150, 200, 250]
}, index=[1, 2, 3, 4])

partition.plot(data, attribute='district')
"""