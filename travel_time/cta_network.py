"""
This module provides a set of functions to help determine and
manipulate ... within a particular
graph. The functions in this module are used internally to ensure
that the geometry data that we are working with is sufficiently
well-defined to be used for analysis.
"""


import pandas as pd
import networkx as nx



def merge(stop_times, trips):
    
    
    # Merge 'stop_times' with 'trips' to include 'route_id' and 'direction_id'
    stop_times = stop_times.merge(trips[['trip_id', 'route_id', 'direction_id']], on='trip_id')
    
    
    # Convert 'arrival_time' to a suitable forma t for time difference calculation, if necessary
    stop_times['arrival_time'] = pd.to_timedelta(stop_times['arrival_time'].astype(str))
    
    
    # Sort and calculate differences
    stop_times.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)
    stop_times['travel_time'] = stop_times.groupby('trip_id')['arrival_time'].diff().fillna(pd.Timedelta(seconds=0))
    
    
    # Pseudocode: Iterate over 'stop_times' to find consecutive stops, map them to graph edges, and assign travel times
    for index, row in stop_times.iterrows():
        start_stop_id = row['stop_id']
        end_stop_id = stop_times.loc[index + 1, 'stop_id'] if index + 1 < len(stop_times) else None
        travel_time = row['travel_time']  
        # Map 'start_stop_id' and 'end_stop_id' to census blocks/nodes in your graph
        start_node = stop_to_census_block_mapping[start_stop_id]  # This mapping needs to be defined based on your data
        end_node = stop_to_census_block_mapping[end_stop_id] if end_stop_id else None   
        if start_node and end_node:
            # Find the edge in your graph that corresponds to this trip and assign the travel time
            # This assumes you have a multi-edge directed graph 'G_multi'
            G_multi[start_node][end_node][0]['travel_time'] = travel_time  # Adjust edge key [0] as necessary
    
    
    return



def to_geodataframe(stops):

    # Convert stops to a GeoDataFrame
    stops_gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs=census_blocks.crs)
    # Perform a spatial join: assign each stop to the census block it falls into
    joined = gpd.sjoin(stops_gdf, census_blocks, how="inner", op='within')
    # Create a mapping from stop_id to census block identifier (assuming 'block_id' as the identifier)
    stop_to_census_block_mapping = pd.Series(joined['block_id'].values, index=joined['stop_id']).to_dict()

    return




def get_digraph(graph):
    
    """
    Converts an undirected graph to a directed graph by duplicating each edge.
    All original node and edge attributes preserved.
    
    Parameters:
    - graph: An undirected NetworkX graph (nx.Graph instance).
    
    Returns:
    - A directed NetworkX graph (nx.DiGraph instance) where each undirected edge
      is represented as two directed edges.
    """
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        digraph.add_edge(u, v, **data)
        digraph.add_edge(v, u, **data)  # Add the reverse edge

    return digraph

def add_stops(graph, stops):
    "Takes a networkx dual graph and Google GTFS stop.txt file and adds stops to the graph as nodes."
    
    
    return



# Creates R-tree spatial index query using Sort-Tile-Recursive (STR) algorithm.
# "STR: A Simple and Efficient Algorithm for R-Tree Packing", Scott Leutenegger et. al., February 1997
def str_tree(geometries): 
    """
    Add ids to geometries and create a STR tree for spatial indexing.
    Use this for all spatial operations!

    :param geometries: A Shapely geometry object to construct the tree from.
    :type geometries: shapely.geometry.BaseGeometry

    :returns: A Sort-Tile-Recursive tree for spatial indexing.
    :rtype: shapely.strtree.STRtree
    """
    from shapely.strtree import STRtree

    try:
        tree = STRtree(geometries)
    except AttributeError:
        tree = STRtree(geometries)
    return tree


# Returns a generator of tuples (id, (ids of neighbors))
def neighboring_geometries(geometries, tree=None): 
    """
    :param geometries: A Shapeley geometry object to construct the tree from
    :type geometries: shapely.geometry.BaseGeometry
    :param tree: A Sort-Tile-Recursive tree for spatial indexing. Default is None.
    :type tree: shapely.strtree.STRtree, optional

    :returns: A generator yielding tuples of the form (id, (ids of neighbors))
    :rtype: Generator
    """
    if tree is None:
        tree = str_tree(geometries)
 
    for geometry_id, geometry in geometries.items():
        possible = tree.query(geometry)
        actual = tuple(
            geometries.index[p]
            for p in possible
            if (not geometries.iloc[p].is_empty) and geometries.index[p] != geometry_id
        )
        yield (geometry_id, actual)
        
        
# Returns a generator of tuples (id, {neighbor_id: intersection})
# Intersection may be empty.
def intersections_with_neighbors(geometries):
    """
    :param geometries: A Shapeley geometry object.
    :type geometries: shapely.geometry.BaseGeometry
    
    :returns: A generator yielding tuples of the form (id, {neighbor_id: intersection})
    :rtype: Generator
    """
    for i, neighbors in neighboring_geometries(geometries):
        intersections = {
            j: geometries[i].intersection(geometries[j]) for j in neighbors
        }
        yield (i, intersections)