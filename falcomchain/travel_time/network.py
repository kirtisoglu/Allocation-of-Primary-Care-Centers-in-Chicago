import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster




def create_multimodal_network(city, state):
    # Create a multimodal network
    G_walk = ox.graph_from_place(f"{city}, {state}", network_type="walk")
    G_transit = ox.graph_from_place(f"{city}, {state}", network_type="all", retain_all=True)
    
    # Combine walk and transit networks
    G = nx.compose(G_walk, G_transit)
    return G


def add_edge_travel_times(G, walking_speed=4.5):
    # Add walking times to edges
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            data['walk_time'] = data['length'] / (walking_speed * 1000 / 3600)  # Convert to hours
    
    # Add transit times (placeholder - replace with actual transit data)
    for u, v, data in G.edges(data=True):
        if data.get('highway') == 'bus_stop':
            data['transit_time'] = 0.25  # Placeholder: 15 minutes between stops
    
    return G

def get_amenities(G, amenity_type):
    amenities = ox.geometries_from_place("Chicago, Illinois", tags={'amenity': amenity_type})
    return amenities


def map_amenities_to_blocks(amenities, blocks):
    amenity_block_map = {}
    for idx, amenity in amenities.iterrows():
        containing_block = blocks.loc[blocks.contains(amenity.geometry)].index
        if not containing_block.empty:
            amenity_block_map[amenity['name'] if 'name' in amenity else idx] = containing_block[0]
    return amenity_block_map


def plot_route(G, start, end, route, map_style='dark'):
    # Create a map
    m = folium.Map(location=[start[0], start[1]], zoom_start=13, tiles='cartodbdark_matter' if map_style == 'dark' else 'OpenStreetMap')
    
    # Plot the route
    locations = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    folium.PolyLine(locations, weight=5, color='red').add_to(m)
    
    # Add markers for start and end
    folium.Marker(start, popup='Start', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end, popup='End', icon=folium.Icon(color='red')).add_to(m)
    
    return m


def main():
    # Create multimodal network
    G = create_multimodal_network("Chicago", "Illinois")
    
    # Add travel times
    G = add_edge_travel_times(G)
    
    # Convert to directed graph
    G = G.to_directed()
    
    # Get hospitals
    hospitals = get_amenities(G, 'hospital')
    
    # Get census blocks (placeholder - replace with actual census data)
    blocks = gpd.read_file('path_to_census_blocks.shp')
    
    # Map hospitals to census blocks
    hospital_block_map = map_amenities_to_blocks(hospitals, blocks)
    
    # Example: Find shortest path between a hospital and a block centroid
    start_node = ox.get_nearest_node(G, (41.8781, -87.6298))  # Example coordinates
    end_node = ox.get_nearest_node(G, (41.8825, -87.6234))    # Example coordinates
    route = nx.shortest_path(G, start_node, end_node, weight='walk_time')
    
    # Plot the route
    m = plot_route(G, (G.nodes[start_node]['y'], G.nodes[start_node]['x']), 
                   (G.nodes[end_node]['y'], G.nodes[end_node]['x']), route)
    m.save('chicago_route.html')

if __name__ == "__main__":
    main()



"""    
Network Accuracy: The street network data from OpenStreetMap is generally quite accurate, but may not always reflect the most up-to-date information1.

Transit Times: The current implementation uses placeholder transit times. In a real-world scenario, you would need to integrate actual transit schedules and real-time data for more accurate results1.

Walking Speeds: The code uses a constant walking speed, which doesn't account for variations due to terrain, pedestrian traffic, or individual differences1.

Multimodal Routing: While the code combines walking and transit networks, sophisticated multimodal routing algorithms would be needed for truly optimal paths1.

Census Data: The code assumes the availability of census block data. The accuracy of results will depend on the quality and currency of this data1.

Amenity Data: OpenStreetMap data for amenities like hospitals can be incomplete or outdated in some areas1.

Time Variability: Travel times can vary significantly based on time of day, day of week, and special events. This implementation doesn't account for such variations1.

Accessibility: The model doesn't consider accessibility issues, which could be crucial for certain users or locations1.

To improve precision and practicality:

Integrate real-time transit data and schedules.
Incorporate traffic data for more accurate travel time estimates.
Consider elevation data for more realistic walking times.
Include accessibility information for a more inclusive model.
Regularly update the underlying data sources.
Validate results against real-world travel times and adjust the model accordingly
"""


"""
To integrate real-time transit data from a GTFS file into the code, you can follow these steps:

    1) Use the gtfs-realtime-translators library. This library allows you to parse GTFS and GTFS-realtime data. Then, use it to load and parse your GTFS data:   


        from gtfs_realtime_translators import GTFSRealtimeTranslator
        translator = GTFSRealtimeTranslator('path/to/your/gtfs.zip')
        feed = translator.translate_from_gtfs_static()
    
    2) Create a custom weight function for edges: Instead of using a fixed transit time, create a function that calculates the travel time based on the GTFS data:
    
        def get_transit_time(u, v, data, current_time):
        stop_pair = (data['from_stop_id'], data['to_stop_id'])
        next_departure = feed.get_next_departure(stop_pair, current_time)
        if next_departure:
            return (next_departure['arrival_time'] - current_time).total_seconds() / 3600
        else:
            return float('inf')  # No service available

    3) Modify the add_edge_travel_times function: Update this function to use the GTFS data:
    
    
        def add_edge_travel_times(G, feed, current_time):
        for u, v, data in G.edges(data=True):
            if data.get('highway') == 'bus_stop':
                data['get_transit_time'] = lambda u, v, data, t: get_transit_time(u, v, data, t)
            else:
                data['walk_time'] = data['length'] / (walking_speed * 1000 / 3600)
        return G
    
    
    4) Update the routing algorithm: Modify the shortest path algorithm to use the dynamic transit times:
    
    
        def time_dependent_shortest_path(G, source, target, initial_time):
            def weight(u, v, d):
                current_time = initial_time + timedelta(hours=d.get('total_time', 0))
                if 'get_transit_time' in G[u][v]:
                    return G[u][v]['get_transit_time'](u, v, G[u][v], current_time)
                else:
                    return G[u][v].get('walk_time', 1)

            return nx.dijkstra_path(G, source, target, weight=weight)


    5) Integrate with the main code: Update your main function to use these new components:    
    
        def main():
    G = create_multimodal_network("Chicago", "Illinois")
    feed = GTFSRealtimeTranslator('path/to/chicago_gtfs.zip').translate_from_gtfs_static()
    current_time = datetime.now()
    G = add_edge_travel_times(G, feed, current_time)
    G = G.to_directed()

    # ... (rest of your code)

    start_node = ox.get_nearest_node(G, (41.8781, -87.6298))
    end_node = ox.get_nearest_node(G, (41.8825, -87.6234))
    route = time_dependent_shortest_path(G, start_node, end_node, current_time)

    # ... (plotting and output)
    
    
Challenges to consider:
Performance: Real-time calculations may be slower than pre-computed weights.
Data management: Requires regular updates of the GTFS file to maintain accuracy.
Complexity: The code becomes more complex, which may make maintenance more challenging.

"""



