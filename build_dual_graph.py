#!/usr/bin/env python3
"""
Filter GeoJSON to keep only assigned districts, build dual graph, and visualize.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union
import networkx as nx
import numpy as np

def load_and_filter_geojson(input_file, output_file):
    """Load GeoJSON and filter to keep only assigned districts (districtr != -1)."""
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['features'])
    print(f"Original features: {original_count}")
    
    # Filter features: exclude districtr == -1 AND districtr == 6
    filtered_features = [
        f for f in data['features'] 
        if f['properties'].get('districtr', -1) not in [-1, 6]
    ]
    
    print(f"Filtered features: {len(filtered_features)}")
    print(f"Removed: {original_count - len(filtered_features)}")
    
    # Create new GeoJSON with filtered features
    filtered_data = {
        'type': 'FeatureCollection',
        'features': filtered_features
    }
    
    # Save filtered GeoJSON
    print(f"Saving filtered data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    return filtered_data

def build_dual_graph(geojson_data):
    """Build dual graph where nodes are polygons and edges connect adjacent polygons."""
    print("\nBuilding dual graph...")
    
    features = geojson_data['features']
    G = nx.Graph()
    
    # Create nodes with attributes
    for i, feature in enumerate(features):
        props = feature['properties']
        districtr_id = props.get('districtr', -1)
        geom = shape(feature['geometry'])
        
        # Calculate centroid for node position
        centroid = geom.centroid
        
        G.add_node(i, 
                   districtr=districtr_id,
                   geometry=geom,
                   centroid=(centroid.x, centroid.y),
                   properties=props)
    
    # Find adjacencies (polygons that share a boundary)
    print("Finding adjacent polygons...")
    for i in range(len(features)):
        geom_i = G.nodes[i]['geometry']
        for j in range(i + 1, len(features)):
            geom_j = G.nodes[j]['geometry']
            
            # Check if polygons touch (share boundary)
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                G.add_edge(i, j)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def plot_visualization(geojson_data, graph, colored=True, output_file='visualization.png'):
    """Plot geodata and dual graph with or without colors, no background elements."""
    print(f"\nCreating {'colored' if colored else 'uncolored'} visualization...")
    
    # Define light, non-clashing pastel colors for districtr values
    districtr_colors = {
        1: '#FFB3BA',  # light pink
        2: '#BAE1FF',  # light blue
        3: '#BAFFC9',  # light mint green
        4: '#E0BBE4',  # light lavender
        6: '#FFD8B3',  # light peach
        7: '#FFFFBA',  # light yellow
        8: '#C1E1C1',  # light sage green
    }
    
    # Rotation angle in degrees (positive = counterclockwise)
    rotation_angle = 1  # Very subtle rotation to make figures more horizontal
    
    def rotate_coords(x, y, angle_deg):
        """Rotate coordinates by angle (in degrees) counterclockwise."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        return x_rot, y_rot
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Plot geodata (polygons)
    for node_id in graph.nodes():
        geom = graph.nodes[node_id]['geometry']
        districtr_id = graph.nodes[node_id]['districtr']
        
        # Use district colors if colored=True, otherwise use pure white
        if colored:
            color = districtr_colors.get(districtr_id, '#cccccc')
            edge_width = 0.5
        else:
            color = '#ffffff'  # pure white for uncolored version
            edge_width = 1.2  # Thicker edges for uncolored version
        
        # Handle both Polygon and MultiPolygon
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
                ax.fill(x_rot, y_rot, color=color, alpha=1.0, edgecolor='black', linewidth=edge_width)
        else:
            x, y = geom.exterior.xy
            x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
            ax.fill(x_rot, y_rot, color=color, alpha=1.0, edgecolor='black', linewidth=edge_width)
    
    # Draw county boundaries only for colored version
    if colored:
        counties = {}
        for node_id in graph.nodes():
            props = graph.nodes[node_id]['properties']
            county_fp = props.get('COUNTYFP20', 'Unknown')
            if county_fp not in counties:
                counties[county_fp] = []
            counties[county_fp].append(graph.nodes[node_id]['geometry'])
        
        # Draw boundary for each county
        for county_fp, geometries in counties.items():
            county_shape = unary_union(geometries)
            
            if isinstance(county_shape, MultiPolygon):
                for poly in county_shape.geoms:
                    x, y = poly.exterior.xy
                    x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
                    ax.plot(x_rot, y_rot, 'k-', linewidth=3, alpha=1.0, zorder=8)
            else:
                x, y = county_shape.exterior.xy
                x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
                ax.plot(x_rot, y_rot, 'k-', linewidth=3, alpha=1.0, zorder=8)
    
    # Plot graph edges - black with reduced opacity (0.5)
    for edge in graph.edges():
        node1, node2 = edge
        x1, y1 = graph.nodes[node1]['centroid']
        x2, y2 = graph.nodes[node2]['centroid']
        x1_rot, y1_rot = rotate_coords(x1, y1, rotation_angle)
        x2_rot, y2_rot = rotate_coords(x2, y2, rotation_angle)
        ax.plot([x1_rot, x2_rot], [y1_rot, y2_rot], 'k-', linewidth=1.5, alpha=0.5, zorder=5)
    
    # Plot graph nodes - colored to match districts in colored version, black in uncolored
    for node_id in graph.nodes():
        x, y = graph.nodes[node_id]['centroid']
        x_rot, y_rot = rotate_coords(x, y, rotation_angle)
        districtr_id = graph.nodes[node_id]['districtr']
        
        # Use district colors for nodes in colored version
        if colored:
            node_color = districtr_colors.get(districtr_id, '#cccccc')
        else:
            node_color = 'black'
        
        ax.plot(x_rot, y_rot, 'o', markersize=8, color=node_color, 
                markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # Remove all background elements
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes, labels, ticks
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Visualization saved to {output_file}")
    plt.close()
    
    return fig

def build_supergraph(dual_graph):
    """Build supergraph where each district is a single node."""
    print("\nBuilding district-level supergraph...")
    
    SG = nx.Graph()
    
    # Create a node for each unique district
    districts = set()
    for node_id in dual_graph.nodes():
        districtr_id = dual_graph.nodes[node_id]['districtr']
        districts.add(districtr_id)
    
    # Add nodes for each district
    for district_id in districts:
        # Count polygons in this district
        polygon_count = sum(1 for n in dual_graph.nodes() 
                           if dual_graph.nodes[n]['districtr'] == district_id)
        SG.add_node(district_id, polygon_count=polygon_count)
    
    # Add edges between adjacent districts
    # Two districts are adjacent if any of their polygons are adjacent
    for edge in dual_graph.edges():
        node1, node2 = edge
        dist1 = dual_graph.nodes[node1]['districtr']
        dist2 = dual_graph.nodes[node2]['districtr']
        
        # Only add edge if districts are different
        if dist1 != dist2:
            SG.add_edge(dist1, dist2)
    
    print(f"Supergraph: {SG.number_of_nodes()} districts, {SG.number_of_edges()} adjacencies")
    return SG

def plot_supergraph(supergraph, dual_graph, output_file='supergraph.png'):
    """Plot the district-level supergraph with nodes at geographic positions."""
    print("\nCreating supergraph visualization...")
    
    # Define light pastel colors
    districtr_colors = {
        1: '#FFB3BA',  # light pink
        2: '#BAE1FF',  # light blue
        3: '#BAFFC9',  # light mint green
        4: '#E0BBE4',  # light lavender
        6: '#FFD8B3',  # light peach
        7: '#FFFFBA',  # light yellow
        8: '#C1E1C1',  # light sage green
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Calculate geographic centroid for each district
    pos = {}
    for district_id in supergraph.nodes():
        # Get all polygons belonging to this district
        district_geoms = []
        for node_id in dual_graph.nodes():
            if dual_graph.nodes[node_id]['districtr'] == district_id:
                district_geoms.append(dual_graph.nodes[node_id]['geometry'])
        
        # Merge all polygons in the district and get centroid
        district_union = unary_union(district_geoms)
        centroid = district_union.centroid
        pos[district_id] = (centroid.x, centroid.y)
    
    # Scale positions to make nodes closer together
    # Find center point
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    
    # Scale around center (smaller factor = nodes closer together)
    scale_factor = 0.3  # Adjust this value (0.1-0.5 recommended)
    scaled_pos = {}
    for node_id, (x, y) in pos.items():
        # Translate to origin, scale, translate back
        scaled_x = center_x + (x - center_x) * scale_factor
        scaled_y = center_y + (y - center_y) * scale_factor
        scaled_pos[node_id] = (scaled_x, scaled_y)
    
    pos = scaled_pos
    
    # Draw edges
    for edge in supergraph.edges():
        node1, node2 = edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5, zorder=1)
    
    # Draw nodes
    for node_id in supergraph.nodes():
        x, y = pos[node_id]
        color = districtr_colors.get(node_id, '#cccccc')
        polygon_count = supergraph.nodes[node_id]['polygon_count']
        
        # Node size proportional to number of polygons
        size = 800 + polygon_count * 100
        
        ax.scatter(x, y, s=size, color=color, 
                  edgecolors='black', linewidths=3, zorder=10)
        
        # Add district label with subscript
        ax.text(x, y, f'$D_{{{node_id}}}$', 
               fontsize=16, ha='center', va='center', 
               weight='bold', zorder=11)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, 
                transparent=True, facecolor='none')
    print(f"Supergraph saved to {output_file}")
    plt.close()
    
    return fig

def plot_districts_with_supergraph(geojson_data, dual_graph, supergraph, output_file='districts_supergraph.png'):
    """Plot geographic districts with supergraph overlay (no dual graph)."""
    print("\nCreating districts + supergraph visualization...")
    
    # Define light pastel colors
    districtr_colors = {
        1: '#FFB3BA',  # light pink
        2: '#BAE1FF',  # light blue
        3: '#BAFFC9',  # light mint green
        4: '#E0BBE4',  # light lavender
        6: '#FFD8B3',  # light peach
        7: '#FFFFBA',  # light yellow
        8: '#C1E1C1',  # light sage green
    }
    
    # Rotation angle (same as other visualizations)
    rotation_angle = 1
    
    def rotate_coords(x, y, angle_deg):
        """Rotate coordinates by angle (in degrees) counterclockwise."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        return x_rot, y_rot
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Plot geographic districts (polygons)
    for node_id in dual_graph.nodes():
        geom = dual_graph.nodes[node_id]['geometry']
        districtr_id = dual_graph.nodes[node_id]['districtr']
        color = districtr_colors.get(districtr_id, '#cccccc')
        
        # Handle both Polygon and MultiPolygon
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
                ax.fill(x_rot, y_rot, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
        else:
            x, y = geom.exterior.xy
            x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
            ax.fill(x_rot, y_rot, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)
    
    # Draw county boundaries
    counties = {}
    for node_id in dual_graph.nodes():
        props = dual_graph.nodes[node_id]['properties']
        county_fp = props.get('COUNTYFP20', 'Unknown')
        if county_fp not in counties:
            counties[county_fp] = []
        counties[county_fp].append(dual_graph.nodes[node_id]['geometry'])
    
    for county_fp, geometries in counties.items():
        county_shape = unary_union(geometries)
        if isinstance(county_shape, MultiPolygon):
            for poly in county_shape.geoms:
                x, y = poly.exterior.xy
                x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
                ax.plot(x_rot, y_rot, 'k-', linewidth=3, alpha=1.0, zorder=8)
        else:
            x, y = county_shape.exterior.xy
            x_rot, y_rot = rotate_coords(np.array(x), np.array(y), rotation_angle)
            ax.plot(x_rot, y_rot, 'k-', linewidth=3, alpha=1.0, zorder=8)
    
    # Calculate geographic positions for supergraph nodes
    pos = {}
    for district_id in supergraph.nodes():
        district_geoms = []
        for node_id in dual_graph.nodes():
            if dual_graph.nodes[node_id]['districtr'] == district_id:
                district_geoms.append(dual_graph.nodes[node_id]['geometry'])
        district_union = unary_union(district_geoms)
        centroid = district_union.centroid
        pos[district_id] = (centroid.x, centroid.y)
    
    # Draw supergraph edges
    for edge in supergraph.edges():
        node1, node2 = edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        x1_rot, y1_rot = rotate_coords(x1, y1, rotation_angle)
        x2_rot, y2_rot = rotate_coords(x2, y2, rotation_angle)
        ax.plot([x1_rot, x2_rot], [y1_rot, y2_rot], 'k-', linewidth=3, alpha=0.7, zorder=15)
    
    # Draw supergraph nodes
    for node_id in supergraph.nodes():
        x, y = pos[node_id]
        x_rot, y_rot = rotate_coords(x, y, rotation_angle)
        color = districtr_colors.get(node_id, '#cccccc')
        polygon_count = supergraph.nodes[node_id]['polygon_count']
        
        # Node size proportional to number of polygons
        size = 1200 + polygon_count * 150
        
        ax.scatter(x_rot, y_rot, s=size, color=color, 
                  edgecolors='black', linewidths=4, zorder=20)
        
        # Add district label with subscript
        ax.text(x_rot, y_rot, f'$D_{{{node_id}}}$', 
               fontsize=18, ha='center', va='center', 
               weight='bold', zorder=21)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, 
                transparent=True, facecolor='none')
    print(f"Districts + supergraph saved to {output_file}")
    plt.close()
    
    return fig

def main():
    # File paths
    filtered_file = 'filtered_districts.geojson'
    
    # Load already filtered data
    print(f"Loading {filtered_file}...")
    with open(filtered_file, 'r') as f:
        filtered_data = json.load(f)
    
    print(f"Loaded features: {len(filtered_data['features'])}")
    
    # Step 2: Build dual graph
    G = build_dual_graph(filtered_data)
    
    # Print graph statistics
    print(f"\nGraph Statistics:")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Is connected: {nx.is_connected(G)}")
    if not nx.is_connected(G):
        print(f"  Number of components: {nx.number_connected_components(G)}")
    
    # Print districtr distribution
    districtr_counts = {}
    for node in G.nodes():
        dist_id = G.nodes[node]['districtr']
        districtr_counts[dist_id] = districtr_counts.get(dist_id, 0) + 1
    print(f"\nDistrict Distribution:")
    for dist_id in sorted(districtr_counts.keys()):
        print(f"  District {dist_id}: {districtr_counts[dist_id]} polygons")
    
    # Print county information
    counties = {}
    for node in G.nodes():
        county_fp = G.nodes[node]['properties'].get('COUNTYFP20', 'Unknown')
        counties[county_fp] = counties.get(county_fp, 0) + 1
    print(f"\nCounty Distribution:")
    for county_fp in sorted(counties.keys()):
        print(f"  County {county_fp}: {counties[county_fp]} polygons")
    
    # Step 3: Create both visualizations
    plot_visualization(filtered_data, G, colored=True, 
                       output_file='dual_graph_colored.png')
    plot_visualization(filtered_data, G, colored=False, 
                       output_file='dual_graph_uncolored.png')
    
    print("\n✅ Both visualizations created successfully!")
    print("   - dual_graph_colored.png (with district colors)")
    print("   - dual_graph_uncolored.png (without colors)")
    
    # Step 4: Build and visualize supergraph
    SG = build_supergraph(G)
    plot_supergraph(SG, G, output_file='supergraph.png')
    
    print("\n✅ Supergraph created!")
    print("   - supergraph.png (district-level adjacency graph)")
    
    # Step 5: Create combined visualization (districts + supergraph)
    plot_districts_with_supergraph(filtered_data, G, SG, output_file='districts_supergraph.png')
    
    print("\n✅ Combined visualization created!")
    print("   - districts_supergraph.png (districts with supergraph overlay)")

if __name__ == "__main__":
    main()
