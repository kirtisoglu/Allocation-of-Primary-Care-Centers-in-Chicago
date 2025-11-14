
import branca
import branca.colormap as cm
import folium
import matplotlib as plt
import matplotlib.pyplot as pltt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from folium import plugins
from libpysal import weights

from .data_handler import DataHandler


class Plot:

    def __init__(self) -> None:

        handler = DataHandler()
        self.geo_data = handler.load_chicago()
        self.geo_candidates = handler.load_geo_candidates()
    

    def basemap(self):

        fig = px.scatter_mapbox(self.geo_data, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],
                                color_discrete_sequence=["fuchsia"], zoom=3, height=300)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        return fig

    # from tree.py
    def plot_map(self, geo_centers, assignment, attr):
        import folium
        import mapclassify
        import matplotlib
        chicago = self.geo_data
        chicago[attr] = [assignment[node] for node in chicago.index]
        regions = chicago.dissolve(by=attr, as_index=False)

        # m = folium.Map([41.85, -87.68], zoom_start=10)
        m = regions.explore(
            column=attr,  # make choropleth based on "district" column
            tooltip=attr,  # show "district" value in tooltip (on hover)
            popup=True,  # show all values in popup (on click)
            tiles="OpenStreetMap",  # use "CartoDB positron" or "OpenStreetMap" tiles
            cmap="Set1",  # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"),  # use black outline
            legend_kwds=dict(colorbar=False),
            # tooltip_kwds=dict(labels=False),  # do not show column label in the tooltip
            # smooth_factor=2,
            # fill_opacity=0.3,  #  transparency of fill colors
            # line_opacity=0.1,  # to de-emphasize border lines
            # fill_color="RdYlGn_r",  # or "YlGn"
            # nan_fill_color="white", # Also see nan_fill_opacity=0.4,
            highlight=True,
            name="chicago",
        )

        # Adds a button to enable/disable zoom scrolling
        folium.plugins.ScrollZoomToggler().add_to(m)

        # To make the map full screen
        folium.plugins.Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)

        geo_centers.explore(
            m=m,  # pass the map object
            color="black",  # use red color on all points
            marker_kwds=dict(radius=3, fill=True),  # make marker radius 10px with fill
            name="Candidates",  # name of the layer in the map
        )
        # folium.TileLayer("CartoDB positron", show=False).add_to(m)
        # use folium to add alternative tiles
        folium.LayerControl().add_to(m)  # use folium to add layer control

        return m, regions, chicago, geo_centers

    
    def plot(self, candidates, color_value=None):
        """
        Plots census blocks from a GeoDataFrame with a uniform color
        and marks given candidates (centers) on the map.

        Args:
            data (gpd.GeoDataFrame): GeoDataFrame containing census block geometries.
            candidates (list): A list of indices from the 'data' GeoDataFrame
                            representing the candidates (centers) to be marked.
            color_value (str or int or float, optional): A value to assign for the color
                                                        of the census blocks. If None,
                                                        a default grey will be used.
        """

        data = self.geo_data
        
        fig = px.choropleth(
            data,
            geojson=data.geometry.__geo_interface__,
            locations=data.index,
            color='id_1',  # This must be a column name from 'data'
            center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
            height=800,
            projection="mercator",
            hover_data=['population'] if 'population' in data.columns else None # Show population data on hover if available
        )



        fig = px.choropleth(df, geojson=geojson, color="winner",
                            locations="district", featureidkey="properties.district",

                        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()
        
        # Add specified candidates (centers) as markers
        for candidate in candidates:
            if candidate in data.index:
                candidate_point = data.loc[candidate].geometry.centroid
                fig.add_scattermapbox(
                    lat=[candidate_point.y],
                    lon=[candidate_point.x],
                    mode='markers',
                    marker=dict(size=15, color='yellow', symbol='star'),
                    name=f'Candidate={candidate}'
                )
            else:
                print(f"Warning: Candidate index '{candidate}' not found in data. Skipping marker.")

        return fig.show()
    

    def plot_districts(self, data, centers, attribute: str, color=None, fake_center=None):
        # Ensure data[attribute] has appropriate type for indexing
        data[attribute] = pd.Categorical(data[attribute])
        
        # Map each cluster to a color using a cycle of the Plotly qualitative palette
        colors = px.colors.qualitative.Plotly  # This is an example palette
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(data[attribute].cat.categories)}
        data['color'] = data[attribute].map(color_map)

        fig = px.choropleth_mapbox(
            data,
            geojson=data.geometry.__geo_interface__,
            locations=data.index,
            color=data['color'],
            mapbox_style="open-street-map",
            center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
            height=800,
            zoom=10,
            opacity=0.5,
            color_discrete_map="identity",  # Ensure this uses the direct mapping of assigned colors
            hover_data=[data['pop']]  # Show population data on hover
        )

        # Add cluster centers as markers
        for center in centers:
            center_point = data.loc[center].geometry.centroid
            fig.add_scattermapbox(
                lat=[center_point.y],
                lon=[center_point.x],
                mode='markers',
                marker=dict(size=10, color='black'),  # Black markers for centers
                name=f'District={center}'
            )

        return fig.show()
    

# Plot initial and final solutions side by side
    def compare(self, initial_partition, final_partition):
        from partition import Partition
        "plots initial and final partitions of recomb chain side by side"
        
        centers = self.geo_candidates.loc[self.geo_candidates['block'].isin(final_partition.centers.values())] 
        others = self.geo_candidates.loc[~self.geo_candidates['block'].isin(final_partition.centers.values())]
        
        self.geo_data['initial_district'] = [final_partition.assignment[node] for node in self.geo_data.index]
        self.geo_data['final_district'] = [initial_partition.assignment[node] for node in self.geo_data.index]
        regions_initial = self.geo_data.dissolve(by='initial_district', as_index=False)
        regions_final = self.geo_data.dissolve(by='final_district', as_index=False) 
        
        regions_initial_new = regions_initial.copy()
        regions_final_new = regions_final.copy()
        regions_initial_new['color'] = [x % 10 for x in range(len(initial_partition))]
        regions_final_new['color'] = [x % 10 for x in range(len(final_partition))]
        del regions_initial_new['centroid']
        del regions_final_new['centroid']
        regions_initial_json = regions_initial_new.to_json()
        regions_final_json = regions_final_new.to_json()

        step = cm.linear.Paired_10.scale(0, 9).to_step(10)

        # empty figures side by side
        fig = branca.element.Figure()
        subplot1 = fig.add_subplot(1, 2, 1)
        subplot2 = fig.add_subplot(1, 2, 2)

        # INITIAL
        m = folium.Map([41.85, -87.68], zoom_start=10, tiles="OpenStreetMap")
        folium.GeoJson(
            regions_initial_json,
            name="Initial Plan",
            tooltip=folium.GeoJsonTooltip(fields=["initial_district"]),
            popup=folium.GeoJsonPopup(fields=["initial_district"]),
            style_function=lambda feature: {
                "fillColor": step(feature['properties']['color']),
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda x: {"fillOpacity": 0.8},
        ).add_to(m)
        
        self.geo_candidates.explore(
            m=m,  # pass the map object
            color='black',  
            name="candidates",  # name of the layer in the map
        )

        folium.plugins.ScrollZoomToggler().add_to(m) #Adds a button to enable/disable zoom scrolling

        m.add_child(plugins.MeasureControl())  # a tool to measure distance and area on the map
        
        folium.plugins.Fullscreen(   # To make the map full screen
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)

        folium.LayerControl().add_to(m)  # use folium to add layer control
        
        # FINAL
        f = folium.Map([41.85, -87.68], zoom_start=10, tiles="OpenStreetMap")
        folium.GeoJson(
            regions_final_json,
            name="Final Plan",
            tooltip=folium.GeoJsonTooltip(fields=["final_district"]),
            popup=folium.GeoJsonPopup(fields=["final_district"]),
            style_function=lambda feature: {
                "fillColor": step(feature['properties']['color']),
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda x: {"fillOpacity": 0.8},
        ).add_to(f)
        
        
        f.add_child(plugins.MeasureControl())

        # To make the map full screen
        folium.plugins.Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(f)

        centers.explore(
            m = f,  # pass the map object
            color='red',  
            name="centers",  # name of the layer in the map
        )
        others.explore(
            m = f,  # pass the map object
            color='black',  
            name="others",  # name of the layer in the map
        )

        folium.LayerControl().add_to(f)  # use folium to add layer control
        
        #add them to the empty figures
        subplot1.add_child(m)
        subplot2.add_child(f)

        return fig, regions_initial_new, regions_final_new, centers, others


# Plot dual graph on geographical map
    def plot_dual_graph(self, geo_data = None):
        
        if geo_data == None:
            geo_data = self.geo_data
        centroids = np.column_stack((geo_data.geometry.centroid.x, geo_data.geometry.centroid.y))
        
        # construct the "Queen" adjacency graph. In geographical applications,
        # the "Queen" adjacency graph considers two polygons as connected if
        # they share a single point on their boundary. 
        queen = weights.Queen.from_dataframe(geo_data.geometry.centroid)
        
        # we can convert the graph to networkx object
        graph = queen.to_networkx()
        
        positions = dict(zip(graph.nodes, centroids))
        # we need to merge the nodes back to their positions in order to plot in networkx
        
        # plot with a nice basemap
        ax = geo_data.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
        ax.axis([-12, 45, 33, 66])
        ax.axis("off")
        nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
        plt.show()





def plot_grid(graph):

    list_nodes = list(graph.nodes)
    line = []
    node_positions = {}
    labels = {}
    font = {}
    
    for node in list_nodes:
        node_positions[node]=node
        
        if graph.nodes[node]["candidate"]== True:
            labels[node] = f"{graph.nodes[node]["population"]}"
            line.append(1)
        else: 
            labels[node]= f"{graph.nodes[node]["population"]}"
            line.append(0)

    
    # nx.draw_networkx_nodes(G_ex, pos, alpha=0.8, node_color=node_color_list)
    nx.draw_networkx_nodes(graph, node_positions, nodelist=list_nodes, node_size=150, alpha=0.8, linewidths=line, edgecolors='red', node_color="tab:blue")
    nx.draw_networkx_edges(graph, node_positions, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, node_positions, labels, font_size=8, font_color="whitesmoke")

    fig = pltt.figure(1, figsize=(12, 6))
    pltt.tight_layout()
    pltt.axis("off")
    pltt.show()
    
    



def plot_grid_with_candidates(Chicago, grid, candidates):
    """
    Plots the census blocks (Chicago), the spatial grid, and selected candidate blocks.
    
    Parameters:
    - Chicago: GeoDataFrame of all census blocks
    - grid: GeoDataFrame of grid cells
    - candidates: GeoDataFrame of selected candidate blocks
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot census blocks in light gray
    Chicago.plot(ax=ax, color='lightgray', linewidth=0.1, edgecolor='white', label='Census Blocks')

    # Plot grid cells with transparent fill and black edges
    grid.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5, alpha=0.5, label='Grid Cells')

    # Plot selected candidates in red
    candidates.plot(ax=ax, color='red', label='Selected Candidates')

    ax.set_title("Geographic Sampling of Census Blocks with Grid Overlay")
    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.show()
