
from libpysal import weights
import networkx as nx
import numpy as np
import geopandas 
from .data_handler import DataHandler

import pandas as pd
import matplotlib as plt
import plotly.graph_objects as go
import plotly.express as px
import branca.colormap as cm
import branca
from .data_handler import DataHandler
from partition import Partition
import folium
from folium import plugins
import geopandas
from datashader.colors import viridis
import matplotlib.pyplot as pltt


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
    
    
    def incomplete_districts():
        return


    def plot(self, data, centers, attribute: str, color=None, fake_center=None):
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
    def compare(self, initial_partition: Partition, final_partition: Partition):
        
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




    def plot_map(assignment, attr):
        import folium
        import matplotlib
        import mapclassify

        handler = DataHandler()
        chicago = handler.load_chicago()
        geo_centers = handler.load_geo_centers()  ## Define a function for that

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
            #tooltip_kwds=dict(labels=False),  # do not show column label in the tooltip
            #smooth_factor=2,
            #fill_opacity=0.3,  #  transparency of fill colors
            #line_opacity=0.1,  # to de-emphasize border lines
            #fill_color="RdYlGn_r",  # or "YlGn"
            #nan_fill_color="white", # Also see nan_fill_opacity=0.4,
            highlight=True,
            name = "chicago"
        )

        #Adds a button to enable/disable zoom scrolling
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
        #folium.TileLayer("CartoDB positron", show=False).add_to(m)  
        # use folium to add alternative tiles
        folium.LayerControl().add_to(m)  # use folium to add layer control


        # Side by side Layers: control=False  to add a layer control to your map
        #m = folium.Map(location=(30, 20), zoom_start=4)

        #layer_right = folium.TileLayer('openstreetmap')
        #layer_left = folium.TileLayer('cartodbpositron')

        #sbs = folium.plugins.SideBySideLayers(layer_left=layer_left, layer_right=layer_right)

        #layer_left.add_to(m)
        #layer_right.add_to(m)
        #sbs.add_to(m)

        return m, regions, chicago, geo_centers 


def plot_grid(graph):
    
    list_nodes = list(graph.nodes)
    #noncandidate_nodes = [item for item in list_nodes if item not in list_candidates]
    
    node_positions = {}
    for i in list_nodes:
        node_positions[i]=i

    nx.draw_networkx_nodes(graph, node_positions, nodelist=list_nodes, node_color="tab:blue")
    
    # edges
    nx.draw_networkx_edges(graph, node_positions, width=1.0, alpha=0.5)
    
    labels = {}
    
    for node in graph.nodes:
        if graph.nodes[node]["candidate"]== True:
            labels[node] = f"C-{graph.nodes[node]["population"]}"    
        else: 
            labels[node]= f"{graph.nodes[node]["population"]}"

    nx.draw_networkx_labels(graph, node_positions, labels, font_size=8, font_color="whitesmoke")

    pltt.tight_layout()
    pltt.axis()
    pltt.show()