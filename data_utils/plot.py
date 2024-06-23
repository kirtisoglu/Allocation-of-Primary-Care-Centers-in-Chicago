


import pandas as pd
import matplotlib as plt
import plotly.graph_objects as go
import plotly.express as px



class Plot:

    def __init__(self) -> None:

        self._create_properties()
        self.show = self.visualize()
        
 
 
        
    def visualize(self):
        return
    

    def _create_properties(self):
        """Dynamically creates properties for each file detected."""
        for name in self.files:
            setattr(self, f"load_{name}", self._create_loader(name))


    def _create_loader(self, name):
        """Creates a loader function for a specific file."""
        def loader():
            return self.load(name)
        return loader
            

    def basemap(self, data):

        fig = px.scatter_mapbox(data, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],
                                color_discrete_sequence=["fuchsia"], zoom=3, height=300)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        return fig
    
    
    def incomplete():
        
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
    
    
"""
def fig_blocks(data):

    data['constant']=2
    
    fig = px.choropleth_mapbox(
        data,
        geojson=data.geometry.__geo_interface__,  # Make sure to convert to GeoJSON correctly
        locations=data.index,  # Index or any unique identifier
        color="constant",  # Use the constant column for coloring
        color_discrete_sequence=["blue"],  # This sets the uniform color
        mapbox_style="open-street-map",
        center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
        height = 800,
        zoom=10,
        opacity=0.2,
        labels={'constant': 'Label here'},  # Adjust label as needed
    )
    
    return fig




def  add_phc(fig, phc_df):
    
    # Add the PHC locations as a new layer to the map
    fig.add_scattermapbox(
    lat=phc_df['lat'],
    lon=phc_df['lon'],
    mode='markers',
    marker=dict(size=5, color='black'))
    
    return fig






def add_routes(chicago, routes, height):
    
    stops = routes['bustime-response']['ptr'][0]['pt'] 

    latitudes = [stop['lat'] for stop in stops]
    longitudes = [stop['lon'] for stop in stops]
    hover_texts = [stop.get("stpnm", f"Stop {stop['seq']}") for stop in stops]
    
    fig1 = plot_census(chicago, height)


    # Plot stops as scatter points on the map
    fig1.add_trace(go.Scattermapbox(
        lat=latitudes,
        lon=longitudes,
        mode='markers+text',  # 'markers' for points, 'text' if you want to show labels
        marker=go.scattermapbox.Marker(size=0.1, color='white'),  # Adjust size and color as needed
        text=hover_texts,
        hoverinfo='text'
    ))

    # Draw lines between consecutive stops to create the route
    fig1.add_trace(go.Scattermapbox(
        lat=latitudes,
        lon=longitudes,
        mode='lines',
        line=dict(width=2, color='red'), 
        hoverinfo='none' 
    ))

    # Update the layout to adjust the viewport if needed
    fig1.update_layout(
        mapbox=dict(
            style='open-street-map', 
            zoom=10, 
            center=dict(lat=sum(latitudes) / len(latitudes), lon=sum(longitudes) / len(longitudes))  # Center map on route
        ),
        showlegend=False
    )

    fig1.show()

"""
