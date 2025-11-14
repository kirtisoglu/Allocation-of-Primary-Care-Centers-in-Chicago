import json
import os
from typing import Dict

import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Point, Polygon


# to plot only census blocks, make stroked = False for grid and add a button for it
# might want to add legend, layer toggles using checkboxes, or dynamic stats panel on hover.
class Plotter:
    """
    MapPlotter handles the visualization of candidate points and population grid.

    Attributes:
        blocks (gpd.GeoDataFrame): Input census blocks.
        real_candidates (gpd.GeoDataFrame): Real candidate points.
        artificial_candidates (gpd.GeoDataFrame): Artificial candidate points.
        grid_metadata (dict): Metadata per grid cell including geometry and population.
    """

    def __init__(
        self,
        blocks: gpd.GeoDataFrame,
        graph,
    ):

        if "centroid" in blocks.columns:
            blocks.drop("centroid", axis=1, inplace=True)

        self.blocks = blocks
        self.artificial_candidates = None
        self.real_candidates = None
        self.prepare_grid()
        self.add_candidates(graph)

    def prepare_grid(self):
        grid_cells = GridProcessor.generate_grid_cells(self.blocks)
        self.grid_cells = grid_cells  # store for later if needed

        self.grid_metadata = {}
        pops = GridProcessor.calculate_grid_populations(self.blocks, grid_cells)

        for cell_name, geom in grid_cells.items():
            self.grid_metadata[cell_name] = {
                "geometry": geom,
                "population": round(pops.get(cell_name, 0)),
            }
        self.grid = gpd.GeoDataFrame(
            [{"geometry": v["geometry"], **v} for v in self.grid_metadata.values()],
            crs="EPSG:4326",
        )

    def add_candidates(self, graph):
        yellow, red = [], []

        for node, data in graph.nodes(data=True):
            if "C_X" in data and "C_Y" in data:
                pt = Point(data["C_X"], data["C_Y"])
            else:
                raise ValueError(
                    f"Node {node} does not have coordinate attribute 'C_X' or 'C_Y'."
                )

            if data.get("real_candidate") == 1:
                yellow.append(pt)
            elif data.get("artificial_candidate") == 1:
                red.append(pt)

        self.real_candidates = gpd.GeoDataFrame(geometry=yellow, crs="EPSG:4326")
        self.artificial_candidates = gpd.GeoDataFrame(geometry=red, crs="EPSG:4326")

    # not in use currently
    def set_grid_metadata(self):
        """
        Adds population and needed candidate info to each grid cell metadata.
        """
        # Calculate populations
        pop_dict = GridProcessor.calculate_grid_populations(
            self.blocks, self.grid_cells
        )

        for cell_name, geom in self.grid_cells.items():
            self.grid_metadata[cell_name]["pop"] = round(pop_dict.get(cell_name, 0))

    # not in use currently
    def add_grid(self, grid_metadata: Dict[Polygon, Dict]):
        records = [{"geometry": geom, **attrs} for geom, attrs in grid_metadata.items()]
        self.grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

    def plot(self):
        layers = []
        blocks_data = json.loads(self.blocks.to_json())

        # Layer 1: Census blocks
        if self.blocks is not None:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=blocks_data,
                    stroked=True,
                    filled=True,
                    get_fill_color="[0, 0, 0, 0]",
                    get_line_color=[0, 0, 0],
                    line_width_min_pixels=1,
                )
            )

        # Layer 2: Grid
        if self.grid is not None:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=self.grid,
                    stroked=True,
                    filled=False,
                    get_line_color=[0, 0, 255],
                    pickable=True,
                    line_width_min_pixels=1,
                )
            )

        # Layer 3: Real candidate points
        if self.real_candidates is not None and not self.real_candidates.empty:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    id="real-candidates",
                    data=self.real_candidates,
                    stroked=False,
                    filled=True,
                    get_fill_color="[255, 255, 0, 255]",  # yellow
                    point_radius_min_pixels=2,
                    visible=True,
                )
            )

        # Layer 4: Artificial candidate points
        if (
            self.artificial_candidates is not None
            and not self.artificial_candidates.empty
        ):
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    id="artificial-candidates",
                    data=self.artificial_candidates,
                    stroked=False,
                    filled=True,
                    get_fill_color="[255, 0, 0, 255]",  # red
                    point_radius_min_pixels=2,
                    visible=True,
                )
            )

        centroids = self.blocks.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
        view_state = pdk.ViewState(
            longitude=centroids.x.mean(), latitude=centroids.y.mean(), zoom=11, pitch=0
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"text": "Population: {pop}"},
        )

        # Save to a fixed path (in temp or project folder)
        output_path = os.path.join(
            os.path.expanduser("~"), "Desktop", "chicago_map.html"
        )

        html = str(deck.to_html(as_string=True))

        # JS to expose deckgl instance
        expose_deck = """
        <script>
        window.deckgl = deckgl;
        </script>
        """

        # Buttons and toggling logic
        buttons_html = """
        <div style="position:absolute; top:10px; left:10px; z-index:1000;">
        <button onclick="toggleLayer('real-candidates')">Toggle Real Candidates</button>
        <button onclick="toggleLayer('artificial-candidates')">Toggle Artificial</button>
        </div>
        <script>
        function toggleLayer(id) {
            const deck = window.deckgl;
            const layers = deck.props.layers.map(layer => {
            if (layer.id === id) {
                return layer.clone({ visible: !layer.props.visible });
            }
            return layer;
            });
            deck.setProps({ layers });
        }
        </script>
        """

        # Inject expose_deck and buttons into HTML
        html = html.replace("</body>", expose_deck + buttons_html + "</body>")

        with open(output_path, "w") as f:
            f.write(html)

        print("Map saved to:", output_path)
