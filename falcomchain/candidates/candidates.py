# Uniform sampling of census blocks
#   Every block has equal probability of being selected.
#   Blocks are selected irrespective of location.
#   The result is not geographically uniform — it reflects the spatial distribution of blocks.

# Consequences
#   Dense areas (e.g., downtown, North Side) have more blocks, so they dominate the sample.
#   Sparse areas (e.g., industrial zones, far South/West sides, or parkland) are underrepresented or ignored.

# Grid based sampling
#   Divides the city into equal-size geographic bins (grid cells).
#   Samples at most one block per cell (until k are chosen).
#   Grid cells are randomly ordered, then sampled.
#   Candidate blocks are selected to represent different geographic zones.
#   Results in spatially dispersed selections.
#   Reduces the risk of over-sampling from one area and missing others.

# Analogy
#   Uniform block sampling	Throwing darts at a histogram of block counts (favoring dense areas)
#   Grid-based sampling	Throwing darts at a map of the city, and picking blocks close to where they land

# Suitable for: Representing population	Representing geography
# Random block sample: clusters around downtown, north side.
# Grid sample: spreads over the entire Chicago area (including southwest, southeast, etc.).
# Represent the population/density	Uniform block sampling
# Ensure wide spatial coverage	Grid-based sampling

import json
import random
from collections import namedtuple
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# import shapely
from shapely.geometry import Point, Polygon, box

from falcomchain.graph.geo import reprojected
from falcomchain.helper import save_pickle

# Block tuple that is used in Cells
Block = namedtuple("Block", "area pop factor is_real includes_centroid")
Block.__doc__ = "Represents a block portion covered by a cell."
Block.area.__doc__ = "area of the block within the cell"
Block.pop.__doc__ = "Estimated population in this block portion within the cell."
Block.factor.__doc__ = "Sharing factor of the block's area covered by this cell."
Block.is_real.__doc__ = "1 if the block is a real candidate, 0 otherwise."
Block.includes_centroid.__doc__ = (
    "True if Block geometry includes the centroid of the original block and is_real = 1"
)


@dataclass
class Cell:
    """
    Represents a single spatial cell in grid.

    Attributes:
        name: Unique identifier for the cell.
        blocks (List[dict]): Records of spatial features (e.g., census blocks) intersecting this cell.
        pop (float): Estimated population in the cell.
        candidates (list): List of node ids of artificial candidates within the cell.
    """

    pop: float
    needs: int
    blocks: dict = field(default_factory=dict)

    real_candidates: list = field(init=False)
    missing: int = field(init=False)
    possible_blocks: list = field(init=False)
    artificial_candidates: list = field(default_factory=list)

    def __post_init__(self):
        self.real_candidates = [
            b
            for b in self.blocks
            if self.blocks[b].is_real == 1 and self.blocks[b].includes_centroid == True
        ]
        self.missing = self.needs - len(self.real_candidates)
        self.possible_blocks = [b for b in self.blocks if self.blocks[b].is_real == 0]
        # self.blocks[b].pop > 0

    def __iter__(self) -> Iterator[Block]:
        """Allow iteration directly over Block objects."""
        return iter(self.blocks.keys())

    def __getattr__(self, name):
        """
        Redirects attribute access to the underlying 'blocks' dictionary.
        This allows for `cell_instance.block` instead of `cell_instance.blocks['block']`.
        """
        if name in self.blocks:
            return self.blocks[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def assign_candidates(self, my_list):
        "Returns a new Cell instance with updated candidates list."
        return replace(self, artificial_candidates=my_list)

    # def __getitem__(self, name) -> Block:
    #    """Access a specific Block by its name."""
    #    return self.blocks[name]


# Notes: For now, I am using graph to check real candidates. we can do this with block_gdf later.
# create real_candidates function.
# how to regenerate a Cells instance? What to save for this?


@dataclass
class Cells:
    """
    Processes a GeoDataFrame into a regular grid of `Cell` objects and
    calculates spatial statistics such as population distribution.
    It is assumed that existing candidates are assigned as graph node
    attributes: "real_candidate" = 0,1.

    Attributes: ( do we need to keep the first two in memory?)
        block_geodata (gpd.GeoDataFrame): The input spatial dataset (e.g., census blocks).
        graph (NetworkX): dual graph of block_data

        grid_size (float): The width/height of each grid cell in CRS units.
        grid_gdf (gpd.GeoDataFrame): geodata of grid cells
        cell_blocks_gdf (gpd.GeoDataFrame): geodata of cell blocks. "id_1" -> "id_2" -> (change this. Use Cell object)
        cells (list(Cell)): Dictionary mapping cell names to `Cell` objects.
    """

    def __init__(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        graph: nx.Graph,
        grid_size: float,
        real_candidates_gdf=None,
        geom_attr=None,
        cell_factor: int = 3,
        workload: int = 1500,
    ):
        self.graph = graph
        self.blocks_gdf = blocks_gdf  # crs checks after accepting data
        self.grid_size = grid_size

        if geom_attr == None:
            if "geometry" not in blocks_gdf.columns:
                raise AttributeError(
                    f'Oops! {blocks_gdf} does not have a "geometry" column. Pass an attribute name for geometries.'
                )
        if "population" not in self.blocks_gdf.columns:
            raise AttributeError(
                f"{self.blocks_gdf} must contain a 'population' column."
            )

        self._assign_real_candidates()  # first assign this. We will use node attributes for real candidates. Or use only geodataframe later.
        self.grid_gdf = self._create_grid_gdf()
        self.cell_blocks_gdf = self._overlay_cell_blocks()
        self.cells = self._create_cells(cell_factor, workload)

    # --------- Core Structure ---------
    def __iter__(self) -> Iterator[Cell]:
        """Allow iteration directly over Cell objects."""
        return iter(self.cells.keys())

    def __getattr__(self, name):
        """
        Redirects attribute access to the underlying 'cells' dictionary.
        This allows for `cells_instance.A1` instead of `cells_instance.cells['A1']`.
        """
        if name in self.cells:
            return self.cells[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _create_grid_gdf(self) -> gpd.GeoDataFrame:
        """
        Generates grid and geodataframe of cells with "id" and "geometry" columns.
        Raises:
            ValueError: If CRS is missing.

        Returns:
            A GeoDataFrame representing the grid cells.
        """
        if self.blocks_gdf.crs is None:
            raise ValueError(f"{self.blocks_gdf} GeoDataFrame must have a CRS.")

        minx, miny, maxx, maxy = self.blocks_gdf.total_bounds
        x_bins = np.arange(minx, maxx, self.grid_size)
        y_bins = np.arange(miny, maxy, self.grid_size)

        rows = []
        for i, x in enumerate(x_bins):
            for j, y in enumerate(y_bins):
                geom = box(x, y, x + self.grid_size, y + self.grid_size)
                rows.append({"id": (i, j), "geometry": geom})

        # construct GeoDataFrame from list of dicts so geometry is a GeoSeries and active geometry is set
        grid_gdf = gpd.GeoDataFrame(rows, crs=self.blocks_gdf.crs, geometry="geometry")

        return grid_gdf

    def _overlay_cell_blocks(self):
        "'id_2' is block id, 'id_1' is cell id"
        cell_blocks_gdf = self.grid_gdf.overlay(self.blocks_gdf, how="intersection")
        cell_blocks_gdf.to_crs(epsg=4326)

        if self.blocks_gdf.crs.is_geographic:
            print("Projecting CRS for area calculations in meters.")
            blocks_projected = reprojected(self.blocks_gdf)
            cell_blocks_projected = reprojected(cell_blocks_gdf)
        else:
            blocks_projected = self.blocks_gdf
            cell_blocks_projected = cell_blocks_gdf

        self.blocks_gdf["area"] = blocks_projected.geometry.area
        cell_blocks_gdf["area"] = cell_blocks_projected.geometry.area

        return cell_blocks_gdf

    def _assign_real_candidates(self):
        return

    def _cell_blocks(self, cell_gdf) -> dict:
        """
        needs cell geo dataset
        returns dict of Block objects keyed by node id in the dual graph
        """
        cell_blocks = {}

        for index, row in cell_gdf.iterrows():
            block_id = row["id_2"]
            node_id = self.blocks_gdf.index[self.blocks_gdf["id"] == block_id].item()
            portion = row["area"] / self.blocks_gdf.loc[node_id]["area"]
            sub_block_pop = portion * self.blocks_gdf.loc[node_id]["population"]
            is_real = self.graph.nodes[node_id]["real_candidate"]

            with_centroid = False
            if is_real == True:
                sub_geometry = row.geometry
                centroid = self.blocks_gdf.at[node_id, "geometry"].centroid
                with_centroid = sub_geometry.contains(centroid)

            cell_blocks[node_id] = Block(
                area=row["area"],
                pop=sub_block_pop,
                factor=portion,
                is_real=is_real,
                includes_centroid=with_centroid,
            )
        return cell_blocks

    def _create_cells(self, cell_factor, workload) -> dict:
        "creates Cell objects for each cell in the grid"
        cells = {}
        grouped_gdf = self.cell_blocks_gdf.groupby("id_1")

        for cell, cell_gdf in grouped_gdf:
            cell_blocks = self._cell_blocks(cell_gdf)
            cell_pop = (
                sum((float(b.pop) for b in cell_blocks.values()))
                if cell_blocks
                else 0.0
            )
            cells[cell] = Cell(
                pop=cell_pop,
                blocks=cell_blocks,
                needs=cell_factor * int(-(cell_pop // -workload)),
            )
        return cells

    def update_cell(self, cell, artificials: list):
        return cell.assign_candidates(artificials)

    def grid_json(self):
        return json.loads(self.grid_gdf.to_json())

    def cell_blocks_json(self):
        return json.loads(self.cell_blocks_gdf.to_json())

    def blocks_json(self):
        return json.loads(self.blocks_gdf.to_json())

    def to_json(self, gdf: gpd.GeoDataFrame):
        return json.loads(gdf.to_json())

    def pop(self) -> int:
        "total population in all cells"
        return sum(cell.pop for cell in self.cells.values())

    def plot(
        self,
        block_type: str = "all",
        color_pop: bool = False,
        candidates: Iterable[Any] | None = None,
    ):
        """_summary_

        Args:
            block_type (str, optional): _description_. Defaults to "all".
            candidates (Iterable[Any] | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Make sure CRS is lat/lon (EPSG:4326) for Plotly
        if block_type == "all":
            geo_data = self.blocks_gdf
            geojson_data = self.blocks_json()

        elif block_type == "non-candidate":
            non_candidates = [n for n in self.graph.nodes if n not in candidates]
            geo_data = self.list_to_geodataframe(non_candidates)
            geojson_data = self.to_json(geo_data)

        if color_pop == True:
            geo_data["pop_cat"] = geo_data["population"].apply(
                lambda x: "Zero Population" if x == 0 else "Nonzero Population"
            )

            fig = px.choropleth_mapbox(
                geo_data,
                geojson=geojson_data,
                locations=geo_data.index,  # link by index (must match GeoJSON feature 'id')
                color="pop_cat",
                category_orders={"pop_cat": ["Zero Population", "Nonzero Population"]},
                color_discrete_map={
                    "Zero Population": "yellow",
                    "Nonzero Population": "blue",
                },
                mapbox_style="carto-positron",
                center={"lat": 41.8781, "lon": -87.6298},
                zoom=9,
                opacity=0.6,  # OK here (Plotly Express wrapper supports it)
                width=1200,
                height=900,
            )

        else:
            fig = px.choropleth_mapbox(
                geo_data,
                geojson=geojson_data,
                locations=geo_data.index,  # link by index (must match GeoJSON feature 'id')
                mapbox_style="carto-positron",
                center={"lat": 41.8781, "lon": -87.6298},
                zoom=9,
                opacity=0.6,  # OK here (Plotly Express wrapper supports it)
                width=1200,
                height=900,
            )

        # --- Overlay: grid cell polygons (outline-only look) ---
        grid_cells_4326 = self.grid_gdf
        grid_cells_4326["gid"] = grid_cells_4326.index.astype(
            str
        )  # property to match on
        grid_geojson = json.loads(grid_cells_4326.to_json())

        fig.add_trace(
            go.Choroplethmapbox(
                geojson=grid_geojson,
                featureidkey="properties.gid",
                locations=grid_cells_4326["gid"],
                z=np.zeros(len(grid_cells_4326)),  # dummy values
                # transparent fill via RGBA colorscale (no 'opacity' arg on this trace type)
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_width=1.5,
                marker_line_color="black",
                name="Grid cells",
                hovertemplate="Grid cell: %{location}<extra></extra>",
                # below='' would force this layer above all mapbox layers if needed
                # below=''
            )
        )

        fig.update_layout(
            title="Chicago Census Blocks + Grid Cells",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )

        return fig

    def list_to_geodataframe(self, my_list):
        gdf = self.blocks_gdf.loc[my_list]
        return gdf.to_crs(epsg=4326)

    def candidates_to_geodataframe(self, candidate_type: str):
        candidates = []
        if candidate_type in ("real", "all"):
            for cell in self.cells.values():
                candidates.extend(cell.real_candidates)
        if candidate_type in ("artificial", "all"):
            for cell in self.cells.values():
                candidates.extend(cell.artificial_candidates)
        print(f"Number of {candidate_type} candidates: {len(candidates)}. Listed.")
        candidates_gdf = self.list_to_geodataframe(candidates)
        print(f"Created {candidate_type} candidates geodataframe.")
        return candidates_gdf

    def plot_add_candidates(self, fig, candidate_type: str):

        if candidate_type not in ["real", "artificial", "all"]:
            raise ValueError(
                f"which must be 'real' or 'artificial' or 'all', got {candidate_type}."
            )

        candidates_gdf = self.candidates_to_geodataframe(candidate_type=candidate_type)
        color_map = {"real": "yellow", "artificial": "red", "all": "green"}

        fig.add_trace(
            go.Scattermapbox(
                lat=candidates_gdf.geometry.centroid.y,
                lon=candidates_gdf.geometry.centroid.x,
                mode="markers",
                marker=dict(size=8, color=color_map[candidate_type], opacity=0.5),
                name="Candidates",
                text=(
                    candidates_gdf["population"]
                    if "population" in candidates_gdf.columns
                    else None
                ),  # hover info
                hoverinfo="text",
            )
        )
        return fig

    # optimize this function
    def assign_artificials_to_graph(self):
        """
        Assigns artificial candidates to the graph nodes based on the artificials dict.

        Args:
            graphhh (nx.Graph): The input graph.
            artificials (Dict[Tuple[int,int], List[int]]): A dictionary mapping cell names to lists of block ids.
        """
        artificials = []
        for cell in self.cells.values():
            artificials.extend(cell.artificial_candidates)

        for node in self.graph.nodes:
            if node in artificials:
                self.graph.nodes[node]["artificial_candidate"] = 1
            else:
                self.graph.nodes[node]["artificial_candidate"] = 0

        ###
        for node in self.graph.nodes:
            if (
                self.graph.nodes[node]["artificial_candidate"] == 1
                or self.graph.nodes[node]["real_candidate"] == 1
            ):
                self.graph.nodes[node]["candidate"] = 1
            else:
                self.graph.nodes[node]["candidate"] = 0

        graph_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/graphhh.pkl"
        save_pickle(self.graph, path=graph_path)

    def save_OD_files(self):
        """
        Saves origins and destinations CSV files for travel time calculations.
        Origins: candidate locations only.
        Destinations: all census blocks.
        """

        self.assign_artificials_to_graph()
        gdf = self.blocks_gdf
        print("Artificial candidates are assigned to the graph as node attributes.")
        # ensure centroids exist
        gdf["centroid"] = gdf.geometry.centroid
        gdf["lon"] = gdf["centroid"].x
        gdf["lat"] = gdf["centroid"].y

        # candidate list from graph
        candidates = [
            node
            for node in self.graph.nodes
            if self.graph.nodes[node].get("candidate", 0) == 1
        ]

        # --- Origins: candidates only ---
        origins = gdf.loc[candidates, ["lon", "lat"]].reset_index()
        origins.rename(columns={"index": "id"}, inplace=True)

        # --- Destinations: all blocks ---
        destinations = gdf[["lon", "lat"]].reset_index()
        destinations.rename(columns={"index": "id"}, inplace=True)

        # Save as CSV
        origins.to_csv(
            "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/network-data/origins.csv",
            index=False,
        )
        destinations.to_csv(
            "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/network-data/destinations.csv",
            index=False,
        )

        print("Saved origins.csv and destinations.csv")

    def total_needed_candidates(self, workload: int = 1500, factor: int = 3) -> int:
        return factor * (-(self.pop() // -workload))


def choose_artificial_candidates(cell, cell_id, possible_blocks):
    """
    Calculates extra number of candidates needed in each cell.
    Randomly selects artificial candidates from non-candidate blocks
    of graph within each cell. Returns a dict mapping cell_name to a list of block ids.
    """
    print(f"Started choosing artificial candidates ramdomly for cell {cell_id}...")
    print(
        f"Cell {cell_id} needs {cell.missing} artificial candidates and {len(possible_blocks)} possible blocks left."
    )

    if cell.missing > len(possible_blocks):
        print(
            f"{cell_id} has {cell.missing - len(possible_blocks)} less blocks than needed candidate locations. Choosing all possible nodes as candidates."
        )
        artificials = list(possible_blocks)

    else:
        if cell.missing > 0:
            chosen = np.random.choice(
                cell.possible_blocks, size=cell.missing, replace=False
            )
            artificials = list(chosen)
        else:
            artificials = []
        print(
            f"Number of chosen artificial candidates for cell {cell_id}: {len(artificials)}"
        )
    return artificials


def assign_artificial_candidates(cells: Cells):
    """"""
    selected_artificials = set()
    i = 1
    num_iterations = len(cells.cells)
    for cell_id in cells:
        cell = cells.cells[cell_id]
        print(f"----------------Processing cell {cell_id}...{i}/{num_iterations}")
        print(
            f"Length of possible blocks before removing selected artificials: {len(cell.possible_blocks)}"
        )
        possible_blocks = set(cell.possible_blocks) - selected_artificials

        artificials = choose_artificial_candidates(cell, cell_id, possible_blocks)

        cells.cells[cell_id] = cells.update_cell(cell, artificials)
        selected_artificials = selected_artificials.union(set(artificials))
        print(f"Total selected artificials so far: {len(selected_artificials)}")
        print("----------------- End of cell processing.\n")
        i += 1
    return cells


def calculate_travel_times(graphhh):
    ### special nodes
    graphhh.nodes[32588]["candidate"] = 0
    graphhh.nodes[32588]["artificial_candidate"] = 0

    import runpy

    def run_mod():
        runpy.run_module("my_module", run_name="__main__")

    return


def from_real_datasets(
    graph: nx.Graph,
    df: gpd.GeoDataFrame,
    candidate_df: gpd.GeoDataFrame,
    coordinate_attr: str,
    geo_artificial_candidates: Optional[bool] = True,
    workload=1500,
    save_graph_to: Optional[str] = None,
    save_data_to: Optional[str] = None,
):
    """
    Args:
        df (gpd.GeoDataFrame): A Geopandas dataframe of census units.
        graph (Graph): A NetworkX graph.
        candidate.df (gpd.GeoDataFrame): A Geopandas dataframe of candidate locations
        coordinate_attr (str): Node attribute name for coordinates.
        save (str):
    Returns:
        None
    """

    real_candidates = Candidates.real_candidates(df, candidate_df, coordinate_attr)
    for node in graph.nodes:
        if node in real_candidates:
            graph.nodes[node]["real_candidate"] = 1
        else:
            graph.nodes[node]["real_candidate"] = 0

    if geo_artificial_candidates == True:
        artificial_candidates = artificial_candidates(df, graph, workload)
        for node in graph.nodes:
            if node in artificial_candidates:
                graph.nodes[node]["artificial_candidate"] = 1
            else:
                graph.nodes[node]["artificial_candidate"] = 0

        for node, data in graph.nodes:
            if data["real_candidate"] == 1 or data["artificial_candidate"] == 1:
                graph.nodes[node]["candidate"] = 1
            else:
                graph.nodes[node]["candidate"] = 0

        if save_graph_to != None:
            save_pickle(graph, save_graph_to)
        if save_data_to != None:
            save_pickle(df, save_data_to)

        if save_graph_to is None and save_data_to is None:
            return graph


def random_weighted(
    graph: nx.Graph,
    num_candidates: int,
    weight_name: str,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    default_weight: float = 0.0,
) -> Set:
    """
    Randomly selects candidate nodes from the graph using node degree or a node attribute as weights.

    Args:
        graph (Graph): A NetworkX graph.
        num_candidates (int): Number of candidates to choose.
        weight_name (str): Node attribute name to use as weight, or 'degree'.
        random_state (int or Generator, optional): Seed or RNG for reproducibility.
        default_weight (float): Default weight to use if attribute is missing on a node.

    Returns:
        Set: A set of candidate node identifiers.
    """
    nodes = list(graph.nodes)

    if weight_name == "degree":
        weights = [graph.degree[node] for node in nodes]
    elif any(weight_name in attrs for _, attrs in graph.nodes(data=True)):
        weights = [graph.nodes[node].get(weight_name, default_weight) for node in nodes]
    else:
        raise AttributeError(
            f"Weight name '{weight_name}' is neither 'degree' nor a node attribute."
        )

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError(
            f"Total weight is zero. Check if attribute '{weight_name}' exists and is nonzero."
        )

    if num_candidates > len(nodes):
        raise ValueError("Number of candidates requested exceeds number of nodes.")

    # Setup reproducible RNG
    rng = np.random.default_rng(random_state)

    return set(
        rng.choice(
            nodes,
            size=num_candidates,
            replace=False,
            p=np.array(weights) / total_weight,
        )
    )


def random_uniformly(nodes: list, num_candidates: int):
    return set(random.sample(population=nodes, k=num_candidates))
