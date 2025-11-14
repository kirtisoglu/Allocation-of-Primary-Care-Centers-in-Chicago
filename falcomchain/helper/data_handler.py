import gzip
import json
import os
import pickle
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

# from shapely.geometry import base, mapping


class DataHandler:  ### Edit initialization for logical consistency. If full_path is given?

    def __init__(self, base_path=None, base_path_2=None, base_path_3=None):

        if base_path == None:
            self.base_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed"
        else:
            self.base_path = base_path

        if base_path_2 == None:
            self.base_path_2 = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed"
        else:
            self.base_path_2 = base_path_2

        if base_path_3 == None:
            self.base_path_3 = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed"
        else:
            self.base_path_3 = base_path_3

        self.files = self.detect_existing_data()
        self._create_properties()

    def get_full_extension(self, filename):
        parts = filename.split(".")
        if len(parts) > 1:
            return "." + ".".join(parts[1:]), parts[0]
        return "", filename

    def detect_existing_data(self):
        """Detects files in the directory of base_path.
        Splits and saves names of files in a dictionary."""
        files = {}

        for path in [
            self.base_path,
            self.base_path_2,
            self.base_path_3,
        ]:  # iterates over files in the directories
            for filename in os.listdir(path):  # iterates over files in the directories
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path):  # checks if full_path is a file
                    extension, root = self.get_full_extension(filename)
                files[root] = (extension, filename, full_path)
        return files

    def _create_properties(self):
        """Dynamically creates properties for each file detected."""
        for name in self.files:
            setattr(self, f"load_{name}", self._create_loader(name))
            setattr(self, f"save_{name}", self._create_saver(name))

    def _create_loader(self, name):
        """Creates a loader function for a specific file."""

        def loader():
            return self.load(name)

        return loader

    def _create_saver(self, name):
        """Creates a saver function for a specific data."""

        def saver(data, new_path=None):
            extension, _ = self.files[name]
            file_path = new_path if new_path else self.files[name][2]
            self.save(data, file_path, extension)

        return saver

    def load(self, name, extension=None) -> None:
        """Load data based on the file name stored in `files` dictionary."""

        if extension == None:
            if name not in self.files:
                raise FileNotFoundError(
                    f"There is no file with name {name} in the directory."
                )
            extension, filename, file_path = self.files[name]

        if extension == ".csv":
            return pd.read_csv(file_path)
        elif extension in {".pkl", ".pickle"}:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        elif extension in {".pkl.gz", ".pickle_gzip"}:
            with gzip.open(file_path, "rb") as file:
                return pickle.load(file)
        elif extension == ".json" or extension == ".json.gz":
            open_func = gzip.open if "gz" in extension else open
            with open_func(file_path, "rt", encoding="utf-8") as file:
                return json.load(file)
        elif extension == ".parquet":
            return pd.read_parquet(file_path)
        elif extension == ".geojson" or extension == ".geojson_gzip":
            open_func = gzip.open if "gz" in extension else open
            with open_func(file_path, "rt", encoding="utf-8") as file:
                return gpd.read_file(file)
        else:
            raise ValueError(f"The file format {extension} is not supported.")

    def save(self, data, name, zip) -> None:
        """Save data to a file with a given name. If a file does not exists
        with that name in the directory, it first creates a file.
        param: zip (boolean): If True, data is zipped.
        """

        if isinstance(data, pd.DataFrame):
            extension = ".csv"
            file_path = os.path.join(self.base_path_2, f"{name}{extension}")
            data.to_csv(file_path, index=False)

        elif isinstance(data, dict):
            extension = ".json.gz" if zip == True else ".json"
            file_path = os.path.join(self.base_path_2, f"{name}{extension}")
            mode += "t" if zip == True else "b"
            opener = gzip.open if zip == True else open
            with opener(
                file_path, mode, encoding=None if "gzip" in name else "utf-8"
            ) as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        elif isinstance(data, gpd.GeoDataFrame):
            extension = ".geojson.gz" if zip == True else ".geojson"
            file_path = os.path.join(self.base_path_2, f"{name}{extension}")
            mode = "wt" if zip == True else "w"
            if zip == True:
                with gzip.open(file_path, mode, encoding="utf-8") as gz_file:
                    data.to_file(gz_file, driver="GeoJSON")
            else:
                data.to_file(file_path, driver="GeoJSON")

        elif isinstance(data, (bytes, bytearray)) or callable(
            getattr(data, "read", None)
        ):
            extension = ".pkl.gz" if zip == True else ".pkl"
            file_path = os.path.join(self.base_path_2, f"{name}{extension}")
            mode = "wb"
            opener = gzip.open if zip == True else open
            with opener(file_path, mode) as file:
                pickle.dump(data, file)

        if not file_path:
            raise ValueError("Unsupported data type for saving.")


def save_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def remove_from_saved_geodata(attr: str, path: str):
    df = load_pickle(path)
    df.drop(attr, axis=1, inplace=True)
    save_pickle(df, path)


def add_to_saved_geodata(attr: str, dictionary, path: str):
    df = load_pickle(path)
    df[attr] = dictionary
    save_pickle(df, path)


def add_to_graph(graph, attr, dict=None):

    if dict is None:
        for node in graph.nodes:
            nx.set_node_attributes(graph, 50, "population")
            nx.set_edge_attributes(graph, 1, "shared_perim")

    # Create a binary mask: 1 if in collection, else 0
    binary_mask = graph.nodes.to_series().apply(lambda node: int(node in dict))

    # Assign to graph
    for node, value in binary_mask.items():
        graph.nodes[node][attr] = value


def add_to_data(data, attr, collection):
    return


def get_column(data=Any, collection=set, column=str):
    """
    Parameters:
    data (Any):
    column (str):

    Returns:
    """

    data["GEOID20"] = data["GEOID20"].astype(str)
    tracts = pd.read_csv("data/chicago_tracts.csv")
    # Convert tracts to string
    tracts = tracts.astype(str)
    # perform filtering operation
    chicago_dhc = data[data["GEOID20"].str[:11].isin(tracts["digits"])]
    # chicago_dhc

    # Create a dictionary from the DataFrame for faster lookup
    df_dict = chicago_dhc.set_index("GEOID20")[column].to_dict()

    return df_dict


def export_graph_with_coords(G, file_path):
    data = {
        "nodes": [
            {
                "id": str(node),
                "x": G.nodes[node]["C_X"],
                "y": G.nodes[node]["C_Y"],
                **{k: v for k, v in G.nodes[node].items() if k not in ["C_X", "C_Y"]},
            }
            for node in G.nodes
        ],
        "links": [{"source": str(u), "target": str(v)} for u, v in G.edges],
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
