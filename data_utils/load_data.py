""" 
    Centralizing caching and reloading logic in a .py file is an excellent practice. 
    It not only promotes code reuse across notebooks but also helps in maintaining and updating the logic in one place.
    
    Preprocessing your data in a separate notebook before analysis or model testing keeps your workflow organized and modular. 
    It also allows you to focus on data exploration and transformation separately from model experimentation.
    
    If the data_processing function in the first notebook is substantial or complex, ensure it's well-documented,
    and consider how it can be efficiently rerun or skipped based on the caching logic in your .py file.
    
    Using nbimporter to import the data_processing function from the first notebook into your main analysis notebook 
    fits well with your iterative development workflow. This method supports your goal of not needing to rerun the first 
    notebook when restarting the second one.

    Returns:
        _type_: _description_
"""

 
import pandas as pd
import numpy as np
import geopandas as gpd
import pickle
import json
import gzip

from pympler import asizeof
from shapely.geometry import mapping, base
from pandas import DataFrame
from typing import Any


# DATA STORAGE NOTES

# JSON: Best for compatibility and when your data is already in JSON format. Compression can help mitigate file size concerns.

# Pandas with Parquet: Ideal for tabular data and when you need fast reads/writes and efficient storage. 
# It's particularly effective for datasets with a mix of numerical data and strings.

# Pickle with Compression: Suitable for complex Python objects that don't fit neatly into a table or when you need to preserve 
# the exact Python data types and structures.

# If results are key-value pairs or lists, JSON is straightforward. If your data is more naturally represented
# in rows and columns (like calculation results across many items or time points), a tabular format might be more efficient.

 # If you need to perform further calculations or analyses on the results, storing them in a format that can be quickly and 
 # efficiently processed by your analysis tools (e.g., Parquet for Pandas) is beneficial.
 
 # For large datasets, consider formats optimized for performance and storage, such as Parquet or HDF5 for tabular data, 
 # or compressed Pickle for more complex Python objects.

# For complex geospatial objects, such as detailed polygons or multi-geometries typical in GeoJSON data, 
# storing them in JSON may be more straightforward. For use cases where human readability and interoperability 
# with web APIs or other tools that directly consume GeoJSON are priorities, JSON shines.

 # If geodata is more tabular, such as a set of points with associated attributes (latitude, longitude, and 
 # additional columns for data like temperature, elevation, etc.), converting this data into a Parquet file 
 # might be more efficient. This approach is especially beneficial for large datasets and when performing analytical queries.
 
 # Hybrid Approach: For scenarios where you have both complex geospatial objects and the need for efficient querying and storage, 
 # a hybrid approach might be suitable. You could store the complex geometries in JSON format to preserve their structure and 
 # the tabular attribute data in Parquet for efficient querying. Link them via a common identifier.
 
 


"""
Transforming your dictionaries into a tabular format, Pandas combined with Parquet 
can offer efficient storage and fast access, especially for numerical data and strings.
"""


"""
For better performance and storage efficiency, consider compressing the JSON file.
Uses gzip compression to reduce the storage space required for the JSON file of large datasets.
"""


"""
Pickle can serialize a wide range of Python objects, including complex nested dictionaries. 
Combining Pickle with compression can optimize storage for large data.
"""


 
def cache_data(data: Any, file_path: str, method: str = None) -> None:
    """
    Cache data to a file using the specified method.
    
    Parameters:
    - data (Any): The data to cache.
    - file_path (str): The path to the file where data should be cached.
    - method (str, optional): The method to use for caching ('parquet', 'json_gzip', 'pickle', or 'pickle_gzip').
      Defaults to None.
    """
    
    if method == 'parquet':
        # Ensure pandas is installed and the data type is appropriate for a DataFrame
        try:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame for parquet method.")
            data.to_parquet(file_path)
        except ImportError:
            raise ImportError("Pandas is required for the 'parquet' method.")
    
    elif method == 'json_gzip':
        # JSON and gzip are part of the standard library
        with gzip.open(file_path, 'wt', encoding='utf-8') as zipf:
            json.dump(data, zipf)
    
    elif method == 'geojson_gzip':
        import geopandas as gpd
        data_json = data.to_json() # Convert the GeoDataFrame to a GeoJSON string
        with gzip.open(file_path, 'wt', encoding='utf-8') as gz_file:
            gz_file.write(data_json)
    
    elif method == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    elif method == 'pickle_gzip':
        with gzip.open(file_path, 'wb') as zipf:
            pickle.dump(data, zipf)
    
    else:
        raise ValueError("Unsupported method specified. Choose 'parquet', 'json_gzip', 'pickle', or 'pickle_gzip'.")
   




def load_data(file_path: str, method: str = None) -> any:
    """
    Load data from a file using the specified method or infer from file extension.
    
    Parameters:
    - file_path (str): The path to the file from which data should be loaded.
    - method (str, optional): The method to use for loading ('parquet', 'pickle_gzip', 'json_gzip', or 'pickle').
      If None, the method is inferred from the file extension.
      
    Returns:
    Loaded data.
    """

    # Infer the loading method from the file extension if not specified
    if method is None:
        if file_path.endswith('.parquet'):
            method = 'parquet'
        elif file_path.endswith('.pkl.gz'):
            method = 'pickle_gzip'
        elif file_path.endswith('.json.gz'):
            method = 'json_gzip'
        elif file_path.endswith('.pkl'):
            method = 'pickle'
        elif file_path.endswith('.geojson.gz'):
            method = 'geojson_gzip'
        else:
            raise ValueError("File extension not recognized and no method specified.")

    # Load data based on the method
    if method == 'parquet':
        try:
            import pandas as pd
            return pd.read_parquet(file_path)
        except ImportError:
            raise ImportError("Pandas is required to load parquet files.")
    
    elif method == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    elif method == 'pickle_gzip':
        with gzip.open(file_path, 'rb') as zipf:
            return pickle.load(zipf)
    
    elif method == 'json_gzip':
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as zipf:
                return json.load(zipf)
        except json.JSONDecodeError as e:
            raise Exception("Failed to decode JSON: " + str(e))
        except Exception as e:
            raise Exception("Failed to load or parse the file: " + str(e))
    
    elif method == 'geojson_gzip':
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
            return gpd.read_file(gz_file)
    
    else:
        raise ValueError("Unsupported method specified.")


 # MEMORY USAGE NOTES
 
# Pandas provides a .memory_usage() method that can be called on a DataFrame to see the memory usage of each column.
# You can use deep=True to get a more accurate measure that accounts for the actual object usage.

# NumPy arrays have a .nbytes attribute that tells you the total bytes consumed by the array:

# For more general Python objects, you can use the getsizeof function from the sys module. 
#This approach gives you the memory usage of the object itself but may not accurately account 
#for the memory used by objects referenced by the object unless you manually account for them.


# For a more detailed analysis of memory usage, especially for complex objects or to track memory usage over time, 
# the pympler library can be very useful. pympler provides a way to get detailed information about the memory footprint 
# of Python objects.



def memory_usage(data: Any, data_type: str = None) -> None:

    """
    Print the memory usage of the given data.

    Parameters:
    data (Any): The data whose memory usage is to be determined.
    data_type (str, optional): The type of the data. Defaults to None, in which case the type is inferred.

    Returns:
    None
    """
    
    # If data_type is not provided, attempt to infer the type
    if data_type is None:
        if pd is not None and isinstance(data, pd.DataFrame):
            data_type = 'DataFrame'
        elif np is not None and isinstance(data, np.ndarray):
            data_type = 'ndarray'
        else:
            data_type = 'object'
    
    if data_type == 'DataFrame':
        if pd is not None and isinstance(data, pd.DataFrame):
            memory_usage = data.memory_usage(deep=True).sum()
            print(f"Total memory usage by the DataFrame: {memory_usage} bytes")
        else:
            raise ValueError("Data is not a pandas DataFrame.")
        
    elif data_type == 'ndarray':
        if np is not None and isinstance(data, np.ndarray):
            print(f"Total memory usage by the NumPy array: {data.nbytes} bytes")
        else:
            raise ValueError("Data is not a numpy ndarray.")
        
    elif data_type == 'object':
        if asizeof is not None:
            print(f"Total memory usage by the object: {asizeof.asizeof(data)} bytes")
        else:
            raise ImportError("Pympler asizeof is not available to measure the object.")
        
    else:
        raise ValueError("Unsupported data type specified.")
    
    

    

def get_column(data = Any, column = str):
    
    """
    Summary?

    Parameters:
    data (Any): The data whose memory usage is to be determined.
    column (str): 
    
    Returns:
    
    """
    
    data['GEOID20'] = data['GEOID20'].astype(str)
    tracts = pd.read_csv('data/chicago_tracts.csv')
    # Convert tracts to string
    tracts = tracts.astype(str)
    # perform filtering operation
    chicago_dhc = data[data['GEOID20'].str[:11].isin(tracts['digits'])]
    #chicago_dhc
    
    # Create a dictionary from the DataFrame for faster lookup
    df_dict = chicago_dhc.set_index('GEOID20')[column].to_dict()
    
    return df_dict




def add_to_graph(graph, dictionary):
    
    """
    Summary?

    Parameters:
    data (Any): The data whose memory usage is to be determined.
    file_path (str):
    column (str): 
    
    Returns:
    
    """
    
    for node in graph.nodes:
        geoid = graph.nodes[node].get('GEOID20')
        
        if geoid in dictionary:
            graph.nodes[node]['pop'] = dictionary[geoid]    

        else:
            print(f'{geoid} not in graph')




def add_to_data(chicago, geoid_dict):
    for index, row in chicago.iterrows():
        geoid = row['GEOID20']
        pop = geoid_dict.get(geoid, 0)  # Use .get to safely handle missing keys, defaulting to 0
        chicago.at[index, 'pop'] = pop  # Correctly update the DataFrame





def to_geojson(df: DataFrame):
    if isinstance(df, gpd.GeoDataFrame) and 'geometry' in df.columns:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "properties": row.drop('geometry').to_dict(),
                 "geometry": mapping(row['geometry'])
                } for idx, row in df.iterrows()
            ]
        }
        return geojson
    else:
        raise ValueError("Input is not a valid GeoDataFrame with a 'geometry' column.")
  
    
    
def projection(data):
    
    """
    Summary?

    Parameters:
    data (Any): The data whose memory usage is to be determined.
    file_path (str):
    column (str): 
    
    Returns:
    """
    
    # reproject the geometries to a suitable projected CRS
    data_projected = data.to_crs(epsg=32616)

    # calculate centroids on the projected geometries and save them as a column in chicago data
    data_projected['centroid'] = data_projected.geometry.centroid

    # centroids in the original geographic CRS, you can project them back
    data['centroid'] = data_projected['centroid'].to_crs(data.crs)

    # calculate the mean latitude and longitude of the centroids
    center_lat = data['centroid'].y.mean()
    center_lon = data['centroid'].x.mean()
    
    return center_lat, center_lon
    



def random_phc(data):  
      
    # Randomly select 1000 census blocks, weighted by population
    selected_blocks = data.sample(n=1000, weights='pop', random_state=42)

    # Step 2: Extract the 'centroid' geometries of these selected blocks to use as PHC locations
    phc_locations = selected_blocks['centroid']

    # Create a GeoDataFrame for PHC locations
    phc_gdf = gpd.GeoDataFrame(geometry=phc_locations, crs=data.crs)

    # Prepare latitude and longitude for plotting
    phc_df = pd.DataFrame({
        'lat': phc_gdf.geometry.y,
        'lon': phc_gdf.geometry.x
    })

    return phc_df, selected_blocks



def add_phc(data, graph, selected_blocks):
    
    sources = []

    for node in graph.nodes:
        if node in selected_blocks.index:
            graph.nodes[node]['phc'] = 1
            sources.append(node)
        else:
            graph.nodes[node]['phc'] = 0

    data['phc'] = 0

    for index in data.index:
        if index in selected_blocks.index:
            data.loc[index, 'phc'] = 1
            
    return sources

        


