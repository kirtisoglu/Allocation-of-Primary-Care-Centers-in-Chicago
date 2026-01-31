#!/usr/bin/env python3
"""
Compare the export-7166.geojson (Missouri data) with chicago_blocks.json to understand the data structure.
"""
import json
import sys

def analyze_file(filepath, file_label):
    """Analyze a GeoJSON or JSON file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {file_label}")
    print(f"File: {filepath}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'r') as f:
            # Try to load first 1000 characters to check structure
            f.seek(0)
            preview = f.read(1000)
            f.seek(0)
            data = json.load(f)
        
        # Check if it's a GeoJSON FeatureCollection
        if isinstance(data, dict):
            if 'type' in data:
                print(f"Type: {data['type']}")
            
            if 'features' in data:
                features = data['features']
                print(f"Total Features: {len(features)}")
                
                if features:
                    # Analyze first feature
                    first_feature = features[0]
                    print(f"\nFirst Feature Structure:")
                    print(f"  Type: {first_feature.get('type', 'N/A')}")
                    
                    # Properties
                    props = first_feature.get('properties', {})
                    print(f"  Properties count: {len(props)}")
                    print(f"  Sample property keys (first 10): {list(props.keys())[:10]}")
                    
                    # Check for districtr
                    if 'districtr' in props:
                        print(f"  ✓ Has 'districtr' property: {props['districtr']}")
                    else:
                        print(f"  ✗ No 'districtr' property")
                    
                    # Check for state/location indicators
                    location_keys = ['STATEFP20', 'COUNTYFP20', 'NAME20', 'NAMELSAD20', 'state', 'county', 'city']
                    print(f"\n  Location Information:")
                    for key in location_keys:
                        if key in props:
                            print(f"    {key}: {props[key]}")
                    
                    # Geometry
                    geom = first_feature.get('geometry', {})
                    print(f"\n  Geometry:")
                    print(f"    Type: {geom.get('type', 'N/A')}")
                    if 'coordinates' in geom:
                        coords = geom['coordinates']
                        print(f"    Has coordinates: Yes")
                        # Try to get a sample coordinate
                        try:
                            sample_coord = coords[0][0][0] if isinstance(coords[0][0], list) else coords[0]
                            print(f"    Sample coordinate: {sample_coord}")
                        except:
                            print(f"    Coordinate structure: Complex/nested")
            
            # For non-FeatureCollection structures
            else:
                print(f"Structure: Not a FeatureCollection")
                print(f"Top-level keys: {list(data.keys())[:10]}")
        
        elif isinstance(data, list):
            print(f"Structure: List/Array")
            print(f"Total items: {len(data)}")
            if data:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())[:10]}")
    
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    # Analyze both files
    analyze_file("export-7166.geojson", "Missouri Voting Districts (GeoJSON)")
    analyze_file("data/primary/chicago_blocks.json", "Chicago Blocks Data")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("Based on the README.md, this project is about optimizing primary care")
    print("facility locations in Chicago using Census block-level data.")
    print()
    print("The 'export-7166.geojson' file contains Missouri voting districts and")
    print("appears to be sample/test data with 'districtr' coloring applied.")
    print()
    print("The actual Chicago data should be in 'data/primary/chicago_blocks.json'")
