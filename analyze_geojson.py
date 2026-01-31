#!/usr/bin/env python3
"""
Analyze the GeoJSON file to understand the districtr property values.
"""
import json
from collections import Counter

def analyze_geojson(filepath):
    """Load and analyze the GeoJSON file."""
    print(f"Loading GeoJSON file: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\nGeoJSON Type: {data.get('type', 'Unknown')}")
    
    # Analyze features
    features = data.get('features', [])
    print(f"Total Features: {len(features)}")
    
    # Extract districtr values
    districtr_values = []
    for feature in features:
        props = feature.get('properties', {})
        districtr = props.get('districtr')
        if districtr is not None:
            districtr_values.append(districtr)
    
    # Count unique values
    districtr_counter = Counter(districtr_values)
    
    print(f"\n{'='*60}")
    print("DISTRICTR VALUE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total features with 'districtr' property: {len(districtr_values)}")
    print(f"Unique districtr values: {sorted(districtr_counter.keys())}")
    print(f"\nDistribution:")
    for value in sorted(districtr_counter.keys()):
        count = districtr_counter[value]
        percentage = (count / len(districtr_values)) * 100
        print(f"  districtr = {value}: {count:5d} features ({percentage:5.2f}%)")
    
    # Sample some properties from first feature
    if features:
        print(f"\n{'='*60}")
        print("SAMPLE FEATURE PROPERTIES (first feature)")
        print(f"{'='*60}")
        sample_props = features[0].get('properties', {})
        for key, value in sorted(sample_props.items()):
            print(f"  {key}: {value}")
    
    # Check geometry types
    geometry_types = Counter()
    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type', 'Unknown')
        geometry_types[geom_type] += 1
    
    print(f"\n{'='*60}")
    print("GEOMETRY TYPES")
    print(f"{'='*60}")
    for geom_type, count in geometry_types.items():
        print(f"  {geom_type}: {count}")

if __name__ == "__main__":
    analyze_geojson("export-7166.geojson")
