import datetime
import gzip
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


def _to_py(v):
    """Convert numpy/pandas/scalars -> builtins for JSON."""
    try:
        import numpy as np

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, (set,)):
        return list(v)
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, (datetime.datetime,)):
        return v.isoformat()
    return v


def _json_dump(path, obj):
    def default(o):
        return _py(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, default=default, separators=(",", ":"), ensure_ascii=False)


def export_tree(tree, iteration, initial):
    G = tree.graph
    nodes = []
    for n, a in G.nodes(data=True):
        # Add/compute the minimal fields we want in the browser
        if tree.supertree == True:
            teams = G.nodes[n]["n_teams"] # accumulated
            attribute_one = "has_teams"
            attribute_two = "compl_teams"
            value_one = 2 <= teams <= tree.capacity_level
            value_two = (2 <= tree.n_teams - teams <= tree.capacity_level or n == tree.root)
        else: 
            attribute_one = "has_candidate"
            attribute_two = "compl_facility"
            value_one = bool(tree.has_facility(n))
            value_two = bool(tree.complement_has_facility(n))
            
        # coordinates
        x = a.get("C_X", a.get("x", 0.0))
        y = a.get("C_Y", a.get("y", 0.0))

        # Copy other small attributes, converting to JSON-safe types
        extra = {
            k: _to_py(v) for k, v in a.items() if k not in ("C_X", "C_Y", "x", "y")
        }

        nodes.append(
            {
                "id": str(n),  # make IDs strings for consistency
                "x": float(x),
                "y": float(y),
                attribute_one: value_one,
                attribute_two: value_two,
                **extra,
            }
        )

    links = [{"source": str(u), "target": str(v)} for u, v in G.edges()]
    # Metadata as a dict (cast scalars)
    metadata = {
        "ideal_pop": _to_py(getattr(tree, "ideal_pop", None)),
        "root": str(getattr(tree, "root", "")),
        "n_teams": _to_py(getattr(tree, "n_teams", None)),
        "epsilon": _to_py(getattr(tree, "epsilon", None)),
        "two_sided": _to_py(getattr(tree, "two_sided", None)),
        "tot_candidates": _to_py(getattr(tree, "tot_candidates", None)),
        "tot_pop": _to_py(getattr(tree, "tot_pop", None)),
        "supergraph": _to_py(getattr(tree, "supertree", None)),
    }

    data = {"nodes": nodes, "links": links, "metadata": metadata}

    if initial:
        tree_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/JS-app/data/trees"
    else:
        tree_path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/js-app/data/int_trees"

    base = Path(tree_path)
    base.mkdir(parents=True, exist_ok=True)

    if iteration == 1:
        for p in base.glob("*.json"): 
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    
    if tree.supertree == True:
        extension = f"supertree_{iteration}.json"
    else: 
        extension = f"tree_{iteration}.json"
    
    path = base / extension
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

        

def export_district_frame(
    root,
    iteration,
    district_nodes,
    hired_teams,
    pop,
    district_id,
    debt,
    merged_ids,
    initial,
    superdistrict
):
    """
    Write one JSON file per (iteration, district).
    - tree: Tree instance (must have attributes used below)
    - district_nodes: iterable of node ids (strings/ints) belonging to this district
    """
    md = {
        "iteration": int(iteration),
        "district_id": str(district_id),
        # "ideal_pop": _py(getattr(tree, "ideal_pop", None)),
        # "epsilon": _py(getattr(tree, "epsilon", None)),
        # "n_teams": _py(getattr(tree, "n_teams", None)),
        "root": _to_py(root),
        "tot_pop": int(pop),
        "hired_teams": int(hired_teams),
        "debt": float(debt),
        "merged_ids": _to_py(merged_ids),
        "superdistrict": _to_py(superdistrict)
    }
    data = {
        "district": [str(n) for n in district_nodes],  # list of node ids
        "metadata": md,
    }
    # Naming: frames/iter_<iiii>/district_<id>.json
    # path = Path(out_dir) / f"iter_{int(iteration):04d}" / f"district_{iteration}.json"
    if initial:
        path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/JS-app/data/districts"
    else:
        path = "/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/js-app/data/int_districts"

    if superdistrict == True:
        extension = f"superdistrict_{iteration}.json"
    else: 
        extension = f"district_{iteration}.json"
        
    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)

    if iteration == 1:
        for p in base.glob("*.json"):  # or "*.json"
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    path = base / extension
    _json_dump(path, data)

    """
    upload district i 
    get its data
    load tree i
    load its data
    compare
    extract expected result and save
    finish the loop and print the result
    """
