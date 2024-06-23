

import networkx as nx
import pandas as pd
import math
import plotly.express as px
from shapely.geometry import Point
from typing import Any, Dict, Optional, Tuple



class District:

    
    __slots__ = (
    "graph",
    "districts",
    "flips",
    #"cut_edges",
    "boundaries",
    "populations",
    "max_travel_times",
    "assignment",
    "parent",
)


    def __init__(self, graph: nx.Graph, candidates: set, d, districts: Dict[str, set], travel_times: Dict[Tuple[int, str], float], 
                 flips: Optional[Dict[int, str]] = None, parent: Optional["District"] = None):
        
        self.graph = graph
        self.closest = self.closest(candidates, travel_times)
        
        for center in candidates:
            self.pop.center = self.population(center)
        
        self.teams = self.doctor_nurse_teams(center, d)
        
        
        
        
        self.flips = flips
        self.assignment = self.generate_assignment(districts)
        #self.cut_edges = self.determine_cut_edges()
        self.populations = self.calculate_total_populations(districts)
        self.total_travel_times = self.calculate_total_travel_time(districts, travel_times)
        
        
    def closest(self, candidates, travel_times):
        districts = {}    
        for node in self.graph.nodes:
            closest_center = min(travel_times[(node, center)] for center in candidates)
            districts[closest_center].add(node)
        return districts

    
    def population(self, center):
        return sum(self.graph.nodes[node]['pop'] for node in self.districts[center])
    
    def doctor_nurse_teams(self, center, d):
        num = math.ceil(self.pop.center / d)
        
        
    def objective_value():
        return
    
    def constraint():
        return
        
        


