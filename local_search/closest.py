

import District
import random



def select_candidates(all_locations, k):
    
    centers = random.choices(all_locations, k=k)
    
    return centers

