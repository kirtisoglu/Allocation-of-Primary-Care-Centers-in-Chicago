import class as cl
import numpy as np
import random 
import travel_time as travel


grid = {}  # dictionary for grid
V_blocks = {}  # set of vertices correspoding to blocks 
V_stops = {}  # set of vertices corresponding to stops
V_existing = {}  # set of vertices corresponding to existing PHCs
V_possible = {}  # set of vertices corresponding to possible PHCs
pop = [] # list of populations


def create_x_axis(name):
    i = name[0]
    noise_x = np.random.normal(0,0.1)   
    pos_x = i + noise_x
    return pos_x

def create_y_axis(name):
    j = name[1]
    noise_y = np.random.normal(0,0.1)
    pos_y = j + noise_y
    return pos_y

def create_population(p_min,p_max):
           
    result =  random.randrange(p_min,p_max)
    
    return result

def create_neighbors(name,n,m):
 
    i = name[0]
    j = name[1]
    neighbors = set()   

    # block-block/stop neighbors
    #if i == 0 or i == n - 1 or j == 0 or j == m - 1:   # corners
    if i != 0:
                neighbors.add((i-1, j))  # bottom neighbor
    if j != m - 1:
                neighbors.add((i, j+1))  # right neighbor
    if i != n - 1:
                neighbors.add((i+1 ,j))  # top neighbor
    if j != 0:
                neighbors.add((i, j-1))  # left neighbor

    
    # stop-stop neighbors
    if name in create_stops(n,m):
        if name != (1,0) and name != (n-1, m-2):   # name is neither origin nor destination.
            neighbors.add((i-1, j-1))   # top-right neighbor
            neighbors.add((i +1, j+1))  # bottom-left neighbor
        if name == (1,0):     # name is the origin.
            neighbors.add((2, 1))  # top-right neighbor
        if name == (n-1 , m-2):   # name is the destination.
            neighbors.add((n-2, m-3))  # bottom-left neighbor

    return neighbors


def create_stops(n, m):
    stops = set()
    for k in range(min(n-1, m-1)):
        i, j = 1 + k, k   # (k+1,k)
        stops.add((i, j))
    return stops

def create_EPHC(n,m,e):
     
    pairs = {(i, j) for i in range(n) for j in range(m)}  # all possible pairs (i, j)
    stops = create_stops(n,m)
    candidates = pairs - stops   # subway nodes cannot be a PHC.
    random_EPHCs = random.sample(list(candidates), e)  # e many random elements as existing PHCs
    existing = set(random_EPHCs)  

    return existing, stops

def create_PPHC(n,m,e,p):

    pairs = {(i, j) for i in range(n) for j in range(m)}  # all possible pairs (i, j)
    existing, stops = create_EPHC(n, m, e)
    forbidden = existing.union(stops)
    remaining = pairs - forbidden
    random_PPHCs = random.sample(list(remaining), p)  # Choose p random elements as possible PHCs
    possible = set(random_PPHCs)

    return existing, possible, stops

def create_distance(name,n,m, d_b, d_s): # distance in minutes between two neighbors. Dictionary.

    result = {}  # key: neighbor. value: distance from name to neighbor

    for neighbor in create_neighbors(name,n,m):
        if name and neighbor in create_stops(n,m): # if both nodes are stops
            dist_neighbor = {neighbor: d_s}  # the distance between consecutive stops is 3 minutes.
            result.update(dist_neighbor)
        else:  
            dist_neighbor = {neighbor: d_b}  # the distance of block to its any neighbor is 5 minutes.
            result.update(dist_neighbor)
        
    return result

def create_grid(n, m, e, p, p_min, p_max, d_b, d_s):

    possibles, existings, stops = create_PPHC(n,m,e,p)

    # Properties: name, ID, x_axis, y_axis, population, distance, neighbors, EPHC, PPHC

    for i in range(n):
        for j in range(m):
            
            # Property EPHC
            EPHC = 0  
            if (i,j) in existings:
                EPHC = 1
            else: EPHC = 0 

            # Property PPHC
            PPHC = 0   
            if (i,j) in possibles:
                PPHC = 1
            else: PPHC = 0 
            
            # Property name
            name = (i,j)

            # Property ID
            if (i,j) in stops:
                id = 1  
            else: 
                id = 0

            # Property x_axis
            x_axis = create_x_axis(name)  # integer

            # Property y_axis
            y_axis = create_y_axis(name)  # integer

            # Property population
            population = create_population(p_min,p_max)  # integer

            # Property neighbors
            neighbors = create_neighbors(name,n,m)  # set

            # Property distance
            distance = create_distance(name,n,m, d_b, d_s)  # dictionary

            # Assign properties of each class.
            node = cl.Node(name, id, x_axis, y_axis, population, 
                        distance, neighbors, EPHC, PPHC) 

            # All nodes
            grid[(i,j)] = node
            
            # V_stops and V_blocks
            if id == 1:
                V_stops[(i,j)] = node  

            # V_existing
            if EPHC == 1:
                V_existing[(i,j)] = node
            
            #V_possible
            if PPHC == 1:
                V_possible[(i,j)] = node       

            if id == 0 and EPHC == 0 and PPHC == 0:
                V_blocks[(i,j)] = node
            
            pop.append(population)
    
    all_facilities = {**V_existing, **V_possible}
    total_pop = np.sum(pop)

    return grid, V_stops, V_blocks, V_existing, V_possible, all_facilities, pop, total_pop