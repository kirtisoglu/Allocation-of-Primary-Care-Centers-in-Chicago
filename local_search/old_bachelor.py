import networkx as nx
<<<<<<< Updated upstream
from gerrychain import Graph, Partition
import random
import copy
import collections

import random
=======
import random
import copy
import collections
import time

import random
import math
>>>>>>> Stashed changes
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Hashable,
    Sequence,
    Tuple,
)

from local_search import tree


def kruskal_tree(graph: nx.Graph) -> nx.Graph:
    """
    Builds a spanning tree chosen by Kruskal's method using random weights.

    :param graph: The input graph to build the spanning tree from. Should be a Networkx Graph.
    :type graph: nx.Graph
    :param weight_dict: Dictionary of weights to add to the random weights used in region-aware
        variants.
    :type weight_dict: Optional[Dict], optional

    :returns: The maximal spanning tree represented as a Networkx Graph.
    :rtype: nx.Graph
    """
    #if weight_dict is None:
    #    weight_dict = dict()

    # for not having the same weights for any two edges
    #for edge in graph.edges():
    #    weight = random.random()
    #    for key, value in weight_dict.items():
    #        if (
    #            graph.nodes[edge[0]][key] == graph.nodes[edge[1]][key]
    #            and graph.nodes[edge[0]][key] is not None
    #        ):
    #            weight += value

    #    graph.edges[edge]["random_weight"] = weight

    spanning_tree = nx.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="time"
    )
    return spanning_tree



def uniform_spanning_tree(graph: nx.Graph, choice: Callable = random.choice) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph. Uses Wilson's algorithm.

    :param graph: Networkx Graph
    :type graph: nx.Graph
    :param choice: :func:`random.choice`. Defaults to :func:`random.choice`.
    :type choice: Callable, optional

    :returns: A spanning tree of the graph chosen uniformly at random.
    :rtype: nx.Graph
    """

    new_graph = graph.copy(as_view=False)

    # remove the edges between stops before sampling.
    """    for edge in new_graph.edges:
        endpoint1, endpoint2 = edge
        if new_graph.nodes[endpoint1]["id"] == 1 and new_graph.nodes[endpoint2]["id"] == 1:
            new_graph.remove_edge(endpoint1, endpoint2)"""

    root = choice(list(new_graph.nodes))
    tree_nodes = set([root])
    next_node = {root: None}

    for node in new_graph.nodes:
        u = node
        while u not in tree_nodes:
            next_node[u] = choice(list(new_graph[u].keys()))
            u = next_node[u]
            
        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    # re-assign the attributes of the nodes.
    for node in G.nodes:
        G.nodes[node].update(graph.nodes[node])

    return G


def uniform_spanning_tree_simplified(graph: nx.Graph, choice: random.choice = random.choice) -> nx.Graph:
    
    root = choice(list(graph.nodes))
    tree_nodes = set([root])
    next_node = {root: None}
    
    for node in graph.nodes:
        if node not in tree_nodes:
            current = node
            path = [current]
            while current not in tree_nodes:
                current = choice(list(graph.neighbors(current)))
                if current in path:
                    path = path[:path.index(current) + 1]
                else:
                    path.append(current)
            for i in range(len(path) - 1):
                next_node[path[i]] = path[i + 1]
                tree_nodes.add(path[i])
    
    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes(data=True))
    for node, next_n in next_node.items():
        if next_n is not None:
            tree.add_edge(node, next_n, **graph.get_edge_data(node, next_n))
    
    return tree


def generate_initial_partition_seed(graph, sources, travel, p):  # cuts random p-1 edges.

    clusterim = {}  #  key: cluster name,  value: list of vertices
    populations = {}
    total_travel = {}

    open_facilities = random.sample(sources, k=p)
    #open_facilities = list(existing.keys()) + openings

    spanning_tree = kruskal_tree(graph)
    # spanning_tree = sample_tree(G)


    tree = spanning_tree.copy(as_view=False)
    setim = list(nx.connected_components(tree)) # list of sets of nodes

    while len(tree.nodes) > 0:  # neden nodelarin sayisi sifirdan buyukse
        
        #print("Num of components in Tree: ", len(S))

        for component in setim: # component is a set of nodes 
            #print("component nodes:", component)
            facilities_in_component = [value for value in open_facilities if value in list(component)]

            if len(facilities_in_component) == 1:
                #print("num of facilities_in_component", len(facilities_in_component))
                #print("facilities_in_component", facilities_in_component)
                clusterim[facilities_in_component[0]] = list(component)
                #print("component", C[facilities_in_component[0]])
                tree.remove_nodes_from(component)
                #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            else:  
                #print("facilities in compoenent (else)", facilities_in_component) 
                #print("num of facilities_in_component (else)", len(facilities_in_component))
                u = random.choice(facilities_in_component)
                #print("u:", u)
                paths = {v: nx.shortest_path(tree, source=u, target=v) for v in facilities_in_component if v != u} # dictionary. key: target node v, value: list of nodes 
                #print("paths", paths)
                lengths = {}
                k = 500
                closest = (0,0)
                for v in paths.keys():   
                   if len(paths[v]) < k:
                        k = len(paths[v])
                        closest = v
                #closest = min((len(paths[v]), v) for v in paths.keys())[1]  # ?
                
            
                #print("closest:", closest)
                path = tree.subgraph(paths[closest])
                #print("path nodes", path.nodes)
                edge = random.choice(list(path.edges))
                #print("removed edge:", edge)
                tree.remove_edge(*edge)  # one edge is removed from the tree. 
            #print("####################################################") 

        setim = list(nx.connected_components(tree)) # list of sets of nodes

    for center in clusterim.keys():
        populations[center] = 0
        total_travel[center] = 0

        for node in clusterim[center]:
            populations[center] += graph.nodes[node].get('pop')
            total_travel[center] += travel[(node, center)]
            

    # Create the partitioned graph
    component = []
    for center in clusterim.keys():
        component.append(nx.induced_subgraph(spanning_tree, clusterim[center]))
    
    partitioned_tree = nx.union(component[0], component[1])
    for index, graph in enumerate(component):
        if index > 1:
            partitioned_tree = nx.union(partitioned_tree, graph)

    return partitioned_tree, spanning_tree, clusterim, populations, total_travel, open_facilities




def generate_initial_partition_seed_simplified(graph, sources, p):
    
    open_facilities = random.sample(sources, k=p)
    spanning_tree = uniform_spanning_tree_simplified(graph)
    clusterim = {}
    populations = {}
    
    for facility in open_facilities:
        reachable = next(comp for comp in nx.connected_components(spanning_tree) if facility in comp)
        for node in reachable:
            clusterim.setdefault(facility, []).append(node)
            populations[facility] = populations.get(facility, 0) + graph.nodes[node].get('pop', 0)
        spanning_tree.remove_nodes_from(reachable)
    
    partitioned_tree = nx.Graph()
    for cluster_nodes in clusterim.values():
        subgraph = graph.subgraph(cluster_nodes)
        partitioned_tree = nx.compose(partitioned_tree, subgraph)
    
    return partitioned_tree, spanning_tree, clusterim, populations, open_facilities




def exact_neighborhood(graph, clusters):  # exact neighborhood is defined for Descent Search and the first Tabu Search implementation.  

    #  migrating node: origin district --> destination district
    #print("--------- ENTERING EXACT NEIGHBORHOOD ---------")
    boundaries = {}  # key: (origin's center, a neighboring cluster's center),  value: nodes in the boundary of origin, and adjacent to the neighboring cluster. Origin is fixed.
    neighborhood = {}  # key: migrating node, value: a dictionary of clusters (clusters -> key: center, value: list of nodes in the cluster of center) 
    
    centers = [center for center in clusters.keys() if len(clusters[center]) > 1] # list of district centers such that discrits have more than 1 node. (Origin district must have at least 2 nodes.)
    #centers = [center for center, nodes in clusters.items() if len(nodes) > 1]
    origin = random.choice(centers)    # choose origin district of migrating node uniformly random.
    #print("origin:", origin)
    #print("clusters[origin] right after choosing origin randomly:", clusters[origin])


    for cluster in clusters.keys():  # each cluster is a candidate of destination. 
        if cluster != origin:
            boundaries[(origin, cluster)] = list(nx.node_boundary(graph, clusters[cluster], clusters[origin])) # return a list of nodes in clusters[cluster] that are adjacent to some nodes in clusters[origin]

            for migrating_node in boundaries[(origin, cluster)]:
                if migrating_node != origin:
                    #print("migrating_node=", migrating_node)
                    new_clusters = copy.deepcopy(clusters)
                    new_clusters[origin].remove(migrating_node)
                    new_clusters[cluster].append(migrating_node)
                    subgraph = nx.induced_subgraph(graph, new_clusters[origin])
                    #print("number of components in origin= ", len(list(nx.connected_components(subgraph))))
                    if len(list(nx.connected_components(subgraph))) == 1:       # remove migrating node from its cluster and check if the cluster is connected
                        #print("subgraph has only 1 component")
                        #print("clusters[origin]:", clusters[origin])
                        #print("migrating node:", migrating_node)
                        #print("new_clusters[origin]:", new_clusters[origin])
                        #print("number of components in origin - after = ", len(list(nx.connected_components(subgraph))))

                        neighborhood[migrating_node] = new_clusters
                    else:
                        continue
    #                    print("nodes=", subgraph.nodes)
    #                    if all(nx.has_path(subgraph, u, v)==True for u, v in subgraph.nodes):
    #                        new_clusters[cluster].append(migrating_node)
    #                        neighborhood[migrating_node] = new_clusters
        else: 
            continue

    #print("--------- LEAVING EXACT NEIGHBORHOOD ---------")

    return origin, boundaries, neighborhood




def largest_population_district(partition):
    """
    Identify the cluster with the largest total population in a partition.

    Parameters:
    - partition: A Gerrychain Partition object.

    Returns:
    - The identifier of the cluster with the largest population.
    """
    
    cluster_populations = {}

    # Sum populations for each cluster
    for node, cluster_id in partition.assignment.items():
        node_population = partition.graph.nodes[node]["pop"]
        if cluster_id in cluster_populations:
            cluster_populations[cluster_id] += node_population
        else:
            cluster_populations[cluster_id] = node_population

    # Identify the cluster with the largest population
    largest_population_cluster = max(cluster_populations, key=cluster_populations.get)
    
    return largest_population_cluster





<<<<<<< Updated upstream
def random_neighbor(graph, clusters, travel, d_1, d_2, current_populations, current_total_travel):
    
    # Create an assignment dict mapping block IDs to district IDs for gerrychain.graph.partition
    assignment = {node: center for center, nodes in clusters.items() for node in nodes}
    # Using partition function in gerrychain library, convert clusters to partition that is suitable to the library
    partition = Partition(graph, assignment=assignment)
    
    endpoints = {}
    for (node_a, node_b) in partition.cut_edges:
        center_a, center_b = assignment[node_a], assignment[node_b]
        if center_a != node_a and center_b != node_b:
            endpoints[(node_a, node_b, center_b)] = ((1/travel[(node_a, center_b)])**d_1)*((1/travel[(node_a, center_a)])**d_2)
            endpoints[(node_b, node_a, center_a)] = (1/travel[(node_b, center_a)]**d_1)*(1/travel[(node_b, center_b)]**d_2)
    
    endpoint_keys, endpoint_values = list(endpoints.keys()), list(endpoints.values())
            
    #print("Entering Edge Validation")
    endpoint_valid = False
    while not endpoint_valid:
        random_endpoint = random.choices(endpoint_keys, weights=endpoint_values, k=1)[0]
        node_origin, node_destination, center_destination = random_endpoint
        center_origin = assignment[node_origin]

        #if node_origin not in clusters: # avoids moving centers. This is solved above with center_a != node_a ...
        clusters[center_origin].remove(node_origin)
        clusters[center_destination].append(node_origin)

        if nx.is_connected(nx.subgraph(graph, clusters[center_origin])):
            current_populations[center_origin] = current_populations[center_origin] - graph.nodes[node_origin].get('pop')
            current_populations[center_destination] + graph.nodes[node_origin].get('pop')
            current_total_travel[center_origin] = current_total_travel[center_origin] - travel[(node_origin, center_origin)]
            current_total_travel[center_destination] = current_total_travel[center_origin] + travel[(node_origin, center_destination)]
=======
def random_neighbor(graph, clusters, travel, d_1, d_2, current_populations, current_total_travel, tabu_set, omega, beta):
    
    # Create an assignment dict mapping block IDs to district IDs for gerrychain.graph.partition
    assignment = {node: center for center, nodes in clusters.items() for node in nodes}
    
    #cut_edges = []
    #for u, v in graph.edges():
    #    if assignment[u] != assignment[v]:
    #        cut_edges.append((u, v))
    
    center_boundary = {}
    moves = {}
    for center in clusters:
        center_boundary[center] = nx.node_boundary(graph, clusters[center])
        for node in center_boundary[center]:
            if node != assignment[node]:
                moves[(node, assignment[node], center)] = (( omega * travel[node, assignment[node]] ) ** d_2 ) / (( beta * travel[(node, center)]) ** d_1 )
    
    endpoint_valid = False
    while not endpoint_valid:
        migration = random.choices(list(moves.keys()), list(moves.values()), k=1)[0]
        migrating_node = migration[0]
        origin = migration[1] 
        destination = migration[2]
        clusters[origin].remove(migrating_node)
        clusters[destination].append(migrating_node)
        
        #if node_origin not in clusters: # avoids moving centers. This is solved above with center_a != node_a ...
        if nx.is_connected(nx.subgraph(graph, clusters[origin])):
            current_populations[origin] -=  graph.nodes[migrating_node].get('pop')
            current_populations[destination] += graph.nodes[migrating_node].get('pop')
            current_total_travel[origin] -=  travel[(migrating_node, origin)]
            current_total_travel[destination] += travel[(migrating_node, destination)]
>>>>>>> Stashed changes
            endpoint_valid = True
    
        else:
            # Restore previous state if flip is invalid
<<<<<<< Updated upstream
            clusters[center_destination].remove(node_origin)
            clusters[center_origin].append(node_origin)        
    return clusters, node_origin, center_origin, center_destination, current_populations, current_total_travel
=======
            clusters[destination].remove(migrating_node)
            clusters[origin].append(migrating_node) 
                   
    return clusters, migrating_node, origin, destination, current_populations, current_total_travel
>>>>>>> Stashed changes






def cut_edges_for_cluster(partition, cluster_id):
    """
    Get cut edges for a specific cluster in a Gerrychain partition.

    Parameters:
    - partition: A Gerrychain Partition object.
    - cluster_id: The identifier of the cluster for which to find cut edges.

    Returns:
    - A set of tuples representing the cut edges for the specified cluster.
    """
    cut_edges_for_cluster = set()
    for edge in partition.cut_edges:
        u, v = edge
        if (partition.assignment[u] == cluster_id and partition.assignment[v] != cluster_id) or \
           (partition.assignment[v] == cluster_id and partition.assignment[u] != cluster_id):
            cut_edges_for_cluster.add(edge)
            
    return cut_edges_for_cluster

def random_neighbor_largest(graph, clusters):
    
    assignment = {node: center for center, nodes in clusters.items() for node in nodes}
    partition = Partition(graph, assignment=assignment)
    largest_cluster = largest_population_district(partition)
    cut_edges = cut_edges_for_cluster(partition, largest_cluster)

    edge_valid = False
    while not edge_valid:
        random_edge = random.choice(cut_edges)
        node_one, node_two = random_edge
        
        # Determine which node is in the largest cluster
        if partition.assignment[node_one] == largest_cluster:
            node_origin, node_destination = node_one, node_two
        else:
            node_origin, node_destination = node_two, node_one


        # Ensure moves involve the largest cluster
        if partition.assignment[node_origin] == largest_cluster or partition.assignment[node_destination] == largest_cluster:
            center_origin = partition.assignment[node_origin]
            center_destination = partition.assignment[node_destination]
            
            if node_origin not in clusters:  # Checks if the origin node is not a center
                clusters[center_origin].remove(node_origin)
                clusters[center_destination].append(node_origin)
                
                # Check connectivity of the resulting cluster
                if nx.is_connected(nx.subgraph(graph, clusters[center_origin])):
                    edge_valid = True
                else:
                    clusters[center_destination].remove(node_origin)  # Revert the change
                    clusters[center_origin].append(node_origin)
            # Additional conditions or handling can be added here
        # It might be useful to consider what to do if neither node is part of the largest cluster

    return clusters





def objective_function(clusters, travel, graph):

    #print("--------- ENTERING OBJECTIVE FUNCTION ---------")

    populations = {}
    travel_radius = {}
    graph_radius = {}
    num_districts = len(clusters)

    for cluster in clusters.keys():
        #print("picked a cluster with the center", cluster)
        #print("nodes in the cluster:", clusters[cluster])
        populations[cluster] = sum(graph.nodes[node].get('pop') for node in clusters[cluster])
        #print("total popuation of the cluster is", populations[cluster])
        #travel_times = [travel[node, cluster] for node in clusters[cluster]]
        #print("travel times from the center are", travel_times)
        travel_radius[cluster] = max(travel[node, cluster] for node in clusters[cluster])
        #print("travel radius of the cluster is", travel_radius[cluster])
        #print("end of the loop for the cluster")
        
        ##subgraph = nx.induced_subgraph(graph, clusters[cluster])
        ##graph_radius[cluster] = nx.radius(subgraph)

    pop_average = sum(populations.values()) / num_districts
    radius_average = sum(travel_radius.values()) / num_districts
    ##graph_radius_average = sum(graph_radius.values()) / num_districts
    ##+ abs(graph_radius[cluster] - graph_radius_average)**2 / (num_districts * graph_radius_average)

    f_1 = sum((abs(populations[cluster] - pop_average) / (num_districts * pop_average)) for cluster in clusters.keys())
    f_2 = sum((abs(travel_radius[cluster] - radius_average)**2 / (num_districts * radius_average)) for cluster in clusters.keys())

    #print("--------- LEAVING OBJECTIVE FUNCTION ---------")

    return f_1, f_2

def objective_function_simplified(clusters, travel, graph):
    
    populations = {}
    #travel_radius = {}
    total_travels = {}
    total_population = 0
    total_travel = 0
    num_districts = len(clusters)

    # Calculate the total population and total travel radius
    for center, cluster_nodes in clusters.items():
        populations[center] = sum(graph.nodes[node].get('pop') for node in cluster_nodes)
        total_population += populations[center] 
        #travel_radius[center] = max(travel[node, center] for node in cluster_nodes)
        total_travels[center] = sum(travel[(node, center)] for node in cluster_nodes)
        total_travel += total_travels[center]

    # Calculate averages
    pop_average = total_population / num_districts
    travel_average = total_travel / num_districts

    # Calculate objective function components
<<<<<<< Updated upstream
    f_1 = sum((abs(populations[center] - pop_average) for center in populations.keys())) / (num_districts * pop_average)
=======
    f_1 = sum((abs(populations[center] - pop_average) ** 2 for center in populations.keys())) / (num_districts * pop_average)
>>>>>>> Stashed changes
    #f_2 = sum((abs(travel_radius[center] - radius_average) ** 2 for center in travel_radius.keys())) / (num_districts * radius_average)
    f_3 = sum((abs(total_travels[center] - travel_average) ** 2 for center in total_travels.keys())) / (num_districts * travel_average)
    return f_1, f_3





<<<<<<< Updated upstream
def multi_old_bachelor_seed(graph, sources, travel, num_iterations, granularity, a, b, c, alpha, num_inner_iterations, p, d_1, d_2, initial_data = None):
=======
def multi_old_bachelor_seed(graph, sources, travel, num_iterations, granularity, a, b, c, alpha, num_inner_iterations, p, d_1, d_2, omega, beta, initial_data = None):
>>>>>>> Stashed changes

    iteration_results = {}
    iteration = 0
    last = 0

    # 3. num_iterations = M in the explonation above. Note that i /leq M-1, and so (1 - i/M) > 0.
    while iteration < num_iterations:

        #print("Outer_iteration = ", iteration)
            
        if initial_data is None:
            partitioned_tree, spanning_tree, initial_solution, initial_populations, initial_total_travel, initial_seeds = generate_initial_partition_seed(graph, sources, travel, p)
        else:
            # Unpack provided initial data
            partitioned_tree, spanning_tree, initial_solution, initial_populations, initial_total_travel, initial_seeds = initial_data

        
        # 1. Generate the same initial solution as in Descent. Change the objective ?
        #partitioned_tree, spanning_tree, initial_solution, initial_populations, initial_seeds = generate_initial_partition_seed(graph, sources, p)
        initial_energy_pop, initial_energy_access = objective_function_simplified(initial_solution, travel, graph) 
        current_solution = initial_solution
        current_energy_pop = initial_energy_pop
        current_energy_access = initial_energy_access
        #current_seeds = initial_seeds
        current_populations = initial_populations
        current_total_travel = initial_total_travel

        # 2. Define an initial threshold T_0.
        threshold = 0
        age = 0
        total_moves = 0
<<<<<<< Updated upstream

        inner_iteration = 0
=======
        function_time = 0
        inner_iteration = 0
        tabu_set = set()
>>>>>>> Stashed changes


        while inner_iteration < num_inner_iterations:

            #if inner_iteration % 250 == 0:
            #    print("inner_iteration = ", inner_iteration)
            print(f"------ SELECTING NEW MIGRATING NODE ------- Inner Iteration={inner_iteration}")
<<<<<<< Updated upstream
            neighbor, migrating_node, center_origin, center_destination, neighbor_populations, neighbor_total_travel = random_neighbor(graph, current_solution, travel, d_1, d_2, current_populations, current_total_travel)
            neighbor_energy_pop, neighbor_energy_access = objective_function_simplified(neighbor, travel, graph)
            print(f'Travel time to Center of Origin={travel[(migrating_node, center_origin)]}', f'Travel Time to Center of Destionation={travel[(migrating_node, center_destination)]}')
            print(f'Current Population Energy={current_energy_pop}', f'Current Access Energy={current_energy_access}')
            print(f'Migrating Node={migrating_node}', f'Neighbor Population Energy={neighbor_energy_pop}', f'Neighbor Access Energy={neighbor_energy_access}')
            print(f'Worsening Access Energy={(1 + alpha) * current_energy_access}')
            print(f'Threshold={threshold}')

            # 3.1. if energy change < T_i, perform the move.
            if neighbor_energy_access <= (1 + alpha) * current_energy_access and neighbor_energy_pop - current_energy_pop <= threshold:   # worsening?
                print("Both of the conditions are satisfied.")
=======
            start_time = time.time()
            neighbor, migrating_node, center_origin, center_destination, neighbor_populations, neighbor_total_travel = random_neighbor(graph, current_solution, travel, d_1, d_2, current_populations, current_total_travel, tabu_set, omega, beta)
            function_time += time.time() - start_time 
            neighbor_energy_pop, neighbor_energy_access = objective_function_simplified(neighbor, travel, graph)
            #print(f'Travel time to Center of Origin={travel[(migrating_node, center_origin)]}', f'Travel Time to Center of Destionation={travel[(migrating_node, center_destination)]}')
            #print(f'Current Population Energy={current_energy_pop}', f'Current Access Energy={current_energy_access}')
            #print(f'Migrating Node={migrating_node}', f'Neighbor Population Energy={neighbor_energy_pop}', f'Neighbor Access Energy={neighbor_energy_access}')
            #print(f'Worsening Access Energy={(1 + alpha) * current_energy_access}')
            #print(f'Threshold={threshold}')

            # 3.1. if energy change < T_i, perform the move.
            if neighbor_energy_access < (1 + alpha) * current_energy_access and neighbor_energy_pop - current_energy_pop <= threshold:   # worsening?
                #print("Both of the conditions are satisfied.")
>>>>>>> Stashed changes
                current_solution = neighbor
                current_energy_pop = neighbor_energy_pop
                current_energy_access = neighbor_energy_access
                current_populations = neighbor_populations
                current_total_travel = neighbor_total_travel
                last = inner_iteration
                total_moves += 1

                # 3.1.1. if energy change < 0  decrease the threshold: T_{i+1}:= T_i − Δ^{-}(i). --> Only close bad moves will be accepted, since we just found a good solution.
                if neighbor_energy_pop - current_energy_pop < 0:
<<<<<<< Updated upstream
                    print("Population equality is improved.")
                    age = 0
                    threshold = ( ( age / a ) ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c 

            # 3.2. Otherwise, Δ >= T_i, increase the threshold: T_{i+1}:= T_i + Δ^{+}(i) --> We should accept worst solutions increasing T_i, since we may be trapped at a local min.
            else: 
                print("None of the conditions are satisfied.")
                age += 1
                threshold = ( (age / a) ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c 
=======
                    #print("Population equality is improved.")
                    age = 0
                    threshold = 0
            # 3.2. Otherwise, Δ >= T_i, increase the threshold: T_{i+1}:= T_i + Δ^{+}(i) --> We should accept worst solutions increasing T_i, since we may be trapped at a local min.
            else: 
                #print("None of the conditions are satisfied.")
                age += 1
                threshold = ( (age / a) ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c 
                #tabu_set.add(migrating_node)
>>>>>>> Stashed changes

            inner_iteration += 1
        
        # Save the result of the curent iteration. 
<<<<<<< Updated upstream
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access, initial_energy_pop, initial_energy_access)
=======
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access, initial_energy_pop, initial_energy_access, function_time)
>>>>>>> Stashed changes

        # increase the iteration number 
        iteration += 1


    # Initialize the best iteration as the first iteration
    current_iteration_initial = iteration_results[0][0]
    current_iteration_solution = iteration_results[0][1]
    current_iteration_energy_pop = iteration_results[0][2]
    current_iteration_energy_access = iteration_results[0][3]
    current_iteration_initial_pop = iteration_results[0][4]
    current_iteration_initial_access = iteration_results[0][5]

    # Compare the results of iterations
    for iteration in range(num_iterations - 1):

        if iteration_results[iteration + 1][3] <= (1 + alpha) * current_iteration_energy_access and iteration_results[iteration + 1][2] < current_iteration_energy_pop:
            current_iteration_initial = iteration_results[iteration + 1][0]
            current_iteration_solution = iteration_results[iteration + 1][1]
            current_iteration_energy_pop = iteration_results[iteration + 1][2]
            current_iteration_energy_access = iteration_results[iteration + 1][3]
            current_iteration_initial_pop = iteration_results[iteration + 1][4]
            current_iteration_initial_access = iteration_results[iteration + 1][5]

<<<<<<< Updated upstream
    # 5. The final solution is the best local optimum found s *
    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access, current_iteration_initial_pop, current_iteration_initial_access, last, current_populations, current_total_travel, total_moves
=======
    print(function_time)
    # 5. The final solution is the best local optimum found s *
    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access, current_iteration_initial_pop, current_iteration_initial_access, last, current_populations, current_total_travel, total_moves, tabu_set

>>>>>>> Stashed changes

def multi_old_bachelor_seed_simplified(graph, sources, travel, num_iterations, granularity, a, b, c, alpha, num_inner_iterations, p):
    
    iteration_results = {}
    
    for iteration in range(num_iterations):
        print(f"Outer_iteration = {iteration}")

        partitioned_tree, spanning_tree, current_solution, initial_populations, initial_seeds = generate_initial_partition_seed_simplified(graph, sources, p)
        current_energy_pop, current_energy_access = objective_function_simplified(current_solution, travel, graph)
        
        threshold = 0
        age = 0
        
        for inner_iteration in range(num_inner_iterations):
            if inner_iteration % 250 == 0:
                print(f"inner_iteration = {inner_iteration}")

            neighbor = random_neighbor(graph, current_solution)
            neighbor_energy_pop, neighbor_energy_access = objective_function_simplified(neighbor, travel, graph)
            
            if neighbor_energy_access <= (1 + alpha) * current_energy_access and neighbor_energy_pop - current_energy_pop < threshold:
                current_solution = neighbor
                current_energy_pop = neighbor_energy_pop
                current_energy_access = neighbor_energy_access
                
                if neighbor_energy_pop - current_energy_pop < 0:
                    age = 0
                else:
                    age += 1
            else:
                age += 1

            threshold = ((age / a) ** b - 1) * granularity * (1 - iteration / num_iterations) ** c

        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access)

    # Choose the best iteration based on energy access and population
    best_iteration = min(iteration_results.items(), key=lambda x: (x[1][3], -x[1][2]))

    return iteration_results, *best_iteration[1]
<<<<<<< Updated upstream
=======



"""
def random_neighbor(graph, clusters, travel, d_1, d_2, current_populations, current_total_travel):
    # Map block IDs to district IDs and initialize clusters as sets
    assignment = {node: center for center, nodes in clusters.items() for node in nodes}
    clusters = {k: set(v) for k, v in clusters.items()}

    # Precompute necessary travel costs and boundaries
    necessary_travel = {}
    boundary_nodes = {}
    for center, nodes in clusters.items():
        boundary = graph.node_boundary(graph, nodes)
        boundary_nodes[center] = boundary
        for node in boundary:
            if (node, center) in travel:
                necessary_travel[(node, center)] = (1 / travel[(node, center)]) ** d_1
            if (node, assignment[node]) in travel:
                necessary_travel[(node, assignment[node])] = (1 / travel[(node, assignment[node])]) ** d_2

    moves = {}
    for center, boundary in boundary_nodes.items():
        for node in boundary:
            origin_center = assignment[node]
            if (node, center) in necessary_travel and (node, origin_center) in necessary_travel:
                moves[(node, origin_center, center)] = necessary_travel[(node, center)] * necessary_travel[(node, origin_center)]
    
    endpoint_valid = False
    while not endpoint_valid:
        migration = random.choices(list(moves.keys()), weights=list(moves.values()), k=1)[0]
        migrating_node, origin, destination = migration
        
        clusters[origin].remove(migrating_node)
        clusters[destination].add(migrating_node)
        
        # Only check connectivity if the origin is potentially non-connected
        if nx.is_connected(graph.subgraph(clusters[origin])):
            update_populations_and_travel(migrating_node, origin, destination, graph, travel, current_populations, current_total_travel)
            endpoint_valid = True
        else:
            clusters[destination].remove(migrating_node)
            clusters[origin].add(migrating_node)

    return {k: list(v) for k, v in clusters.items()}, migrating_node, origin, destination, current_populations, current_total_travel

def update_populations_and_travel(node, origin, destination, graph, travel, populations, total_travel):
    node_population = graph.nodes[node].get('pop', 0)
    populations[origin] -= node_population
    populations[destination] += node_population
    total_travel[origin] -= travel.get((node, origin), 0)
    total_travel[destination] += travel.get((node, destination), 0)
"""

>>>>>>> Stashed changes
