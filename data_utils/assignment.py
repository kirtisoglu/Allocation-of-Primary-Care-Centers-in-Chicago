


def assign(graph, data, clusters, attribute: str, question: str):  
    
    " Assign Cluster IDs to Each Node "
    cluster_id = 0
    for center, nodes in clusters.items():
        # Assign the cluster ID to the center and its nodes
        graph.nodes[center][attribute] = cluster_id   # attribute='initial_cluster'
        for node in nodes:
            graph.nodes[node][attribute] = cluster_id
        cluster_id += 1
        
    for center in clusters.keys():
        graph.nodes[center][question] = True  # question = 'is_initial_center'
        
    " Prepare Data for Plotting "
    data[attribute] = [graph.nodes[node][attribute] for node in data.index]
    data[question] = [graph.nodes[node].get(question, False) for node in data.index]
    
<<<<<<< Updated upstream
    return
=======
>>>>>>> Stashed changes




def results(graph, initial_solution, final_solution, travel_time, initial_pop, initial_access, energy_pop, energy_access, 
            populations_initial, populations_final, total_travel_initial, total_travel_final, last, total_moves):

    print("--- TOTAL TRAVEL TIMES ---")
    print("Initial:",  total_travel_initial)
    print("final:", total_travel_final)
    
    print("       ")
    
    print("--- TOTAL POPULATIONS ---")
    print("Initial:", populations_initial)
    print("Final:", populations_final)

    print("       ")
    
    l = len(populations_final.keys())
    pop_avg = 0
    travel_avg = 0
    pop_avg = sum(populations_final[center] for center in populations_final.keys()) / l
    travel_avg = sum(total_travel_final[center] for center in total_travel_final.keys()) / l

    print("       ") 
    
    print("Population Average:", pop_avg)
    print("Total Travel Time Average:", travel_avg)

    print("       ")    
 
    print("Initial pop energy: ", initial_pop)
    print("Initial access energy: ", initial_access)
    print("Final pop energy of the best iteration: ", energy_pop)
    print("Final access energy of the best iteration: ", energy_access)
    
    print("       ")
    
    print(f"Number of Succesful Moves={total_moves}")
    print(f"Last Succesful Move={last}")