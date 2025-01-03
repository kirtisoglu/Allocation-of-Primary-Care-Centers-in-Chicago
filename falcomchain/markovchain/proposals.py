import random
from partition import Partition
from tree import (capacitated_recursive_tree, ReselectException)


class MetagraphError(Exception):
    """
    Raised when the partition we are trying to split is a low degree
    node in the metagraph.
    """

    pass


class ValueWarning(UserWarning):
    """
    Raised whe a particular value is technically valid, but may
    cause issues with the algorithm.
    """

    pass



def recom( # Note: recomb is called for each state of the chain. Parameters must be static for the states.
    partition: Partition,
    capacity_level: int,
    pop_target: int,
    column_names: tuple[str],
    epsilon: float,
    density: float = None,
) -> Partition:
    """
    ReCom (short for ReCombination) is a Markov Chain Monte Carlo (MCMC) algorithm
    used for redistricting. At each step of the algorithm, a pair of adjacent districts
    is selected at random and merged into a single district. The region is then split
    into two new districts by generating a spanning tree using the Kruskal/Karger
    algorithm and cutting an edge at random. The edge is checked to ensure that it
    separates the region into two new districts that are population balanced, and,
    if not, a new edge is selected at random and the process is repeated.

    :param partition: The initial partition.
    :type partition: Partition
    :param pop_col: The name of the population column.
    :type pop_col: str
    :param pop_target: The target population for each district.
    :type pop_target: Union[int,float]
    :param epsilon: The epsilon value for population deviation as a percentage of the
        target population.
    :type epsilon: float
    :param node_repeats: The number of times to repeat the bipartitioning step. Default is 1.
    :type node_repeats: int, optional

    :returns: The new partition resulting from the ReCom algorithm.
    :rtype: Partition
    """
    bad_district_pairs = set()
    n_parts = len(partition)
    tot_pairs = n_parts * (n_parts - 1) / 2  # n choose 2

    while len(bad_district_pairs) < tot_pairs:
        try:
            while True:
                edge = random.choice(tuple(partition["cut_edges"]))
                # Need to sort the tuple so that the order is consistent in the bad_district_pairs set
                part_one, part_two = partition.assignment.mapping[edge[0]], partition.assignment.mapping[edge[1]]
                parts_to_merge = [part_one, part_two]
                parts_to_merge.sort()

                if tuple(parts_to_merge) not in bad_district_pairs:
                    break

            n_teams = partition.teams[part_one] + partition.teams[part_two]
            subgraph = partition.graph.subgraph(partition.parts[part_one] | partition.parts[part_two])
            filtered_parts = {part: partition.parts[part] for part in parts_to_merge}

            flips, new_teams = capacitated_recursive_tree(
                graph = subgraph.graph,
                filtered_parts = filtered_parts,
                n_parts = n_parts,
                column_names = column_names,
                n_teams=n_teams,
                pop_target=pop_target,
                epsilon=epsilon,
                capacity_level=capacity_level,
                density = density)
            break
   
        except Exception as e:
            if isinstance(e, ReselectException):
                bad_district_pairs.add(tuple(parts_to_merge))
                continue
            else:
                raise

    if len(bad_district_pairs) == tot_pairs:
        raise MetagraphError(
            f"Bipartitioning failed for all {tot_pairs} district pairs."
            f"Consider rerunning the chain with a different random seed."
        )
                
    return partition.flip(flips, new_teams)



def propose_chunk_flip(partition: Partition) -> Partition:
    """
    Chooses a random boundary node and proposes to flip it and all of its neighbors

    :param partition: The current partition to propose a flip from.
    :type partition: Partition

    :returns: A possible next `~gerrychain.Partition`
    :rtype: Partition
    """
    flips = dict()

    edge = random.choice(tuple(partition["cut_edges"]))
    index = random.choice((0, 1))

    flipped_node = edge[index]

    valid_flips = [
        nbr
        for nbr in partition.graph.neighbors(flipped_node)
        if partition.assignment.mapping[nbr]
        != partition.assignment.mapping[flipped_node]
    ]

    for flipped_neighbor in valid_flips:
        flips.update({flipped_neighbor: partition.assignment.mapping[flipped_node]})

    return partition.flip(flips)



def propose_random_flip(partition: Partition) -> Partition:
    """
    Proposes a random boundary flip from the partition.

    :param partition: The current partition to propose a flip from.
    :type partition: Partition

    :returns: A possible next `~gerrychain.Partition`
    :rtype: Partition
    """
    if len(partition["cut_edges"]) == 0:
        return partition
    edge = random.choice(tuple(partition["cut_edges"]))
    index = random.choice((0, 1))
    flipped_node, other_node = edge[index], edge[1 - index]
    flip = {flipped_node: partition.assignment.mapping[other_node]}
    return partition.flip(flip)




class Supergraph:
    """
    Generates a supergraph of parts in a partition to pick a random set of parts that does not 
    exceed the maximum number of teams in total. In each state, we update merged and re-splitted 
    parts in the metagraph locally.All possible sets of possible neighboring parts are produced 
    and one of them is uniformly selected.
    """
    
    
    def __init__(self, partition: Partition, max):
        
        self.nodes = set(partition.parts.keys())
        self.neighborhood = {node: {} for node in self.nodes}
        
        for edge in partition.cut_edges:
            self.neighborhood[partition.assignment.mapping[edge[0]]].add(partition.assignment.mapping[edge[1]])
            self.neighborhood[partition.assignment.mapping[edge[1]]].add(partition.assignment.mapping[edge[0]])

        self.node_subsets = self.parts_to_merge()
        
        sum = sum(partition.radius[part] for part in partition.radius.keys())
        
        def __repr__(self):
            pass
        
        def parts_to_merge(self):
            
            return
        
        
        def node_ant(self):
            
            return
        
        
        def update_supergraph():
            return



def weigthted_selection_by_travel_time(partition: Partition):
   
    population = list(partition.parts.keys())
    weight_list = [partition.radius[part] for part in population]

    first_part = random.choices(population, weights=weight_list, k=1)
        
    return
    
    
    
def uniform_selection_from_neighborhoods(partition: Partition):

    population = list(partition.parts.keys())

    first_part = random.choice(population)
    cuts = {edge for edge in partition.cut_edges if first_part in edge}
    
    return        
    
    