import random
from partition import Partition
from tree import (capacitated_recursive_tree, ReselectException)
from partition import cut_edges, cut_edges_by_part, put_edges_into_parts, bipartition_supertree
from collections import deque

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



def recom( # Note: recomb is called for each state of the chain. Parameters must be static for the states. (should we cache some of them in Partition?)
    partition: Partition,
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
    tot_pairs = n_parts * (n_parts - 1) / 2  # n choose 2  (isn't it too big? no adjacency between any two districts. it should be # of super cut edges)
    ids = set(partition.parts.keys())
    
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

            flips, new_teams = capacitated_recursive_tree(
                graph = subgraph.graph,
                column_names = column_names,
                n_teams=n_teams,
                pop_target=pop_target,
                epsilon=epsilon,
                capacity_level=partition.capacity_level,
                density = density,
                assignments = partition.assignment, 
                merged_parts = parts_to_merge,
                ids=ids)
            break

        except Exception as e:
            if isinstance(e, ReselectException):  # if there is no balanced cut after max_attempt in bipartition_tree, then the pair is a bad district pair.
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



def hierarchical_recomb(partition: Partition,
    pop_target: int,
    column_names: tuple[str],
    epsilon: float,
    density: float = None,
) -> Partition:
    """_summary_

    Args:
        partition (Partition): _description_
        pop_target (int): _description_
        column_names (tuple[str]): _description_
        epsilon (float): _description_
        density (float, optional): _description_. Defaults to None.
        supergraph (str, optional): Local or global. Defaults to None.

    Returns:
        Partition: _description_
    """

    bad_selection = set()
    n_parts = len(partition)
    tot_pairs = n_parts * (n_parts - 1) / 2 
    ids = set(partition.parts.keys())

    while len(bad_selection) < tot_pairs:  # we need to change this to something related to supergraph
        try:
            while True:
                parts_to_merge = bipartition_supertree(
                    supergraph=partition.supergraph)
                parts_to_merge.sort() # Need to sort the tuple so that the order is consistent in the bad_selection set
                
                if tuple(parts_to_merge) not in  bad_selection:
                    break

            n_teams = sum(partition.teams[part] for part in parts_to_merge)
            subgraph = partition.graph.subgraph(set.union(*(set(partition.parts[part]) for part in parts_to_merge)))

            flips, new_teams, remaining_ids, new_ids = capacitated_recursive_tree(
                graph = subgraph.graph,
                column_names = column_names,
                n_teams=n_teams,
                pop_target=pop_target,
                epsilon=epsilon,
                capacity_level=partition.capacity_level,
                density = density,
                assignments = partition.assignment, # ??
                merged_ids=ids, # ??
                max_id = max_id # ??
            )
            break

        except Exception as e:
            if isinstance(e, ReselectException):  # if there is no balanced cut after max_attempt in bipartition_tree, then the pair is a bad district pair.
                bad_selection.add(tuple(parts_to_merge))
                continue
            else:
                raise

    if len(bad_selection) == tot_pairs:
        raise MetagraphError(
            f"Bipartitioning failed for {tot_pairs} of district collections."
            f"Consider rerunning the chain with a different random seed."
        )
                
    return partition.flip(flips, new_teams, merged_ids, new_ids)



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

