
import random
from typing import Callable, Union

import Partition
from tree import (epsilon_tree_bipartition, bipartition_tree, ReselectException)



class MetagraphError(Exception):
    """
    Raised when the partition we are trying to split is a low degree
    node in the metagraph.
    """
    pass



def recom(
    partition: Partition,
    pop_target: Union[int, float],
    epsilon: float,
    node_repeats: int = 1,
    method: Callable = bipartition_tree,
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
    :param pop_target: The target population for each district.
    :type pop_target: Union[int,float]
    :param epsilon: The epsilon value for population deviation as a percentage of the
        target population.
    :type epsilon: float
    :param node_repeats: The number of times to repeat the bipartitioning step. Default is 1.
    :type node_repeats: int, optional
    :param method: The method used for bipartitioning the tree.
    :type method: Callable, optional

    :returns: The new partition resulting from the ReCom algorithm.
    :rtype: Partition
    """

    bad_district_pairs = set()
    n = len(partition.districts)
    tot_pairs = n * (n - 1) // 2 # integer division of n choose 2


    while len(bad_district_pairs) < tot_pairs:
        try:
            while True:
                edge = random.choice(tuple(partition.cut_edges))
                # Need to sort the tuple so that the order is consistent
                districts_to_merge = tuple(sorted((partition.assignment[edge[0]], partition.assignment[edge[1]])))

                if districts_to_merge not in bad_district_pairs:
                    break
            # induced subgraph by the nodes of two districts
            subgraph = partition.graph.subgraph(partition.districts[districts_to_merge[0]] | partition.districts[districts_to_merge[1]])

            flips = epsilon_tree_bipartition(
                subgraph,
                districts_to_merge,
                partition.populations,
                pop_target=pop_target,
                epsilon=epsilon,
                node_repeats=node_repeats,
                method=method,
            )
            break

        except ReselectException:
            bad_district_pairs.add(districts_to_merge)
            continue

    if len(bad_district_pairs) == tot_pairs:
        raise MetagraphError(
            f"Bipartitioning failed for all {tot_pairs} district pairs."
            f"Consider rerunning the chain with a different random seed."
        )
        
    return partition.flip(flips)



