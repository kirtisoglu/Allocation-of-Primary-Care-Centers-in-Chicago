
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




def recom_candidate(
    partition: Partition,
    pop_col: str,
    pop_target: Union[int, float],
    epsilon: float,
    node_repeats: int = 1,
    region_surcharge: Optional[Dict] = None,
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

    Example usage:

    .. code-block:: python

        from functools import partial
        from gerrychain import MarkovChain
        from gerrychain.proposals import recom

        # ...define constraints, accept, partition, total_steps here...

        # Ideal population:
        pop_target = sum(partition["population"].values()) / len(partition)

        proposal = partial(
            recom, pop_col="POP10", pop_target=pop_target, epsilon=.05, node_repeats=10
        )

        chain = MarkovChain(proposal, constraints, accept, partition, total_steps)

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
    :param region_surcharge: The surcharge dictionary for the graph used for region-aware
        partitioning of the grid. Default is None.
    :type region_surcharge: Optional[Dict], optional
    :param method: The method used for bipartitioning the tree. Default is
        :func:`~gerrychain.tree.bipartition_tree`.
    :type method: Callable, optional

    :returns: The new partition resulting from the ReCom algorithm.
    :rtype: Partition
    """

    bad_district_pairs = set()
    n_parts = len(partition)
    tot_pairs = n_parts * (n_parts - 1) / 2  # n choose 2

    # Try to add the region aware in if the method accepts the surcharge dictionary
    if "region_surcharge" in signature(method).parameters:
        method = partial(method, region_surcharge=region_surcharge)

    while len(bad_district_pairs) < tot_pairs:
        try:
            while True:
                edge = random.choice(tuple(partition["cut_edges"]))
                # Need to sort the tuple so that the order is consistent
                # in the bad_district_pairs set
                parts_to_merge = [
                    partition.assignment.mapping[edge[0]],
                    partition.assignment.mapping[edge[1]],
                ]
                parts_to_merge.sort()

                if tuple(parts_to_merge) not in bad_district_pairs:
                    break

            subgraph = partition.graph.subgraph(
                partition.parts[parts_to_merge[0]] | partition.parts[parts_to_merge[1]]
            )

            flips = epsilon_tree_bipartition(
                subgraph.graph,
                parts_to_merge,
                pop_col=pop_col,
                pop_target=pop_target,
                epsilon=epsilon,
                node_repeats=node_repeats,
                method=method,
            )
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

    return partition.flip(flips)




class ReCom:
    """
    ReCom (short for ReCombination) is a class that represents a ReCom proposal
    for redistricting. It is used to create new partitions by recombining existing
    districts while maintaining population balance.

    """

    def __init__(
        self,
        pop_col: str,
        ideal_pop: Union[int, float],
        epsilon: float,
        method: Callable = bipartition_tree_random,
    ):
        """
        :param pop_col: The name of the column in the partition that contains the population data.
        :type pop_col: str
        :param ideal_pop: The ideal population for each district.
        :type ideal_pop: Union[int,float]
        :param epsilon: The epsilon value for population deviation as a percentage of the
            target population.
        :type epsilon: float
        :param method: The method used for bipartitioning the tree.
            Defaults to `bipartition_tree_random`.
        :type method: function, optional
        """
        self.pop_col = pop_col
        self.ideal_pop = ideal_pop
        self.epsilon = epsilon
        self.method = method

    def __call__(self, partition: Partition):
        return recom(
            partition, self.pop_col, self.ideal_pop, self.epsilon, method=self.method
        )
        
        
        import requests