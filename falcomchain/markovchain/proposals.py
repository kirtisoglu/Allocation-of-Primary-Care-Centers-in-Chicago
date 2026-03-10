import random
from collections import namedtuple
from functools import partial
from typing import Optional, Tuple

from falcomchain.partition import Partition
from falcomchain.tree.tree import (
    Cut,
    Flip,
    ReselectException,
    bipartition_tree,
    capacitated_recursive_tree,
)


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


def hierarchical_recom(
    partition: Partition,
    epsilon: float,
    demand_target: float,
    density: Optional[float] = None,
    snapshot: bool = True,
) -> Partition:
    """
    Proposes a new partition via two-level hierarchical ReCom.

    At the upper level, two adjacent super-districts are selected from the supergraph
    and merged into a single superdistrict. At the lower level, the merged region is
    re-split using a capacitated spanning tree such that the total team capacity of the
    new districts equals the total capacity of the merged superdistrict.

    :param partition: The current partition.
    :type partition: Partition
    :param epsilon: Maximum relative demand deviation allowed, as a fraction of demand_target.
    :type epsilon: float
    :param demand_target: Target demand per district.
    :type demand_target: float
    :param density: Optional density parameter passed to the tree method.
    :type density: float, optional
    :param snapshot: If True, saves tree snapshots for debugging. Defaults to True.
    :type snapshot: bool
    :returns: The new partition after the hierarchical flip.
    :rtype: Partition
    """

    method = partial(
        capacitated_recursive_tree,
        capacity_level=partition.capacity_level,
        density=density,
    )

    # UPPER LEVEL: selecting districts from supergraph to merge

    all_teams = sum(team for team in partition.teams.values())

    ##superflip = method(
    ##    graph=partition.supergraph, n_teams=all_teams, supergraph=True
    ##)
    acut_object = bipartition_tree(
        graph=partition.supergraph.copy(),
        demand_target=1500,
        capacity_level=partition.capacity_level,
        n_teams=sum(partition.teams.values()),
        epsilon=epsilon / 2,
        two_sided=False,
        supergraph=True,
        density=False,
        snapshot=snapshot,
        max_attempts=10,
        iteration=partition.step,
    )

    # LOWER LEVEL: resplitting merged districts
    new_demand_target = acut_object.demand / acut_object.assigned_teams
    # print("demand in superdistrict", acut_object.demand)

    merge = frozenset(acut_object.subnodes)
    superflip = Flip(merged_ids=merge, super_cut_object=acut_object)

    subgraph = partition.graph.graph.subgraph(
        set.union(*(set(partition.parts[part]) for part in merge))
    )

    max_id = max(district for district in partition.parts)
    sub_assignments = {
        node: partition.assignment.mapping[node] for node in subgraph.nodes
    }

    flip = method(
        graph=subgraph,
        n_teams=int(acut_object.assigned_teams),
        merged_ids=set(merge.copy()),
        assignments=sub_assignments,
        max_id=max_id,
        demand_target=new_demand_target,
        epsilon=epsilon,
        snapshot=snapshot,
        debt=(acut_object.demand - acut_object.assigned_teams * 1500),
        iteration=partition.step,
    )
    flip = flip.add_merged_ids(merge)

    return partition.perform_flip(flipp=flip, superflipp=superflip)


def recom(  # Note: recomb is called for each state of the chain. Parameters must be static for the states. (should we cache some of them in Partition?)
    partition: Partition,
    demand_target: int,
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
    separates the region into two new districts that are demand balanced, and,
    if not, a new edge is selected at random and the process is repeated.

    :param partition: The initial partition.
    :type partition: Partition
    :param demand_col: The name of the demand column.
    :type demand_col: str
    :param demand_target: The target demand for each district.
    :type demand_target: Union[int,float]
    :param epsilon: The epsilon value for demand deviation as a percentage of the
        target demand.
    :type epsilon: float
    :param node_repeats: The number of times to repeat the bipartitioning step. Default is 1.
    :type node_repeats: int, optional

    :returns: The new partition resulting from the ReCom algorithm.
    :rtype: Partition
    """
    bad_district_pairs = set()
    n_parts = len(partition)
    tot_pairs = (
        n_parts * (n_parts - 1) / 2
    )  # n choose 2  (isn't it too big? no adjacency between any two districts. it should be # of super cut edges)
    ids = set(partition.parts.keys())

    while len(bad_district_pairs) < tot_pairs:
        try:
            while True:
                edge = random.choice(tuple(partition["cut_edges"]))
                # Need to sort the tuple so that the order is consistent in the bad_district_pairs set
                part_one, part_two = (
                    partition.assignment.mapping[edge[0]],
                    partition.assignment.mapping[edge[1]],
                )
                parts_to_merge = [part_one, part_two]
                parts_to_merge.sort()

                if tuple(parts_to_merge) not in bad_district_pairs:
                    break

            n_teams = partition.teams[part_one] + partition.teams[part_two]
            subgraph = partition.graph.subgraph(
                partition.parts[part_one] | partition.parts[part_two]
            )

            flips, new_teams = capacitated_recursive_tree(
                graph=subgraph.graph,
                column_names=column_names,
                n_teams=n_teams,
                demand_target=demand_target,
                epsilon=epsilon,
                capacity_level=partition.capacity_level,
                density=density,
                assignments=partition.assignment,
                merged_parts=parts_to_merge,
                ids=ids,
            )
            break

        except Exception as e:
            if isinstance(
                e, ReselectException
            ):  # if there is no balanced cut after max_attempt in bipartition_tree, then the pair is a bad district pair.
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

    :returns: A possible next `~falcomchain.partition.Partition`
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

    :returns: A possible next `~falcomchain.partition.Partition`
    :rtype: Partition
    """
    if len(partition["cut_edges"]) == 0:
        return partition
    edge = random.choice(tuple(partition["cut_edges"]))
    index = random.choice((0, 1))
    flipped_node, other_node = edge[index], edge[1 - index]
    flip = {flipped_node: partition.assignment.mapping[other_node]}
    return partition.flip(flip)
