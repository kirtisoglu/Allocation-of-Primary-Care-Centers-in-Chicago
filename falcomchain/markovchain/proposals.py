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
    pop_target: float,
    density: Optional[float] = None,
    snapshot: bool = True,
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
    try:
        acut_object = bipartition_tree(
            graph=partition.supergraph.copy(),
            pop_target=1500,
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
    except Exception:
        raise
    
    
    # ("Cut", "node subnodes assigned_teams pop")
    # For now, we merge only one super district picking it randomly
    ##super_partition = {
    ##    super_district: set() for super_district in set(superflip.flips.values())
    ##}
    ##for supernode in superflip.flips.keys():
    ##    super_partition[superflip.flips[supernode]].add(supernode)
    ##superdistrict_to_merge = random.choice(list(super_partition.keys()))
    ##superflip.add_merged_ids(super_partition[superdistrict_to_merge])

    # LOWER LEVEL: resplitting merged districts

    ##subgraph = partition.graph.graph.subgraph(
    ##    set.union(*(set(partition.parts[part]) for part in superflip.merged_ids)))

    new_pop_target = acut_object.pop / acut_object.assigned_teams
    # print("pop in superdistrict", acut_object.pop)

    merge = frozenset(acut_object.subnodes)
    superflip = Flip(merged_ids=merge, super_cut_object=acut_object)

    subgraph = partition.graph.graph.subgraph(
        set.union(*(set(partition.parts[part]) for part in merge))
    )

    max_id = max(district for district in partition.parts)
    sub_assignments = {
        node: partition.assignment.mapping[node] for node in subgraph.nodes
    }

    try:
        flip = method(
            graph=subgraph,
            n_teams=int(
                acut_object.assigned_teams
            ),  ##sum(superflip.team_flips.values()),
            merged_ids=set(merge.copy()),  ##superflip.merged_ids,
            assignments=sub_assignments,
            max_id=max_id,
            pop_target=new_pop_target,
            epsilon=epsilon,
            snapshot=snapshot,
            debt=(acut_object.pop - acut_object.assigned_teams * 1500),
            iteration=partition.step
        )
    except Exception:
        raise

    flip = flip.add_merged_ids(merge)

    return partition.perform_flip(flipp=flip, superflipp=superflip)


def recom(  # Note: recomb is called for each state of the chain. Parameters must be static for the states. (should we cache some of them in Partition?)
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
                pop_target=pop_target,
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
