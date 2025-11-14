## Functions

# put_edges_into_parts: {district: set of outgoing edges} - does not call any function - not used anywhere
# new_cuts: set of cut edges that were not cut, but now are - calls crosses_parts, neighbor_flips - used in cut_edges
# obsolete_cuts:
# initialize_cut_edges
# cut_edges_by_part
# cut_edges
#
#
# TODO:


import collections
from typing import Dict, List, Set, Tuple

from .flows import neighbor_flips

"""# using for cut edges
def new_cuts(partition) -> Set[Tuple]:

    :param partition: A partition of a Graph
    :type partition: :class:`~gerrychain.partition.Partition`

    :returns: The set of edges that were not cut, but now are.
    :rtype: Set[Tuple]

    return {
        (node, neighbor)
        for node, neighbor in neighbor_flips(partition)
        if partition.crosses_parts((node, neighbor))
    }"""


# using for cut edges
"""def obsolete_cuts(partition) -> Set[Tuple]:

    :param partition: A partition of a Graph
    :type partition: :class:`~gerrychain.partition.Partition`

    :returns: The set of edges that were cut, but now are not.
    :rtype: Set[Tuple]

    return {
        (node, neighbor)
        for node, neighbor in neighbor_flips(partition)
        if partition.parent.crosses_parts((node, neighbor))
        and not partition.crosses_parts((node, neighbor))
    }"""


"""# main function: using for supergraph edges
def cut_edges(partition):
    
    :param partition: A partition of a Graph
    :type partition: :class:`~gerrychain.partition.Partition`

    :returns: The set of edges that are cut by the given partition.
    :rtype: Set[Tuple]

    parent = partition.parent

    if not parent:
        return {
            tuple(sorted(edge))
            for edge in partition.graph.edges
            if partition.crosses_parts(edge)
        }
    # Edges that weren't cut, but now are cut
    # We sort the tuples to make sure we don't accidentally end
    # up with both (4,5) and (5,4) (for example) in it
    new, obsolete = new_cuts(partition), obsolete_cuts(partition)

    return (parent.cut_edges | new) - obsolete"""
