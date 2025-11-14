## Functions

# neighbot_flips(partition) -> Set[Tuple]: The set of edges that were flipped in the given partition. Uses only partition.flips
# create_flow(): helper fuction which returns a dict {"in": set(), "out: set()"
# id_flows(merged_parts, new_ids): similar to create flow, this function creates a flow dictionary but for district ids.
# flows_from_changes(old_partition, new_partition) -> Dict: Mmapping each node that changed assignment between the previous and
#        current partitions to a dictionary of the form `{'in': <set of nodes that flowed in>, 'out': <set of nodes that flowed out>}`
# on_flow(initializer, alias) -> : A decorator to create an updater that responds to flows of nodes between parts of the partition. Uses only flows.
# compute_edge_flow(partition) -> Dict: A flow dictionary containing the flow from the parent of this partition to this partition.
#        Uses neighbor_flips. Used for calculating cut edges in Partition class.
# on_edge_flow() -> :


# TODO: Call and return a single dataclass for all flows

import collections
import functools
from typing import Callable, Dict, Set, Tuple


@functools.lru_cache(maxsize=2)
def neighbor_flips(partition) -> Set[Tuple]:
    """
    :param partition: A partition of a Graph
    :type partition: :class:`~gerrychain.partition.Partition`

    :returns: The set of edges that were flipped in the given partition.
    :rtype: Set[Tuple]
    """
    return {
        tuple(sorted((node, neighbor)))
        for node in partition.flip.flips
        for neighbor in partition.graph.neighbors(node)
        if neighbor not in partition.part_flows["out"]
    }


def create_flow():
    return {"in": set(), "out": set()}


def compute_part_flows(merged_parts, new_ids):
    outgoing_ids = merged_parts.difference(
        new_ids
    )  # keys that will be removed from partition.parts.keys()
    incoming_ids = new_ids.difference(
        merged_parts
    )  # will be added to partition.parts.keys()
    part_flows = {"in": set(incoming_ids), "out": set(outgoing_ids)}
    return part_flows


def on_flow(initializer: Callable, alias: str) -> Callable:
    """
    Use this decorator to create an updater that responds to flows of nodes
    between parts of the partition.

    Decorate a function that takes:
    - The partition
    - The previous value of the updater on a fixed part P_i
    - The new nodes that are just joining P_i at this step
    - The old nodes that are just leaving P_i at this step
    and returns:
    - The new value of the updater for the fixed part P_i.

    This will create an updater whose values are dictionaries of the
    form `{part: <value of the given function on the part>}`.

    The initializer, by contrast, should take the entire partition and
    return the entire `{part: <value>}` dictionary.

    Example:

    .. code-block:: python

        @on_flow(initializer, alias='my_updater')
        def my_updater(partition, previous, new_nodes, old_nodes):
            # return new value for the part

    :param initializer: A function that takes the partition and returns a
        dictionary of the form `{part: <value>}`.
    :type initializer: Callable
    :param alias: The name of the updater to be created.
    :type alias: str

    :returns: A decorator that takes a function as input and returns a
        wrapped function.
    :rtype: Callable
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapped(partition, previous=None):
            if partition.parent is None:
                return initializer(partition)

            if previous is None:
                previous = partition.parent[alias]

            new_values = previous.copy()

            for part in partition.id_flow["in"]:
                new_values[part] = set()

            for part, flow in partition.flows.items():
                new_values[part] = function(
                    partition, previous[part], flow["in"], flow["out"]
                )

            for part in partition.id_flow["out"]:
                new_values.pop(part, None)

            return new_values

        return wrapped

    return decorator


@functools.lru_cache(maxsize=2)
def compute_node_flows(old_partition, new_partition) -> Dict:
    """
    :param old_partition: A partition of a Graph representing dz    the previous step.
    :type old_partition: :class:`~gerrychain.partition.Partition`
    :param new_partition: A partition of a Graph representing the current step.
    :type new_partition: :class:`~gerrychain.partition.Partition`

    :returns: A dictionary mapping each node that changed assignment between
        the previous and current partitions to a dictionary of the form
        `{'in': <set of nodes that flowed in>, 'out': <set of nodes that flowed out>}`.
    :rtype: Dict
    """
    node_flows = collections.defaultdict(create_flow)

    for node, target in new_partition.flip.flips.items():
        source = old_partition.assignment.mapping[node]
        if source != target:
            node_flows[target]["in"].add(node)
            node_flows[source]["out"].add(node)
    return node_flows


def compute_candidate_flows(partition):
    candidate_flows = collections.defaultdict(create_flow)
    for part in partition.node_flows:
        candidate_flows[part] = {
            "in": {
                node
                for node in partition.node_flows[part]["in"]
                if partition.graph.nodes[node]["candidate"] == 1
            },
            "out": {
                node
                for node in partition.node_flows[part]["out"]
                if partition.graph.nodes[node]["candidate"] == 1
            },
        }
    return candidate_flows


class WrongFunction(Exception):
    """Raised when an unused function is called."""
