from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, Union, Optional, DefaultDict, Set, Type, Tuple
from graph import Graph

import pandas


class Assignment(Mapping):
    """
    An assignment of nodes into parts.

    The goal of :class:`Assignment` is to provide an interface that mirrors a
    dictionary (what we have been using for assigning nodes to districts) while making it
    convenient/cheap to access the set of nodes in each part.

    An :class:`Assignment` has a ``parts`` property that is a dictionary of the form
    ``{part: <frozenset of nodes in part>}``.
    """

    __slots__ = ["parts", "mapping", "centers", "radius", "candidates"]

    def __init__(
        self, parts: Dict, candidates: Dict, travel_times: Dict, mapping: Optional[Dict] = None, validate: bool = True
    ) -> None:
        """
        :param parts: Dictionary mapping partition assignments frozensets of nodes.
        :type parts: Dict
        :param centers:
        :type parts: Dict
        :param radius:
        :type parts: Dict
        :param candidates:
        :type parts: Dict 
        :param mapping: Dictionary mapping nodes to partition assignments. Default is None.
        :type mapping: Optional[Dict], optional
        :param validate: Whether to validate the assignment. Default is True.
        :type validate: bool, optional

        :returns: None

        :raises ValueError: if the keys of ``parts`` are not unique
        :raises TypeError: if the values of ``parts`` are not frozensets
        """
        if validate:
            number_of_keys = sum(len(keys) for keys in parts.values())
            number_of_unique_keys = len(set().union(*parts.values()))
            if number_of_keys != number_of_unique_keys:
                raise ValueError("Keys must have unique assignments.")
            if not all(isinstance(keys, frozenset) for keys in parts.values()):
                raise TypeError("Level sets must be frozensets")
        
        self.parts = parts
        self.candidates = candidates
        
        self.centers = {} 
        self.radius = {}
        for part in parts.keys():
            self.centers[part], self.radius[part] = self.facility_assignment(part, travel_times)

        if not mapping:
            self.mapping = {}
            for part, nodes in self.parts.items():
                for node in nodes:
                    self.mapping[node] = part
        else:
            self.mapping = mapping

    def __repr__(self):
        return "<Assignment [{} keys, {} parts]>".format(len(self), len(self.parts))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(len(keys) for keys in self.parts.values())

    def __getitem__(self, node):
        return self.mapping[node]

    # add travel times 
    def copy(self, travel_times):
        """
        Returns a copy of the assignment.
        Does not duplicate the frozensets of nodes, just the parts dictionary.
        """
        return Assignment(self.parts.copy(), self.candidates.copy(), travel_times, self.mapping.copy(), validate=False)


    def candidate_flows(self, flow):
        print("candidates values", self.candidates.values())
        listoffrozensets = [s for s in self.candidates.values()]
        all_candidates = frozenset().union(*listoffrozensets)
        print("all candidates after union", all_candidates)
        return flow["in"] & all_candidates 
     

    def update_flows(self, flows, travel_times):
        """
        Update the assignment for some nodes using the given flows.
        """
        for part, flow in flows.items():
            print("part", part)
            print("flow", flow)
            print("parts[part]", self.parts[part])
            # Union between frozenset and set returns an object whose type
            # matches the object on the left, which here is a frozenset
            print("flow out", flow['out'])
            print("flow in", flow["in"])
            self.parts[part] = (self.parts[part] - flow["out"]) | flow["in"]
            print("updated parts[part]", self.parts[part])
            self.candidates[part] = (self.candidates[part] - flow["out"]) | self.candidate_flows(flow)
            self.centers[part], self.radius[part] = self.facility_assignment(part, travel_times)
            print("updated candidates[part]", self.candidates[part])
            for node in flow["in"]:
                self.mapping[node] = part
                
                
    def items(self):
        """
        Iterate over ``(node, part)`` tuples, where ``node`` is assigned to ``part``.
        """
        yield from self.mapping.items()

    def keys(self):
        yield from self.mapping.keys()

    def values(self):
        yield from self.mapping.values()

    def update_parts(self, new_parts: Dict) -> None:
        """
        Update some parts of the assignment. Does not check that every node is
        still assigned to a part.

        :param new_parts: dictionary mapping (some) parts to their new sets or
            frozensets of nodes
        :type new_parts: Dict

        :returns: None
        """
        for part, nodes in new_parts.items():
            self.parts[part] = frozenset(nodes)

            for node in nodes:
                self.mapping[node] = part

    def to_series(self) -> pandas.Series:
        """
        :returns: The assignment as a :class:`pandas.Series`.
        :rtype: pandas.Series
        """
        groups = [
            pandas.Series(data=part, index=nodes) for part, nodes in self.parts.items()
        ]
        return pandas.concat(groups)

    def to_dict(self) -> Dict:
        """
        :returns: The assignment as a ``{node: part}`` dictionary.
        :rtype: Dict
        """
        return self.mapping

    @classmethod
    def from_dict(cls, assignment: Dict, graph: Graph, column_names: Tuple[str], travel_times: Dict) -> "Assignment":
        """
        Create an :class:`Assignment` from a dictionary. This is probably the method you want
        to use to create a new assignment.

        This also works for :class:`pandas.Series`.

        :param assignment: dictionary mapping nodes to partition assignments
        :type assignment: Dict

        :returns: A new instance of :class:`Assignment` with the same assignments as the
            passed-in dictionary.
        :rtype: Assignment
        """
        sets, facilities = level_sets(assignment, graph, column_names)
        parts = {part: frozenset(keys) for part, keys in sets.items()}
        candidates = {part: frozenset(keys) for part, keys in facilities.items()}

        return cls(parts, candidates, travel_times)

    def facility_assignment(self, part, travel_times):
    
        save_candidates_radius = {}

        for candidate in self.candidates[part]:
            candidate_radius = max(travel_times[(node, candidate)] for node in self.parts[part])
            save_candidates_radius[candidate] = candidate_radius
            
        best_candidate = min(save_candidates_radius, key=save_candidates_radius.get) # center of the part
        
        return best_candidate, save_candidates_radius[best_candidate]


def get_assignment(
    part_assignment: Union[str, Dict, Assignment], graph: Optional[Graph], column_names: list[str], travel_times: Dict
) -> Assignment:
    """
    Either extracts an :class:`Assignment` object from the input graph
    using the provided key or attempts to convert part_assignment into
    an :class:`Assignment` object.

    :param part_assignment: A node attribute key, dictionary, or
        :class:`Assignment` object corresponding to the desired assignment.
    :type part_assignment: str
    :param graph: The graph from which to extract the assignment.
        Default is None.
    :type graph: Optional[Graph], optional

    :returns: An :class:`Assignment` object containing the assignment
        corresponding to the part_assignment input
    :rtype: Assignment

    :raises TypeError: If the part_assignment is a string and the graph
        is not provided.
    :raises TypeError: If the part_assignment is not a string or dictionary.
    """
    #if isinstance(part_assignment, str):
    #    if graph is None:
    #        raise TypeError(
    #            "You must provide a graph when using a node attribute for the part_assignment"
    #        )
    #    return Assignment.from_dict(
    #        {node: graph.nodes[node][part_assignment] for node in graph}
    #    )
    # Check if assignment is a dict or a mapping type
    #elif callable(getattr(part_assignment, "items", None)):
    return Assignment.from_dict(part_assignment, graph, column_names, travel_times)
    #elif isinstance(part_assignment, Assignment):
    #    return part_assignment
    #else:
    #    raise TypeError("Assignment must be a dict or a node attribute key")


def level_sets(assignment: dict, graph, column_names: list, container: type[Set] = set) -> DefaultDict:
    """
    Inverts a dictionary. ``{key: value}`` becomes
    ``{value: <container of keys that map to value>}``.

    :param mapping: A dictionary to invert. Keys and values can be of any type.
    :type mapping: Dict
    :param container: A container type used to collect keys that map to the same value.
        By default, the container type is ``set``.
    :type container: Type[Set], optional

    :return: A dictionary where each key is a value from the original dictionary,
        and the corresponding value is a container (by default, a set) of keys from
        the original dictionary that mapped to this value.
    :rtype: DefaultDict

    Example usage::

    .. code_block:: python

        >>> level_sets({'a': 1, 'b': 1, 'c': 2})
        defaultdict(<class 'set'>, {1: {'a', 'b'}, 2: {'c'}})
    """
    sets: Dict = defaultdict(container)
    candidates: Dict = defaultdict(container)
    facility_attr = 'real_phc'

    for node, part in assignment.items():
        sets[part].add(node)
        if graph.nodes[node][facility_attr]==True:
            candidates[part].add(node)  
            
    for part in sets.keys():
        if not any(graph.nodes[node][facility_attr]==True for node in sets[part]):
            print(f"Part {part} does not have a candidate.")
            print(sets[part])
                 
    return sets, candidates
