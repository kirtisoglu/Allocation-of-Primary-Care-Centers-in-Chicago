# treecut_tally.py

"""
logger.py

This module provides data structures and methods to track and record each step of 
a recursive tree partitioning process for later visualization and analysis.

Classes:
- CutLog: Stores evaluation data for each node considered as a cut.
- SpanningTreeStep: Represents a drawn spanning tree with metadata.
- AcceptedCutStep: Represents an accepted cut and its assignment.
- TreeCutTally: Main class for logging and serializing the full recursive tree cut.

Typical usage:
    logger = TreeCutTally(path="logs/treecut.json")
    logger.log_spanning_tree(edges, meta)
    logger.log_cut(CutLog(node, pop, facility))
    logger.log_accepted_cut(AcceptedCutStep(...))
    logger.mark_accepted_nodes(accepted_nodes)
    logger.save()
"""

import json
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

# from falcomchain.helper import save_pickle
from falcomchain.tree import Cut

# StateLog is the entering function for recursively cutting a spanning tree.
#
# StateLog = list[recursion]
#
# where
#
# recursion = {recursion_info: dictionary, recursion: list[steps]} and
#
# recursion_info = {'target_pop': float, 'epsilon': float, 'n_teams': int,
#                          'two_sided': bool, 'selected_cut': tuple, 'debt': float,
#                          'remaining_ids': set, 'hired_teams': int,
#                          'district_id': int}
#
#   Within a step, we save selected_nodes in order and some static info for the step (tree edges, tree pop and ideal pop, n_teams,...)
#   Every step is a tree generation in bipartition_tree and find_cut_edges process on it.
#
#   step = {tree: dict, cut_nodes: list}
#
#       tree: {edges: set(), ideal_pop: float, n_teams: int, two_sided: bool, root, epsilon, pop_target}
#       selected_nodes: list(Cut)

#           node: {pop: int, facility: bool, complement: bool, accepted: int}

#               if accepted = -1, then not accepted. Else, it is the district id.
#               n_teams?

# aslinda komple spanning tree'yi tutuyorum gibi. Gereksiz attributelar atilirsa direk tree kaydetmek olur.


# RecursionLog
#       ├── Recursion (global info about the recursion level)
#       └── Steps: List[Step]
#       │      ├── StepLog
#       │      │     ├── TreeLog (global info about the tree)
#       │      │     └── List[CutLog] (node-level evaluations)
#


# https://python.plainenglish.io/five-unknown-tricks-for-python-classes-ba5edd29a108
# 1. @property
# 2. iter and next
# 3. __setitem__
# 4. __getitem__


@dataclass
class CutLog:
    """
    For tracking each node evaluation in find_edge_cuts.

    Attributes:
    - node (tuple): ID of the evaluated node
    - pop (int): Population in the subtree beneath 'node'
    - facility (int): Facility count in the subtree
    - accepted (bool): Whether this cut was accepted
    """

    node: Tuple
    subnodes: list
    pop: int
    facility: int
    accepted: str

    @classmethod
    def from_cut(cls, cut) -> "CutLog":

        return cls(
            node=cut.node,
            subnodes=cut.subnodes,
            pop=cut.pop,
            facility=cut.facility,
            accepted=cut.accepted,
        )

    def mark_accepted(self, string):
        """Used later to mark nodes accepted in a successful cut"""
        self.accepted = string

    def to_dict(self):
        return asdict(self)


@dataclass
class TreeLog:
    """
    Captures some of the attributes of a SpanningTree class.

    Includes edges and parameters relevant for cut decision-making.
    """

    edges: List[Tuple]
    root: Tuple

    @classmethod
    def from_spanning_tree(cls, tree) -> "TreeLog":
        return cls(
            edges=list(tree.graph.edges),
            root=tree.root,
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class StepLog:
    """
    One tree step in bipartition_tree function.

    Includes:
    - a randomly generated spanning tree (TreeLog)
    - evaluations of candidate cut nodes (CutLog)
    """

    tree: TreeLog
    cut_nodes: List[CutLog]

    @classmethod
    def from_bipartition_tree(cls, tree) -> "StepLog":
        return cls(tree=TreeLog.from_spanning_tree(tree), cut_nodes=[])

    def add_cut_node(self, cut: Cut):
        self.cut_nodes.append(CutLog.from_cut(cut))

    def to_dict(self):
        return {
            "tree": self.tree.to_dict(),
            "cut_nodes": [log.to_dict() for log in self.cut_nodes],
        }


@dataclass
class RecursionLog:
    target_pop: float
    epsilon: float
    n_teams: float
    two_sided: bool
    cut_node: Tuple
    debt: float
    remaining_ids: List[int]
    hired_teams: float
    district_id: int
    steps: List[StepLog] = field(default_factory=list)

    def add_step(self, tree):
        self.steps.append(StepLog.from_bipartition_tree(tree))

    def to_dict(self):
        return {
            "target_pop": self.target_pop,
            "epsilon": self.epsilon,
            "n_teams": self.n_teams,
            "two_sided": self.two_sided,
            "selected_cut": self.cut_node,
            "debt": self.debt,
            "remaining_ids": self.remaining_ids,
            "hired_teams": self.hired_teams,
            "district_id": self.district_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    def update(self, teams, dist_id, node):
        self.hired_teams = teams
        self.district_id = dist_id
        self.cut_node = node


@dataclass
class StateLog:
    recursions: List[RecursionLog] = field(default_factory=list)

    def __len__(self):
        return len(self.recursions)

    def add_recursion(
        self,
        target_pop,
        epsilon,
        n_teams,
        two_sided,
        cut_node,
        debt,
        remaining_ids,
        hired_teams,
        district_id,
    ):
        self.recursions.append(
            RecursionLog(
                target_pop,
                epsilon,
                n_teams,
                two_sided,
                cut_node,
                debt,
                remaining_ids,
                hired_teams,
                district_id,
            )
        )

    def add_step(self, tree):
        self.recursions[-1].add_step(tree)

    def to_dict(self):
        return [rec.to_dict() for rec in self.recursions]


class ChainLog:

    def __init__(self, path: str, batch_size: int = 10, is_initial: bool = False):

        self.path = path
        self.batch_size = batch_size
        self.is_initial = is_initial
        self.states: List[StateLog] = []

        if os.path.exists(path) and not is_initial:
            with open(path, "rb") as f:
                self.all_states: List[dict] = pickle.load(f)
        else:
            self.all_states: List[dict] = []

    def __len__(self):
        return len(self.states)

    def add_state(self):
        self.states.append(StateLog())

    def add_recursion(self, *args, **kwargs):
        if not self.states:
            raise RuntimeError("No state exists. Call `add_state()` first.")
        self.states[-1].add_recursion(*args, **kwargs)

    def update_recursion(self, teams, dist_id, node):
        self.states[-1].recursions[-1].update(teams, dist_id, node)

    def add_step(self, tree):
        if not self.states:
            raise RuntimeError("No state exists.")
        self.states[-1].add_step(tree)

    def new_cut(
        self, cut
    ):  # change this later on. define a cut list temporarily then add to step later for not making many calls everytime.
        self.states[-1].recursions[-1].steps[-1].cut_nodes.append(CutLog.from_cut(cut))

    def append(self, state: StateLog):
        self.states.append(state)
        if self.is_initial:
            self._save_all(self.states)
            self.states = []
        elif len(self.states) >= self.batch_size:
            self._flush()

    def _flush(self):
        self.all_states.extend(self.states)
        self._save_all(self.all_states)
        self.states = []

    def _save_all(self, data: List[StateLog | dict]):
        if os.path.exists(self.path):
            shutil.copy(self.path, self.path.replace(".pkl", "_backup.pkl"))
        serialized = [s.to_dict() if isinstance(s, StateLog) else s for s in data]
        with open(self.path, "wb") as f:
            pickle.dump(serialized, f)

    @classmethod
    def load_pickle(cls, path: str, deserialize: bool = False):
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(path)
        instance.all_states = data
        if deserialize:
            instance.all_states = [
                StateLog(
                    [
                        RecursionLog(
                            **{
                                **rec,
                                "steps": [
                                    StepLog(
                                        TreeLog(**step["tree"]),
                                        [CutLog(**cut) for cut in step["cut_nodes"]],
                                    )
                                    for step in rec["steps"]
                                ],
                            }
                        )
                        for rec in state
                    ]
                )
                for state in data
            ]
        return instance

    def save_json(self, json_path: str):
        all_data = [
            s.to_dict() if isinstance(s, StateLog) else s
            for s in self.all_states + self.states
        ]
        with open(json_path, "w") as f:
            json.dump(all_data, f, indent=2)

    def finalize(self):
        if self.states:
            self._flush()

    def flatten_steps(self) -> List[dict]:
        flat = []
        for state_index, state in enumerate(self.all_states + self.states):
            if isinstance(state, StateLog):
                state_data = state
            else:
                continue  # skip unconverted dicts

            for recursion_index, recursion in enumerate(state_data.recursions):
                for step_index, step in enumerate(recursion.steps):
                    flat.append(
                        {
                            "state_index": state_index,
                            "recursion_index": recursion_index,
                            "step_index": step_index,
                            "cut_node": recursion.cut_node,
                            "tree_root": step.tree.root,
                            "n_cut_nodes": len(step.cut_nodes),
                            "target_pop": recursion.target_pop,
                            "district_id": recursion.district_id,
                        }
                    )
        return flat

    def summary_table(self) -> List[dict]:
        summary = []
        for state_index, state in enumerate(self.all_states + self.states):
            if isinstance(state, StateLog):
                for recursion in state.recursions:
                    summary.append(
                        {
                            "state_index": state_index,
                            "district_id": recursion.district_id,
                            "target_pop": recursion.target_pop,
                            "num_steps": len(recursion.steps),
                        }
                    )
        return summary

    def filter_steps(self, **conditions) -> List[dict]:
        return [
            step
            for step in self.flatten_steps()
            if all(step.get(k) == v for k, v in conditions.items())
        ]
