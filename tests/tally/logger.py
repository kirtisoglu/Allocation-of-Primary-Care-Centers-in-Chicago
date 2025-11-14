import pytest

from falcomchain.tally import AcceptedCutStep, CutLog, SpanningTreeStep


def test_cutlog_instantiation_and_acceptance():
    log = CutLog(node=(5,), pop=300, facility=2)
    assert log.accepted is False
    assert log.pop == 300
    assert log.node == (5,)

    log.mark_accepted()
    assert log.accepted is True

    d = log.to_dict()
    assert isinstance(d, dict)
    assert d["node"] == (5,)
    assert d["accepted"] is True


def test_spanning_tree_step_serialization():
    step = SpanningTreeStep(
        edges=[(1, 2), (2, 3)], meta={"pop": 500, "teams": 3, "epsilon": 0.05}
    )
    d = step.to_dict()
    assert d["type"] == "spanning_tree"
    assert d["edges"] == [(1, 2), (2, 3)]
    assert "pop" in d["meta"]
    assert d["meta"]["teams"] == 3


def test_accepted_cut_step_serialization():
    cut = AcceptedCutStep(
        nodes={1, 2, 3}, assigned_district=7, assigned_teams=2, pop=980
    )

    d = cut.to_dict()
    assert d["type"] == "cut_accepted"
    assert sorted(d["nodes"]) == [1, 2, 3]  # because we convert a set to list
    assert d["district"] == 7
    assert d["teams"] == 2
    assert d["pop"] == 980
