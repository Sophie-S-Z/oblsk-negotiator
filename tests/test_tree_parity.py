"""
Parity and config-tree tests.

The big one (test_tree_parity) is a characterization test: it replays a fixed
battery of scenarios through the current YAML-driven tree and asserts the result
is identical, decision for decision and state for state, to the golden record
captured from the tree before it was made config-driven. That is the proof the
refactor changed structure without changing behavior. The rest check the config
layer itself: the spec is total and every name in it resolves.

Regenerate the golden only via tests/generate_golden.py, on purpose.
"""

from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from oblsk_negotiator.behavior_tree import (GUARDS, HANDLERS, _load_tree_spec,
                                           _validate_spec, run_tree)
from tests.parity_harness import run_battery

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_decisions.json")


def _normalize(records):
    # Round-trip through JSON so tuples/ints compare the same way the golden does.
    return json.loads(json.dumps(records, sort_keys=True))


def test_tree_parity():
    with open(GOLDEN) as f:
        golden = json.load(f)
    fresh = _normalize(run_battery())
    assert len(fresh) == len(golden), "scenario count drifted from the golden record"
    for g, f in zip(golden, fresh):
        assert g == f, (
            f"behavior drift in scenario {g.get('name')!r}: the config-driven tree "
            f"no longer matches the recorded decisions. If intended, regenerate "
            f"with tests/generate_golden.py.")


def test_spec_is_total_and_resolves():
    spec = _load_tree_spec()
    assert spec, "tree_spec.yaml loaded empty"
    for step in spec:
        assert step["guard"] in GUARDS, f"guard {step['guard']!r} not registered"
        assert step["handler"] in HANDLERS, f"handler {step['handler']!r} not registered"
    # A total guard must close the list so some branch always fires.
    assert spec[-1]["guard"] == "always"


def test_validate_rejects_unknown_guard():
    with pytest.raises(ValueError):
        _validate_spec([{"guard": "nope_not_a_guard", "handler": "pause_human"}])


def test_validate_rejects_unknown_handler():
    with pytest.raises(ValueError):
        _validate_spec([{"guard": "always", "handler": "nope_not_a_handler"}])


def test_run_tree_requires_a_match():
    # An empty spec matches nothing; the interpreter must refuse rather than
    # return a bogus decision.
    with pytest.raises(RuntimeError):
        run_tree(object(), [])
