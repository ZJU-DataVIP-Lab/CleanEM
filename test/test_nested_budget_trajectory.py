from types import SimpleNamespace

import numpy as np

from adaptive_mechanism_discovery import (
    MechanismCandidate,
    MechanismDiscoveryReport,
)
from cleanem_inference import EvidenceMatrix, select_active_queries
from main import _build_nested_active_query_trajectory


def _matrix(size=60):
    keys = [f"{index}::value" for index in range(size)]
    values = np.eye(size, dtype=float)
    return EvidenceMatrix(
        cell_keys=keys,
        feature_names=[f"feature_{index}" for index in range(size)],
        values=values,
        key_to_index={key: index for index, key in enumerate(keys)},
    )


def _candidate(index):
    members = [f"{2 * index}::value", f"{2 * index + 1}::value"]
    return MechanismCandidate(
        mechanism_id=f"mechanism_{index}",
        label=f"discovered:mechanism_{index}",
        name=f"mechanism_{index}",
        explanation="test",
        signature=f"cluster:v1:test:{index}",
        archetype="open_entity",
        column="value",
        columns=["value"],
        family="pattern",
        source_kind="discovered_pattern",
        support=2,
        row_support=2,
        strength=1.0,
        cross_row_stability=1.0,
        archetype_consistency=1.0,
        repairability=0.0,
        repair_success_rate=None,
        uncertainty=1.0,
        representative_cells=list(members),
        member_cells=list(members),
    )


def test_d_optimal_extension_preserves_initial_prefix_and_exclusions():
    matrix = _matrix(8)
    selected = select_active_queries(
        matrix,
        4,
        initial_cell_keys=["0::value", "1::value"],
        protected_cell_keys={"0::value", "1::value"},
        excluded_cell_keys={"2::value"},
    )

    assert selected[:2] == ["0::value", "1::value"]
    assert len(selected) == 4
    assert "2::value" not in selected


def test_nested_trajectory_uses_proportional_mechanism_cap():
    matrix = _matrix()
    report = MechanismDiscoveryReport(
        candidates=[_candidate(index) for index in range(12)]
    )
    args = SimpleNamespace(
        active_query_strategy="d_optimal",
        adaptive_mechanism_query_budget=0,
        adaptive_mechanism_min_labeled_support=2,
        adaptive_mechanism_query_fraction=0.60,
        adaptive_mechanism_max_candidates_per_round=0,
        active_query_clean_proxy_fraction=0.0,
        enable_structural_missing_query_coverage=False,
    )
    checkpoints = [0, 4, 8, 10, 20, 40]

    payload = _build_nested_active_query_trajectory(
        matrix,
        report,
        checkpoints,
        args,
        metadata={},
        cell_registry={},
        cell_observations={},
    )

    expected_mechanisms = {0: 0, 4: 1, 8: 2, 10: 3, 20: 6, 40: 12}
    previous = []
    for budget in checkpoints:
        cells = payload["checkpoint_cell_keys"][str(budget)]
        summary = payload["checkpoint_summaries"][str(budget)]
        assert len(cells) == budget
        assert cells[:len(previous)] == previous
        assert summary["allocated_mechanisms"] == expected_mechanisms[budget]
        assert summary["mechanism_label_count"] == 2 * expected_mechanisms[budget]
        previous = cells
