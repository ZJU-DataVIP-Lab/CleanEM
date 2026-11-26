import pytest

from cleanem_inference import build_cell_evidence_matrix
from cleanem_models import CellRecord, EvidenceObservation


def _record(cell_key: str, column: str, archetype: str) -> CellRecord:
    return CellRecord(
        key=cell_key,
        row_index=0,
        column=column,
        raw_value="value",
        value_key=f"{column}::value",
        archetype=archetype,
    )


@pytest.mark.parametrize(
    "family",
    ["outlier", "pattern", "prototype_lexical", "relationship", "contextual"],
)
def test_semantic_evidence_is_namespaced_by_archetype(family):
    registry = {
        "0::entity": _record("0::entity", "entity", "open_entity"),
        "0::date": _record("0::date", "date", "temporal_measure"),
    }
    observations = {
        cell_key: [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="shared_source",
            family=family,
            polarity="dirty",
            strength=1.0,
            hard=False,
            reason_code="test",
        )]
        for cell_key in registry
    }

    matrix = build_cell_evidence_matrix(registry, {}, observations)

    assert set(matrix.feature_names) == {
        f"cell:{family}:shared_source:archetype:open_entity",
        f"cell:{family}:shared_source:archetype:temporal_measure",
    }
    assert matrix.values.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_missing_evidence_remains_global_across_archetypes():
    registry = {
        "0::entity": _record("0::entity", "entity", "open_entity"),
        "0::date": _record("0::date", "date", "temporal_measure"),
    }
    observations = {
        cell_key: [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="missing_token",
            family="missing",
            polarity="dirty",
            strength=1.0,
            hard=True,
            reason_code="missing_token",
        )]
        for cell_key in registry
    }

    matrix = build_cell_evidence_matrix(registry, {}, observations)

    assert matrix.feature_names == ["cell:missing:missing_token"]
    assert matrix.values.tolist() == [[1.0], [1.0]]
