import numpy as np

from adaptive_mechanism_discovery import (
    allocate_mechanism_queries,
    annotate_discovered_observations,
    discover_candidate_mechanisms,
    enable_validated_mechanisms,
    validate_mechanism_candidates,
)
from cleanem_inference import build_cell_evidence_matrix, select_active_queries
from cleanem_models import CellRecord, EvidenceObservation, RepairCandidate
from main import _infer_deterministic_surface_repairs, _levenshtein_distance


def test_banded_levenshtein_matches_exact_distance_within_cutoff():
    cases = [
        ("kitten", "sitting", 3),
        ("creator,writer", "creator,writer", 0),
        ("description..", "description.", 1),
        ("abcd", "abdc", 2),
    ]
    for left, right, cutoff in cases:
        assert _levenshtein_distance(
            left,
            right,
            max_distance=cutoff,
        ) == _levenshtein_distance(left, right)


def test_banded_levenshtein_returns_cutoff_sentinel_when_exceeded():
    assert _levenshtein_distance("short", "a much longer value", 2) == 3
    assert _levenshtein_distance("abcdefgh", "ijklmnop", 2) == 3


def _context_fixture(column="city", determinant="country", dependent="city", size=4):
    registry = {}
    observations = {}
    for row_index in range(size):
        cell_key = f"{row_index}::{column}"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column=column,
            raw_value=f"unexpected-{row_index}",
            value_key=f"{column}::unexpected-{row_index}",
            archetype="geo_name",
        )
        observations[cell_key] = [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="conditional_disagreement",
            family="contextual",
            polarity="dirty",
            strength=0.9,
            hard=False,
            reason_code="context_conflict",
            metadata={
                "determinant_column": determinant,
                "dependent_column": dependent,
                "relation_type": "functional_dependency",
            },
        )]
    return registry, observations


def _discover_context(column="city", determinant="country", dependent="city", size=4):
    registry, observations = _context_fixture(column, determinant, dependent, size)
    report = discover_candidate_mechanisms(
        {}, registry, {}, observations, min_support=2
    )
    assert len(report.candidates) == 1
    return registry, observations, report


def test_same_context_mechanism_has_stable_id_across_fields():
    _, _, first = _discover_context("city", "country", "city")
    _, _, second = _discover_context("employer", "person", "employer")

    assert first.candidates[0].mechanism_id == second.candidates[0].mechanism_id
    assert first.candidates[0].columns == ["city"]
    assert second.candidates[0].columns == ["employer"]


def test_unknown_mechanism_keeps_discovered_identity():
    registry, observations = _context_fixture(size=3)
    metadata = {
        "city": {
            "semantics": {
                "possible_error_mechanisms": ["fixed_typo"],
            }
        }
    }
    report = discover_candidate_mechanisms(
        metadata, registry, {}, observations, min_support=2
    )

    candidate = report.candidates[0]
    assert candidate.novel
    assert candidate.status == "discovered"
    assert candidate.label == f"discovered:{candidate.mechanism_id}"
    assert candidate.label != "fixed_typo"


def test_cross_archetype_signature_is_validated_per_role():
    registry = {}
    observations = {}
    for row_index, archetype in enumerate(("open_entity", "closed_enum")):
        cell_key = f"{row_index}::value"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="value",
            raw_value="xvariant",
            value_key=f"value::xvariant::{row_index}",
            archetype=archetype,
        )
        observations[cell_key] = [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="repeated_character_corruption",
            family="canonicalization",
            polarity="dirty",
            strength=0.9,
            hard=False,
            reason_code="repeated_character_corruption",
            metadata={"discovery_signature": "repeated_character:x"},
        )]
    report = discover_candidate_mechanisms(
        {}, registry, {}, observations, min_support=1
    )

    assert len(report.candidates) == 2
    assert all(candidate.archetype_consistency == 1.0 for candidate in report.candidates)


def test_unvalidated_discovery_does_not_change_inference_matrix():
    registry, observations, report = _discover_context(size=3)
    before = build_cell_evidence_matrix(registry, {}, observations)

    annotated = annotate_discovered_observations(report, {}, observations)
    activation = enable_validated_mechanisms(report, {}, observations)
    after = build_cell_evidence_matrix(registry, {}, observations)

    assert annotated == 3
    assert activation["added"] == 0
    assert before.feature_names == after.feature_names
    np.testing.assert_allclose(before.values, after.values)


def test_query_only_discovery_evidence_is_available_to_sampling_only():
    registry, observations, report = _discover_context(size=3)
    for cell_items in observations.values():
        cell_items[0].metadata["exclude_from_inference"] = True
    before = build_cell_evidence_matrix(registry, {}, observations)
    assert before.values.shape[1] == 0
    assert before.query_values is not None
    assert before.query_values.shape[1] == 1
    assert len(select_active_queries(before, 1)) == 1


def test_query_allocation_gets_two_labels_without_growing_budget():
    _, _, report = _discover_context(size=4)
    selected, summary = allocate_mechanism_queries(
        ["base-0", "base-1", "base-2", "base-3"],
        budget=4,
        report=report,
        fraction=0.5,
        verification_samples=2,
        allow_replacement=True,
    )

    members = set(report.candidates[0].member_cells)
    assert len(selected) == 4
    assert len(members.intersection(selected)) == 2
    assert summary["covered_mechanisms"] == [report.candidates[0].label]


def test_query_allocation_does_not_replace_base_calibration_by_default():
    _, _, report = _discover_context(size=4)
    selected, summary = allocate_mechanism_queries(
        ["base-0", "base-1", "base-2", "base-3"],
        budget=4,
        report=report,
        fraction=0.5,
        verification_samples=2,
    )

    assert selected == ["base-0", "base-1", "base-2", "base-3"]
    assert summary["covered_mechanisms"] == []
    assert summary["replaced"] == []
    assert summary["skipped_replacement_required"] == [report.candidates[0].label]
    assert summary["replacement_enabled"] is False


def test_explicit_mechanism_budget_appends_without_touching_calibration():
    _, _, report = _discover_context(size=8)
    base = ["base-0", "base-1", "base-2", "base-3"]
    selected, summary = allocate_mechanism_queries(
        base,
        budget=9,
        report=report,
        verification_samples=5,
        max_candidates_per_round=1,
        mechanism_budget=5,
    )

    assert selected[:4] == base
    assert len(selected) == 9
    assert summary["reserved_slots"] == 5
    assert summary["mechanism_budget"] == 5
    assert summary["replaced"] == []
    assert len(summary["appended_without_replacement"]) == 5


def test_query_allocation_does_not_evict_protected_coverage():
    _, _, report = _discover_context(size=4)
    original = ["base-0", "base-1", "base-2", "base-3"]
    selected, summary = allocate_mechanism_queries(
        original,
        budget=4,
        report=report,
        protected_cell_keys={"base-0", "base-1", "base-2"},
        fraction=0.5,
        verification_samples=2,
    )

    assert selected == original
    assert summary["covered_mechanisms"] == []
    assert summary["protected_count"] == 3


def test_two_dirty_labels_validate_and_materialize_new_evidence():
    _, observations, report = _discover_context(size=3)
    member_cells = report.candidates[0].member_cells
    validate_mechanism_candidates(
        report,
        {member_cells[0]: 1, member_cells[1]: 1},
    )
    activation = enable_validated_mechanisms(report, {}, observations)

    candidate = report.candidates[0]
    assert candidate.status == "validated"
    assert candidate.label_purity == 1.0
    assert candidate.false_positive_risk == 0.0
    assert activation["added"] == 3
    assert all(
        any(item.source_id == candidate.label for item in cell_items)
        for cell_items in observations.values()
    )


def test_unallocated_overlap_does_not_promote_candidate():
    _, observations, report = _discover_context(size=3)
    member_cells = report.candidates[0].member_cells
    validate_mechanism_candidates(
        report,
        {member_cells[0]: 1, member_cells[1]: 1},
        eligible_mechanisms=set(),
    )
    assert report.candidates[0].status == "discovered"
    assert report.candidates[0].validation_reason == "not_allocated_this_round"
    assert enable_validated_mechanisms(report, {}, observations)["added"] == 0


def test_clean_counterexample_rejects_candidate():
    _, observations, report = _discover_context(size=3)
    member_cells = report.candidates[0].member_cells
    validate_mechanism_candidates(
        report,
        {member_cells[0]: 1, member_cells[1]: 0},
    )
    activation = enable_validated_mechanisms(report, {}, observations)

    assert report.candidates[0].status == "rejected"
    assert report.candidates[0].false_positive_risk == 0.5
    assert activation["added"] == 0


def test_repair_success_rate_uses_only_queried_corrected_values():
    registry = {}
    observations = {}
    repairs = {}
    for row_index in range(3):
        cell_key = f"{row_index}::amount"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="amount",
            raw_value="1O",
            value_key="amount::1O",
            archetype="numeric_measure",
        )
        observations[cell_key] = []
        repairs[cell_key] = [RepairCandidate(
            cell_key=cell_key,
            row_index=row_index,
            column="amount",
            mechanism="ocr_character_substitution",
            candidate_value="10",
            detection_confidence=0.9,
            repairability=0.9,
        )]
    report = discover_candidate_mechanisms(
        {}, registry, {}, observations, repairs, min_support=2
    )
    member_cells = report.candidates[0].member_cells
    validate_mechanism_candidates(
        report,
        {member_cells[0]: 1, member_cells[1]: 1},
        corrected_values={member_cells[0]: "10", member_cells[1]: "11"},
    )

    candidate = report.candidates[0]
    assert candidate.repair_labeled_support == 2
    assert candidate.repair_success_rate == 0.5
    assert candidate.repair_validated
    assert candidate.status == "validated"


def test_failed_repair_does_not_reject_valid_detection_mechanism():
    registry = {}
    repairs = {}
    for row_index in range(3):
        cell_key = f"{row_index}::amount"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="amount",
            raw_value="1O",
            value_key="amount::1O",
            archetype="numeric_measure",
        )
        repairs[cell_key] = [RepairCandidate(
            cell_key=cell_key,
            row_index=row_index,
            column="amount",
            mechanism="ocr_character_substitution",
            candidate_value="10",
            detection_confidence=0.9,
            repairability=0.9,
        )]
    report = discover_candidate_mechanisms(
        {}, registry, {}, {}, repairs, min_support=2
    )
    member_cells = report.candidates[0].member_cells
    validate_mechanism_candidates(
        report,
        {member_cells[0]: 1, member_cells[1]: 1},
        corrected_values={member_cells[0]: "11", member_cells[1]: "12"},
    )

    candidate = report.candidates[0]
    assert candidate.status == "validated"
    assert not candidate.repair_validated
    assert candidate.repair_validation_reason == "corrected_value_validation_failed"


def test_deterministic_surface_repairs_are_query_only_and_auditable():
    values = {
        "0::text": "A description...",
        "1::text": "Alice,Alice",
        "2::text": "one  two  three",
        "3::text": "one  intentional",
    }
    registry = {
        key: CellRecord(
            key=key,
            row_index=int(key.split("::", 1)[0]),
            column="text",
            raw_value=value,
            value_key=f"text::{value}",
            archetype="free_text",
        )
        for key, value in values.items()
    }

    repairs, counts = _infer_deterministic_surface_repairs(registry)

    assert counts == {
        "excess_internal_whitespace": 2,
        "repeated_list_segment": 1,
        "repeated_terminal_punctuation": 1,
    }
    assert "3::text" not in repairs
    assert repairs["0::text"][0].candidate_value == "A description."
    assert repairs["1::text"][0].candidate_value == "Alice"
    assert {
        item.candidate_value for item in repairs["2::text"]
    } == {"one two  three", "one  two three"}
    assert all(
        item.metadata.get("exclude_from_inference") is True
        for items in repairs.values()
        for item in items
    )


def test_specific_repair_candidate_gets_query_priority():
    registry, observations = _context_fixture(size=8)
    repairs = {}
    for row_index in range(3):
        cell_key = f"{row_index}::summary"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="summary",
            raw_value="summary...",
            value_key="summary::summary...",
            archetype="free_text",
        )
        repairs[cell_key] = [RepairCandidate(
            cell_key=cell_key,
            row_index=row_index,
            column="summary",
            mechanism="repeated_terminal_punctuation",
            candidate_value="summary.",
            detection_confidence=0.98,
            repairability=0.99,
            metadata={
                "exclude_from_inference": True,
                "discovery_signature": "repair:repeated_terminal_punctuation",
            },
        )]
    report = discover_candidate_mechanisms(
        {}, registry, {}, observations, repairs, min_support=2
    )

    selected, summary = allocate_mechanism_queries(
        ["base-0", "base-1"],
        budget=2,
        report=report,
        fraction=0.5,
        verification_samples=1,
        max_candidates_per_round=1,
        allow_replacement=True,
    )

    repair_candidate = next(
        item for item in report.candidates
        if "repeated_terminal_punctuation" in item.signature
    )
    assert summary["covered_mechanisms"] == [repair_candidate.label]
    assert set(selected).intersection(repair_candidate.member_cells)


def test_multiple_explicit_mechanisms_on_one_cell_keep_separate_identities():
    registry = {}
    observations = {}
    repairs = {}
    for row_index in range(3):
        cell_key = f"{row_index}::text"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="text",
            raw_value="xvalue...",
            value_key="text::xvalue...",
            archetype="free_text",
        )
        observations[cell_key] = [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="repeated_character_corruption",
            family="canonicalization",
            polarity="dirty",
            strength=0.9,
            hard=False,
            reason_code="repeated_character_corruption",
            metadata={"discovery_signature": "repeated_character:x"},
        )]
        repairs[cell_key] = [RepairCandidate(
            cell_key=cell_key,
            row_index=row_index,
            column="text",
            mechanism="repeated_terminal_punctuation",
            candidate_value="xvalue.",
            detection_confidence=0.98,
            repairability=0.99,
            metadata={
                "exclude_from_inference": True,
                "discovery_signature": "repair:repeated_terminal_punctuation",
            },
        )]

    report = discover_candidate_mechanisms(
        {}, registry, {}, observations, repairs, min_support=2
    )
    signatures = {item.signature for item in report.candidates}

    assert len(report.candidates) == 2
    assert any("repeated_character:x" in item for item in signatures)
    assert any("repeated_terminal_punctuation" in item for item in signatures)
    assert all("+" not in item for item in signatures)
