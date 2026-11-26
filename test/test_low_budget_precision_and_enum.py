import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

from agent import BaseAgent, ColumnArchetypeAgent
from cleanem_inference import (
    build_cell_evidence_matrix,
    estimate_calibrated_posteriors,
    identify_unvalidated_broad_features,
    select_active_queries,
)
from cleanem_models import CellRecord, EvidenceObservation
from main import (
    _build_corrected_value_standardization_candidates,
    _ensure_core_mechanism_query_coverage,
    _ensure_structural_missing_query_coverage,
    _gate_observation,
    _has_pagination_contamination,
    _has_trusted_numeric_contract,
    _infer_percent_decimal_contract,
    _is_full_string_numeric,
    _infer_closed_enum_normalizations,
    _surface_role_conflict,
    _unicode_corruption_kind,
    _geo_suffix_shift_observations,
)
from profiler import PandasProfiler


def _metadata(archetype):
    return {"semantics": {"archetype": archetype, "confidence": 0.9}}


def test_llm_context_error_retries_with_derived_output_budget():
    requested_budgets = []

    class FakeCompletions:
        def create(self, **kwargs):
            requested_budgets.append(kwargs["max_tokens"])
            if len(requested_budgets) == 1:
                raise RuntimeError(
                    "This model's maximum context length is 8192 tokens. "
                    "However, you requested 2048 output tokens and your prompt "
                    "contains at least 6145 input tokens, for a total of at "
                    "least 8193 tokens."
                )
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok": true}')
            )])

    agent = object.__new__(BaseAgent)
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    agent.model = "test-model"
    agent.last_prompt = None
    agent.last_response = None
    agent.last_system_prompt = None

    assert agent._call_llm("prompt", max_tokens=2048) == '{"ok": true}'
    assert requested_budgets == [2048, 1024]


def test_field_roles_override_numeric_storage_type():
    agent = object.__new__(ColumnArchetypeAgent)
    profile = {
        "type": "numeric",
        "unique_ratio": 0.2,
        "top_value_coverage": 0.2,
    }
    assert agent._heuristic_fallback("Rank", profile).archetype == "identifier"
    assert agent._heuristic_fallback("Company Founded", profile).archetype == "temporal_measure"


def test_geo_suffix_shift_is_name_agnostic_and_requires_reference_support():
    frame = pd.DataFrame({"locality": ["North Point CA", "Plainville", "Support", "Other"], "region_code": [None, "TX", "CA", None]})
    metadata = {
        "locality": _metadata("geo_name"),
        "region_code": {"semantics": {"archetype": "closed_enum"}},
    }
    registry = {
        f"{row}::{column}": CellRecord(
            key=f"{row}::{column}", row_index=row, column=column,
            raw_value=frame.loc[row, column], value_key=f"{column}::{frame.loc[row, column]}"
        )
        for row in range(len(frame)) for column in frame.columns
    }
    observations = {}
    summary = _geo_suffix_shift_observations(frame, metadata, registry, observations)
    assert summary["emitted_cells"] == 2
    assert observations["0::locality"][0].source_id == "geo_suffix_pollution"
    assert observations["0::region_code"][0].source_id == "displaced_region_missing"
    assert "1::locality" not in observations


def test_open_taxonomy_profile_overrides_closed_enum_prediction():
    agent = object.__new__(ColumnArchetypeAgent)
    profile = {
        "type": "text",
        "non_null_count": 2541,
        "unique_count": 564,
        "unique_ratio": 0.222,
        "singleton_ratio": 0.539,
        "top_value_coverage": 0.344,
    }
    heuristic = agent._heuristic_fallback("Company Sector", profile)
    assert heuristic.archetype == "open_entity"
    assert agent._reconcile_archetype(
        "Company Sector", profile, "closed_enum", heuristic
    ) == (
        "open_entity",
        "profile_override:closed_enum->open_entity",
    )


def test_concentrated_category_profile_remains_closed_enum():
    agent = object.__new__(ColumnArchetypeAgent)
    profile = {
        "type": "categorical",
        "non_null_count": 2514,
        "unique_count": 148,
        "unique_ratio": 0.059,
        "singleton_ratio": 0.703,
        "top_value_coverage": 0.910,
    }
    heuristic = agent._heuristic_fallback("Company Relationship", profile)
    assert heuristic.archetype == "closed_enum"
    assert agent._reconcile_archetype(
        "Company Relationship", profile, "closed_enum", heuristic
    ) == ("closed_enum", None)


def test_surface_role_contract_detects_numeric_and_shape_conflicts():
    closed_enum = {
        "type": "categorical",
        "non_null_count": 1000,
        "full_numeric_surface_ratio": 0.04,
        "alphabetic_surface_ratio": 0.94,
        "semantics": {"archetype": "closed_enum"},
        "normalized_top_values": [
            {"value": "financial", "ratio": 0.90},
        ],
        "length_distribution": [{"length": 1, "ratio": 0.02}],
    }
    assert _surface_role_conflict("-7", closed_enum)["source_id"] == "type_role_conflict"
    assert _surface_role_conflict("22222222", closed_enum)["metadata"]["numeric_shape"] == "repeated_digits"
    assert _surface_role_conflict("19991231", closed_enum)["metadata"]["numeric_shape"] == "compact_date"
    assert _surface_role_conflict("x", closed_enum)["source_id"] == "surface_shape_conflict"

    open_entity = {
        "type": "text",
        "non_null_count": 1000,
        "full_numeric_surface_ratio": 0.02,
        "alphabetic_surface_ratio": 0.96,
        "semantics": {"archetype": "open_entity"},
        "normalized_top_values": [],
        "length_distribution": [{"length": 1, "ratio": 0.02}],
    }
    assert _surface_role_conflict("42", open_entity)["source_id"] == "type_role_conflict"
    assert _surface_role_conflict("x", open_entity)["source_id"] == "surface_shape_conflict"


def test_surface_role_contract_abstains_from_legitimate_numeric_roles():
    for archetype in (
        "identifier",
        "numeric_measure",
        "unit_measure",
        "temporal_measure",
    ):
        metadata = {
            "type": "categorical",
            "non_null_count": 1000,
            "full_numeric_surface_ratio": 0.02,
            "alphabetic_surface_ratio": 0.96,
            "semantics": {"archetype": archetype},
        }
        assert _surface_role_conflict("-7", metadata) is None
        assert _surface_role_conflict("19991231", metadata) is None

    closed_enum = {
        "type": "categorical",
        "non_null_count": 1000,
        "full_numeric_surface_ratio": 0.08,
        "alphabetic_surface_ratio": 0.85,
        "semantics": {"archetype": "closed_enum"},
        "normalized_top_values": [
            {"value": "0", "ratio": 0.04},
        ],
        "length_distribution": [{"length": 1, "ratio": 0.08}],
    }
    assert _surface_role_conflict("0", closed_enum) is None
    assert _surface_role_conflict("y", closed_enum) is None


def test_direct_decision_evidence_is_excluded_from_calibration_matrix():
    registry = {
        "0::sector": CellRecord(
            key="0::sector",
            row_index=0,
            column="sector",
            raw_value="-1",
            value_key="sector::-1",
        ),
        "1::sector": CellRecord(
            key="1::sector",
            row_index=1,
            column="sector",
            raw_value=None,
            value_key="sector::<na>",
        ),
    }
    direct = EvidenceObservation(
        target_scope="cell",
        target_key="0::sector",
        source_id="type_role_conflict",
        family="outlier",
        polarity="dirty",
        strength=1.0,
        hard=True,
        reason_code="numeric_surface_role_conflict",
        metadata={"calibration_role": "direct_decision"},
    )
    calibratable = EvidenceObservation(
        target_scope="cell",
        target_key="1::sector",
        source_id="missing_token",
        family="missing",
        polarity="dirty",
        strength=1.0,
        hard=True,
        reason_code="missing_token",
    )
    matrix = build_cell_evidence_matrix(
        registry,
        {},
        {"0::sector": [direct], "1::sector": [calibratable]},
    )
    assert matrix.cell_keys == ["0::sector", "1::sector"]
    assert matrix.feature_names == ["cell:missing:missing_token"]
    assert matrix.values.tolist() == [[0.0], [1.0]]


def test_passive_evidence_does_not_drive_active_query_design():
    registry = {
        "0::role": CellRecord(
            key="0::role",
            row_index=0,
            column="role",
            raw_value="42",
            value_key="role::42",
        ),
        "1::role": CellRecord(
            key="1::role",
            row_index=1,
            column="role",
            raw_value=None,
            value_key="role::<na>",
        ),
    }
    passive = EvidenceObservation(
        target_scope="cell",
        target_key="0::role",
        source_id="type_role_conflict",
        family="outlier",
        polarity="dirty",
        strength=0.9,
        hard=False,
        reason_code="numeric_surface_role_conflict",
        metadata={"exclude_from_query_design": True},
    )
    active = EvidenceObservation(
        target_scope="cell",
        target_key="1::role",
        source_id="missing_token",
        family="missing",
        polarity="dirty",
        strength=0.2,
        hard=False,
        reason_code="structural_missing_token",
    )
    matrix = build_cell_evidence_matrix(
        registry,
        {},
        {"0::role": [passive], "1::role": [active]},
    )
    assert matrix.values.shape == (2, 2)
    assert matrix.query_excluded_features == {
        "cell:outlier:type_role_conflict"
    }
    assert select_active_queries(matrix, 1) == ["1::role"]


def test_hard_numeric_contract_requires_scalar_measurement_semantics():
    page_metadata = {
        "type": "numeric",
        "non_null_count": 892,
        "numeric_count": 889,
        "fully_numeric_count": 21,
        "native_numeric_dtype": False,
        "semantics": {"archetype": "identifier"},
    }
    assert not _has_trusted_numeric_contract(page_metadata)

    date_metadata = {
        "type": "numeric",
        "non_null_count": 913,
        "numeric_count": 913,
        "fully_numeric_count": 0,
        "native_numeric_dtype": False,
        "semantics": {"archetype": "temporal_measure"},
    }
    assert not _has_trusted_numeric_contract(date_metadata)

    measure_metadata = {
        "type": "numeric",
        "non_null_count": 100,
        "fully_numeric_count": 98,
        "native_numeric_dtype": False,
        "semantics": {"archetype": "numeric_measure"},
    }
    assert _has_trusted_numeric_contract(measure_metadata)

    native_id_metadata = {
        "type": "numeric",
        "native_numeric_dtype": True,
        "semantics": {"archetype": "identifier"},
    }
    assert _has_trusted_numeric_contract(native_id_metadata)


def test_profiler_distinguishes_embedded_from_fully_numeric_values(tmp_path):
    data_path = tmp_path / "page_ranges.csv"
    pd.DataFrame({"pages": ["79-105", "S40", "25"]}).to_csv(
        data_path, index=False
    )
    metadata = PandasProfiler(str(data_path)).get_metadata()["pages"]
    assert metadata["numeric_count"] == 3
    assert metadata["fully_numeric_count"] == 1
    assert metadata["native_numeric_dtype"] is False


def test_profiler_emits_surface_type_contract_for_text_columns(tmp_path):
    data_path = tmp_path / "surface_roles.csv"
    pd.DataFrame({"role": ["alpha", "beta", "0", "-7"]}).to_csv(
        data_path, index=False
    )
    metadata = PandasProfiler(str(data_path)).get_metadata()["role"]
    assert metadata["full_numeric_surface_count"] == 2
    assert metadata["full_numeric_surface_ratio"] == 0.5
    assert metadata["alphabetic_surface_ratio"] == 0.5


def test_pagination_contamination_detector_is_narrow_and_semantic():
    metadata = {"semantics": {"archetype": "identifier"}}
    for value in (
        "27-32 ST  - Wound care-",
        "51-54 4p",
        "98--100",
        "728-730-",
    ):
        assert _has_pagination_contamination(
            "article_pagination", value, metadata
        )

    for value in (
        "79-105", "S40", "S331-", "iii-vii", "233-43; discussion 243",
    ):
        assert not _has_pagination_contamination(
            "article_pagination", value, metadata
        )
    assert not _has_pagination_contamination(
        "article_title", "27-32 ST  - Wound care-", metadata
    )


def test_percent_decimal_contract_requires_dominant_fraction_scale():
    metadata = _metadata("numeric_measure")
    contract = _infer_percent_decimal_contract(
        pd.Series(["0.05"] * 15 + ["0.13%"] * 5),
        metadata,
    )
    assert contract is not None
    assert contract["bare_ratio"] == 0.75
    assert contract["percent_count"] == 5

    assert _infer_percent_decimal_contract(
        pd.Series(["5%"] * 20), metadata
    ) is None
    assert _infer_percent_decimal_contract(
        pd.Series(["12"] * 15 + ["13%"] * 5), metadata
    ) is None
    assert _infer_percent_decimal_contract(
        pd.Series(["0.05"] * 15 + ["0.13%"] * 5),
        _metadata("identifier"),
    ) is None


def test_full_string_numeric_parse_excludes_extracted_payloads():
    for value in (0.13, "0.13", ".13", "1.", "1,250.5", "-2.5e-3"):
        assert _is_full_string_numeric(value)
    for value in ("0.13%", "$0.13", "12 oz", "(0.13)"):
        assert not _is_full_string_numeric(value)


def test_unicode_corruption_distinguishes_replacement_and_normal_accents():
    assert _unicode_corruption_kind("J\ufffdrgen") == "replacement_character"
    assert _unicode_corruption_kind("Ceki\u0324") == "unresolved_combining_mark"
    assert _unicode_corruption_kind("Jos\u00e9") is None
    assert _unicode_corruption_kind("Jose\u0301") is None


def test_core_mechanism_coverage_replaces_within_fixed_budget():
    values = [
        "missing", "other", "1/1/14", "J\ufffdrgen", "Ceki\u0324", "spare"
    ]
    columns = ["title", "id", "created_at", "author", "author", "title"]
    registry = {
        f"{row}::{column}": CellRecord(
            key=f"{row}::{column}",
            row_index=row,
            column=column,
            raw_value=value,
            value_key=f"{column}::{value}",
        )
        for row, (column, value) in enumerate(zip(columns, values))
    }
    selected, summary = _ensure_core_mechanism_query_coverage(
        ["0::title", "1::id", "5::title"],
        3,
        {
            "title": _metadata("free_text"),
            "id": _metadata("identifier"),
            "created_at": _metadata("temporal_measure"),
            "author": _metadata("open_entity"),
        },
        registry,
        {},
    )
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert summary["inserted"] == [
        "2::created_at", "4::author", "3::author"
    ]
    assert [item["kind"] for item in summary["unicode_coverage"]] == [
        "unresolved_combining_mark", "replacement_character"
    ]
    assert len(summary["replaced"]) == 3


def test_structural_missing_coverage_replaces_low_priority_within_budget():
    columns = ["src", "flight", "act_arr_time", "act_arr_time"]
    values = ["iad", "AA-1-A-B", "6:13 p.m.", None]
    registry = {
        f"{row}::{column}": CellRecord(
            key=f"{row}::{column}",
            row_index=row,
            column=column,
            raw_value=value,
            value_key=f"{column}::{value}",
        )
        for row, (column, value) in enumerate(zip(columns, values))
    }
    structural = EvidenceObservation(
        target_scope="cell",
        target_key="3::act_arr_time",
        source_id="missing_token",
        family="missing",
        polarity="dirty",
        strength=0.25,
        hard=False,
        reason_code="structural_missing_token",
    )
    selected, summary = _ensure_structural_missing_query_coverage(
        ["0::src", "1::flight", "2::act_arr_time"],
        3,
        {"3::act_arr_time": [structural]},
        registry,
        {
            "src": _metadata("closed_enum"),
            "flight": _metadata("identifier"),
            "act_arr_time": _metadata("temporal_measure"),
        },
    )

    assert selected == ["0::src", "1::flight", "3::act_arr_time"]
    assert len(selected) == 3
    assert summary["inserted"] == ["3::act_arr_time"]
    assert summary["replaced"] == ["2::act_arr_time"]

    covered, covered_summary = _ensure_structural_missing_query_coverage(
        selected,
        3,
        {"3::act_arr_time": [structural]},
        registry,
        {
            "src": _metadata("closed_enum"),
            "flight": _metadata("identifier"),
            "act_arr_time": _metadata("temporal_measure"),
        },
    )
    assert covered == selected
    assert covered_summary["inserted"] == []
    assert covered_summary["replaced"] == []


def test_labeled_temporal_component_mapping_propagates_exact_values():
    dirty = pd.DataFrame({"created_at": ["1/1/14", "1/1/14", "2/3/04"]})
    clean = pd.DataFrame({"created_at": ["1/14/01", "1/14/01", "2/3/04"]})
    registry = {
        f"{row}::created_at": CellRecord(
            key=f"{row}::created_at",
            row_index=row,
            column="created_at",
            raw_value=dirty.loc[row, "created_at"],
            value_key=f"created_at::{dirty.loc[row, 'created_at']}",
        )
        for row in range(len(dirty))
    }
    observations = {}
    _, summary = _build_corrected_value_standardization_candidates(
        dirty,
        clean,
        registry,
        ["0::created_at"],
        {"0::created_at": 1},
        detection_observations=observations,
        metadata={
            "created_at": {"semantics": {"archetype": "temporal_measure"}}
        },
    )
    assert summary["component_permutation_contracts"] == 1
    assert summary["component_permutation_detection"] == 2
    assert all(
        items[0].source_id == "labeled_component_permutation"
        and items[0].hard
        for items in observations.values()
    )


def test_closed_enum_pattern_guard_requires_same_column_support():
    observation = EvidenceObservation(
        target_scope="cell",
        target_key="1::sector",
        source_id="pattern_mismatch",
        family="pattern",
        polarity="dirty",
        strength=1.0,
        hard=False,
        reason_code="pattern_mismatch",
    )
    gated, _ = _gate_observation(
        observation,
        "sector",
        {"sector": _metadata("closed_enum")},
        policy_support={
            ("closed_enum", "pattern_mismatch"): {"clean": 3, "dirty": 3},
        },
        gating_mode="weak_prior",
    )
    assert gated.strength == 0.25
    assert gated.metadata["gate_reason"] == "column_closed_enum_pattern_precision_guard"


def test_numeric_rarity_guard_requires_same_column_support():
    observation = EvidenceObservation(
        target_scope="value",
        target_key="rank::100",
        source_id="rarity_high",
        family="outlier",
        polarity="dirty",
        strength=0.8,
        hard=False,
        reason_code="numeric_rarity_high",
    )
    gated, _ = _gate_observation(
        observation,
        "rank",
        {"rank": _metadata("numeric_measure")},
        policy_support={},
        gating_mode="weak_prior",
    )
    assert gated.strength == 0.2
    assert gated.metadata["gate_reason"] == "column_numeric_rarity_precision_guard"


def _broad_feature_fixture(archetype, family, source_id, trigger_count):
    registry = {
        f"{row}::field": CellRecord(
            key=f"{row}::field",
            row_index=row,
            column="field",
            raw_value=f"value-{row}",
            value_key=f"field::value-{row}",
            archetype=archetype,
        )
        for row in range(10)
    }
    observations = {
        f"{row}::field": [EvidenceObservation(
            target_scope="cell",
            target_key=f"{row}::field",
            source_id=source_id,
            family=family,
            polarity="dirty",
            strength=1.0,
            hard=False,
            reason_code="test_trigger",
        )]
        for row in range(trigger_count)
    }
    return registry, build_cell_evidence_matrix(registry, {}, observations)


def test_broad_unstructured_feature_requires_clean_counterexample():
    registry, matrix = _broad_feature_fixture(
        "free_text", "pattern", "pattern_mismatch", 9
    )
    feature_name = matrix.feature_names[0]
    report = identify_unvalidated_broad_features(
        matrix, registry, {"0::field": 1}
    )
    assert report[feature_name]["reason"] == (
        "broad_unstructured_without_clean_counterexample"
    )
    assert report[feature_name]["max_column_coverage"] == 0.9

    validated = identify_unvalidated_broad_features(
        matrix,
        registry,
        {"0::field": 1, "1::field": 0},
    )
    assert feature_name not in validated


def test_unvalidated_repair_and_enum_distribution_are_guarded():
    repair_registry, repair_matrix = _broad_feature_fixture(
        "open_entity", "canonicalization", "surface_variant", 2
    )
    repair_feature = repair_matrix.feature_names[0]
    repair_report = identify_unvalidated_broad_features(
        repair_matrix, repair_registry, {}
    )
    assert repair_report[repair_feature]["reason"] == (
        "broad_repair_candidate_without_clean_counterexample"
    )

    enum_registry, enum_matrix = _broad_feature_fixture(
        "closed_enum", "prototype_lexical", "prototype_far", 2
    )
    enum_feature = enum_matrix.feature_names[0]
    enum_report = identify_unvalidated_broad_features(
        enum_matrix, enum_registry, {"0::field": 1}
    )
    assert enum_report[enum_feature]["reason"] == (
        "broad_enum_distribution_without_clean_counterexample"
    )


def test_structured_broad_feature_is_not_guarded_and_weight_cap_is_applied():
    registry, matrix = _broad_feature_fixture(
        "temporal_measure", "pattern", "pattern_mismatch", 10
    )
    feature_name = matrix.feature_names[0]
    assert identify_unvalidated_broad_features(
        matrix, registry, {"0::field": 1}
    ) == {}

    _, _, trace = estimate_calibrated_posteriors(
        {},
        registry,
        matrix,
        ["0::field"],
        {"0::field": 1},
        0.1,
        feature_weight_caps={feature_name: 0.0},
    )
    assert feature_name not in trace.feature_weights
    assert "broad_feature_guard" in trace.reason


def test_active_query_class_balance_cannot_make_baseline_dirty():
    registry = {}
    observations = {}
    selected = []
    labels = {}
    for row_index in range(10):
        cell_key = f"{row_index}::field"
        registry[cell_key] = CellRecord(
            key=cell_key,
            row_index=row_index,
            column="field",
            raw_value=f"value-{row_index}",
            value_key=f"field::value-{row_index}",
            archetype="open_entity",
        )
        observations[cell_key] = [EvidenceObservation(
            target_scope="cell",
            target_key=cell_key,
            source_id="candidate_signal",
            family="discovered_mechanism",
            polarity="dirty",
            strength=1.0 if row_index < 9 else 0.1,
            hard=False,
            reason_code="candidate_signal",
        )]
        selected.append(cell_key)
        labels[cell_key] = 1 if row_index < 9 else 0
    matrix = build_cell_evidence_matrix(registry, {}, observations)

    _, _, trace = estimate_calibrated_posteriors(
        {}, registry, matrix, selected, labels, 0.05
    )

    assert trace.intercept <= np.log(0.30 / 0.70) + 1e-9
    assert "capped_intercept" in trace.reason


def test_validated_feature_floor_survives_single_label_shrinkage():
    registry = {
        f"{row}::field": CellRecord(
            key=f"{row}::field",
            row_index=row,
            column="field",
            raw_value=f"value-{row}",
            value_key=f"field::value-{row}",
            archetype="open_entity",
        )
        for row in range(2)
    }
    observations = {
        "0::field": [EvidenceObservation(
            target_scope="cell",
            target_key="0::field",
            source_id="discovered:test",
            family="discovered_mechanism",
            polarity="dirty",
            strength=1.0,
            hard=False,
            reason_code="validated_discovered_mechanism",
        )]
    }
    matrix = build_cell_evidence_matrix(registry, {}, observations)
    feature_name = matrix.feature_names[0]

    _, posteriors, trace = estimate_calibrated_posteriors(
        {},
        registry,
        matrix,
        ["0::field"],
        {"0::field": 1},
        0.05,
        feature_weight_floors={feature_name: np.log(4.0)},
    )

    assert trace.feature_weights[feature_name] >= np.log(4.0)
    assert posteriors["0::field"]["posterior"] > 0.5
    assert "validated_feature_floor" in trace.reason


def test_closed_enum_normalization_is_directional_and_conservative():
    frame = pd.DataFrame({
        "company_type": ["new"] * 20
        + ["neŊw"] * 2
        + ["aquired"] * 5
        + ["acquired"] * 5,
        "flag": ["TRUE"] * 30 + ["EURT"] * 2,
    })
    metadata = {
        "company_type": _metadata("closed_enum"),
        "flag": _metadata("closed_enum"),
    }
    proposals = _infer_closed_enum_normalizations(
        frame,
        metadata,
        lexicon={"new", "acquired", "true"},
    )
    assert proposals["company_type"]["neŋw"]["candidate_value"] == "new"
    assert proposals["company_type"]["aquired"]["candidate_value"] == "acquired"
    assert proposals["flag"]["eurt"]["candidate_value"] == "true"

    without_dictionary = _infer_closed_enum_normalizations(
        frame,
        metadata,
        lexicon=set(),
    )
    assert "aquired" not in without_dictionary["company_type"]


def test_closed_enum_normalization_abstains_from_reversed_short_codes():
    frame = pd.DataFrame({"state": ["ND"] * 20 + ["DN"] * 5})
    proposals = _infer_closed_enum_normalizations(
        frame,
        {"state": _metadata("closed_enum")},
        lexicon=set(),
    )
    assert "state" not in proposals
