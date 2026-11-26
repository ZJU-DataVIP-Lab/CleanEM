import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from cleanem_models import CellRecord, EvidenceContribution, EvidenceObservation, ValueRecord


@dataclass
class EvidenceMatrix:
    """Dense feature view over per-cell weak evidence."""

    cell_keys: List[str]
    feature_names: List[str]
    values: np.ndarray
    key_to_index: Dict[str, int]
    query_excluded_features: Set[str] = field(default_factory=set)
    # Discovery evidence may be useful for choosing an oracle query before it
    # has been trusted for inference. Keep that signal in a separate view so
    # it can influence sampling without changing calibrated posteriors.
    query_feature_names: List[str] = field(default_factory=list)
    query_values: Optional[np.ndarray] = None
    feature_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CalibrationTrace:
    """Diagnostics for active-learning calibration."""

    selected_cell_keys: List[str] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    selected_labels: Dict[str, int] = field(default_factory=dict)
    queried_cells: List[Dict[str, object]] = field(default_factory=list)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    fitted: bool = False
    reason: str = ""
    feature_count: int = 0
    label_count: int = 0


def _clip_probability(value: float, low: float = 1e-4, high: float = 1 - 1e-4) -> float:
    return float(min(high, max(low, value)))


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return float(math.log(p / (1.0 - p)))


def _sigmoid_array(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits.astype(float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _sigmoid_scalar(logit: float) -> float:
    return float(_sigmoid_array(np.asarray([logit], dtype=float))[0])


def _prediction_confidence(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    return np.clip(1.0 - entropy, 0.0, 1.0)


def _signed_strength(observation: EvidenceObservation) -> float:
    if observation.polarity == "dirty":
        return float(observation.strength)
    if observation.polarity == "clean":
        return -float(observation.strength)
    return 0.0


_ARCHETYPE_NAMESPACED_FAMILIES = {
    # These families encode contracts whose meaning depends on the semantic
    # role of a column. Missingness remains global because its contract is
    # intentionally shared across roles.
    "outlier",
    "pattern",
    "prototype_lexical",
    "relationship",
    "contextual",
}


def _default_feature_namespace(record: CellRecord, observation: EvidenceObservation) -> str:
    """Choose the narrowest safe namespace for an evidence observation.

    Agent-generated pattern/prototype rules describe a column's observed
    contract, so their calibration must remain column-local. Explicitly
    identified cross-column mechanisms such as Unicode artifacts can share a
    semantic archetype prior. Other archetype-dependent evidence uses the
    broader role namespace for statistical strength.
    """
    archetype = str(record.archetype or "").strip()
    source = str(observation.source_id or "").lower()
    if observation.family in {"pattern", "prototype_lexical"}:
        if source.startswith("unicode_"):
            return f"archetype:{archetype}" if archetype else ""
        # Only Agent-generated rule outcomes describe a column-local contract.
        # Generic parser/regex/prototype evidence must share the archetype
        # namespace; splitting those sources by column makes sparse columns
        # uncalibratable and can change active-query composition drastically.
        if (
            observation.family == "pattern"
            and observation.metadata.get("rule_pool_source")
            == "agentic_clean_rule_pool"
            and record.column
        ):
            if archetype:
                return f"column:{record.column}:archetype:{archetype}"
            return f"column:{record.column}"
    return f"archetype:{archetype}" if archetype else ""


def _feature_name(
    scope: str,
    observation: EvidenceObservation,
    default_namespace: str = "",
) -> str:
    base = f"{scope}:{observation.family}:{observation.source_id}"
    metadata = observation.metadata or {}
    namespace = str(metadata.get("feature_namespace") or "").strip()
    if not namespace and observation.family in _ARCHETYPE_NAMESPACED_FAMILIES:
        namespace = str(default_namespace or "").strip()
    return f"{base}:{namespace}" if namespace else base


def build_cell_evidence_matrix(
    cell_registry: Dict[str, CellRecord],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> EvidenceMatrix:
    """Convert value-level and cell-level observations into signed evidence features.

    A positive feature value means the source emitted dirty evidence; a negative
    value means the same source emitted clean evidence. The model learns whether
    that signed source is reliable for the current dataset.
    """

    cell_keys = list(cell_registry.keys())
    feature_index: Dict[str, int] = {}
    query_excluded_features: Set[str] = set()
    sparse_rows: List[Dict[int, float]] = []
    query_feature_index: Dict[str, int] = {}
    sparse_query_rows: List[Dict[int, float]] = []
    feature_metadata: Dict[str, Dict[str, Any]] = {}

    for cell_key in cell_keys:
        record = cell_registry[cell_key]
        row_features: Dict[int, float] = {}
        observations = []
        observations.extend(value_observations.get(record.value_key, []))
        observations.extend(cell_observations.get(cell_key, []))

        query_row_features: Dict[int, float] = {}
        for observation in observations:
            # Deterministic contracts may directly decide their target without
            # teaching a dataset-wide source weight. Other weak evidence on
            # the same cell remains available to active query selection.
            excluded_from_inference = (
                observation.metadata.get("calibration_role") == "direct_decision"
                or observation.metadata.get("exclude_from_inference")
            )
            if excluded_from_inference:
                # Preserve explicitly discovered but unvalidated mechanisms for
                # query design only. A caller can suppress even this use with
                # exclude_from_query_design, which is important for broad or
                # unstable detectors.
                if observation.metadata.get("exclude_from_query_design"):
                    continue
                signed = _signed_strength(observation)
                if signed == 0.0:
                    continue
                name = _feature_name(
                    observation.target_scope,
                    observation,
                    _default_feature_namespace(record, observation),
                )
                idx = query_feature_index.setdefault(name, len(query_feature_index))
                query_row_features[idx] = query_row_features.get(idx, 0.0) + signed
                continue
            signed = _signed_strength(observation)
            if signed == 0.0:
                continue
            name = _feature_name(
                observation.target_scope,
                observation,
                _default_feature_namespace(record, observation),
            )
            descriptor = feature_metadata.setdefault(name, {
                "family": observation.family,
                "source_id": observation.source_id,
                "archetypes": set(),
                "columns": set(),
                "has_hard_evidence": False,
            })
            if record.archetype:
                descriptor["archetypes"].add(str(record.archetype))
            if record.column:
                descriptor["columns"].add(str(record.column))
            descriptor["has_hard_evidence"] = bool(
                descriptor["has_hard_evidence"] or observation.hard
            )
            if observation.metadata.get("exclude_from_query_design"):
                query_excluded_features.add(name)
            idx = feature_index.setdefault(name, len(feature_index))
            row_features[idx] = row_features.get(idx, 0.0) + signed
        sparse_rows.append(row_features)
        sparse_query_rows.append(query_row_features)

    values = np.zeros((len(cell_keys), len(feature_index)), dtype=float)
    for row_idx, row_features in enumerate(sparse_rows):
        for feature_idx, feature_value in row_features.items():
            values[row_idx, feature_idx] = feature_value

    feature_names = [None] * len(feature_index)
    for name, idx in feature_index.items():
        feature_names[idx] = name

    query_values = np.zeros((len(cell_keys), len(query_feature_index)), dtype=float)
    for row_idx, row_features in enumerate(sparse_query_rows):
        for feature_idx, feature_value in row_features.items():
            query_values[row_idx, feature_idx] = feature_value
    query_feature_names = [None] * len(query_feature_index)
    for name, idx in query_feature_index.items():
        query_feature_names[idx] = name

    return EvidenceMatrix(
        cell_keys=cell_keys,
        feature_names=[str(name) for name in feature_names],
        values=values,
        key_to_index={key: idx for idx, key in enumerate(cell_keys)},
        query_excluded_features=query_excluded_features,
        query_feature_names=[str(name) for name in query_feature_names],
        query_values=query_values,
        feature_metadata={
            name: {
                **descriptor,
                "archetypes": sorted(descriptor["archetypes"]),
                "columns": sorted(descriptor["columns"]),
            }
            for name, descriptor in feature_metadata.items()
        },
    )


def identify_unvalidated_broad_features(
    evidence_matrix: EvidenceMatrix,
    cell_registry: Dict[str, CellRecord],
    selected_labels: Dict[str, int],
    broad_unstructured_coverage: float = 0.90,
    repair_candidate_coverage: float = 0.10,
    distributional_coverage: float = 0.10,
) -> Dict[str, Dict[str, Any]]:
    """Find broad soft features that lack a labeled clean counterexample.

    A single positive label cannot establish the precision of a detector that
    fires on a large part of a column. Such features remain in the evidence
    matrix for active-query design, but callers can keep them out of inference
    until a trigger-positive clean example has tested their false-positive
    risk. The policy is defined by evidence family and semantic archetype, not
    by dataset or column names.
    """
    if evidence_matrix.values.size == 0:
        return {}

    column_sizes: Dict[str, int] = {}
    for record in cell_registry.values():
        column_sizes[record.column] = column_sizes.get(record.column, 0) + 1

    guarded: Dict[str, Dict[str, Any]] = {}
    unstructured_archetypes = {"open_entity", "geo_name", "free_text"}
    epsilon = 1e-9

    for feature_idx, feature_name in enumerate(evidence_matrix.feature_names):
        values = evidence_matrix.values[:, feature_idx]
        positive_indices = np.flatnonzero(values > epsilon)
        if positive_indices.size == 0:
            continue

        descriptor = evidence_matrix.feature_metadata.get(feature_name, {})
        if descriptor.get("has_hard_evidence"):
            continue
        family = str(descriptor.get("family") or "")
        source_id = str(descriptor.get("source_id") or "")
        archetypes = set(descriptor.get("archetypes") or [])

        support_by_column: Dict[str, int] = {}
        clean_support = 0
        dirty_support = 0
        for row_idx in positive_indices.tolist():
            cell_key = evidence_matrix.cell_keys[row_idx]
            record = cell_registry.get(cell_key)
            if record is not None:
                support_by_column[record.column] = (
                    support_by_column.get(record.column, 0) + 1
                )
            label = selected_labels.get(cell_key)
            if label == 0:
                clean_support += 1
            elif label == 1:
                dirty_support += 1

        if clean_support > 0:
            continue
        coverage_by_column = {
            column: count / max(1, column_sizes.get(column, 0))
            for column, count in support_by_column.items()
        }
        max_coverage = max(coverage_by_column.values(), default=0.0)

        reason = ""
        if (
            archetypes.intersection(unstructured_archetypes)
            and max_coverage >= float(broad_unstructured_coverage)
        ):
            reason = "broad_unstructured_without_clean_counterexample"
        elif (
            family == "canonicalization"
            and archetypes.intersection(unstructured_archetypes)
            and max_coverage >= float(repair_candidate_coverage)
        ):
            reason = "broad_repair_candidate_without_clean_counterexample"
        elif (
            "closed_enum" in archetypes
            and source_id in {"prototype_far", "rarity_high"}
            and max_coverage >= float(distributional_coverage)
        ):
            reason = "broad_enum_distribution_without_clean_counterexample"

        if reason:
            guarded[feature_name] = {
                "reason": reason,
                "family": family,
                "source_id": source_id,
                "archetypes": sorted(archetypes),
                "positive_support": int(positive_indices.size),
                "dirty_label_support": dirty_support,
                "clean_label_support": clean_support,
                "max_column_coverage": float(max_coverage),
                "coverage_by_column": dict(sorted(coverage_by_column.items())),
            }

    return guarded


def select_active_queries(
    evidence_matrix: EvidenceMatrix,
    budget: int,
    ridge: float = 1.0,
    clean_proxy_fraction: float = 0.0,
    include_query_features: bool = True,
    strategy: str = "d_optimal",
    random_seed: Optional[int] = None,
    initial_cell_keys: Optional[Sequence[str]] = None,
    protected_cell_keys: Optional[Set[str]] = None,
    excluded_cell_keys: Optional[Set[str]] = None,
) -> List[str]:
    """Select cells by D-optimal design or a matched random baseline.

    ``initial_cell_keys`` seeds the information matrix before new cells are
    selected. This makes repeated calls an exact nested D-optimal trajectory:
    each extension is conditioned on mechanism representatives and all prior
    calibration queries. Cells in ``protected_cell_keys`` cannot be evicted by
    the clean-proxy diversity pass. ``excluded_cell_keys`` are unavailable for
    new selection, which lets callers reserve future mechanism representatives.

    ``random`` samples from the same nonzero evidence-signature representative
    pool as D-optimal selection. It deliberately skips the clean-proxy rewrite
    below so every base query is uniformly sampled before any caller-level
    mechanism-validation policy is applied.
    """

    if strategy not in {"d_optimal", "random"}:
        raise ValueError(f"Unknown active-query strategy: {strategy}")

    budget = max(0, int(budget))
    initial_keys = list(dict.fromkeys(str(key) for key in (initial_cell_keys or [])))
    if len(initial_keys) > budget:
        raise ValueError("Initial active-query set exceeds the requested budget")
    protected_keys = set(str(key) for key in (protected_cell_keys or set()))
    excluded_keys = set(str(key) for key in (excluded_cell_keys or set()))
    query_values = evidence_matrix.query_values
    has_query_features = (
        include_query_features
        and query_values is not None
        and query_values.size > 0
    )
    if budget == 0 or (
        evidence_matrix.values.size == 0 and not has_query_features
    ):
        return initial_keys

    x_all = evidence_matrix.values.copy()
    query_names = list(evidence_matrix.query_feature_names)
    if has_query_features:
        x_all = np.column_stack([x_all, query_values])
    excluded_indices = [
        idx
        for idx, name in enumerate(evidence_matrix.feature_names)
        if name in evidence_matrix.query_excluded_features
    ]
    if has_query_features:
        excluded_indices.extend(
            len(evidence_matrix.feature_names) + idx
            for idx, name in enumerate(query_names)
            if name in evidence_matrix.query_excluded_features
        )
    if excluded_indices:
        x_all[:, excluded_indices] = 0.0
    nonzero_mask = np.linalg.norm(x_all, axis=1) > 1e-9
    if not np.any(nonzero_mask):
        return initial_keys

    signature_by_index: Dict[int, Tuple[Tuple[int, float], ...]] = {}
    pattern_representatives: Dict[Tuple[Tuple[int, float], ...], int] = {}
    for row_idx in np.where(nonzero_mask)[0].tolist():
        nz = np.nonzero(np.abs(x_all[row_idx]) > 1e-9)[0]
        signature = tuple((int(idx), round(float(x_all[row_idx, idx]), 3)) for idx in nz)
        signature_by_index[int(row_idx)] = signature
        pattern_representatives.setdefault(signature, int(row_idx))

    key_to_index = {
        str(key): index for index, key in enumerate(evidence_matrix.cell_keys)
    }
    unknown_initial = [key for key in initial_keys if key not in key_to_index]
    if unknown_initial:
        raise ValueError(
            f"Unknown initial active-query cell keys: {unknown_initial[:3]}"
        )
    initial_indices = [key_to_index[key] for key in initial_keys]
    initial_signatures = {
        signature_by_index[index]
        for index in initial_indices
        if index in signature_by_index
    }
    candidate_indices = [
        index
        for signature, index in pattern_representatives.items()
        if signature not in initial_signatures
        and evidence_matrix.cell_keys[index] not in excluded_keys
    ]
    if not candidate_indices:
        return initial_keys

    if strategy == "random":
        if random_seed is None:
            seed_text = os.environ.get("CLEANEM_LLM_SEED", "").strip()
            try:
                random_seed = int(seed_text) if seed_text else None
            except ValueError:
                random_seed = None
        rng = np.random.default_rng(random_seed)
        sample_size = min(max(0, budget - len(initial_indices)), len(candidate_indices))
        sampled = rng.choice(
            np.asarray(candidate_indices, dtype=int),
            size=sample_size,
            replace=False,
        )
        return initial_keys + [
            evidence_matrix.cell_keys[int(idx)] for idx in sampled.tolist()
        ]

    dim = x_all.shape[1]
    inverse_information = np.eye(dim, dtype=float) / max(float(ridge), 1e-6)
    selected: List[int] = list(initial_indices)
    remaining = set(candidate_indices)

    for selected_idx in initial_indices:
        x = x_all[selected_idx]
        x_col = x.reshape(-1, 1)
        denom = 1.0 + float(x @ inverse_information @ x.T)
        if denom > 1e-12:
            inverse_information = inverse_information - (
                inverse_information @ x_col @ x_col.T @ inverse_information
            ) / denom

    additions_needed = max(0, budget - len(selected))
    for _ in range(min(additions_needed, len(candidate_indices))):
        best_idx = None
        best_score = -float("inf")
        for candidate_idx in remaining:
            x = x_all[candidate_idx]
            score = float(x @ inverse_information @ x.T)
            if score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is None or best_score <= 1e-12:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        x = x_all[best_idx]
        x_col = x.reshape(-1, 1)
        denom = 1.0 + float(x @ inverse_information @ x.T)
        if denom > 1e-12:
            inverse_information = inverse_information - (
                inverse_information @ x_col @ x_col.T @ inverse_information
            ) / denom

    # D-optimal design is label-free and can legitimately concentrate on one
    # polarity when the table is heavily corrupted.  Reserve a small fraction
    # for clean-looking representatives so the downstream logistic calibrator
    # can estimate both signs of each weak feature.  This is only a sampling
    # policy; no proxy is treated as a label.
    fraction = float(min(0.5, max(0.0, clean_proxy_fraction)))
    target_clean = int(round(len(selected) * fraction))
    if target_clean > 0 and selected:
        net_scores = np.sum(x_all, axis=1)
        clean_candidates = [
            idx for idx in candidate_indices
            if idx not in selected
            and evidence_matrix.cell_keys[idx] not in excluded_keys
            and float(net_scores[idx]) < -1e-9
        ]
        # In a heavily corrupted table every candidate may still have a
        # positive net score. Reserve the least suspicious representatives in
        # that case; this remains a proxy for query diversity, never an
        # assumed clean label.
        fallback_clean_proxy = not clean_candidates
        if fallback_clean_proxy:
            clean_candidates = [
                idx for idx in candidate_indices
                if idx not in selected
                and evidence_matrix.cell_keys[idx] not in excluded_keys
            ]
        clean_candidates.sort(key=lambda idx: (float(net_scores[idx]), idx))
        current_clean = (
            sum(float(net_scores[idx]) < -1e-9 for idx in selected)
            if not fallback_clean_proxy
            else 0
        )
        for candidate_idx in clean_candidates:
            if current_clean >= target_clean:
                break
            replace_pos = next(
                (
                    pos
                    for pos, idx in reversed(list(enumerate(selected)))
                    if float(net_scores[idx]) >= -1e-9
                    and evidence_matrix.cell_keys[idx] not in protected_keys
                ),
                None,
            )
            if replace_pos is None:
                break
            selected[replace_pos] = candidate_idx
            current_clean += 1

    return [evidence_matrix.cell_keys[idx] for idx in selected]


def _fit_prior_centered_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_prior: float,
    l2: float = 0.25,
    intercept_l2: float = 0.25,
    max_intercept_prior: float = 0.30,
    max_iters: int = 2000,
) -> Tuple[float, np.ndarray, str]:
    n_features = x_train.shape[1]
    base_logit = _logit(base_prior)
    theta = np.zeros(n_features + 1, dtype=float)
    theta[0] = base_logit
    prior_theta = np.zeros_like(theta)
    prior_theta[0] = base_logit

    x_aug = np.column_stack([np.ones(x_train.shape[0], dtype=float), x_train])
    penalties = np.full(n_features + 1, float(l2), dtype=float)
    penalties[0] = float(intercept_l2)

    spectral_norm = float(np.linalg.norm(x_aug, ord=2)) if x_aug.size else 1.0
    lipschitz = 0.25 * spectral_norm * spectral_norm + float(np.max(penalties))
    step_size = 1.0 / max(lipschitz, 1e-6)
    intercept_ceiling = _logit(max(base_prior, min(0.30, max_intercept_prior)))
    reason = "projected_gradient_converged_capped_intercept"

    for _ in range(max_iters):
        logits = np.clip(x_aug @ theta, -30.0, 30.0)
        probs = _sigmoid_array(logits)
        gradient = x_aug.T @ (probs - y_train) + penalties * (theta - prior_theta)
        old_theta = theta.copy()
        theta -= step_size * gradient
        # Active-query labels are deliberately enriched for informative
        # anomalies and are not a prevalence sample. Let them adjust the
        # baseline within a conservative envelope, but never turn the absence
        # of evidence itself into a dirty decision.
        theta[0] = min(theta[0], intercept_ceiling)
        theta[1:] = np.maximum(theta[1:], 0.0)
        if float(np.linalg.norm(theta - old_theta)) < 1e-6:
            break
    else:
        reason = "max_iters"

    return float(theta[0]), theta[1:], reason


def estimate_calibrated_posteriors(
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    evidence_matrix: EvidenceMatrix,
    selected_cell_keys: List[str],
    selected_labels: Dict[str, int],
    base_prior: float,
    queried_cells: List[Dict[str, object]] = None,
    fixed_calibration: Optional[CalibrationTrace] = None,
    unsupported_feature_weight_priors: Optional[Dict[str, float]] = None,
    feature_weight_floors: Optional[Dict[str, float]] = None,
    feature_weight_caps: Optional[Dict[str, float]] = None,
    learn_evidence_weights: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], CalibrationTrace]:
    """Fit a calibrated label model, or use fixed equal evidence weights."""

    queried_cells = queried_cells or []
    clipped_prior = _clip_probability(min(0.30, max(0.01, base_prior)))
    base_logit = _logit(clipped_prior)
    x_all = evidence_matrix.values
    n_cells = len(evidence_matrix.cell_keys)

    trace = CalibrationTrace(
        selected_cell_keys=list(selected_cell_keys),
        selected_indices=[
            evidence_matrix.key_to_index[key]
            for key in selected_cell_keys
            if key in evidence_matrix.key_to_index
        ],
        selected_labels=dict(selected_labels),
        queried_cells=list(queried_cells),
        intercept=base_logit,
        fitted=False,
        reason="no_active_labels",
        feature_count=len(evidence_matrix.feature_names),
        label_count=len(selected_labels),
    )

    train_indices = [
        evidence_matrix.key_to_index[key]
        for key in selected_cell_keys
        if key in selected_labels and key in evidence_matrix.key_to_index
    ]
    weight_floors = feature_weight_floors or {}
    weight_caps = feature_weight_caps or {}

    def apply_weight_constraints(weights: np.ndarray) -> np.ndarray:
        constrained = weights.copy()
        for feature_idx, feature_name in enumerate(evidence_matrix.feature_names):
            if feature_name in weight_floors:
                constrained[feature_idx] = max(
                    float(constrained[feature_idx]),
                    max(0.0, float(weight_floors[feature_name])),
                )
            if feature_name in weight_caps:
                constrained[feature_idx] = min(
                    float(constrained[feature_idx]),
                    max(0.0, float(weight_caps[feature_name])),
                )
        return constrained

    if (
        learn_evidence_weights
        and fixed_calibration is not None
        and fixed_calibration.fitted
        and x_all.shape[1] > 0
    ):
        weights = np.array(
            [float(fixed_calibration.feature_weights.get(name, 0.0)) for name in evidence_matrix.feature_names],
            dtype=float,
        )
        weights = apply_weight_constraints(weights)
        logits = float(fixed_calibration.intercept) + x_all @ weights
        probabilities = _sigmoid_array(logits)
        trace.intercept = float(fixed_calibration.intercept)
        trace.feature_weights = {
            name: float(weight)
            for name, weight in zip(evidence_matrix.feature_names, weights)
            if abs(float(weight)) > 1e-8
        }
        trace.fitted = True
        trace.reason = f"fixed_after_gating:{fixed_calibration.reason}"
        if weight_caps:
            trace.reason = f"{trace.reason};broad_feature_guard"
    elif learn_evidence_weights and train_indices and x_all.shape[1] > 0:
        y_train = np.array([float(selected_labels[evidence_matrix.cell_keys[idx]]) for idx in train_indices])
        x_train = x_all[train_indices]
        intercept, weights, reason = _fit_prior_centered_logistic(x_train, y_train, clipped_prior)
        priors = unsupported_feature_weight_priors or {}
        for feature_idx, feature_name in enumerate(evidence_matrix.feature_names):
            if feature_name in priors and not np.any(np.abs(x_train[:, feature_idx]) > 1e-9):
                weights[feature_idx] = max(0.0, float(priors[feature_name]))
                reason = f"{reason};unsupported_feature_prior"
        weights = apply_weight_constraints(weights)
        if weight_floors:
            reason = f"{reason};validated_feature_floor"
        if weight_caps:
            reason = f"{reason};broad_feature_guard"
        logits = intercept + x_all @ weights
        probabilities = _sigmoid_array(logits)
        trace.intercept = float(intercept)
        trace.feature_weights = {
            name: float(weight)
            for name, weight in zip(evidence_matrix.feature_names, weights)
            if abs(float(weight)) > 1e-8
        }
        trace.fitted = True
        trace.reason = reason
        if len(set(int(label) for label in selected_labels.values())) == 1:
            trace.reason = f"{reason};single_class_labels"
    elif x_all.shape[1] > 0:
        # Weight-learning ablation: keep the extracted evidence matrix and
        # label queries, but assign every source the same unit weight. Static
        # safety floors/caps remain active because they are evidence-policy
        # constraints rather than fitted weights.
        weights = apply_weight_constraints(np.ones(x_all.shape[1], dtype=float))
        logits = base_logit + x_all @ weights
        probabilities = _sigmoid_array(logits)
        trace.intercept = base_logit
        trace.feature_weights = {
            name: float(weight)
            for name, weight in zip(evidence_matrix.feature_names, weights)
            if abs(float(weight)) > 1e-8
        }
        trace.fitted = True
        trace.reason = "fixed_equal_evidence_weights"
    else:
        probabilities = np.full(n_cells, clipped_prior, dtype=float)
        logits = np.full(n_cells, base_logit, dtype=float)
        if x_all.shape[1] == 0:
            trace.reason = "no_evidence_features"

    confidences = _prediction_confidence(probabilities)
    cell_posteriors: Dict[str, Dict[str, float]] = {}
    for idx, cell_key in enumerate(evidence_matrix.cell_keys):
        cell_posteriors[cell_key] = {
            "posterior": float(probabilities[idx]),
            "confidence": float(confidences[idx]),
            # Keep the exact evaluated model logit for faithful counterfactual
            # explanations. Reconstructing it from a clipped probability loses
            # information for saturated predictions.
            "logit": float(np.clip(logits[idx], -30.0, 30.0))
            if n_cells and x_all.shape[1] > 0
            else float(base_logit),
        }

    value_priors: Dict[str, Dict[str, float]] = {}
    for value_key, record in value_registry.items():
        posterior_values = []
        confidence_values = []
        for row_idx in record.row_indices:
            cell_key = f"{row_idx}::{record.column}"
            bundle = cell_posteriors.get(cell_key)
            if bundle:
                posterior_values.append(bundle["posterior"])
                confidence_values.append(bundle["confidence"])
        if posterior_values:
            value_priors[value_key] = {
                "prior": float(np.mean(posterior_values)),
                "confidence": float(np.mean(confidence_values)),
            }
        else:
            value_priors[value_key] = {"prior": clipped_prior, "confidence": 0.0}

    return value_priors, cell_posteriors, trace


def compute_leave_one_evidence_out_contributions(
    cell_registry: Dict[str, CellRecord],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_posteriors: Dict[str, Dict[str, float]],
    calibration_trace: CalibrationTrace,
) -> Dict[str, List[EvidenceContribution]]:
    """Measure each observation's exact posterior contribution with LOEO.

    The fitted calibration parameters remain fixed. For observation ``e`` with
    signed feature value ``x_e`` and learned weight ``w_e``, its contribution is
    ``p(full) - p(full logit - w_e*x_e)``. Positive values support the dirty
    class, while negative values support the clean class. When several
    observations share a feature, each is removed independently rather than
    dropping the entire aggregated feature.
    """

    contributions: Dict[str, List[EvidenceContribution]] = {}
    weights = calibration_trace.feature_weights or {}

    for cell_key, record in cell_registry.items():
        posterior_bundle = cell_posteriors.get(cell_key, {})
        posterior = float(posterior_bundle.get("posterior", 0.5))
        full_logit = float(posterior_bundle.get("logit", _logit(posterior)))
        observations: List[EvidenceObservation] = []
        observations.extend(value_observations.get(record.value_key, []))
        observations.extend(cell_observations.get(cell_key, []))

        cell_contributions: List[EvidenceContribution] = []
        for observation in observations:
            feature_name = _feature_name(
                observation.target_scope,
                observation,
                _default_feature_namespace(record, observation),
            )
            signed_value = _signed_strength(observation)
            feature_weight = float(weights.get(feature_name, 0.0))
            weighted_logit = float(feature_weight * signed_value)
            posterior_without = _sigmoid_scalar(full_logit - weighted_logit)
            cell_contributions.append(EvidenceContribution(
                target_scope=observation.target_scope,
                source_id=observation.source_id,
                family=observation.family,
                polarity=observation.polarity,
                reason_code=observation.reason_code,
                strength=float(observation.strength),
                hard=bool(observation.hard),
                feature_name=feature_name,
                feature_weight=feature_weight,
                signed_feature_value=float(signed_value),
                weighted_logit=weighted_logit,
                posterior_without=posterior_without,
                posterior_contribution=float(posterior - posterior_without),
                metadata=dict(observation.metadata or {}),
            ))

        cell_contributions.sort(
            key=lambda item: (
                abs(item.posterior_contribution),
                item.hard,
                item.strength,
            ),
            reverse=True,
        )
        contributions[cell_key] = cell_contributions

    return contributions
