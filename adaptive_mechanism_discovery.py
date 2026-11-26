"""Data-driven discovery and validation of error mechanisms.

This module deliberately does not contain dataset names or a fixed mechanism
vocabulary. Existing evidence is clustered into auditable candidate patterns;
only candidates supported by active labels are materialized as new evidence
features. Unknown candidates retain a stable ``discovered:<id>`` identity until
they are validated.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from cleanem_models import EvidenceObservation


@dataclass
class MechanismCandidate:
    mechanism_id: str
    label: str
    name: str
    explanation: str
    signature: str
    archetype: str
    column: str
    columns: List[str]
    family: str
    source_kind: str
    support: int
    row_support: int
    strength: float
    cross_row_stability: float
    archetype_consistency: float
    repairability: float
    repair_success_rate: Optional[float]
    uncertainty: float
    label_purity: Optional[float] = None
    agent_proposals: List[str] = field(default_factory=list)
    false_positive_risk: Optional[float] = None
    labeled_support: int = 0
    repair_labeled_support: int = 0
    repair_validated: bool = False
    repair_validation_reason: str = "not_evaluated"
    status: str = "discovered"
    novel: bool = True
    enabled: bool = False
    representative_cells: List[str] = field(default_factory=list)
    member_cells: List[str] = field(default_factory=list, repr=False)
    repair_predictions: Dict[str, List[Any]] = field(default_factory=dict, repr=False)
    validation_reason: str = "awaiting_active_labels"

    @property
    def potential_repair_gain(self) -> float:
        return float(self.support * self.repairability * (0.5 + 0.5 * self.uncertainty))

    def to_dict(self, include_members: bool = False) -> Dict[str, Any]:
        payload = asdict(self)
        if not include_members:
            payload.pop("member_cells", None)
        payload.pop("repair_predictions", None)
        payload["potential_repair_gain"] = self.potential_repair_gain
        return payload


@dataclass
class MechanismDiscoveryReport:
    candidates: List[MechanismCandidate] = field(default_factory=list)
    observation_to_mechanism: Dict[str, str] = field(default_factory=dict)
    repair_to_mechanism: Dict[str, str] = field(default_factory=dict)
    cell_to_mechanism: Dict[str, str] = field(default_factory=dict)
    value_to_mechanism: Dict[str, str] = field(default_factory=dict)
    naming_source: str = "deterministic"

    def candidate_by_id(self) -> Dict[str, MechanismCandidate]:
        return {item.mechanism_id: item for item in self.candidates}

    def novel_candidates(self) -> List[MechanismCandidate]:
        return [item for item in self.candidates if item.novel]

    def validated_candidates(self) -> List[MechanismCandidate]:
        return [item for item in self.candidates if item.status == "validated"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "naming_source": self.naming_source,
            "candidate_count": len(self.candidates),
            "novel_candidate_count": len(self.novel_candidates()),
            "validated_candidate_count": len(self.validated_candidates()),
            "candidates": [item.to_dict() for item in self.candidates],
        }

    def agent_payload(
        self,
        limit: int = 40,
        cell_registry: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return data-derived evidence examples for the naming Agent."""
        ranked = sorted(
            self.novel_candidates(),
            key=lambda item: (-item.support, -item.repairability, item.mechanism_id),
        )
        payload = []
        for item in ranked[: max(0, int(limit))]:
            values = []
            if cell_registry is not None:
                for cell_key in item.representative_cells[:4]:
                    record = cell_registry.get(cell_key)
                    if record is None:
                        continue
                    values.append({
                        "cell_key": cell_key,
                        "raw_value": str(getattr(record, "raw_value", ""))[:240],
                        "row_index": int(getattr(record, "row_index", -1)),
                    })
            repair_examples = []
            for cell_key, predictions in list(item.repair_predictions.items())[:4]:
                repair_examples.append({
                    "cell_key": cell_key,
                    "predictions": [str(value)[:240] for value in predictions[:3]],
                })
            payload.append({
                "mechanism_id": item.label,
                "archetype": item.archetype,
                "column": item.column,
                "family": item.family,
                "source_kind": item.source_kind,
                "signature": item.signature,
                "support": item.support,
                "row_support": item.row_support,
                "repairability": round(item.repairability, 4),
                "agent_proposals": list(item.agent_proposals),
                "representative_values": values,
                "repair_examples": repair_examples,
            })
        return payload

    def apply_names(self, names: Mapping[str, Mapping[str, Any]]) -> None:
        """Apply Agent names without allowing it to alter mechanism identity."""
        for item in self.candidates:
            payload = names.get(item.label) or names.get(item.mechanism_id)
            if not isinstance(payload, Mapping):
                continue
            name = _safe_name(payload.get("name"))
            explanation = str(payload.get("explanation") or payload.get("description") or "").strip()
            if name:
                item.name = name
            if explanation:
                item.explanation = explanation
        self.naming_source = "agent"


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower()).strip("_")
    return text[:64]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _surface_shape(value: Any) -> str:
    text = _text(value)
    if not text:
        return "empty"
    if any(ord(char) > 127 for char in text) and any(char.isascii() and char.isalpha() for char in text):
        return "mixed_script"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return "numeric"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*%", text):
        return "percent_numeric"
    if re.fullmatch(r"[A-Za-z]+(?:[\s\-_/][A-Za-z0-9]+)*", text):
        return "latin_words"
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        return "alphanumeric"
    return "other_surface"


def _archetype_for(record: Any, metadata: Mapping[str, Mapping[str, Any]]) -> str:
    direct = _text(getattr(record, "archetype", ""))
    if direct:
        return direct
    info = metadata.get(getattr(record, "column", ""), {}) or {}
    semantics = info.get("semantics") if isinstance(info, Mapping) else {}
    if isinstance(semantics, Mapping):
        return _text(semantics.get("archetype")) or "unknown"
    return "unknown"

def _agent_proposals_for(
    metadata: Mapping[str, Mapping[str, Any]],
    archetype: str,
    columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Collect column-local Agent hypotheses for audit and prioritization."""
    proposals: Set[str] = set()
    allowed_columns = set(columns or [])
    for column, info in metadata.items():
        if allowed_columns and column not in allowed_columns:
            continue
        semantics = info.get("semantics") if isinstance(info, Mapping) else {}
        if not isinstance(semantics, Mapping):
            continue
        if _text(semantics.get("archetype")) not in {"", archetype}:
            continue
        for value in semantics.get("possible_error_mechanisms") or []:
            normalized = _safe_name(value)
            if normalized:
                proposals.add(normalized)
        for contract in semantics.get("mechanism_contracts") or []:
            if isinstance(contract, Mapping):
                normalized = _safe_name(contract.get("mechanism"))
                if normalized:
                    proposals.add(normalized)
    return sorted(proposals)


def _context_signature(metadata: Mapping[str, Any]) -> Optional[str]:
    determinant = _text(metadata.get("determinant_column") or metadata.get("context_column"))
    dependent = _text(metadata.get("dependent_column"))
    relation = _safe_name(metadata.get("relation_type"))
    if determinant and dependent:
        # The concrete columns are deliberately excluded. They are useful in
        # the audit explanation, but would make the same relationship anomaly
        # acquire a different identity in every table.
        return f"context:{relation or 'conditional'}:determinant_dependent_conflict"
    return None


def _event_signature(
    observation: Any,
    record: Any,
    metadata: Mapping[str, Any],
    repair_mechanism: str = "",
    candidate_value: Any = None,
) -> Tuple[str, str, str]:
    """Return (stable signature, source kind, normalized mechanism hint)."""
    family = _safe_name(getattr(observation, "family", "evidence")) or "evidence"
    source = _safe_name(getattr(observation, "source_id", "unknown")) or "unknown"
    reason = _safe_name(getattr(observation, "reason_code", ""))
    observed = getattr(record, "raw_value", None)
    if repair_mechanism:
        mechanism = _safe_name(repair_mechanism) or "repair"
        shape = _surface_shape(candidate_value if candidate_value is not None else observed)
        return f"repair:{mechanism}:{shape}", "repair", mechanism

    # Deterministic detectors may provide a column-independent signature for a
    # repeated surface transformation. Respect it so all instances are
    # validated as one mechanism cluster instead of creating one candidate per
    # concrete value shape. The signature is supplied by the detector, never by
    # an interactive label or a dataset-specific column name.
    discovery_signature = _text(
        metadata.get("discovery_signature")
        or metadata.get("mechanism_signature")
    )
    if discovery_signature:
        hint = _safe_name(
            metadata.get("mechanism_hint")
            or source
            or reason
        ) or source
        return f"discovered:{discovery_signature}", "discovered_pattern", hint

    context = _context_signature(metadata)
    if context and ("disagreement" in source or "conflict" in reason or family in {"contextual", "relationship"}):
        return context, "context_conflict", context.split(":", 1)[0]
    shape = _surface_shape(observed)
    signature = f"evidence:v1:{family}:{source}:{reason or source}:{shape}"
    # Source/reason names describe observations only. They never become a
    # trusted mechanism identity without clustering and active labels.
    return signature, "evidence_source", source


def _stable_id(signature: str, archetype: str) -> str:
    # Dataset and column names are intentionally excluded. The same observed
    # mechanism obtains the same provisional identity on a new table.
    payload = f"{archetype}|{signature}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _default_name(source_kind: str, signature: str) -> str:
    tail = signature.split(":")[-1]
    if source_kind == "compound_evidence":
        return "compound_anomaly_pattern"
    if source_kind == "context_conflict":
        return "contextual_consensus_conflict"
    if source_kind == "repair":
        return f"repair_{_safe_name(signature.split(':', 2)[1]) or 'surface_variant'}"
    if source_kind == "format_difference":
        if "mixed_script" in signature:
            return "mixed_script_surface_difference"
        return "format_surface_difference"
    if source_kind == "missingness":
        return "structural_missingness_pattern"
    return f"observed_{_safe_name(tail) or 'evidence_pattern'}"


def _default_explanation(source_kind: str, signature: str, archetype: str) -> str:
    descriptions = {
        "context_conflict": "A repeated cross-column context is inconsistent with the observed value.",
        "repair": "A validated repair candidate indicates a repeatable non-canonical value transformation.",
        "format_difference": "The value surface differs from the dominant format pattern for this semantic type.",
        "missingness": "The value follows a structural or explicit missingness pattern.",
        "evidence_source": "The same evidence source fires repeatedly on cells of this semantic type.",
    }
    return f"{descriptions.get(source_kind, descriptions['evidence_source'])} Archetype: {archetype}. Signature: {signature}."


def _representative_cells(events: Sequence[Dict[str, Any]], limit: int = 8) -> List[str]:
    ranked = sorted(
        events,
        key=lambda item: (-float(item.get("strength", 0.0)), str(item.get("cell_key", ""))),
    )
    return list(dict.fromkeys(item["cell_key"] for item in ranked if item.get("cell_key")))[:limit]


def discover_candidate_mechanisms(
    metadata: Mapping[str, Mapping[str, Any]],
    cell_registry: Mapping[str, Any],
    value_observations: Mapping[str, Sequence[Any]],
    cell_observations: Mapping[str, Sequence[Any]],
    repair_candidates: Optional[Mapping[str, Sequence[Any]]] = None,
    min_support: int = 2,
) -> MechanismDiscoveryReport:
    """Cluster candidate cells by their combined anomaly evidence fingerprint.

    This is label-free. A cluster is therefore kept as ``discovered:<id>`` and
    cannot affect inference until :func:`validate_mechanism_candidates` passes.
    """
    repair_candidates = repair_candidates or {}
    events_by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cells_by_value: Dict[str, List[str]] = defaultdict(list)
    for cell_key, record in cell_registry.items():
        cells_by_value[str(getattr(record, "value_key", ""))].append(cell_key)

    def add(cell_key: str, observation: Any, repairability: float = 0.0,
            repair_mechanism: str = "", candidate_value: Any = None) -> None:
        if _text(getattr(observation, "polarity", "")) != "dirty":
            return
        record = cell_registry.get(cell_key)
        if record is None:
            return
        signature, source_kind, hint = _event_signature(
            observation,
            record,
            getattr(observation, "metadata", {}) or {},
            repair_mechanism,
            candidate_value,
        )
        events_by_cell[cell_key].append({
            "cell_key": cell_key,
            "token": signature,
            "source_kind": source_kind,
            "hint": hint,
            "family": _text(getattr(observation, "family", "")) or source_kind,
            "strength": float(max(0.0, min(1.0, float(getattr(observation, "strength", 0.0))))),
            "repairability": float(max(0.0, min(1.0, repairability))),
            "candidate_value": candidate_value if repair_mechanism else None,
            "discovery_signature": _text(
                (getattr(observation, "metadata", {}) or {}).get(
                    "discovery_signature"
                )
                or (getattr(observation, "metadata", {}) or {}).get(
                    "mechanism_signature"
                )
            ),
        })

    for value_key, observations in value_observations.items():
        for cell_key in cells_by_value.get(str(value_key), []):
            for observation in observations:
                add(cell_key, observation)
    for cell_key, observations in cell_observations.items():
        for observation in observations:
            add(str(cell_key), observation)
    for cell_key, candidates in repair_candidates.items():
        for candidate in candidates:
            mechanism = _text(getattr(candidate, "mechanism", "")) or "surface_variant"
            candidate_metadata = dict(getattr(candidate, "metadata", {}) or {})
            candidate_metadata.setdefault(
                "discovery_signature", f"repair:{mechanism}"
            )
            candidate_metadata.setdefault("mechanism_hint", mechanism)
            pseudo = EvidenceObservation(
                target_scope="cell",
                target_key=str(cell_key),
                source_id=mechanism,
                family="canonicalization",
                polarity="dirty",
                strength=float(getattr(candidate, "detection_confidence", 0.0)),
                hard=False,
                reason_code=f"canonicalization_{mechanism}",
                metadata=candidate_metadata,
            )
            add(
                str(cell_key),
                pseudo,
                repairability=float(getattr(candidate, "repairability", 0.0)),
                repair_mechanism=mechanism,
                candidate_value=getattr(candidate, "candidate_value", None),
            )

    # One cell fingerprint combines all available clues. This prevents the
    # discovery layer from merely renaming individual evidence sources.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    global_signature_archetypes: Dict[str, Counter] = defaultdict(Counter)
    for cell_key, events in events_by_cell.items():
        record = cell_registry[cell_key]
        archetype = _archetype_for(record, metadata)
        # Each explicit detector signature is an independent candidate. A
        # cell may exhibit multiple mechanisms at once (for example repeated
        # punctuation plus a repeated-character mutation); combining those
        # signatures would erase both identities and make validation depend
        # on their accidental co-occurrence.
        explicit_tokens = sorted({
            event["token"]
            for event in events
            if event.get("discovery_signature")
        })
        event_groups = (
            [
                (token, [event for event in events if event["token"] == token])
                for token in explicit_tokens
            ]
            if explicit_tokens
            else [("+".join(sorted({event["token"] for event in events})), events)]
        )
        for token, mechanism_events in event_groups:
            if not token or not mechanism_events:
                continue
            signature = "cluster:v1:" + token
            summary = {
                "cell_key": cell_key,
                "column": _text(getattr(record, "column", "")) or "unknown",
                "row_index": int(getattr(record, "row_index", -1)),
                "events": mechanism_events,
                "signature": signature,
            }
            grouped[(archetype, signature)].append(summary)
    candidates: List[MechanismCandidate] = []
    cell_to_mechanism: Dict[str, str] = {}
    for (archetype, signature), members in grouped.items():
        support = len(members)
        if support < max(1, int(min_support)):
            continue
        columns = sorted({item["column"] for item in members})
        rows = {item["row_index"] for item in members if item["row_index"] >= 0}
        events = [event for item in members for event in item["events"]]
        mechanism_events = [
            event for event in events if event.get("discovery_signature")
        ]
        effective_events = mechanism_events or events
        strengths = [event["strength"] for event in effective_events]
        source_kinds = Counter(event["source_kind"] for event in effective_events)
        source_kind = "compound_evidence" if len(source_kinds) > 1 else source_kinds.most_common(1)[0][0]
        mechanism_id = _stable_id(signature, archetype)
        agent_proposals = _agent_proposals_for(metadata, archetype, columns)
        # previous calculation divided by the support of the same signature
        # in *other* archetypes and consequently rejected a valid mechanism
        # merely because it appeared across several roles. Keep the metric
        # meaningful by measuring purity within the candidate's own role; the
        # cross-role prevalence remains available in the audit payload.
        archetype_consistency = support / max(1, support)
        repair_events = [event for event in events if event["repairability"] > 0.0]
        repairability = (
            sum(event["repairability"] for event in repair_events) / len(repair_events)
            if repair_events else 0.0
        )
        repair_predictions: Dict[str, List[Any]] = defaultdict(list)
        for item in members:
            for event in item["events"]:
                if event["source_kind"] != "repair":
                    continue
                value = event.get("candidate_value")
                if not any(type(value) is type(existing) and value == existing
                           for existing in repair_predictions[item["cell_key"]]):
                    repair_predictions[item["cell_key"]].append(value)
        candidate = MechanismCandidate(
            mechanism_id=mechanism_id,
            label=f"discovered:{mechanism_id}",
            name=_default_name(source_kind, signature),
            explanation=_default_explanation(source_kind, signature, archetype),
            signature=signature,
            archetype=archetype,
            column=columns[0] if len(columns) == 1 else "*",
            columns=columns,
            family=(
                "compound"
                if len({event["family"] for event in effective_events}) > 1
                else effective_events[0]["family"]
            ),
            source_kind=source_kind,
            support=support,
            row_support=len(rows),
            strength=float(sum(strengths) / max(1, len(strengths))),
            cross_row_stability=float(min(1.0, len(rows) / max(3.0, float(support)))),
            archetype_consistency=float(archetype_consistency),
            repairability=float(repairability),
            repair_success_rate=None,
            uncertainty=1.0,
            agent_proposals=agent_proposals,
            status="discovered",
            novel=True,
            representative_cells=_representative_cells([
                {
                    "cell_key": item["cell_key"],
                    "strength": max(event["strength"] for event in item["events"]),
                }
                for item in members
            ]),
            member_cells=[item["cell_key"] for item in members],
            repair_predictions=dict(repair_predictions),
            validation_reason="awaiting_active_labels",
        )
        candidates.append(candidate)
        for cell_key in candidate.member_cells:
            cell_to_mechanism.setdefault(cell_key, mechanism_id)

    candidates.sort(key=lambda item: (-item.novel, -item.support, -item.repairability, item.label))
    value_to_mechanism: Dict[str, str] = {}
    for value_key, cell_keys in cells_by_value.items():
        mechanism_ids = {
            cell_to_mechanism[cell_key]
            for cell_key in cell_keys
            if cell_key in cell_to_mechanism
        }
        if len(mechanism_ids) == 1:
            value_to_mechanism[value_key] = next(iter(mechanism_ids))

    return MechanismDiscoveryReport(
        candidates=candidates,
        cell_to_mechanism=cell_to_mechanism,
        value_to_mechanism=value_to_mechanism,
    )


def _members_for(candidate: MechanismCandidate) -> List[str]:
    return candidate.member_cells or candidate.representative_cells


def rank_mechanism_candidates(
    report: MechanismDiscoveryReport,
) -> List[MechanismCandidate]:
    """Return the deterministic priority order used for active validation."""
    return sorted(
        report.novel_candidates(),
        key=lambda item: (
            0 if (
                item.source_kind == "repair"
                and "repair:surface_variant:" not in item.signature
                and item.repairability > 0.0
            ) else 1,
            0 if item.source_kind == "discovered_pattern" else 1,
            -float(item.uncertainty),
            -float(item.potential_repair_gain),
            -float(item.support),
            item.label,
        ),
    )


def validate_mechanism_candidates(
    report: MechanismDiscoveryReport,
    active_labels: Mapping[str, int],
    eligible_mechanisms: Optional[Set[str]] = None,
    corrected_values: Optional[Mapping[str, Any]] = None,
    values_equal: Optional[Callable[[Any, Any], bool]] = None,
    min_labeled_support: int = 2,
    min_label_purity: float = 0.75,
    max_false_positive_risk: float = 0.25,
    min_cross_row_stability: float = 0.50,
    min_archetype_consistency: float = 0.60,
    min_repair_success_rate: float = 0.50,
) -> MechanismDiscoveryReport:
    """Validate candidates using only labels selected by active querying."""
    corrected_values = corrected_values or {}
    equal = values_equal or (lambda left, right: _text(left) == _text(right))
    for candidate in report.candidates:
        if not candidate.novel:
            candidate.status = "existing"
            candidate.enabled = False
            continue
        # Query allocation is an explicit part of the validation contract.
        # Labels that happen to overlap another candidate's members must not
        # silently validate that candidate as well; otherwise one reserved
        # mechanism can promote an entire family in the same round.
        if eligible_mechanisms is not None and candidate.label not in eligible_mechanisms:
            candidate.status = "discovered"
            candidate.enabled = False
            candidate.validation_reason = "not_allocated_this_round"
            candidate.labeled_support = 0
            continue
        labels = [int(active_labels[key]) for key in _members_for(candidate) if key in active_labels]
        candidate.labeled_support = len(labels)
        if not labels:
            candidate.validation_reason = "no_active_label_for_candidate"
            continue
        dirty = sum(label == 1 for label in labels)
        clean = len(labels) - dirty
        candidate.label_purity = float(dirty / len(labels))
        candidate.false_positive_risk = float(clean / len(labels))
        candidate.uncertainty = float(1.0 - abs(candidate.label_purity - 0.5) * 2.0)
        repair_attempts = 0
        repair_successes = 0
        for cell_key in _members_for(candidate):
            if int(active_labels.get(cell_key, 0)) != 1 or cell_key not in corrected_values:
                continue
            predictions = candidate.repair_predictions.get(cell_key, [])
            if not predictions:
                continue
            repair_attempts += 1
            if any(equal(prediction, corrected_values[cell_key]) for prediction in predictions):
                repair_successes += 1
        candidate.repair_labeled_support = repair_attempts
        candidate.repair_success_rate = (
            float(repair_successes / repair_attempts) if repair_attempts else None
        )
        if not candidate.repair_predictions:
            candidate.repair_validation_reason = "not_applicable"
        elif not repair_attempts:
            candidate.repair_validation_reason = "no_labeled_repair_prediction"
        elif candidate.repair_success_rate is not None and candidate.repair_success_rate >= float(min_repair_success_rate):
            candidate.repair_validated = True
            candidate.repair_validation_reason = "corrected_value_validation_passed"
        else:
            candidate.repair_validation_reason = "corrected_value_validation_failed"
        if len(labels) < max(1, int(min_labeled_support)):
            candidate.validation_reason = "insufficient_labeled_support"
            continue
        if candidate.cross_row_stability < float(min_cross_row_stability):
            candidate.status = "rejected"
            candidate.validation_reason = "unstable_across_rows"
            continue
        if candidate.archetype_consistency < float(min_archetype_consistency):
            candidate.status = "rejected"
            candidate.validation_reason = "low_archetype_consistency"
            continue
        if candidate.label_purity >= float(min_label_purity) and candidate.false_positive_risk <= float(max_false_positive_risk):
            candidate.status = "validated"
            candidate.enabled = True
            candidate.validation_reason = "active_label_validation_passed"
        else:
            candidate.status = "rejected"
            candidate.validation_reason = "label_purity_or_false_positive_risk_failed"
    # Transfer validation for the same explicit detector signature across
    # semantic roles after a strict source-label validation.
    validated_by_signature = {}
    for candidate in report.candidates:
        if (
            candidate.status == "validated"
            and candidate.source_kind == "discovered_pattern"
            and candidate.signature
            and candidate.label_purity is not None
            and candidate.false_positive_risk is not None
            and candidate.labeled_support >= max(1, int(min_labeled_support))
            and candidate.label_purity >= float(min_label_purity)
            and candidate.false_positive_risk <= float(max_false_positive_risk)
        ):
            validated_by_signature.setdefault(candidate.signature, candidate)
    for candidate in report.candidates:
        source = validated_by_signature.get(candidate.signature)
        if (
            source is None
            or not candidate.novel
            or candidate.status != "discovered"
            or candidate.source_kind != "discovered_pattern"
            or candidate.support < 3
            or candidate.cross_row_stability < float(min_cross_row_stability)
            or candidate.archetype_consistency < float(min_archetype_consistency)
        ):
            continue
        candidate.status = "validated"
        candidate.enabled = True
        candidate.label_purity = source.label_purity
        candidate.false_positive_risk = source.false_positive_risk
        candidate.uncertainty = source.uncertainty
        candidate.validation_reason = "shared_signature_transfer:" + source.label
    return report


def allocate_mechanism_queries(
    selected_cell_keys: Sequence[str],
    budget: int,
    report: MechanismDiscoveryReport,
    protected_cell_keys: Optional[Set[str]] = None,
    fraction: float = 0.5,
    verification_samples: int = 2,
    max_candidates_per_round: int = 1,
    allow_replacement: bool = False,
    mechanism_budget: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reserve part of the fixed active-query budget for unverified mechanisms.

    A round validates at most one candidate by default. The calibration model
    is deliberately data-efficient, so validating several novel clusters at
    once can replace complementary labels and cause a large, unobservable
    change in all learned evidence weights. With ``mechanism_budget`` set,
    the caller supplies extra slots that are appended after the base query
    set; otherwise the historical fractional reservation behavior is used.
    Replacing a base calibration label remains opt-in for controlled
    ablations.
    """
    budget = max(0, int(budget))
    selected = list(dict.fromkeys(selected_cell_keys))[:budget]
    protected = set(protected_cell_keys or set())
    verification_samples = max(1, int(verification_samples))
    explicit_budget = (
        max(0, int(mechanism_budget))
        if mechanism_budget is not None
        else None
    )
    reserved_slots = (
        min(explicit_budget, budget)
        if explicit_budget is not None
        else int(round(max(0.0, min(1.0, float(fraction))) * budget))
    )
    proportional_candidate_limit = reserved_slots // verification_samples
    max_candidates = (
        proportional_candidate_limit
        if int(max_candidates_per_round) <= 0
        else max(1, int(max_candidates_per_round))
    )
    target = min(
        len(report.novel_candidates()),
        reserved_slots // verification_samples,
        max_candidates,
    )
    ranked = rank_mechanism_candidates(report)
    inserted: List[str] = []
    replaced: List[str] = []
    covered: List[str] = []
    covered_naturally: List[str] = []
    skipped_replacement_required: List[str] = []
    appended_without_replacement: List[str] = []
    reserved = set(protected)
    for candidate in ranked:
        if len(covered) >= target:
            break
        members = list(dict.fromkeys(candidate.representative_cells + _members_for(candidate)))
        if len(members) < verification_samples:
            continue
        already_selected = [key for key in selected if key in set(members)]
        needed = verification_samples - len(already_selected)
        if needed <= 0:
            reserved.update(already_selected)
            covered.append(candidate.label)
            covered_naturally.append(candidate.label)
            continue
        additions = [key for key in members if key not in selected][:needed]
        if len(additions) < needed:
            continue
        append_count = min(needed, max(0, budget - len(selected)))
        if not allow_replacement and append_count < needed:
            skipped_replacement_required.append(candidate.label)
            continue
        replacement_capacity = sum(
            1
            for key in selected
            if key not in reserved and key not in members
        )
        if replacement_capacity < needed - append_count:
            continue
        added = 0
        for representative in additions:
            used_replacement = False
            if len(selected) < budget:
                selected.append(representative)
            else:
                # Recompute after every replacement. Earlier code retained a
                # stale index list when an addition filled the last free slot,
                # causing an IndexError and aborting the entire pipeline.
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index] not in reserved
                        and selected[index] not in members
                    ),
                    None,
                )
                if replace_index is None:
                    break
                replaced.append(selected[replace_index])
                selected[replace_index] = representative
                used_replacement = True
            inserted.append(representative)
            if not used_replacement:
                appended_without_replacement.append(representative)
            added += 1
        if added == needed:
            reserved.update(key for key in selected if key in set(members))
            covered.append(candidate.label)
    return list(dict.fromkeys(selected))[: max(0, int(budget))], {
        "target": target,
        "reserved_slots": reserved_slots,
        "verification_samples": verification_samples,
        "max_candidates_per_round": max_candidates,
        "inserted": inserted,
        "replaced": replaced,
        "covered_mechanisms": covered,
        "covered_naturally": covered_naturally,
        "skipped_replacement_required": skipped_replacement_required,
        "appended_without_replacement": appended_without_replacement,
        "replacement_enabled": bool(allow_replacement),
        "mechanism_budget": explicit_budget,
        "candidate_count": len(report.novel_candidates()),
        "protected_count": len(protected),
    }


def annotate_discovered_observations(
    report: MechanismDiscoveryReport,
    value_observations: Mapping[str, Sequence[Any]],
    cell_observations: Mapping[str, Sequence[Any]],
) -> int:
    """Attach stable identities for audit without changing inference yet."""
    count = 0
    for observations_by_key, identity_map in (
        (value_observations, report.value_to_mechanism),
        (cell_observations, report.cell_to_mechanism),
    ):
        for target_key, observations in observations_by_key.items():
            mechanism_id = identity_map.get(str(target_key))
            if not mechanism_id:
                continue
            for observation in observations:
                if _text(getattr(observation, "polarity", "")) != "dirty":
                    continue
                candidate = report.candidate_by_id().get(mechanism_id)
                if candidate is None or not candidate.novel:
                    continue
                metadata = dict(getattr(observation, "metadata", {}) or {})
                metadata.update({
                    "discovered_mechanism": candidate.label,
                    "discovered_mechanism_name": candidate.name,
                    "discovered_mechanism_status": candidate.status,
                })
                observation.metadata = metadata
                count += 1
    return count


def enable_validated_mechanisms(
    report: MechanismDiscoveryReport,
    value_observations: Dict[str, List[Any]],
    cell_observations: Dict[str, List[Any]],
) -> Dict[str, int]:
    """Materialize validated candidates as isolated calibratable features."""
    added = 0
    skipped = 0
    del value_observations
    for candidate in report.validated_candidates():
        source_id = candidate.label
        namespace = f"mechanism:{candidate.label}:archetype:{candidate.archetype}"
        # Membership is cell-level in the report, so append only cell features.
        for cell_key in _members_for(candidate):
            observations = cell_observations.setdefault(cell_key, [])
            if any(getattr(item, "source_id", "") == source_id for item in observations):
                skipped += 1
                continue
            observation = EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id=source_id,
                family="discovered_mechanism",
                polarity="dirty",
                strength=float(candidate.strength),
                hard=False,
                reason_code="validated_discovered_mechanism",
                metadata={
                    "feature_namespace": namespace,
                    "discovered_mechanism": candidate.label,
                    "mechanism_name": candidate.name,
                    "mechanism_explanation": candidate.explanation,
                    "propagation_scope": "cluster_members",
                    "validation": candidate.validation_reason,
                    "support": candidate.support,
                    "label_purity": candidate.label_purity,
                    "false_positive_risk": candidate.false_positive_risk,
                    "repair_success_rate": candidate.repair_success_rate,
                    "repair_validated": candidate.repair_validated,
                    "repair_validation_reason": candidate.repair_validation_reason,
                },
            )
            observations.append(observation)
            added += 1
    return {"added": added, "skipped": skipped, "validated": len(report.validated_candidates())}
