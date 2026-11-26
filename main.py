"""
Command-line entry point for the agentic error detector.

Supports a dual verification pipeline with clean rule-level refinement.
"""
import argparse
import ast
import difflib
import hashlib
import json
import logging
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime

from dateutil import parser as date_parser

import numpy as np
import pandas as pd

from judge import Judge
from agent import AgentFactory, DirtyExampleAgent, CleanRuleReflectionAgent
from profiler import PandasProfiler, DEFAULT_MISSING_TOKEN_SET
from validator import DisjointnessValidator
from core.utils import safe_dict
from cleanem_inference import (
    build_cell_evidence_matrix,
    compute_leave_one_evidence_out_contributions,
    estimate_calibrated_posteriors,
    identify_unvalidated_broad_features,
    select_active_queries,
)
from adaptive_mechanism_discovery import (
    allocate_mechanism_queries,
    annotate_discovered_observations,
    discover_candidate_mechanisms,
    enable_validated_mechanisms,
    rank_mechanism_candidates,
    validate_mechanism_candidates,
)
from cleanem_models import (
    CellRecord,
    EvidenceContribution,
    EvidenceObservation,
    ExplanationTrace,
    HardEvidenceLabel,
    RepairCandidate,
    ValueRecord,
)
from canonicalization import (
    PlanDrivenCanonicalizationAgent,
    build_canonicalization_profiles,
    flatten_repair_candidates,
)


# CleanEM Logger
cleanem_logger = logging.getLogger("CleanEM")

FAMILIES = ["missing", "outlier", "pattern", "relationship", "prototype_lexical"]
ARCHETYPE_EVIDENCE_GATES = {
    "open_entity": {
        "rarity_high": 0.10,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.30,
    },
    "identifier": {
        "rarity_high": 0.00,
        "prototype_far": 0.00,
        "pattern_mismatch": 1.00,
    },
    "closed_enum": {
        "rarity_high": 0.25,
        "prototype_far": 0.30,
        "pattern_mismatch": 0.60,
    },
    "unit_measure": {
        "rarity_high": 0.20,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.60,
    },
    "numeric_measure": {
        "rarity_high": 0.50,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.80,
    },
    # Temporal values rely on their format/range/order evidence. Preserve the
    # generic sources until dedicated temporal evidence sources are available.
    "temporal_measure": {
        "rarity_high": 1.00,
        "prototype_far": 1.00,
        "pattern_mismatch": 1.00,
    },
    "geo_name": {
        "rarity_high": 0.10,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.30,
    },
    "free_text": {
        "rarity_high": 0.05,
        "prototype_far": 0.10,
        "pattern_mismatch": 0.20,
    },
}
FAMILY_SCORE_KEYS = {
    "missing": "S_missing",
    "outlier": "S_outlier",
    "pattern": "S_pattern",
    "relationship": "S_relationship",
    "prototype_lexical": "S_prototype_lexical",
}
ANCHOR_HARD_CLEAN = "hard_clean"
ANCHOR_HARD_DIRTY = "hard_dirty"
ANCHOR_UNLABELED = "unlabeled"
ANCHOR_ABSTAIN = "abstain"

# Dataset-agnostic reliability requirements.
STRUCTURAL_MISSING_MIN_SUPPORT = 5
STRUCTURAL_MISSING_MIN_RATE = 0.02
RULE_MIN_PASS_RATE = 0.50
CONTEXT_MIN_GROUP_SUPPORT = 3
CONTEXT_MIN_EXPECTED_PROBABILITY = 0.80
CONTEXT_MIN_PREDICTIVENESS = 0.20
CONTEXT_IDENTIFIER_CLEAN_MIN_GROUP_SUPPORT = 3
CONTEXT_IDENTIFIER_CLEAN_MIN_PREDICTIVENESS = 0.20
LLM_FD_MIN_GROUP_SUPPORT = 3
LLM_FD_MIN_EXPECTED_SUPPORT = 2
LLM_FD_MIN_EXPECTED_PROBABILITY = 0.50
MULTISOURCE_MIN_DISTINCT_SOURCES = 3
MULTISOURCE_MIN_DOMINANT_WEIGHT_RATIO = 0.60
MULTISOURCE_TIME_TOLERANCE_MINUTES = 10.0
MULTISOURCE_MIN_SOURCE_RELIABILITY = 0.75


@dataclass
class GroupRecord:
    normalized_signature: str
    row_indices: List[int]
    weight: int
    rule_outputs_by_family: Dict[str, np.ndarray]
    anchor_state: str
    anchor_source: str
    representative_value: Any
    dirty_prior: float = 0.5
    anchor_weight: float = 0.0
    support_count: int = 0


@dataclass
class FamilyRunStatus:
    status: str
    reason: str = ""
    coverage_rows: int = 0
    coverage_groups: int = 0
    reliability: float = float("nan")
    mode: str = "abstained"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    score_key: str
    uses_rules: bool = True
    allow_synthetic: bool = True


FAMILY_REGISTRY: Dict[str, FamilySpec] = {
    family: FamilySpec(
        name=family,
        score_key=FAMILY_SCORE_KEYS[family],
        uses_rules=family != "prototype_lexical",
        allow_synthetic=family in {"missing", "outlier", "pattern"},
    )
    for family in FAMILIES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI with dual and clean+EM modes."
    )
    parser.add_argument(
        "--dirty_csv",
        default="data/beers_error-01.csv",
        help="Path to the dirty/error-prone CSV that needs inspection."
    )
    parser.add_argument(
        "--clean_csv",
        default="data/beers_clean.csv",
        help="Optional clean CSV for evaluation against ground truth."
    )
    parser.add_argument(
        "--output_dir",
        default="results/agentic_error_detector",
        help="Directory for serialized outputs."
    )
    parser.add_argument(
        "--llm_usage_log",
        default=None,
        help=(
            "Optional JSONL path for per-request LLM token usage. "
            "Defaults to <output_dir>/llm_usage.jsonl."
        ),
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Max refinement rounds for dual verification."
    )
    parser.add_argument(
        "--grey_tolerance",
        type=float,
        default=0.01,
        help="Allowed grey-zone rate when evaluating dual rules."
    )
    parser.add_argument(
        "--vr_threshold",
        type=float,
        default=0.6,
        help="Violation-rate threshold for legacy VR mode."
    )
    parser.add_argument(
        "--mode",
        choices=["dual", "clean_em"],
        default="clean_em",
        help="Pipeline mode: 'dual' uses P_clean/P_dirty, 'clean_em' uses clean rules + EM."
    )
    parser.add_argument(
        "--base_url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint for rule-generating LLMs."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the Legislator factory."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum parallel workers for per-column LLM calls."
    )
    parser.add_argument(
        "--clean_seed_percent",
        type=float,
        default=0.1,
        help="Top percentile per column used as high-confidence clean seeds (0-1)."
    )
    parser.add_argument(
        "--synthetic_per_family",
        type=int,
        default=50,
        help="Synthetic dirty examples per family per column."
    )
    parser.add_argument(
        "--em_max_iters",
        type=int,
        default=10,
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
    )
    parser.add_argument(
        "--em_prior_dirty",
        type=float,
        default=0.05,
        help="Base prior dirty rate used for value-prior initialization."
    )
    parser.add_argument(
        "--calib_min_clean_pass",
        type=float,
        default=0.8,
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
    )
    parser.add_argument(
        "--calib_max_dirty_pass",
        type=float,
        default=0.3,
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-K highest scoring cells to display."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold on cell_posterior to treat a cell as error in clean_em mode."
    )
    parser.add_argument(
        "--rule_min_pass_rate",
        type=float,
        default=RULE_MIN_PASS_RATE,
        help=(
            "Minimum empirical pass rate required for a non-missing Agent rule "
            f"during rule-pool validation (default: {RULE_MIN_PASS_RATE})."
        ),
    )
    parser.add_argument(
        "--disable_column_thresholds",
        action="store_true",
        help="Disable conservative per-column thresholds learned from active labels."
    )
    parser.add_argument(
        "--column_threshold_min_labels",
        type=int,
        default=2,
        help="Minimum active labels in a column before learning its decision threshold."
    )
    parser.add_argument(
        "--active_label_budget",
        type=int,
        default=10,
        help="Max oracle cell labels used to calibrate clean_em evidence sources."
    )
    parser.add_argument(
        "--active_query_strategy",
        choices=["d_optimal", "random"],
        default="d_optimal",
        help="Active-query selection strategy (D-optimal or matched random)."
    )
    parser.add_argument(
        "--active_query_clean_proxy_fraction",
        type=float,
        default=0.30,
        help=(
            "Fraction of the fixed query budget reserved for clean-looking "
            "evidence representatives used to calibrate both polarities."
        ),
    )
    parser.add_argument(
        "--enable_structural_missing_query_coverage",
        dest="enable_structural_missing_query_coverage",
        action="store_true",
        default=True,
        help=(
            "Ensure D-optimal selection covers structural missingness by "
            "replacing a lower-priority query within the fixed budget "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--disable_structural_missing_query_coverage",
        dest="enable_structural_missing_query_coverage",
        action="store_false",
        help=(
            "Disable the default fixed-budget structural-missing query "
            "coverage mechanism for ablation runs."
        ),
    )
    parser.add_argument(
        "--enable_adaptive_mechanism_discovery",
        action="store_true",
        default=True,
        help=(
            "Discover evidence/repair/context clusters, query representatives, "
            "and enable only label-validated mechanisms (enabled by default; "
            "use --disable_adaptive_mechanism_discovery for ablations)."
        ),
    )
    parser.add_argument(
        "--disable_adaptive_mechanism_discovery",
        action="store_true",
        help=(
            "Disable candidate mechanism clustering, mechanism-query allocation, "
            "and validated mechanism activation while retaining generic evidence."
        ),
    )
    parser.add_argument(
        "--enable_adaptive_all_mechanisms",
        action="store_true",
        help=(
            "Defer specialized detector evidence to adaptive mechanism "
            "validation. Explicit missing and parser failures remain direct "
            "evidence; all other mechanism candidates require active labels."
        ),
    )
    parser.add_argument(
        "--mechanism_mode",
        choices=["adaptive_agent", "agent_rules_only"],
        default="adaptive_agent",
        help=(
            "Use Agent-proposed mechanisms with unsupervised clustering and "
            "active validation (default), or disable adaptive mechanism discovery "
            "while retaining only generic evidence and the Agent rule pool."
        ),
    )
    parser.add_argument(
        "--adaptive_mechanism_query_fraction",
        type=float,
        default=0.40,
        help="Fraction of the fixed active-label budget reserved for unverified mechanisms.",
    )
    parser.add_argument(
        "--adaptive_mechanism_query_budget",
        type=int,
        default=0,
        help=(
            "Extra oracle-query slots reserved for mechanism validation. "
            "The base --active_label_budget remains dedicated to D-optimal "
            "calibration; zero keeps the legacy fractional reservation mode."
        ),
    )
    parser.add_argument(
        "--adaptive_mechanism_max_candidates_per_round",
        type=int,
        default=0,
        help=(
            "Maximum number of newly discovered mechanisms validated in one "
            "active-query round. Zero derives the limit from the mechanism "
            "label fraction and per-mechanism validation support."
        ),
    )
    parser.add_argument(
        "--adaptive_mechanism_allow_query_replacement",
        action="store_true",
        default=True,
        help=(
            "Allow discovered-mechanism validation to replace base calibration "
            "queries inside the fixed active-label budget. Enabled by default; "
            "use --disable_adaptive_mechanism_query_replacement only for "
            "controlled ablations."
        ),
    )
    parser.add_argument(
        "--disable_adaptive_mechanism_query_replacement",
        dest="adaptive_mechanism_allow_query_replacement",
        action="store_false",
        help=(
            "Keep the base calibration queries unchanged when validating "
            "discovered mechanisms (ablation only)."
        ),
    )
    parser.add_argument(
        "--adaptive_mechanism_min_support",
        type=int,
        default=3,
        help="Minimum distinct cells supporting a discovered mechanism cluster.",
    )
    parser.add_argument(
        "--adaptive_mechanism_min_labeled_support",
        type=int,
        default=2,
        help=(
            "Minimum active labels required before a discovered mechanism can "
            "enabled (default: 2 labels; purity, false-positive risk, row "
            "stability, and archetype checks still apply)."
        ),
    )
    parser.add_argument(
        "--adaptive_mechanism_min_label_purity",
        type=float,
        default=0.75,
        help="Minimum dirty-label purity for enabling a discovered mechanism.",
    )
    parser.add_argument(
        "--adaptive_mechanism_max_fp_risk",
        type=float,
        default=0.25,
        help="Maximum clean-label risk allowed for an enabled discovered mechanism.",
    )
    parser.add_argument(
        "--adaptive_mechanism_min_archetype_consistency",
        type=float,
        default=0.60,
        help="Minimum semantic-archetype consistency for enabling a discovered mechanism.",
    )
    parser.add_argument(
        "--adaptive_mechanism_min_repair_success_rate",
        type=float,
        default=0.50,
        help="Minimum observed corrected-value match rate when repair predictions are available.",
    )
    parser.add_argument(
        "--disable_agent_mechanism_naming",
        action="store_true",
        help="Use deterministic mechanism names without the naming-only Agent.",
    )
    parser.add_argument(
        "--enable_core_mechanism_query_coverage",
        action="store_true",
        help=(
            "Reserve active labels for non-missing temporal component patterns "
            "and Unicode corruption when those mechanisms are present."
        ),
    )
    parser.add_argument(
        "--active_query_rounds",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--active_query_batch_size",
        type=int,
        default=10,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable_evidence_gating",
        action="store_true",
        help="Disable archetype-conditioned evidence scaling for ablation runs."
    )
    parser.add_argument(
        "--disable_evidence_weight_learning",
        action="store_true",
        help=(
            "Disable label-fitted evidence weights and use fixed equal weights "
            "while retaining the extracted evidence matrix."
        ),
    )
    parser.add_argument(
        "--disable_broad_evidence_guard",
        action="store_true",
        help=(
            "Disable the clean-counterexample requirement for broad soft "
            "evidence (ablation only)."
        ),
    )
    parser.add_argument(
        "--broad_evidence_coverage_threshold",
        type=float,
        default=0.75,
        help=(
            "Column coverage at which unstructured soft evidence requires a "
            "trigger-positive clean counterexample (default: 0.75)."
        ),
    )
    parser.add_argument(
        "--unvalidated_evidence_coverage_threshold",
        type=float,
        default=0.10,
        help=(
            "Coverage threshold for unvalidated repair and closed-enum "
            "distributional evidence (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--unvalidated_broad_feature_cap",
        type=float,
        default=1.5,
        help=(
            "Maximum likelihood ratio for broad evidence without a clean "
            "counterexample; values above 1 keep it as weak inference evidence."
        ),
    )
    parser.add_argument(
        "--evidence_gating_mode",
        choices=["hard", "soft", "weak_prior"],
        default="hard",
        help=(
            "Evidence gate policy: hard threshold, legacy label-supported soft "
            "gate, or an always-on weak archetype prior updated by labels."
        )
    )
    parser.add_argument(
        "--hard_gate_min_clean_support",
        type=int,
        default=1,
        help="Distinct clean counterexamples required to activate a hard gate."
    )
    parser.add_argument(
        "--gate_prior_strength",
        type=float,
        default=0.25,
        help="Weak-prior trust multiplier in [0, 1] (default: 0.25)."
    )
    parser.add_argument(
        "--gate_prior_pseudocount",
        type=float,
        default=2.0,
        help="Pseudo-count assigned to the archetype gate prior (default: 2)."
    )
    parser.add_argument(
        "--disable_canonicalization_evidence",
        action="store_true",
        help=(
            "Generate repair candidates for audit output but exclude their "
            "canonicalization evidence from calibrated inference."
        )
    )
    parser.add_argument(
        "--disable_closed_enum_normalization",
        action="store_true",
        help=(
            "Disable conservative reversal, Unicode-contamination, and "
            "dictionary-edit evidence for closed-enum values."
        ),
    )
    parser.add_argument(
        "--agent_pattern_rule_scale",
        type=float,
        default=1.0,
        help=(
            "Post-calibration multiplier applied only to pattern evidence "
            "originating from the agentic clean-rule pool (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--rule_pool_cache",
        default=None,
        help=(
            "Optional JSON cache for column semantics and generated rules. "
            "Load it when present; otherwise generate and save it."
        )
    )
    parser.add_argument(
        "--disable_agent_rules",
        action="store_true",
        help=(
            "Ablate Agent-generated rule evidence: skip the per-column clean-rule "
            "pool and table-level LLM functional-dependency rules while retaining "
            "mechanism-neutral evidence and adaptive discovery."
        ),
    )
    parser.add_argument(
        "--active_query_cache",
        default=None,
        help=(
            "Optional JSON cache for selected active-query cell keys. "
            "Load it when present; otherwise select and save it."
        )
    )
    parser.add_argument(
        "--active_query_seed_cache",
        default=None,
        help=(
            "Optional active-query cache used as a stable selection base. "
            "Coverage constraints may replace low-information seed queries, "
            "but never increase --active_label_budget."
        ),
    )
    parser.add_argument(
        "--active_query_trajectory_cache",
        default=None,
        help=(
            "Optional cache for one nested active-query trajectory shared by "
            "all label-budget checkpoints. The first run creates it; later "
            "runs load an exact prefix and verify evidence/mechanism fingerprints."
        ),
    )
    parser.add_argument(
        "--active_query_checkpoints",
        default="0,4,8,10,20,40",
        help=(
            "Comma-separated budgets represented in the nested trajectory "
            "cache (default: 0,4,8,10,20,40)."
        ),
    )
    parser.add_argument(
        "--canonicalization_plan_cache",
        default=None,
        help=(
            "Optional JSON cache for LLM-generated column canonicalization plans. "
            "Load it when present; otherwise generate, validate, and save it."
        )
    )
    parser.add_argument(
        "--relationship_rule_cache",
        default=None,
        help=(
            "Optional JSON cache for LLM-proposed and data-validated table-level "
            "relationship rules. Load it when present; otherwise generate and save it."
        ),
    )
    parser.add_argument(
        "--disable_llm_relationship_rules",
        action="store_true",
        help="Disable table-level LLM functional-dependency evidence for ablation runs.",
    )
    parser.add_argument(
        "--disable_multisource_temporal_consensus",
        action="store_true",
        help=(
            "Disable schema-role-driven consensus evidence for repeated temporal "
            "observations reported by multiple sources."
        ),
    )
    parser.add_argument(
        "--multisource_min_sources",
        type=int,
        default=MULTISOURCE_MIN_DISTINCT_SOURCES,
    )
    parser.add_argument(
        "--multisource_min_ratio",
        type=float,
        default=MULTISOURCE_MIN_DOMINANT_WEIGHT_RATIO,
    )
    parser.add_argument(
        "--multisource_time_tolerance",
        type=float,
        default=MULTISOURCE_TIME_TOLERANCE_MINUTES,
    )
    parser.add_argument(
        "--multisource_min_source_reliability",
        type=float,
        default=MULTISOURCE_MIN_SOURCE_RELIABILITY,
    )
    parser.add_argument(
        "--enable_corrected_value_standardization",
        action="store_true",
        help=(
            "Use corrected values returned for active queries to learn a format "
            "contract, feed match/mismatch evidence into the second detection pass, "
            "and export standardization candidates."
        ),
    )
    return parser.parse_args()


def _invoke_rule(rule_func, value, row) -> bool:
    try:
        return bool(rule_func(value, row))
    except TypeError:
        try:
            return bool(rule_func(value))
        except Exception:
            return False
    except Exception:
        return False


def _representative_rule_validation_indices(
    df: pd.DataFrame,
    column: str,
    max_rows: int = 2048,
) -> List[Any]:
    """Select deterministic common and distinct values for rule smoke tests."""
    non_null = df[column].dropna()
    if non_null.empty:
        return []

    def evenly_spaced(indices: List[Any], limit: int) -> List[Any]:
        if len(indices) <= limit:
            return indices
        positions = np.linspace(0, len(indices) - 1, num=limit, dtype=int)
        return [indices[int(position)] for position in positions]

    half = max(1, int(max_rows) // 2)
    row_indices = evenly_spaced(list(non_null.index), half)
    distinct_indices = evenly_spaced(
        list(non_null.astype(str).drop_duplicates().index),
        half,
    )
    return list(dict.fromkeys(row_indices + distinct_indices))


def _validate_rule_candidate(
    df: pd.DataFrame,
    column: str,
    family: str,
    rule_str: str,
    min_pass_rate: float = RULE_MIN_PASS_RATE,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Compile and smoke-test an LLM rule before it enters the rule pool."""
    report: Dict[str, Any] = {
        "accepted": False,
        "reason": "unknown",
        "tested": 0,
        "pass_rate": None,
    }
    try:
        tree = ast.parse(rule_str, mode="eval")
    except SyntaxError as exc:
        report["reason"] = f"syntax_error:{exc.msg}"
        return None, report

    if any(
        isinstance(node, ast.Constant) and node.value is Ellipsis
        for node in ast.walk(tree)
    ):
        report["reason"] = "placeholder_ellipsis"
        return None, report

    try:
        rule_func = eval(compile(tree, "<generated_clean_rule>", "eval"), safe_dict)
    except Exception as exc:
        report["reason"] = f"compile_error:{type(exc).__name__}"
        return None, report
    if not callable(rule_func):
        report["reason"] = "not_callable"
        return None, report

    indices = _representative_rule_validation_indices(df, column)
    if not indices:
        if family == "missing":
            report.update({"accepted": True, "reason": "empty_column_missing_rule"})
            return rule_func, report
        report["reason"] = "no_non_null_samples"
        return None, report

    outcomes: List[bool] = []
    for row_index in indices:
        row = df.loc[row_index]
        value = row[column]
        try:
            outcome = bool(rule_func(value, row))
        except TypeError:
            try:
                outcome = bool(rule_func(value))
            except Exception as exc:
                report.update({
                    "reason": f"runtime_error:{type(exc).__name__}",
                    "tested": len(outcomes),
                })
                return None, report
        except Exception as exc:
            report.update({
                "reason": f"runtime_error:{type(exc).__name__}",
                "tested": len(outcomes),
            })
            return None, report
        outcomes.append(outcome)

    pass_rate = float(np.mean(outcomes))
    violation_rate = 1.0 - pass_rate
    report.update({
        "tested": len(outcomes),
        "pass_rate": pass_rate,
        "violation_rate": violation_rate,
        "validation_status": "rejected",
    })
    if family != "missing":
        if all(outcomes) or not any(outcomes):
            report["reason"] = "degenerate_constant_result"
            return None, report
        if pass_rate < float(min_pass_rate):
            report["reason"] = "excessive_empirical_conflict"
            return None, report

    report.update({
        "accepted": True,
        "reason": "data_validated",
        "validation_status": "accepted",
    })
    return rule_func, report


def _log_rule_rejection(
    column: str,
    rule_name: str,
    rule_str: str,
    report: Dict[str, Any],
) -> None:
    cleanem_logger.warning(
        "Rejected clean rule: column=%s, rule=%s, reason=%s, tested=%s, "
        "pass_rate=%s, expression=%s",
        column,
        rule_name,
        report.get("reason"),
        report.get("tested"),
        report.get("pass_rate"),
        rule_str,
    )


def _build_clean_rule_pool(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    factory: AgentFactory,
    rule_min_pass_rate: float = RULE_MIN_PASS_RATE,
) -> Dict[str, List[Dict[str, Any]]]:
    if getattr(factory, "_disable_agent_rules", False):
        return {}, {column: [] for column in metadata}
    clean_rules, _ = factory.generate_clean_rules_per_column(metadata)
    family_map = {
        "completeness": "missing",
        "accuracy": "outlier",
        "pattern_consistency": "pattern",
        "column_relationship": "relationship",
    }
    pool: Dict[str, List[Dict[str, Any]]] = {}
    raw_proposals: Dict[str, List[Dict[str, Any]]] = {}
    for column, rules in clean_rules.items():
        col_pool: List[Dict[str, Any]] = []
        col_proposals: List[Dict[str, Any]] = []
        for idx, (agent_name, rule_str) in enumerate(rules):
            family = family_map.get(agent_name)
            if not family:
                continue
            rule_name = f"{agent_name}_{idx}"
            rule_func, validation = _validate_rule_candidate(
                df,
                column,
                family,
                rule_str,
                min_pass_rate=rule_min_pass_rate,
            )
            col_proposals.append({"agent": agent_name, "family": family, "rule_name": rule_name, "rule_str": rule_str, "validation": dict(validation)})
            if rule_func is None:
                _log_rule_rejection(column, rule_name, rule_str, validation)
                continue
            col_pool.append(
                {
                    "agent": agent_name,
                    "family": family,
                    "rule_name": rule_name,
                    "rule_str": rule_str,
                    "rule_func": rule_func,
                    "proposal_source": "llm",
                    "validation": dict(validation),
                }
            )
        raw_proposals[column] = col_proposals
        if col_pool:
            pool[column] = col_pool
    return pool, raw_proposals


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    schema = "|".join(f"{column}:{df[column].dtype}" for column in df.columns)
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    digest.update(hashed)
    return digest.hexdigest()


def _save_relationship_rule_cache(
    path: str,
    df: pd.DataFrame,
    payload: Dict[str, Any],
) -> None:
    _write_json_atomic(path, {
        "version": 1,
        "dataframe_fingerprint": _dataframe_fingerprint(df),
        "columns": list(df.columns),
        **payload,
    })


def _load_relationship_rule_cache(
    path: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != 1:
        raise ValueError(
            f"Unsupported relationship-rule cache version: {payload.get('version')}"
        )
    if payload.get("columns") != list(df.columns):
        raise ValueError("Relationship-rule cache columns do not match the dataset")
    if payload.get("dataframe_fingerprint") != _dataframe_fingerprint(df):
        raise ValueError("Relationship-rule cache fingerprint does not match the dataset")
    rules = payload.get("validated_rules") or []
    if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
        raise ValueError("Relationship-rule cache contains invalid validated rules")
    return payload



def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = f"{absolute_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary_path, absolute_path)


def _save_rule_pool_cache(
    path: str,
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    rule_pool: Dict[str, List[Dict[str, Any]]],
) -> None:
    payload = {
        "version": 1,
        "dataframe_fingerprint": _dataframe_fingerprint(df),
        "columns": list(df.columns),
        "semantics": {
            column: dict(metadata.get(column, {}).get("semantics") or {})
            for column in df.columns
        },
        "rules": {
            column: [
                {
                    "agent": rule.get("agent"),
                    "family": rule.get("family"),
                    "rule_name": rule.get("rule_name"),
                    "rule_str": rule.get("rule_str"),
                    "proposal_source": rule.get("proposal_source", "llm"),
                    "validation": rule.get("validation") or {},
                }
                for rule in rules
            ]
            for column, rules in rule_pool.items()
        },
    }
    _write_json_atomic(path, payload)


def _load_rule_pool_cache(
    path: str,
    df: pd.DataFrame,
    rule_min_pass_rate: float = RULE_MIN_PASS_RATE,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("version") != 1:
        raise ValueError(f"Unsupported rule-pool cache version: {payload.get('version')}")
    if payload.get("columns") != list(df.columns):
        raise ValueError("Rule-pool cache columns do not match the current dataset")
    if payload.get("dataframe_fingerprint") != _dataframe_fingerprint(df):
        raise ValueError("Rule-pool cache fingerprint does not match the current dataset")

    cached_semantics = payload.get("semantics") or {}
    if any(column not in cached_semantics for column in df.columns):
        raise ValueError("Rule-pool cache is missing column semantics")

    pool: Dict[str, List[Dict[str, Any]]] = {}
    for column, rules in (payload.get("rules") or {}).items():
        if column not in df.columns:
            raise ValueError(f"Rule-pool cache contains unknown column: {column}")
        compiled_rules = []
        for item in rules:
            rule_str = str(item.get("rule_str") or "").strip()
            if not rule_str:
                raise ValueError(f"Cached rule has no rule_str for column {column}")
            family = str(item.get("family") or "")
            rule_name = str(item.get("rule_name") or "cached_rule")
            rule_func, validation = _validate_rule_candidate(
                df,
                column,
                family,
                rule_str,
                min_pass_rate=rule_min_pass_rate,
            )
            if rule_func is None:
                _log_rule_rejection(column, rule_name, rule_str, validation)
                continue
            compiled_rules.append({
                "agent": item.get("agent"),
                "family": family,
                "rule_name": rule_name,
                "rule_str": rule_str,
                "rule_func": rule_func,
                "proposal_source": item.get("proposal_source", "llm"),
                "validation": dict(validation),
            })
        if compiled_rules:
            pool[column] = compiled_rules
    return pool, cached_semantics


def _save_canonicalization_plan_cache(
    path: str,
    df: pd.DataFrame,
    plans: Dict[str, Dict[str, Any]],
) -> None:
    _write_json_atomic(path, {
        "version": 1,
        "dataframe_fingerprint": _dataframe_fingerprint(df),
        "columns": list(df.columns),
        "plans": plans,
    })


def _load_canonicalization_plan_cache(
    path: str,
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != 1:
        raise ValueError(
            f"Unsupported canonicalization-plan cache version: {payload.get('version')}"
        )
    if payload.get("columns") != list(df.columns):
        raise ValueError("Canonicalization-plan cache columns do not match the dataset")
    if payload.get("dataframe_fingerprint") != _dataframe_fingerprint(df):
        raise ValueError("Canonicalization-plan cache fingerprint does not match the dataset")
    plans = payload.get("plans") or {}
    if any(column not in plans for column in df.columns):
        raise ValueError("Canonicalization-plan cache is missing one or more columns")
    if any(not isinstance(plans[column], dict) for column in df.columns):
        raise ValueError("Canonicalization-plan cache contains an invalid plan")
    return {column: dict(plans[column]) for column in df.columns}


def _save_active_query_cache(
    path: str,
    df: pd.DataFrame,
    budget: int,
    cell_keys: List[str],
    strategy: str,
    rounds: int,
    batch_size: int,
    query_rounds: Optional[List[List[str]]] = None,
) -> None:
    _write_json_atomic(path, {
        "version": 4,
        "dataframe_fingerprint": _dataframe_fingerprint(df),
        "budget": int(budget),
        "strategy": str(strategy),
        "rounds": int(rounds),
        "batch_size": int(batch_size),
        "cell_keys": list(cell_keys),
        "query_rounds": [list(batch) for batch in (query_rounds or [])],
    })


def _load_active_query_cache(
    path: str,
    df: pd.DataFrame,
    budget: int,
    valid_cell_keys: set,
    strategy: str,
    rounds: int,
    batch_size: int,
) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != 4:
        raise ValueError(
            "Unsupported active-query cache version: "
            f"{payload.get('version')}; delete this cache and regenerate it"
        )
    if payload.get("dataframe_fingerprint") != _dataframe_fingerprint(df):
        raise ValueError("Active-query cache fingerprint does not match the current dataset")
    if int(payload.get("budget", -1)) != int(budget):
        raise ValueError("Active-query cache budget does not match --active_label_budget")
    if str(payload.get("strategy")) != str(strategy):
        raise ValueError("Active-query cache strategy does not match --active_query_strategy")
    if int(payload.get("rounds", -1)) != int(rounds):
        raise ValueError("Active-query cache rounds do not match --active_query_rounds")
    if int(payload.get("batch_size", -1)) != int(batch_size):
        raise ValueError(
            "Active-query cache batch size does not match --active_query_batch_size"
        )

    cell_keys = [str(key) for key in payload.get("cell_keys") or []]
    if len(cell_keys) != len(set(cell_keys)):
        raise ValueError("Active-query cache contains duplicate cell keys")
    if len(cell_keys) > int(budget):
        raise ValueError("Active-query cache contains more cells than the configured budget")
    unknown = [key for key in cell_keys if key not in valid_cell_keys]
    if unknown:
        raise ValueError(f"Active-query cache contains unknown cell keys: {unknown[:3]}")
    return cell_keys


def _load_active_query_seed_cache(
    path: str,
    df: pd.DataFrame,
    target_budget: int,
    valid_cell_keys: set,
    strategy: str,
    rounds: int,
    batch_size: int,
) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    seed_budget = int(payload.get("budget", -1))
    if seed_budget < 0 or seed_budget > int(target_budget):
        raise ValueError(
            "Active-query seed budget must be non-negative and no larger "
            "than --active_label_budget"
        )
    return _load_active_query_cache(
        path,
        df,
        seed_budget,
        valid_cell_keys,
        strategy,
        rounds,
        batch_size,
    )


def _parse_active_query_checkpoints(raw: str) -> List[int]:
    checkpoints = sorted({
        int(item.strip())
        for item in str(raw or "").split(",")
        if item.strip()
    })
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("Active-query checkpoints must be non-negative integers")
    if 0 not in checkpoints:
        checkpoints.insert(0, 0)
    return checkpoints


def _evidence_matrix_fingerprint(evidence_matrix: Any) -> str:
    digest = hashlib.sha256()
    for values in (
        evidence_matrix.cell_keys,
        evidence_matrix.feature_names,
        evidence_matrix.query_feature_names,
        sorted(evidence_matrix.query_excluded_features),
    ):
        digest.update(json.dumps(list(values), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
    for matrix in (evidence_matrix.values, evidence_matrix.query_values):
        if matrix is None:
            digest.update(b"none\0")
            continue
        array = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mechanism_pool_fingerprint(adaptive_report: Any) -> str:
    if adaptive_report is None:
        return "none"
    payload = []
    for candidate in rank_mechanism_candidates(adaptive_report):
        payload.append({
            "label": candidate.label,
            "signature": candidate.signature,
            "source_kind": candidate.source_kind,
            "support": int(candidate.support),
            "row_support": int(candidate.row_support),
            "strength": round(float(candidate.strength), 12),
            "uncertainty": round(float(candidate.uncertainty), 12),
            "repairability": round(float(candidate.repairability), 12),
            "representative_cells": list(candidate.representative_cells),
            "member_cells": list(candidate.member_cells),
        })
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_nested_active_query_trajectory(
    evidence_matrix: Any,
    adaptive_report: Any,
    checkpoints: List[int],
    args: argparse.Namespace,
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, Any]:
    """Build one nested trajectory using D-optimal selection then replacement.

    The historical Full protocol first selected a D-optimal calibration set
    and then replaced a bounded number of its cells with representatives from
    validated mechanism candidates. Budget checkpoints retain that order,
    while the final sets remain prefix-nested: a later checkpoint extends the
    previous final set with new D-optimal cells before applying only the new
    mechanism replacements.
    """
    if args.active_query_strategy != "d_optimal":
        raise ValueError("Nested budget trajectories currently require d_optimal selection")
    if int(args.adaptive_mechanism_query_budget) != 0:
        raise ValueError(
            "Nested budget trajectories require adaptive_mechanism_query_budget=0"
        )
    if adaptive_report is None:
        raise ValueError("Nested budget trajectories require adaptive mechanism discovery")

    verification_samples = max(1, int(args.adaptive_mechanism_min_labeled_support))
    mechanism_fraction = max(
        0.0, min(1.0, float(args.adaptive_mechanism_query_fraction))
    )
    explicit_candidate_cap = int(args.adaptive_mechanism_max_candidates_per_round)
    valid_cell_keys = set(evidence_matrix.cell_keys)
    ranked_candidates = rank_mechanism_candidates(adaptive_report)
    candidate_members: Dict[str, List[str]] = {}
    for candidate in ranked_candidates:
        members = list(dict.fromkeys(
            list(candidate.representative_cells) + list(candidate.member_cells)
        ))
        candidate_members[candidate.label] = [
            key for key in members if key in valid_cell_keys
        ]

    # First construct the historical D-optimal trajectory without allowing
    # future mechanism candidates to bias the base design. The previous base
    # prefix is protected so clean-proxy replacement cannot alter a completed
    # checkpoint.
    base_selected: List[str] = []
    base_roles: Dict[str, str] = {}
    base_checkpoint_cells: Dict[str, List[str]] = {}
    base_checkpoint_roles: Dict[str, Dict[str, str]] = {}
    base_checkpoint_coverage: Dict[str, Dict[str, Any]] = {}
    for checkpoint in checkpoints:
        checkpoint = max(0, int(checkpoint))
        previous_base = list(base_selected)
        base_selected = select_active_queries(
            evidence_matrix,
            checkpoint,
            clean_proxy_fraction=args.active_query_clean_proxy_fraction,
            include_query_features=False,
            strategy=args.active_query_strategy,
            initial_cell_keys=base_selected,
            protected_cell_keys=set(previous_base),
        )
        base_added = [key for key in base_selected if key not in previous_base]
        for key in base_added:
            base_roles[key] = "d_optimal"

        structural_summary: Dict[str, Any] = {
            "inserted": [],
            "replaced": [],
            "structural_archetypes": [],
        }
        if args.enable_structural_missing_query_coverage and checkpoint > 0:
            before_coverage = list(base_selected)
            base_selected, structural_summary = _ensure_structural_missing_query_coverage(
                base_selected,
                checkpoint,
                cell_observations,
                cell_registry,
                metadata,
                protected_cell_keys=set(previous_base),
            )
            for key in structural_summary.get("replaced", []):
                base_roles.pop(key, None)
            for key in structural_summary.get("inserted", []):
                base_roles[key] = "d_optimal_structural_coverage"
            if base_selected[:len(previous_base)] != previous_base:
                raise RuntimeError(
                    "Structural coverage modified a completed D-optimal prefix"
                )
            if len(base_selected) != len(before_coverage):
                raise RuntimeError("Structural coverage changed the query budget")

        if len(base_selected) != checkpoint:
            raise RuntimeError(
                f"Could not fill D-optimal checkpoint {checkpoint}: "
                f"selected={len(base_selected)}"
            )
        if base_selected[:len(previous_base)] != previous_base:
            raise RuntimeError(
                f"D-optimal trajectory invariant failed at checkpoint {checkpoint}"
            )
        base_checkpoint_cells[str(checkpoint)] = list(base_selected)
        base_checkpoint_roles[str(checkpoint)] = {
            key: base_roles[key] for key in base_selected
        }
        base_checkpoint_coverage[str(checkpoint)] = structural_summary

    # Then apply the old fixed-budget mechanism replacement policy to each
    # newly extended suffix. Only suffix cells may be replaced, which keeps
    # the final checkpoint sets strictly prefix-nested.
    selected: List[str] = []
    roles: Dict[str, str] = {}
    eligible_mechanisms: List[str] = []
    pending_candidates = list(ranked_candidates)
    checkpoint_cells: Dict[str, List[str]] = {}
    checkpoint_mechanisms: Dict[str, List[str]] = {}
    checkpoint_roles: Dict[str, Dict[str, str]] = {}
    checkpoint_summaries: Dict[str, Dict[str, Any]] = {}

    for checkpoint in checkpoints:
        checkpoint = max(0, int(checkpoint))
        previous_prefix = list(selected)
        base_prefix = base_checkpoint_cells[str(checkpoint)]
        d_optimal_added: List[str] = []

        # Extend the previous final set using the already-generated D-optimal
        # trajectory. Replaced base cells can reappear at a later checkpoint;
        # they are new suffix labels and this preserves both fixed budgets and
        # the historical replacement semantics.
        for key in base_prefix:
            if len(selected) >= checkpoint:
                break
            if key in selected:
                continue
            selected.append(key)
            roles[key] = base_checkpoint_roles[str(checkpoint)].get(
                key, "d_optimal"
            )
            d_optimal_added.append(key)
        if len(selected) < checkpoint:
            raise RuntimeError(
                f"Could not extend nested checkpoint {checkpoint} from D-optimal trajectory"
            )

        proportional_target = int(
            math.floor(checkpoint * mechanism_fraction / verification_samples)
        )
        target_mechanisms = (
            min(proportional_target, explicit_candidate_cap)
            if explicit_candidate_cap > 0
            else proportional_target
        )
        mechanism_cells_added: List[str] = []
        replaced_cells: List[str] = []
        skipped_candidates: List[str] = []

        # Revisit pending candidates at later checkpoints when a previous
        # suffix was too small to provide all required replacement slots.
        while len(eligible_mechanisms) < target_mechanisms:
            progress = False
            next_pending: List[MechanismCandidate] = []
            for candidate in pending_candidates:
                if len(eligible_mechanisms) >= target_mechanisms:
                    next_pending.append(candidate)
                    continue
                members = candidate_members.get(candidate.label, [])
                member_set = set(members)
                already_selected = [key for key in selected if key in member_set]
                if len(already_selected) >= verification_samples:
                    for key in already_selected[:verification_samples]:
                        roles[key] = f"mechanism:{candidate.label}"
                    eligible_mechanisms.append(candidate.label)
                    progress = True
                    continue

                needed = verification_samples - len(already_selected)
                additions = [
                    key for key in members if key not in selected
                ][:needed]
                suffix_positions = [
                    index
                    for index in range(len(selected) - 1, len(previous_prefix) - 1, -1)
                    if selected[index] not in member_set
                    and not str(roles.get(selected[index], "")).startswith("mechanism:")
                ]
                if len(additions) < needed or len(suffix_positions) < needed:
                    skipped_candidates.append(candidate.label)
                    next_pending.append(candidate)
                    continue

                for key in already_selected[:verification_samples]:
                    roles[key] = f"mechanism:{candidate.label}"
                for addition, replace_index in zip(additions, suffix_positions):
                    replaced = selected[replace_index]
                    replaced_cells.append(replaced)
                    roles.pop(replaced, None)
                    selected[replace_index] = addition
                    roles[addition] = f"mechanism:{candidate.label}"
                    mechanism_cells_added.append(addition)
                eligible_mechanisms.append(candidate.label)
                progress = True

            pending_candidates = next_pending
            if not progress:
                break

        if len(selected) != checkpoint:
            raise RuntimeError(
                f"Could not fill nested query checkpoint {checkpoint}: "
                f"selected={len(selected)}"
            )
        if selected[:len(previous_prefix)] != previous_prefix:
            raise RuntimeError(
                f"Nested query invariant failed at checkpoint {checkpoint}"
            )
        if len(selected) != len(set(selected)):
            raise RuntimeError(
                f"Nested query checkpoint {checkpoint} contains duplicate cells"
            )

        checkpoint_key = str(checkpoint)
        checkpoint_cells[checkpoint_key] = list(selected)
        checkpoint_mechanisms[checkpoint_key] = list(eligible_mechanisms)
        checkpoint_roles[checkpoint_key] = {
            key: roles[key] for key in selected
        }
        checkpoint_summaries[checkpoint_key] = {
            "budget": checkpoint,
            "selection_order": "d_optimal_then_mechanism_replacement",
            "mechanism_fraction": mechanism_fraction,
            "verification_samples": verification_samples,
            "target_mechanisms": target_mechanisms,
            "allocated_mechanisms": len(eligible_mechanisms),
            "mechanism_label_count": sum(
                str(roles.get(key, "")).startswith("mechanism:")
                for key in selected
            ),
            "d_optimal_label_count": sum(
                not str(roles.get(key, "")).startswith("mechanism:")
                for key in selected
            ),
            "mechanism_cells_added": mechanism_cells_added,
            "mechanism_cells_replaced": replaced_cells,
            "d_optimal_cells_added": d_optimal_added,
            "skipped_mechanisms": sorted(set(skipped_candidates)),
            "structural_coverage": base_checkpoint_coverage[checkpoint_key],
        }

    for earlier, later in zip(checkpoints, checkpoints[1:]):
        earlier_cells = checkpoint_cells[str(earlier)]
        later_cells = checkpoint_cells[str(later)]
        if later_cells[:len(earlier_cells)] != earlier_cells:
            raise RuntimeError(
                f"Trajectory is not nested: S{earlier} is not a prefix of S{later}"
            )

    return {
        "version": 2,
        "selection_order": "d_optimal_then_mechanism_replacement",
        "dataframe_fingerprint": None,
        "evidence_matrix_fingerprint": _evidence_matrix_fingerprint(evidence_matrix),
        "mechanism_pool_fingerprint": _mechanism_pool_fingerprint(adaptive_report),
        "seed": os.environ.get("CLEANEM_LLM_SEED", ""),
        "strategy": args.active_query_strategy,
        "clean_proxy_fraction": float(args.active_query_clean_proxy_fraction),
        "mechanism_label_fraction": mechanism_fraction,
        "verification_samples": verification_samples,
        "checkpoints": list(checkpoints),
        "cell_keys": list(selected),
        "checkpoint_cell_keys": checkpoint_cells,
        "checkpoint_eligible_mechanisms": checkpoint_mechanisms,
        "checkpoint_roles": checkpoint_roles,
        "checkpoint_summaries": checkpoint_summaries,
    }


def _save_active_query_trajectory_cache(
    path: str,
    df: pd.DataFrame,
    payload: Dict[str, Any],
) -> None:
    serialized = dict(payload)
    serialized["dataframe_fingerprint"] = _dataframe_fingerprint(df)
    _write_json_atomic(path, serialized)


def _load_active_query_trajectory_cache(
    path: str,
    df: pd.DataFrame,
    evidence_matrix: Any,
    adaptive_report: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != 2:
        raise ValueError("Unsupported active-query trajectory cache version")
    if payload.get("selection_order") != "d_optimal_then_mechanism_replacement":
        raise ValueError("Trajectory cache selection order mismatch")
    if payload.get("dataframe_fingerprint") != _dataframe_fingerprint(df):
        raise ValueError("Trajectory cache dataframe fingerprint mismatch")
    if payload.get("evidence_matrix_fingerprint") != _evidence_matrix_fingerprint(
        evidence_matrix
    ):
        raise ValueError("Trajectory cache evidence-matrix fingerprint mismatch")
    if payload.get("mechanism_pool_fingerprint") != _mechanism_pool_fingerprint(
        adaptive_report
    ):
        raise ValueError("Trajectory cache mechanism-pool fingerprint mismatch")
    if payload.get("strategy") != args.active_query_strategy:
        raise ValueError("Trajectory cache query strategy mismatch")
    if float(payload.get("clean_proxy_fraction", -1.0)) != float(
        args.active_query_clean_proxy_fraction
    ):
        raise ValueError("Trajectory cache clean-proxy fraction mismatch")
    if float(payload.get("mechanism_label_fraction", -1.0)) != float(
        args.adaptive_mechanism_query_fraction
    ):
        raise ValueError("Trajectory cache mechanism fraction mismatch")
    if int(payload.get("verification_samples", -1)) != int(
        args.adaptive_mechanism_min_labeled_support
    ):
        raise ValueError("Trajectory cache mechanism support mismatch")

    checkpoints = [int(value) for value in payload.get("checkpoints") or []]
    expected_checkpoints = _parse_active_query_checkpoints(
        args.active_query_checkpoints
    )
    if checkpoints != expected_checkpoints:
        raise ValueError("Trajectory cache checkpoints mismatch")
    valid_cell_keys = set(evidence_matrix.cell_keys)
    previous: List[str] = []
    for checkpoint in checkpoints:
        cells = list(
            (payload.get("checkpoint_cell_keys") or {}).get(str(checkpoint)) or []
        )
        if len(cells) != checkpoint or len(cells) != len(set(cells)):
            raise ValueError(f"Invalid trajectory checkpoint {checkpoint}")
        if any(key not in valid_cell_keys for key in cells):
            raise ValueError(f"Unknown cell in trajectory checkpoint {checkpoint}")
        if cells[:len(previous)] != previous:
            raise ValueError(f"Trajectory checkpoint {checkpoint} is not nested")
        previous = cells
    return payload


def _compute_rule_outputs_for_column(
    df: pd.DataFrame,
    column: str,
    rules: List[Dict[str, Any]],
) -> np.ndarray:
    n_rows = len(df)
    if n_rows == 0 or not rules:
        return np.zeros((0, len(rules)), dtype=int)
    outputs = np.zeros((n_rows, len(rules)), dtype=int)
    for j, rule in enumerate(rules):
        func = rule["rule_func"]
        for i, (_, row) in enumerate(df.iterrows()):
            value = row[column]
            outputs[i, j] = int(_invoke_rule(func, value, row))
    return outputs


def _normalize_signature(value: Any) -> str:
    if value is None:
        return "<na>"
    if isinstance(value, float) and np.isnan(value):
        return "<na>"
    normalized = str(value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized if normalized else "<empty>"


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    normalized = str(value).strip().lower()
    return re.sub(r"\s+", " ", normalized)


_EXPLICIT_MISSING_MARKER_RE = re.compile(
    r"^(?:\{\s*(?:null|none|nan|n/a|na)\s*\}|"
    r"\[\s*(?:null|none|nan|n/a|na)\s*\]|"
    r"<\s*(?:null|none|nan|n/a|na)\s*>|"
    r"\(\s*(?:null|none|nan|n/a|na)\s*\))$"
)


def _is_explicit_missing_marker(value: Any) -> bool:
    """Recognize serialized null markers without treating arbitrary text as missing."""
    return bool(_EXPLICIT_MISSING_MARKER_RE.fullmatch(_normalize_token(value)))


def _shape_signature(value: Any) -> str:
    if value is None:
        return "<empty>"
    tokens: List[str] = []
    for char in str(value).strip():
        if char.isdigit():
            tokens.append("D")
        elif char.isalpha():
            tokens.append("A")
        elif char in "-_/":
            tokens.append(char)
        elif char.isspace():
            tokens.append("S")
        else:
            tokens.append(char)
    return "".join(tokens) if tokens else "<empty>"


def _is_missing_like(value: Any, metadata: Dict[str, Any]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    normalized = _normalize_token(value)
    dominant_tokens = set(metadata.get("dominant_missing_tokens") or [])
    return (
        normalized in DEFAULT_MISSING_TOKEN_SET
        or normalized in dominant_tokens
        or _is_explicit_missing_marker(value)
    )


def _geo_suffix_shift_observations(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, int]:
    """Detect generic geo-name + missing-region pairs using observed values."""
    geo_columns = [
        column for column in df.columns
        if _column_archetype(metadata.get(column, {})) == "geo_name"
    ]
    region_columns = [
        column for column in df.columns
        if _column_archetype(metadata.get(column, {})) in {"closed_enum", "geo_name"}
    ]
    emitted = 0
    pairs = 0
    for geo_column in geo_columns:
        for region_column in region_columns:
            if geo_column == region_column:
                continue
            # Use the complete observed reference vocabulary. Profiler
            # ``top_values`` is intentionally truncated and is insufficient
            # for long-tail region codes.
            legal = {
                _normalize_signature(raw)
                for raw in df[region_column].tolist()
                if not _is_missing_like(raw, metadata.get(region_column, {}))
            }
            if not legal:
                continue
            pairs += 1
            for row_index in range(len(df)):
                geo_value = df.iloc[row_index][geo_column]
                region_value = df.iloc[row_index][region_column]
                if _is_missing_like(geo_value, metadata.get(geo_column, {})):
                    continue
                if not _is_missing_like(region_value, metadata.get(region_column, {})):
                    continue
                match = re.match(r"^(.+?)\s+([A-Za-z]{2,})$", str(geo_value).strip())
                if not match:
                    continue
                prefix = match.group(1).strip()
                suffix = _normalize_signature(match.group(2))
                if not prefix or suffix not in legal:
                    continue
                geo_key = f"{row_index}::{geo_column}"
                region_key = f"{row_index}::{region_column}"
                if geo_key not in cell_registry or region_key not in cell_registry:
                    continue
                common = {
                    "reference_column": region_column,
                    "candidate_region": match.group(2).strip(),
                    "candidate_geo_value": prefix,
                    "row_index": row_index,
                    "generic_mechanism": True,
                }
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell", target_key=geo_key,
                    source_id="geo_suffix_pollution", family="relationship",
                    polarity="dirty", strength=1.0, hard=True,
                    reason_code="geo_suffix_shift", metadata=common,
                ))
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell", target_key=region_key,
                    source_id="displaced_region_missing", family="relationship",
                    polarity="dirty", strength=1.0, hard=True,
                    reason_code="geo_suffix_shift", metadata=common,
                ))
                emitted += 2
    return {"pairs": pairs, "emitted_cells": emitted}


def _is_structural_missing_group(support: int, total_rows: int) -> bool:
    """Frequent placeholders represent allowed absence, not an error alone."""
    return (
        int(support) >= STRUCTURAL_MISSING_MIN_SUPPORT
        and float(support) / max(int(total_rows), 1) >= STRUCTURAL_MISSING_MIN_RATE
    )


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            numeric = float(value)
        except Exception:
            return None
        return None if np.isnan(numeric) else numeric
    text = str(value).strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.startswith("$"):
        text = text[1:].strip()
    text = text.replace(",", "")
    if not text:
        return None
    if negative:
        text = f"-{text}"
    try:
        numeric = float(text)
    except Exception:
        return None
    return None if np.isnan(numeric) else numeric


def _is_full_string_numeric(value: Any) -> bool:
    """Return true only when the complete value is a scalar numeric token."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    text = str(value).strip()
    if not text:
        return False
    if not re.fullmatch(
        r"[-+]?(?:(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        text,
    ):
        return False
    try:
        return bool(np.isfinite(float(text.replace(",", ""))))
    except (TypeError, ValueError):
        return False


def _matching_regex_rate(value: Any, regex_candidates: List[Dict[str, Any]]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    best_rate = 0.0
    for candidate in regex_candidates or []:
        pattern = str(candidate.get("pattern", ""))
        if not pattern:
            continue
        try:
            if re.fullmatch(pattern, text):
                best_rate = max(best_rate, float(candidate.get("match_rate", 0.0)))
        except re.error:
            continue
    return float(best_rate)


def _best_regex_contract_rate(metadata: Dict[str, Any]) -> float:
    regex_candidates = metadata.get("regex_candidates") or []
    if not regex_candidates:
        return 0.0
    return float(max(float(candidate.get("match_rate", 0.0)) for candidate in regex_candidates))


def _has_trusted_numeric_contract(metadata: Dict[str, Any]) -> bool:
    if metadata.get("type") != "numeric":
        return False
    if bool(metadata.get("native_numeric_dtype")):
        return True

    semantics = metadata.get("semantics") or {}
    archetype = str(
        semantics.get("archetype", "unknown")
        if isinstance(semantics, dict)
        else getattr(semantics, "archetype", "unknown")
    )
    # Numeric-looking identifiers, dates, and page ranges may contain digits
    # in nearly every value without being scalar measurements. Only a
    # measurement archetype may promote string parse failures to hard labels.
    if archetype not in {"numeric_measure", "unit_measure"}:
        return False

    fully_numeric_count = int(metadata.get("fully_numeric_count") or 0)
    non_null_count = int(metadata.get("non_null_count") or 0)
    fully_numeric_ratio = (
        float(fully_numeric_count / non_null_count)
        if non_null_count > 0 else 0.0
    )
    return bool(fully_numeric_ratio >= 0.80)


def _has_pagination_contamination(
    column: str,
    value: Any,
    metadata: Dict[str, Any],
) -> bool:
    """Detect narrow, schema-specific suffix and delimiter corruption."""
    semantics = metadata.get("semantics") or {}
    archetype = str(
        semantics.get("archetype", "unknown")
        if isinstance(semantics, dict)
        else getattr(semantics, "archetype", "unknown")
    )
    normalized_column = re.sub(r"[^a-z0-9]+", "_", str(column).lower())
    if archetype != "identifier" or not any(
        token in normalized_column for token in ("page", "pagination")
    ):
        return False
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if not text or text == "-":
        return False
    return bool(
        re.search(r"\s+(?:ST|DO)\s+-", text, flags=re.IGNORECASE)
        or re.search(r"\s+\d+p$", text, flags=re.IGNORECASE)
        or "--" in text
        or (text[0].isdigit() and text.endswith("-"))
    )


def _profile_value_support_ratio(
    value: Any,
    metadata: Dict[str, Any],
) -> float:
    normalized = _normalize_token(value)
    for item in metadata.get("normalized_top_values") or []:
        if _normalize_token(item.get("value")) != normalized:
            continue
        try:
            return float(item.get("ratio") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _surface_role_conflict(
    value: Any,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Detect a surface type/shape that conflicts with a learned column role."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    archetype = _column_archetype(metadata)
    if archetype in {
        "identifier",
        "numeric_measure",
        "unit_measure",
        "temporal_measure",
    }:
        return None
    if str(metadata.get("type", "text")).lower() not in {"text", "categorical"}:
        return None

    text = str(value).strip()
    non_null_count = int(metadata.get("non_null_count") or 0)
    numeric_ratio = float(metadata.get("full_numeric_surface_ratio") or 0.0)
    alphabetic_ratio = float(metadata.get("alphabetic_surface_ratio") or 0.0)
    if non_null_count < 20 or numeric_ratio > 0.10 or alphabetic_ratio < 0.80:
        return None

    full_numeric = _is_full_string_numeric(value)
    if full_numeric:
        numeric = _coerce_numeric(value)
        repeated_digits = bool(re.fullmatch(r"(\d)\1{5,}", text))
        compact_date_shape = bool(re.fullmatch(
            r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])",
            text,
        ))
        signed_scalar = bool(numeric is not None and numeric < 0.0)
        value_support = _profile_value_support_ratio(value, metadata)
        strong_role_conflict = archetype in {"open_entity", "geo_name"}
        anomalous_numeric_shape = bool(
            repeated_digits or compact_date_shape or signed_scalar
        )
        if not (
            strong_role_conflict
            or anomalous_numeric_shape
            or value_support <= 0.005
        ):
            return None
        # A numeric surface is a high-confidence role violation when the
        # column is demonstrably lexical.  This remains generic: the role is
        # inferred from the profile, while identifiers and numeric/temporal
        # columns are excluded above.  The old rule required an anomalous
        # numeric shape as well, which let repeated values such as ``-1`` pass
        # through as soft evidence and depressed recall after calibration.
        hard = bool(
            alphabetic_ratio >= 0.85
            and numeric_ratio <= 0.08
            and (strong_role_conflict or anomalous_numeric_shape)
        )
        return {
            "source_id": "type_role_conflict",
            "reason_code": "numeric_surface_role_conflict",
            "strength": 1.0 if hard else (0.90 if strong_role_conflict else 0.80),
            "hard": hard,
            "metadata": {
                "observed_surface_type": "numeric",
                "column_archetype": archetype,
                "full_numeric_surface_ratio": numeric_ratio,
                "alphabetic_surface_ratio": alphabetic_ratio,
                "value_support_ratio": value_support,
                "numeric_shape": (
                    "repeated_digits" if repeated_digits
                    else "compact_date" if compact_date_shape
                    else "signed_scalar" if signed_scalar
                    else "scalar"
                ),
                "calibration_role": (
                    "direct_decision" if hard else "learned_evidence"
                ),
                "exclude_from_query_design": not hard,
            },
        }

    if len(text) == 1 and text.isalpha():
        length_one_ratio = 0.0
        for item in metadata.get("length_distribution") or []:
            if int(item.get("length") or 0) == 1:
                length_one_ratio = float(item.get("ratio") or 0.0)
                break
        dominant_values = metadata.get("normalized_top_values") or []
        dominant = dominant_values[0] if dominant_values else {}
        dominant_value = str(dominant.get("value") or "")
        try:
            dominant_ratio = float(dominant.get("ratio") or 0.0)
        except (TypeError, ValueError):
            dominant_ratio = 0.0
        strong_role_conflict = archetype == "open_entity"
        concentrated_longer_shape = bool(
            len(dominant_value) >= 3 and dominant_ratio >= 0.80
        )
        if length_one_ratio <= 0.05 and (
            strong_role_conflict or concentrated_longer_shape
        ):
            hard = bool(alphabetic_ratio >= 0.90)
            return {
                "source_id": "surface_shape_conflict",
                "reason_code": "single_character_shape_conflict",
                "strength": 1.0 if hard else 0.80,
                "hard": hard,
                "metadata": {
                    "observed_surface_type": "single_character_text",
                    "column_archetype": archetype,
                    "alphabetic_surface_ratio": alphabetic_ratio,
                    "length_one_ratio": length_one_ratio,
                    "dominant_value_length": len(dominant_value),
                    "dominant_value_ratio": dominant_ratio,
                    "calibration_role": (
                        "direct_decision" if hard else "learned_evidence"
                    ),
                    "exclude_from_query_design": not hard,
                },
            }
    return None


def _temporal_range_conflict(
    value: Any,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Detect implausible numeric values in a profiled temporal column.

    Parsing success only establishes that a value is numeric.  For year-like
    columns, an extreme value can still be a corruption.  The bounds are
    derived from the observed distribution and broad calendar limits; no
    dataset-specific years or column names are required.
    """
    if _column_archetype(metadata) != "temporal_measure":
        return None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None

    # A four-digit calendar surface is the common temporal contract.  Keep a
    # broad safety envelope so legitimate historical/future dates are not
    # rejected solely because they are rare.
    is_year_surface = bool(re.fullmatch(r"[-+]?\d{4}", text))
    if not is_year_surface:
        return None
    current_year = datetime.now().year
    broad_invalid = numeric < 1000 or numeric > current_year + 2
    quantiles = metadata.get("quantiles") or {}
    try:
        p01 = float(quantiles.get("p01"))
        p99 = float(quantiles.get("p99"))
        p05 = float(quantiles.get("p05"))
        p95 = float(quantiles.get("p95"))
    except (TypeError, ValueError):
        p01 = p99 = p05 = p95 = float("nan")
    robust_invalid = False
    if all(np.isfinite(item) for item in (p01, p99, p05, p95)):
        spread = max(p95 - p05, 1.0)
        robust_invalid = numeric < p01 - max(5.0, 2.0 * spread) or numeric > p99 + max(5.0, 2.0 * spread)
    if not (broad_invalid or robust_invalid):
        return None
    return {
        "source_id": "temporal_range_violation",
        "reason_code": "temporal_range_violation",
        "strength": 1.0 if broad_invalid else 0.85,
        # Broad calendar violations are deterministic. Distribution-only
        # violations remain soft because the observed range can be truncated.
        "hard": broad_invalid,
        "metadata": {
            "observed_value": numeric,
            "broad_calendar_violation": broad_invalid,
            "distribution_violation": robust_invalid,
            "quantiles": {
                "p01": p01,
                "p99": p99,
                "p05": p05,
                "p95": p95,
            },
            "calibration_role": "direct_decision" if broad_invalid else "learned_evidence",
        },
    }


def _infer_percent_decimal_contract(
    series: pd.Series,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Infer a dominant bare-decimal contract with percent contamination."""
    if _column_archetype(metadata) not in {"numeric_measure", "unit_measure"}:
        return None

    bare_values: List[float] = []
    percent_values: List[float] = []
    considered = 0
    for value in series.dropna().tolist():
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        considered += 1
        bare_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
        percent_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*%", text)
        if bare_match:
            bare_values.append(float(text))
        elif percent_match:
            percent_values.append(float(percent_match.group(1)))

    structured_count = len(bare_values) + len(percent_values)
    if considered < 20 or not percent_values or structured_count <= 0:
        return None
    structured_coverage = float(structured_count / considered)
    bare_ratio = float(len(bare_values) / structured_count)
    fraction_scale_ratio = float(
        sum(abs(value) <= 1.0 for value in bare_values) / max(len(bare_values), 1)
    )
    if (
        structured_coverage < 0.90
        or bare_ratio < 0.70
        or fraction_scale_ratio < 0.95
    ):
        return None
    return {
        "considered": considered,
        "structured_count": structured_count,
        "structured_coverage": structured_coverage,
        "bare_count": len(bare_values),
        "percent_count": len(percent_values),
        "bare_ratio": bare_ratio,
        "fraction_scale_ratio": fraction_scale_ratio,
    }


def _unicode_corruption_kind(value: Any) -> Optional[str]:
    """Return a narrow Unicode corruption class, or abstain."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    if "\ufffd" in text:
        return "replacement_character"
    # Canonically decomposed accents compose under NFC. Marks that remain
    # unattached/uncomposed are suspicious but not intrinsically invalid.
    normalized = unicodedata.normalize("NFC", text)
    if any(unicodedata.category(char).startswith("M") for char in normalized):
        return "unresolved_combining_mark"
    # Mixed-script substitutions are a common encoding/homoglyph artifact:
    # preserve all-Latin names and genuinely non-Latin values, but flag a
    # small non-Latin insertion inside an otherwise Latin surface. Requiring
    # at least three Latin letters keeps short codes from becoming noisy.
    latin_letters = [
        char for char in normalized
        if char.isalpha() and "LATIN" in unicodedata.name(char, "")
    ]
    non_latin_letters = [
        char for char in normalized
        if char.isalpha() and "LATIN" not in unicodedata.name(char, "")
    ]
    if (
        len(latin_letters) >= 3
        and non_latin_letters
        and len(non_latin_letters) * 3 <= len(latin_letters)
    ):
        return "non_latin_mixed_script"
    return None


_REPRESENTATION_TEXT_ARCHETYPES = {"free_text", "open_entity"}
_REPRESENTATION_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "de", "du", "et", "for",
    "from", "in", "la", "le", "les", "of", "on", "the", "to", "with",
}


def _representation_tokens(value: Any) -> List[str]:
    """Tokenize a lexical value for conservative abbreviation matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    # NFKD makes accents comparable while the token matcher remains ASCII
    # based; this is only used as a relation signal, never as a repair.
    text = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return [
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _REPRESENTATION_STOPWORDS
    ]


def _abbreviation_coverage(short_value: Any, long_value: Any) -> float:
    """Return the fraction of short-form tokens explained by a long form."""
    short_tokens = _representation_tokens(short_value)
    long_tokens = _representation_tokens(long_value)
    if not short_tokens or not long_tokens:
        return 0.0
    used: Set[int] = set()
    matched = 0
    for short_token in short_tokens:
        candidates = [
            (len(short_token) / max(len(long_token), 1), index)
            for index, long_token in enumerate(long_tokens)
            if index not in used and long_token.startswith(short_token)
        ]
        if not candidates:
            continue
        _, index = max(candidates)
        used.add(index)
        matched += 1
    return float(matched / max(len(short_tokens), 1))


def _normalized_representation_text(value: Any) -> str:
    """Normalize lexical text for the prefix-extension guard."""
    return "".join(_representation_tokens(value))


def _add_representation_role_conflict_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, Any]:
    """Detect swapped long/short representations between open-entity columns.

    The relation is learned from column statistics and row-level token-prefix
    agreement.  It deliberately requires substantial support and a clear
    length asymmetry, so unrelated lexical columns (for example categories and
    names) do not become accidental relationships.
    """
    eligible_columns = [
        column for column in df.columns
        if _column_archetype(metadata.get(column, {}))
        in _REPRESENTATION_TEXT_ARCHETYPES
    ]
    emitted_cells = 0
    accepted_pairs = 0
    candidate_rows = 0
    for left_index, left_column in enumerate(eligible_columns):
        for right_column in eligible_columns[left_index + 1:]:
            joint_rows: List[Tuple[int, str, str]] = []
            for row_index in range(len(df)):
                left_value = df.iloc[row_index][left_column]
                right_value = df.iloc[row_index][right_column]
                if _is_missing_like(left_value, metadata.get(left_column, {})):
                    continue
                if _is_missing_like(right_value, metadata.get(right_column, {})):
                    continue
                left_text = str(left_value).strip()
                right_text = str(right_value).strip()
                if left_text and right_text:
                    joint_rows.append((row_index, left_text, right_text))
            if len(joint_rows) < 30:
                continue

            left_median = float(np.median([len(item[1]) for item in joint_rows]))
            right_median = float(np.median([len(item[2]) for item in joint_rows]))
            shorter_median = min(left_median, right_median)
            longer_median = max(left_median, right_median)
            if shorter_median <= 0.0 or longer_median / shorter_median < 1.40:
                continue
            if left_median >= right_median:
                long_column, short_column = left_column, right_column
            else:
                long_column, short_column = right_column, left_column

            related_rows: List[Tuple[int, bool]] = []
            for row_index, left_text, right_text in joint_rows:
                long_value = left_text if long_column == left_column else right_text
                short_value = right_text if short_column == right_column else left_text
                # A row is inverted when the nominal long column has the
                # shorter representation and the nominal short column has the
                # longer one. Prefix extensions are retained at low cardinality
                # but only mark the long-column cell, avoiding a false claim
                # that the extension itself is dirty.
                if len(long_value) >= len(short_value):
                    continue
                if len(short_value) < 1.05 * max(len(long_value), 1):
                    continue
                coverage = _abbreviation_coverage(long_value, short_value)
                if coverage < 0.50:
                    continue
                short_norm = _normalized_representation_text(long_value)
                long_norm = _normalized_representation_text(short_value)
                prefix_extension = bool(
                    short_norm and long_norm and short_norm in long_norm
                )
                related_rows.append((row_index, prefix_extension))

            if len(related_rows) < 10 or len(related_rows) / len(joint_rows) < 0.05:
                continue
            accepted_pairs += 1
            candidate_rows += len(related_rows)
            for row_index, prefix_extension in related_rows:
                targets = [f"{row_index}::{long_column}"]
                if not prefix_extension:
                    targets.append(f"{row_index}::{short_column}")
                relation_metadata = {
                    "generic_mechanism": True,
                    "relation_type": "representation_role_conflict",
                    "long_column": long_column,
                    "short_column": short_column,
                    "joint_support": len(joint_rows),
                    "inverted_support": len(related_rows),
                    "inverted_ratio": float(len(related_rows) / len(joint_rows)),
                    "prefix_extension": prefix_extension,
                }
                for cell_key in targets:
                    if cell_key not in cell_registry:
                        continue
                    _record_observation(cell_observations, EvidenceObservation(
                        target_scope="cell",
                        target_key=cell_key,
                        source_id="representation_role_conflict",
                        family="relationship",
                        polarity="dirty",
                        strength=1.0,
                        hard=True,
                        reason_code="long_short_representation_swap",
                        metadata=relation_metadata,
                    ))
                    emitted_cells += 1
    return {
        "accepted_pairs": accepted_pairs,
        "candidate_rows": candidate_rows,
        "emitted_cells": emitted_cells,
    }


def _coerce_numeric_with_metadata(value: Any, metadata: Dict[str, Any]) -> Optional[float]:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return numeric

    regex_rate = _matching_regex_rate(value, metadata.get("regex_candidates") or [])
    if regex_rate <= 0.0:
        return None

    text = str(value).strip().replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        numeric = float(match.group(1))
    except Exception:
        return None
    return None if np.isnan(numeric) else numeric


def _build_signature_weights(df: pd.DataFrame, column: str) -> np.ndarray:
    signatures = [_normalize_signature(value) for value in df[column].tolist()]
    counts: Dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    return np.array([counts[signature] for signature in signatures], dtype=int)


def _build_anchor_groups(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
    family: str,
) -> Tuple[List[GroupRecord], FamilyRunStatus]:
    groups_by_signature: Dict[str, GroupRecord] = {}
    for row_idx, value in enumerate(df[column].tolist()):
        signature = _normalize_signature(value)
        record = groups_by_signature.get(signature)
        if record is None:
            record = GroupRecord(
                normalized_signature=signature,
                row_indices=[],
                weight=0,
                rule_outputs_by_family={},
                anchor_state=ANCHOR_UNLABELED,
                anchor_source="unlabeled",
                representative_value=value,
            )
            groups_by_signature[signature] = record
        record.row_indices.append(row_idx)
        record.weight += 1

    groups = list(groups_by_signature.values())
    if not groups:
        return [], FamilyRunStatus(status="abstained", reason="empty_column")

    if family == "missing":
        for record in groups:
            if _is_missing_like(record.representative_value, metadata):
                if _is_structural_missing_group(record.weight, len(df)):
                    record.anchor_state = ANCHOR_ABSTAIN
                    record.anchor_source = "structural_missing_token"
                else:
                    record.anchor_state = ANCHOR_HARD_DIRTY
                    record.anchor_source = "real_missing_token"
            else:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "real_non_missing"
        return groups, FamilyRunStatus(status="ran")

    if family == "outlier":
        if metadata.get("type") != "numeric":
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "categorical_outlier_disabled"
            return groups, FamilyRunStatus(status="abstained", reason="categorical_outlier_disabled")
        quantiles = metadata.get("quantiles") or {}
        p25 = quantiles.get("p25")
        p75 = quantiles.get("p75")
        p05 = quantiles.get("p05")
        p95 = quantiles.get("p95")
        if p25 is None or p75 is None:
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "missing_numeric_stats"
            return groups, FamilyRunStatus(status="abstained", reason="missing_numeric_stats")
        iqr = float(metadata.get("iqr") or 0.0)
        inner_low = float(p25)
        inner_high = float(p75)
        if inner_low == inner_high:
            inner_low = float(p05 if p05 is not None else p25)
            inner_high = float(p95 if p95 is not None else p75)
        if iqr > 0:
            outer_low = float(p25) - 6.0 * iqr
            outer_high = float(p75) + 6.0 * iqr
        else:
            outer_low = float(p05 if p05 is not None else p25)
            outer_high = float(p95 if p95 is not None else p75)
        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "missing_reserved_for_missing_family"
                continue
            numeric = _coerce_numeric(value)
            if numeric is None:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "non_numeric_value"
            elif inner_low <= numeric <= inner_high:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "central_numeric_band"
            elif numeric < outer_low or numeric > outer_high:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "extreme_numeric_value"
            else:
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "numeric_borderline"
        return groups, FamilyRunStatus(status="ran")

    if family == "pattern":
        length_distribution = sorted(
            metadata.get("length_distribution") or [],
            key=lambda item: item.get("count", 0),
            reverse=True,
        )
        shape_distribution = sorted(
            metadata.get("shape_distribution") or [],
            key=lambda item: item.get("count", 0),
            reverse=True,
        )
        top_lengths = [int(item["length"]) for item in length_distribution[:2] if item.get("ratio", 0.0) >= 0.15]
        top_shapes = [str(item["shape"]) for item in shape_distribution[:2] if item.get("ratio", 0.0) >= 0.15]
        length_coverage = float(sum(item.get("ratio", 0.0) for item in length_distribution[:2]))
        shape_coverage = float(sum(item.get("ratio", 0.0) for item in shape_distribution[:2]))
        if length_coverage < 0.75 and shape_coverage < 0.75:
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "weak_structure_evidence"
            return groups, FamilyRunStatus(status="abstained", reason="weak_structure_evidence")
        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "missing_reserved_for_missing_family"
                continue
            text = str(value).strip()
            length_ok = True if not top_lengths else len(text) in top_lengths
            shape_ok = True if not top_shapes else _shape_signature(text) in top_shapes
            if length_ok and shape_ok:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "stable_structure"
            elif not length_ok or not shape_ok:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "structure_violation"
            else:
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "weak_structure_match"
        return groups, FamilyRunStatus(status="ran")

    for record in groups:
        record.anchor_state = ANCHOR_ABSTAIN
        record.anchor_source = "unsupported_family"
    return groups, FamilyRunStatus(status="abstained", reason="unsupported_family")


def _build_anchor_state_arrays(n_rows: int, anchor_groups: List[GroupRecord]) -> Tuple[np.ndarray, np.ndarray]:
    anchor_states = np.full(n_rows, ANCHOR_UNLABELED, dtype=object)
    anchor_sources = np.full(n_rows, "unlabeled", dtype=object)
    for record in anchor_groups:
        for row_idx in record.row_indices:
            anchor_states[row_idx] = record.anchor_state
            anchor_sources[row_idx] = record.anchor_source
    return anchor_states, anchor_sources


def _build_em_groups(
    anchor_groups: List[GroupRecord],
    family: str,
    family_outputs: np.ndarray,
    df: pd.DataFrame,
    column: str,
) -> List[GroupRecord]:
    em_groups: Dict[Tuple[str, Tuple[int, ...], str, str], GroupRecord] = {}
    for anchor_group in anchor_groups:
        if anchor_group.anchor_state == ANCHOR_ABSTAIN:
            continue
        for row_idx in anchor_group.row_indices:
            obs = tuple(int(v) for v in family_outputs[row_idx].tolist()) if family_outputs.size else tuple()
            key = (
                anchor_group.normalized_signature,
                obs,
                anchor_group.anchor_state,
                anchor_group.anchor_source,
            )
            if key not in em_groups:
                em_groups[key] = GroupRecord(
                    normalized_signature=anchor_group.normalized_signature,
                    row_indices=[],
                    weight=0,
                    rule_outputs_by_family={family: np.array(obs, dtype=int)},
                    anchor_state=anchor_group.anchor_state,
                    anchor_source=anchor_group.anchor_source,
                    representative_value=df.iloc[row_idx][column],
                )
            em_groups[key].row_indices.append(int(row_idx))
            em_groups[key].weight += 1
    return list(em_groups.values())


def _weighted_binary_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0:
        return float("nan")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.dot(values.astype(float), weights.astype(float)) / total_weight)


def _select_representative_values(anchor_groups: List[GroupRecord], limit: int = 20) -> List[Any]:
    clean_groups = [record for record in anchor_groups if record.anchor_state == ANCHOR_HARD_CLEAN]
    clean_groups.sort(key=lambda record: record.weight, reverse=True)
    selected: List[Any] = []
    seen_signatures = set()
    for record in clean_groups:
        if record.normalized_signature in seen_signatures:
            continue
        seen_signatures.add(record.normalized_signature)
        selected.append(record.representative_value)
        if len(selected) >= limit:
            break
    return selected


def _run_weighted_em_for_family(
    group_z: np.ndarray,
    group_weights: np.ndarray,
    anchor_states: np.ndarray,
    max_iters: int,
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_groups = group_z.shape[0]
    n_rules = group_z.shape[1] if group_z.ndim == 2 else 0
    eps = 1e-3
    if n_groups == 0 or n_rules == 0:
        return np.zeros(n_groups), np.zeros(n_rules), np.zeros(n_rules), float("nan")

    clean_mask = anchor_states == ANCHOR_HARD_CLEAN
    dirty_mask = anchor_states == ANCHOR_HARD_DIRTY
    unlabeled_mask = anchor_states == ANCHOR_UNLABELED

    alpha = np.full(n_rules, 0.9)
    beta = np.full(n_rules, 0.3)
    if clean_mask.any():
        alpha = np.clip(
            (group_z[clean_mask] * group_weights[clean_mask, None]).sum(axis=0) / max(group_weights[clean_mask].sum(), eps),
            eps,
            1 - eps,
        )
    if dirty_mask.any():
        beta = np.clip(
            (group_z[dirty_mask] * group_weights[dirty_mask, None]).sum(axis=0) / max(group_weights[dirty_mask].sum(), eps),
            eps,
            1 - eps,
        )

    pi1 = min(max(prior_dirty, eps), 1 - eps)
    gamma = np.zeros(n_groups, dtype=float)
    gamma[dirty_mask] = 1.0
    gamma[clean_mask] = 0.0
    gamma[unlabeled_mask] = pi1

    for _ in range(max_iters):
        if unlabeled_mask.any():
            z_u = group_z[unlabeled_mask]
            log_p0 = np.full(z_u.shape[0], np.log(1.0 - pi1))
            log_p1 = np.full(z_u.shape[0], np.log(pi1))
            for rule_idx in range(n_rules):
                zr = z_u[:, rule_idx]
                log_p0 += zr * np.log(alpha[rule_idx] + eps) + (1 - zr) * np.log(1.0 - alpha[rule_idx] + eps)
                log_p1 += zr * np.log(beta[rule_idx] + eps) + (1 - zr) * np.log(1.0 - beta[rule_idx] + eps)
            max_log = np.maximum(log_p0, log_p1)
            p0 = np.exp(log_p0 - max_log)
            p1 = np.exp(log_p1 - max_log)
            gamma[unlabeled_mask] = p1 / (p0 + p1 + eps)

        gamma[clean_mask] = 0.0
        gamma[dirty_mask] = 1.0

        y1_total = float((group_weights * gamma).sum())
        y0_total = float((group_weights * (1.0 - gamma)).sum())
        if y0_total <= 0 or y1_total <= 0:
            break

        weighted_clean = (group_weights * (1.0 - gamma))[:, None]
        weighted_dirty = (group_weights * gamma)[:, None]
        alpha = np.clip(weighted_clean.T.dot(group_z).ravel() / y0_total, eps, 1 - eps)
        beta = np.clip(weighted_dirty.T.dot(group_z).ravel() / y1_total, eps, 1 - eps)
        total_weight = y0_total + y1_total
        pi1 = min(max(y1_total / max(total_weight, eps), eps), 1 - eps)

    return gamma, alpha, beta, pi1


def _run_em_for_family(
    clean_z: np.ndarray,
    dirty_z: np.ndarray,
    unlabeled_z: np.ndarray,
    max_iters: int,
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rules = unlabeled_z.shape[1] if unlabeled_z.size > 0 else clean_z.shape[1]
    eps = 1e-3
    if n_rules == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    alpha = np.full(n_rules, 0.9)
    beta = np.full(n_rules, 0.3)
    if clean_z.size > 0:
        alpha = np.clip(clean_z.mean(axis=0), eps, 1 - eps)
    if dirty_z.size > 0:
        beta = np.clip(dirty_z.mean(axis=0), eps, 1 - eps)
    pi1 = min(max(prior_dirty, eps), 1 - eps)
    pi0 = 1.0 - pi1
    n_unlabeled = unlabeled_z.shape[0]
    gamma = np.full(n_unlabeled, pi1)
    for _ in range(max_iters):
        if n_unlabeled > 0:
            log_p0 = np.log(pi0)
            log_p1 = np.log(pi1)
            for r in range(n_rules):
                zr = unlabeled_z[:, r]
                p_z1_y0 = alpha[r]
                p_z0_y0 = 1.0 - alpha[r]
                p_z1_y1 = beta[r]
                p_z0_y1 = 1.0 - beta[r]
                log_p0 += zr * np.log(p_z1_y0 + eps) + (1 - zr) * np.log(p_z0_y0 + eps)
                log_p1 += zr * np.log(p_z1_y1 + eps) + (1 - zr) * np.log(p_z0_y1 + eps)
            max_log = np.maximum(log_p0, log_p1)
            p0 = np.exp(log_p0 - max_log)
            p1 = np.exp(log_p1 - max_log)
            denom = p0 + p1 + eps
            gamma = p1 / denom
        y0_clean = clean_z.shape[0]
        y1_dirty = dirty_z.shape[0]
        y0_unlabeled = float(n_unlabeled) - gamma.sum()
        y1_unlabeled = gamma.sum()
        y0_total = y0_clean + y0_unlabeled
        y1_total = y1_dirty + y1_unlabeled
        if y0_total <= 0 or y1_total <= 0:
            break
        num_alpha = np.zeros(n_rules)
        num_beta = np.zeros(n_rules)
        if clean_z.size > 0:
            num_alpha += clean_z.sum(axis=0)
        if dirty_z.size > 0:
            num_beta += dirty_z.sum(axis=0)
        if n_unlabeled > 0:
            for r in range(n_rules):
                zr = unlabeled_z[:, r]
                num_alpha[r] += float(((1.0 - gamma) * zr).sum())
                num_beta[r] += float((gamma * zr).sum())
        alpha = np.clip(num_alpha / y0_total, eps, 1 - eps)
        beta = np.clip(num_beta / y1_total, eps, 1 - eps)
        total_points = y0_total + y1_total
        pi1 = min(max(y1_total / total_points, eps), 1 - eps)
        pi0 = 1.0 - pi1
    return gamma, alpha, beta


def _setup_cleanem_logger(output_dir: str, dataset_name: str) -> logging.Logger:
    """Setup logger for clean_em mode with both console and file handlers."""
    logger = logging.getLogger("CleanEM")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler - INFO level (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - DEBUG level (detailed format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{timestamp}_{dataset_name}_cleanem.log")
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def _clip_probability(value: float, lower: float = 1e-3, upper: float = 1 - 1e-3) -> float:
    return float(min(max(value, lower), upper))


def _format_float(value: float) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


def _set_group_anchor(
    record: GroupRecord,
    state: str,
    source: str,
    dirty_prior: float,
    anchor_weight: float,
) -> None:
    record.anchor_state = state
    record.anchor_source = source
    record.dirty_prior = _clip_probability(dirty_prior)
    record.anchor_weight = float(np.clip(anchor_weight, 0.0, 1.0))
    record.support_count = record.weight


def _build_value_groups(df: pd.DataFrame, column: str) -> List[GroupRecord]:
    groups_by_signature: Dict[str, GroupRecord] = {}
    for row_idx, value in enumerate(df[column].tolist()):
        signature = _normalize_signature(value)
        record = groups_by_signature.get(signature)
        if record is None:
            record = GroupRecord(
                normalized_signature=signature,
                row_indices=[],
                weight=0,
                rule_outputs_by_family={},
                anchor_state=ANCHOR_UNLABELED,
                anchor_source="unlabeled",
                representative_value=value,
            )
            groups_by_signature[signature] = record
        record.row_indices.append(row_idx)
        record.weight += 1
    return list(groups_by_signature.values())


def _segment_signature(value: str) -> str:
    if value == "":
        return "<empty>"
    if value.isspace():
        return f"S{len(value)}"
    if all(char.isdigit() for char in value):
        return f"D{len(value)}"
    if all(char.isalpha() for char in value):
        return f"A{len(value)}"
    if all(char.isalnum() for char in value):
        return f"M{len(value)}"
    return f"X{len(value)}"


def _grammar_signature(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "<empty>"
    parts = re.split(r'([\-_/.:\s]+)', text)
    signature_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r'[\-_/.:\s]+', part):
            delimiter = part[0]
            if delimiter.isspace():
                delimiter = ' '
            signature_parts.append(f"<{delimiter}:{len(part)}>" )
        else:
            signature_parts.append(_segment_signature(part))
    return "|".join(signature_parts) if signature_parts else "<empty>"


def _relationship_predicate(value: Any, other_value: Any, constraint_type: str) -> bool:
    if value is None or other_value is None:
        return True
    value_str = str(value).strip()
    other_str = str(other_value).strip()
    if not value_str or not other_str:
        return True
    constraint_type = (constraint_type or "").lower()
    if constraint_type == "prefix_match":
        return value_str.lower().startswith(other_str.lower())
    if constraint_type == "contains":
        return other_str.lower() in value_str.lower()
    if constraint_type == "stateavg_format":
        return value_str.lower().startswith(f"{other_str.lower()}_")
    if constraint_type == "zip_prefix":
        return value_str[:3] == other_str[:3]
    return True


def _build_soft_anchor_groups(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
    family: str,
    prior_dirty: float,
) -> Tuple[List[GroupRecord], FamilyRunStatus]:
    groups = _build_value_groups(df, column)
    if not groups:
        return [], FamilyRunStatus(status="abstained", reason="empty_column")

    status = FamilyRunStatus(status="ran", mode="ran")

    if family == "missing":
        for record in groups:
            if _is_missing_like(record.representative_value, metadata):
                if _is_structural_missing_group(record.weight, len(df)):
                    _set_group_anchor(
                        record, ANCHOR_ABSTAIN, "structural_missing_token",
                        prior_dirty, 0.0,
                    )
                else:
                    _set_group_anchor(
                        record, ANCHOR_HARD_DIRTY, "real_missing_token",
                        0.99, 1.0,
                    )
            else:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "real_non_missing", 0.01, 1.0)
        return groups, status

    if family == "outlier":
        if metadata.get("type") != "numeric":
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "categorical_outlier_disabled", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="categorical_outlier_disabled")

        quantiles = metadata.get("quantiles") or {}
        p25 = quantiles.get("p25")
        p75 = quantiles.get("p75")
        p05 = quantiles.get("p05")
        p95 = quantiles.get("p95")
        if p25 is None or p75 is None:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "missing_numeric_stats", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="missing_numeric_stats")

        iqr = float(metadata.get("iqr") or 0.0)
        inner_low = float(p25)
        inner_high = float(p75)
        if inner_low == inner_high:
            inner_low = float(p05 if p05 is not None else p25)
            inner_high = float(p95 if p95 is not None else p75)
        if iqr > 0:
            outer_low = float(p25) - 6.0 * iqr
            outer_high = float(p75) + 6.0 * iqr
        else:
            outer_low = float(p05 if p05 is not None else p25)
            outer_high = float(p95 if p95 is not None else p75)

        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                _set_group_anchor(record, ANCHOR_UNLABELED, "missing_reserved_for_missing_family", prior_dirty, 0.15)
                continue
            numeric = _coerce_numeric(value)
            if numeric is None:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "non_numeric_value", 0.99, 1.0)
            elif inner_low <= numeric <= inner_high:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "central_numeric_band", 0.02, 0.95)
            elif numeric < outer_low or numeric > outer_high:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "extreme_numeric_value", 0.98, 0.95)
            else:
                offset = 0.5
                if outer_high > inner_high and numeric > inner_high:
                    offset = min(0.85, 0.35 + (numeric - inner_high) / max(outer_high - inner_high, 1e-6))
                elif inner_low > outer_low and numeric < inner_low:
                    offset = min(0.85, 0.35 + (inner_low - numeric) / max(inner_low - outer_low, 1e-6))
                _set_group_anchor(record, ANCHOR_UNLABELED, "numeric_borderline", offset, 0.4)
        return groups, status

    if family == "pattern":
        grammar_variants = metadata.get("grammar_variants") or []
        accepted_variants = [
            str(item.get("signature"))
            for item in grammar_variants
            if float(item.get("ratio", 0.0)) >= 0.05
        ]
        grammar_coverage = float(sum(
            float(item.get("ratio", 0.0))
            for item in grammar_variants
            if float(item.get("ratio", 0.0)) >= 0.05
        ))
        shape_distribution = metadata.get("shape_distribution") or []
        accepted_shapes = {
            str(item.get("shape"))
            for item in shape_distribution
            if float(item.get("ratio", 0.0)) >= 0.05
        }
        regex_candidates = [
            str(item.get("pattern"))
            for item in metadata.get("regex_candidates") or []
            if float(item.get("match_rate", 0.0)) >= 0.35
        ]

        if not accepted_variants and not accepted_shapes and not regex_candidates:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "weak_structure_evidence", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="weak_structure_evidence")

        if grammar_coverage < 0.35 and len(regex_candidates) == 0:
            status.reason = "weak_structure_evidence"
            status.mode = "provisional"

        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                _set_group_anchor(record, ANCHOR_UNLABELED, "missing_reserved_for_missing_family", prior_dirty, 0.15)
                continue

            text = str(value).strip()
            grammar_sig = _grammar_signature(text)
            grammar_match = grammar_sig in accepted_variants if accepted_variants else False
            shape_match = _shape_signature(text) in accepted_shapes if accepted_shapes else False
            regex_match = False
            for pattern in regex_candidates:
                try:
                    if re.fullmatch(pattern, text):
                        regex_match = True
                        break
                except re.error:
                    continue

            if grammar_match:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "stable_grammar", 0.02, 0.95)
            elif regex_match:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "regex_consistent", 0.08, 0.75)
            elif shape_match:
                _set_group_anchor(record, ANCHOR_UNLABELED, "coarse_shape_match", 0.35, 0.35)
            else:
                source = "grammar_violation" if accepted_variants or regex_candidates else "structure_violation"
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, source, 0.95, 0.85)
        return groups, status

    if family == "relationship":
        profiles = metadata.get("relationship_profiles") or []
        strong_profiles = [
            profile for profile in profiles
            if float(profile.get("violation_rate", 1.0)) <= 0.25
            and int(profile.get("applicable_count", 0)) >= max(5, int(0.02 * max(len(df), 1)))
        ]
        if not strong_profiles:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "weak_relationship_evidence", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="weak_relationship_evidence")

        active_groups = 0
        for record in groups:
            applicable = 0
            valid = 0
            for row_idx in record.row_indices:
                row = df.iloc[row_idx]
                for profile in strong_profiles:
                    other_column = profile.get("other_column")
                    if not other_column or other_column not in df.columns:
                        continue
                    current_value = row[column]
                    other_value = row[other_column]
                    if pd.isna(current_value) or pd.isna(other_value):
                        continue
                    if str(current_value).strip() == "" or str(other_value).strip() == "":
                        continue
                    applicable += 1
                    valid += int(_relationship_predicate(current_value, other_value, profile.get("type", "")))
            if applicable == 0:
                _set_group_anchor(record, ANCHOR_UNLABELED, "no_applicable_relationship", prior_dirty, 0.0)
                continue
            active_groups += 1
            satisfaction = valid / max(applicable, 1)
            weight = min(1.0, applicable / max(len(strong_profiles), 1))
            if satisfaction >= 0.95:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "relationship_consistent", 0.02, weight)
            elif satisfaction <= 0.2:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "relationship_violation", 0.98, weight)
            else:
                _set_group_anchor(record, ANCHOR_UNLABELED, "relationship_mixed", 1.0 - satisfaction, min(0.6, weight))

        if active_groups == 0:
            return groups, FamilyRunStatus(status="abstained", reason="no_applicable_relationship")
        if active_groups < max(3, int(0.1 * len(groups))):
            return groups, FamilyRunStatus(status="ran", reason="sparse_relationship_support", mode="provisional")
        return groups, status

    for record in groups:
        _set_group_anchor(record, ANCHOR_ABSTAIN, "unsupported_family", prior_dirty, 0.0)
    return groups, FamilyRunStatus(status="abstained", reason="unsupported_family")


def _build_anchor_arrays_v2(
    n_rows: int,
    anchor_groups: List[GroupRecord],
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_states = np.full(n_rows, ANCHOR_UNLABELED, dtype=object)
    anchor_sources = np.full(n_rows, "unlabeled", dtype=object)
    anchor_priors = np.full(n_rows, float(prior_dirty), dtype=float)
    anchor_weights = np.zeros(n_rows, dtype=float)
    for record in anchor_groups:
        for row_idx in record.row_indices:
            anchor_states[row_idx] = record.anchor_state
            anchor_sources[row_idx] = record.anchor_source
            anchor_priors[row_idx] = record.dirty_prior
            anchor_weights[row_idx] = record.anchor_weight
    return anchor_states, anchor_sources, anchor_priors, anchor_weights


def _build_em_groups_v2(
    anchor_groups: List[GroupRecord],
    family: str,
    family_outputs: np.ndarray,
    df: pd.DataFrame,
    column: str,
) -> List[GroupRecord]:
    em_groups: Dict[Tuple[str, Tuple[int, ...], str, str, float, float], GroupRecord] = {}
    for anchor_group in anchor_groups:
        if anchor_group.anchor_state == ANCHOR_ABSTAIN:
            continue
        for row_idx in anchor_group.row_indices:
            obs = tuple(int(v) for v in family_outputs[row_idx].tolist()) if family_outputs.size else tuple()
            key = (
                anchor_group.normalized_signature,
                obs,
                anchor_group.anchor_state,
                anchor_group.anchor_source,
                round(float(anchor_group.dirty_prior), 3),
                round(float(anchor_group.anchor_weight), 3),
            )
            if key not in em_groups:
                em_groups[key] = GroupRecord(
                    normalized_signature=anchor_group.normalized_signature,
                    row_indices=[],
                    weight=0,
                    rule_outputs_by_family={family: np.array(obs, dtype=int)},
                    anchor_state=anchor_group.anchor_state,
                    anchor_source=anchor_group.anchor_source,
                    representative_value=df.iloc[row_idx][column],
                    dirty_prior=anchor_group.dirty_prior,
                    anchor_weight=anchor_group.anchor_weight,
                    support_count=anchor_group.support_count,
                )
            em_groups[key].row_indices.append(int(row_idx))
            em_groups[key].weight += 1
    return list(em_groups.values())


def _select_representative_values_v2(anchor_groups: List[GroupRecord], limit: int = 20) -> List[Any]:
    clean_groups = [
        record for record in anchor_groups
        if record.anchor_weight >= 0.5 and record.dirty_prior <= 0.25
    ]
    clean_groups.sort(key=lambda record: record.weight * max(record.anchor_weight, 1e-3), reverse=True)
    selected: List[Any] = []
    seen_signatures = set()
    for record in clean_groups:
        if record.normalized_signature in seen_signatures:
            continue
        seen_signatures.add(record.normalized_signature)
        selected.append(record.representative_value)
        if len(selected) >= limit:
            break
    return selected


def _weighted_pass_rate(outputs: np.ndarray, weights: np.ndarray) -> float:
    if outputs.size == 0 or weights.size == 0:
        return float("nan")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.dot(outputs.astype(float), weights.astype(float)) / total_weight)


def _compute_family_reliability(
    rule_quality: float,
    coverage_rows: int,
    coverage_groups: int,
    total_rows: int,
    mode: str,
) -> float:
    coverage_ratio = coverage_rows / max(total_rows, 1)
    group_ratio = min(1.0, coverage_groups / max(3.0, np.sqrt(max(total_rows, 1))))
    reliability = 0.45 * max(rule_quality, 0.0) + 0.35 * coverage_ratio + 0.20 * group_ratio
    if mode == "provisional":
        reliability *= 0.65
    return float(np.clip(reliability, 0.05, 0.99))


def _compute_family_weight(status: FamilyRunStatus, total_rows: int) -> float:
    if status.status == "abstained" or pd.isna(status.reliability):
        return 0.0
    coverage_confidence = min(1.0, status.coverage_rows / max(1, int(0.1 * max(total_rows, 1))))
    mode_factor = 1.0 if status.mode == "ran" else 0.6
    return float(np.clip(status.reliability * max(0.25, coverage_confidence) * mode_factor, 0.0, 1.0))


def _levenshtein_distance(
    left: str,
    right: str,
    max_distance: Optional[int] = None,
) -> int:
    """Return edit distance, stopping at ``max_distance + 1`` when exceeded."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if max_distance is not None:
        cutoff = max(0, int(max_distance))
        if abs(len(left) - len(right)) > cutoff:
            return cutoff + 1
        sentinel = cutoff + 1
        previous = [
            index if index <= cutoff else sentinel
            for index in range(len(right) + 1)
        ]
        for i, left_char in enumerate(left, start=1):
            start = max(1, i - cutoff)
            end = min(len(right), i + cutoff)
            current = [sentinel] * (len(right) + 1)
            if i <= cutoff:
                current[0] = i
            for j in range(start, end + 1):
                cost = 0 if left_char == right[j - 1] else 1
                current[j] = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            if min(current[start:end + 1], default=sentinel) > cutoff:
                return sentinel
            previous = current
        return previous[-1] if previous[-1] <= cutoff else sentinel
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def _char_ngrams(value: str, n: int = 3) -> set:
    text = value.lower()
    if len(text) < n:
        return {text} if text else set()
    return {text[idx: idx + n] for idx in range(len(text) - n + 1)}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def _prototype_similarity(left: str, right: str) -> Tuple[float, float, float, int]:
    if not left or not right:
        return 0.0, 0.0, 0.0, max(len(left), len(right))
    edit_distance = _levenshtein_distance(left.lower(), right.lower())
    edit_similarity = 1.0 - (edit_distance / max(len(left), len(right), 1))
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    ngram_similarity = float(len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)) if left_ngrams and right_ngrams else 0.0
    token_similarity = _token_jaccard(left, right)
    return edit_similarity, ngram_similarity, token_similarity, edit_distance


def _score_prototype_lexical_family(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, FamilyRunStatus]:
    n_rows = len(df)
    scores = np.full(n_rows, np.nan, dtype=float)
    sources = np.full(n_rows, "prototype_unavailable", dtype=object)

    if metadata.get("type") not in {"categorical", "pattern", "text"}:
        return scores, sources, FamilyRunStatus(status="abstained", reason="prototype_not_applicable")

    groups = _build_value_groups(df, column)
    valid_groups = [group for group in groups if not _is_missing_like(group.representative_value, metadata)]
    if not valid_groups:
        return scores, sources, FamilyRunStatus(status="abstained", reason="empty_column")

    prototype_threshold = max(2, int(np.ceil(0.01 * max(n_rows, 1))))
    prototypes = [
        group for group in valid_groups
        if group.weight >= prototype_threshold or (
            group.weight >= 2 and (group.weight / max(n_rows, 1)) >= 0.02
        )
    ]
    prototypes.sort(key=lambda record: record.weight, reverse=True)
    prototypes = prototypes[:25]
    prototype_coverage = sum(record.weight for record in prototypes) / max(n_rows, 1)

    if len(prototypes) < 2 or prototype_coverage < 0.12:
        return scores, sources, FamilyRunStatus(status="abstained", reason="insufficient_prototype_support")

    for group in valid_groups:
        group_text = str(group.representative_value).strip()
        if group.normalized_signature in {prototype.normalized_signature for prototype in prototypes}:
            group_score = 0.0
            source = "prototype_bank"
        else:
            best_prototype = None
            best_score = 0.0
            best_distance = 0
            for prototype in prototypes:
                if prototype.normalized_signature == group.normalized_signature:
                    continue
                edit_similarity, ngram_similarity, token_similarity, edit_distance = _prototype_similarity(
                    group_text,
                    str(prototype.representative_value).strip(),
                )
                similarity = 0.55 * edit_similarity + 0.25 * ngram_similarity + 0.20 * token_similarity
                if similarity > best_score:
                    best_score = similarity
                    best_prototype = prototype
                    best_distance = edit_distance

            if best_prototype is None or best_score < 0.55:
                group_score = 0.0
                source = "no_close_prototype"
            else:
                support_factor = min(1.0, best_prototype.weight / max(3.0, 0.05 * max(n_rows, 1)))
                rarity_factor = 1.0 - min(0.85, group.weight / max(best_prototype.weight, 1))
                group_score = best_score * support_factor * max(rarity_factor, 0.15)
                if best_distance <= 2 and abs(len(group_text) - len(str(best_prototype.representative_value).strip())) <= 2:
                    group_score += 0.10
                if _shape_signature(group_text) == _shape_signature(str(best_prototype.representative_value).strip()):
                    group_score += 0.05
                group_score = float(np.clip(group_score, 0.0, 1.0))
                source = f"closest_prototype:{best_prototype.normalized_signature}"

        for row_idx in group.row_indices:
            scores[row_idx] = group_score
            sources[row_idx] = source

    coverage_rows = int(np.sum(pd.notna(scores)))
    coverage_groups = len(valid_groups)
    mode = "ran" if prototype_coverage >= 0.25 else "provisional"
    reliability = float(np.clip(0.45 * prototype_coverage + 0.25 * min(1.0, len(prototypes) / 6.0) + 0.30, 0.10, 0.90))
    if mode == "provisional":
        reliability *= 0.7
    return scores, sources, FamilyRunStatus(
        status="ran",
        reason="",
        coverage_rows=coverage_rows,
        coverage_groups=coverage_groups,
        reliability=reliability,
        mode=mode,
    )


def _run_weighted_em_for_family_v2(
    group_z: np.ndarray,
    group_weights: np.ndarray,
    dirty_priors: np.ndarray,
    anchor_weights: np.ndarray,
    max_iters: int,
    prior_dirty: float,
    rule_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_groups = group_z.shape[0]
    n_rules = group_z.shape[1] if group_z.ndim == 2 else 0
    eps = 1e-3
    if n_groups == 0 or n_rules == 0:
        return np.zeros(n_groups), np.zeros(n_rules), np.zeros(n_rules), float("nan")

    dirty_priors = np.clip(dirty_priors.astype(float), eps, 1 - eps)
    anchor_weights = np.clip(anchor_weights.astype(float), 0.0, 1.0)
    if rule_weights is None or rule_weights.size == 0:
        rule_weights = np.ones(n_rules, dtype=float)
    else:
        rule_weights = np.clip(rule_weights.astype(float), 0.1, 1.0)

    pi1 = _clip_probability(float(prior_dirty))
    gamma = anchor_weights * dirty_priors + (1.0 - anchor_weights) * pi1
    gamma = np.clip(gamma, eps, 1 - eps)

    alpha = np.full(n_rules, 0.9, dtype=float)
    beta = np.full(n_rules, 0.3, dtype=float)

    for _ in range(max_iters):
        clean_weights = group_weights * (1.0 - gamma)
        dirty_weights = group_weights * gamma
        clean_total = float(clean_weights.sum())
        dirty_total = float(dirty_weights.sum())
        if clean_total <= 0 or dirty_total <= 0:
            break

        alpha = np.clip(clean_weights.T.dot(group_z).ravel() / clean_total, eps, 1 - eps)
        beta = np.clip(dirty_weights.T.dot(group_z).ravel() / dirty_total, eps, 1 - eps)

        log_p0 = np.full(n_groups, np.log(1.0 - pi1), dtype=float)
        log_p1 = np.full(n_groups, np.log(pi1), dtype=float)
        for rule_idx in range(n_rules):
            zr = group_z[:, rule_idx]
            weight = rule_weights[rule_idx]
            log_p0 += weight * (zr * np.log(alpha[rule_idx] + eps) + (1 - zr) * np.log(1.0 - alpha[rule_idx] + eps))
            log_p1 += weight * (zr * np.log(beta[rule_idx] + eps) + (1 - zr) * np.log(1.0 - beta[rule_idx] + eps))

        max_log = np.maximum(log_p0, log_p1)
        p0 = np.exp(log_p0 - max_log)
        p1 = np.exp(log_p1 - max_log)
        posterior = p1 / (p0 + p1 + eps)
        gamma = anchor_weights * dirty_priors + (1.0 - anchor_weights) * posterior
        gamma = np.clip(gamma, eps, 1 - eps)
        pi1 = _clip_probability(float(np.dot(group_weights, gamma) / max(group_weights.sum(), eps)))

    return gamma, alpha, beta, pi1


def _record_observation(
    observation_map: Dict[str, List[EvidenceObservation]],
    observation: EvidenceObservation,
) -> None:
    target_list = observation_map.setdefault(observation.target_key, [])
    for idx, existing in enumerate(target_list):
        if (
            existing.source_id == observation.source_id
            and existing.polarity == observation.polarity
            and existing.hard == observation.hard
            and existing.reason_code == observation.reason_code
        ):
            if observation.strength > existing.strength:
                target_list[idx] = observation
            return
    target_list.append(observation)


def _column_archetype(column_metadata: Dict[str, Any]) -> str:
    semantics = column_metadata.get("semantics") or {}
    if isinstance(semantics, dict):
        return str(semantics.get("archetype") or "unknown")
    return str(getattr(semantics, "archetype", "unknown"))


def _gate_observation(
    observation: EvidenceObservation,
    column: str,
    metadata: Dict[str, Dict[str, Any]],
    supported_policies: Optional[Set[Tuple[str, str]]] = None,
    policy_support: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
    gating_mode: str = "hard",
    gate_prior_strength: float = 0.25,
    gate_prior_pseudocount: float = 2.0,
) -> Tuple[EvidenceObservation, str]:
    """Apply a fixed semantic gate while retaining full audit provenance."""
    observation.metadata = dict(observation.metadata or {})
    if observation.metadata.get("gate_applied"):
        return observation, "already_gated"

    archetype = _column_archetype(metadata.get(column, {}))
    raw_strength = float(observation.strength)

    if observation.hard:
        scale = 1.0
        gate_reason = "hard_evidence_preserved"
        outcome = "hard_preserved"
    else:
        policy = ARCHETYPE_EVIDENCE_GATES.get(archetype, {})
        policy_key = (archetype, observation.source_id)
        policy_requested = observation.source_id in policy
        if policy_requested and gating_mode == "weak_prior":
            support = (policy_support or {}).get(policy_key, {})
            column_support = (policy_support or {}).get(
                (f"column:{column}", observation.source_id),
                {},
            )
            clean_count = max(0, int(support.get("clean", 0)))
            dirty_count = max(0, int(support.get("dirty", 0)))
            column_clean_count = max(0, int(column_support.get("clean", 0)))
            column_dirty_count = max(0, int(column_support.get("dirty", 0)))
            semantics = metadata.get(column, {}).get("semantics") or {}
            semantics_confidence = float(np.clip(
                semantics.get("confidence", 0.0) if isinstance(semantics, dict)
                else getattr(semantics, "confidence", 0.0),
                0.0,
                1.0,
            ))
            prior_strength = float(np.clip(gate_prior_strength, 0.0, 1.0))
            prior_count = max(0.0, float(gate_prior_pseudocount))
            prior_trust = prior_strength * semantics_confidence
            denominator = prior_count + clean_count + dirty_count
            if denominator > 0.0:
                posterior_trust = (
                    prior_count * prior_trust + clean_count
                ) / denominator
            else:
                posterior_trust = prior_trust
            target_scale = float(policy[observation.source_id])
            scale = 1.0 - posterior_trust * (1.0 - target_scale)
            gate_reason = "weak_archetype_prior_with_label_update"
            outcome = "attenuated" if scale < 1.0 else "unchanged"
            # Open-ended text has legitimate long-tail values, so retain the
            # semantic weak prior above. Do not collapse these signals to the
            # raw target scale merely because the tiny active-label sample did
            # not contain a clean counterexample; that makes an unqueried text
            # column effectively invisible. Explicit column namespaces and
            # active labels still control learned rule weights.
            # Evidence authorization is column-local. At low budgets, a label
            # from one enum or numeric column must not unlock broad rarity or
            # regex evidence for every column with the same archetype.
            if (
                archetype == "numeric_measure"
                and observation.source_id == "rarity_high"
                and (column_clean_count <= 0 or column_dirty_count <= 0)
            ):
                scale = min(scale, 0.25)
                gate_reason = "column_numeric_rarity_precision_guard"
                outcome = "attenuated" if scale < 1.0 else "unchanged"
            if (
                archetype == "closed_enum"
                and observation.source_id == "pattern_mismatch"
                and (column_clean_count <= 0 or column_dirty_count <= 0)
            ):
                scale = min(scale, 0.25)
                gate_reason = "column_closed_enum_pattern_precision_guard"
                outcome = "attenuated" if scale < 1.0 else "unchanged"
            observation.metadata.update({
                "gate_clean_support": clean_count,
                "gate_dirty_support": dirty_count,
                "gate_prior_trust": prior_trust,
                "gate_posterior_trust": posterior_trust,
                "gate_target_scale": target_scale,
                "gate_column_clean_support": column_clean_count,
                "gate_column_dirty_support": column_dirty_count,
            })
        elif policy_requested and gating_mode == "soft":
            support = (policy_support or {}).get(policy_key, {})
            clean_count = max(0, int(support.get("clean", 0)))
            dirty_count = max(0, int(support.get("dirty", 0)))
            semantics = metadata.get(column, {}).get("semantics") or {}
            semantics_confidence = float(np.clip(
                semantics.get("confidence", 0.0) if isinstance(semantics, dict)
                else getattr(semantics, "confidence", 0.0),
                0.0,
                1.0,
            ))
            if clean_count <= 0:
                scale = 1.0
                gate_reason = "no_clean_label_support"
                outcome = "support_fallback"
            else:
                trust = semantics_confidence * clean_count / (
                    clean_count + dirty_count + 1.0
                )
                target_scale = float(policy[observation.source_id])
                scale = 1.0 - trust * (1.0 - target_scale)
                gate_reason = "label_supported_soft_policy"
                outcome = "attenuated" if scale < 1.0 else "unchanged"
        else:
            policy_supported = (
                supported_policies is None or policy_key in supported_policies
            )
            if policy_requested and not policy_supported:
                # Open-ended lexical and geographic surfaces are especially
                # vulnerable to stochastic active-label composition: a fresh
                # run may contain no clean counterexample for the archetype,
                # which would otherwise release a broad pattern detector at
                # full strength and mark an entire legitimate vocabulary as
                # dirty. Keep the generic archetype safety prior active until
                # labels authorize a stronger policy. Structured roles retain
                # the historical support fallback because their contracts are
                # intentionally stricter.
                conservative_unvalidated_surface = (
                    archetype in {"open_entity", "geo_name"}
                    and observation.source_id in {
                        "pattern_mismatch",
                        "rarity_high",
                        "prototype_far",
                    }
                )
                if conservative_unvalidated_surface:
                    scale = float(policy[observation.source_id])
                    gate_reason = "archetype_policy_without_clean_counterexample"
                    outcome = "attenuated" if scale < 1.0 else "unchanged"
                else:
                    scale = 1.0
                    gate_reason = "insufficient_clean_label_support"
                    outcome = "support_fallback"
            else:
                scale = float(policy.get(observation.source_id, 1.0))
                gate_reason = "archetype_policy" if policy_requested else "source_default"
                if scale <= 0.0:
                    outcome = "disabled"
                elif scale < 1.0:
                    outcome = "attenuated"
                else:
                    outcome = "unchanged"

    observation.metadata.update({
        "gate_applied": True,
        "gate_archetype": archetype,
        "raw_strength": raw_strength,
        "gate_scale": scale,
        "gate_reason": gate_reason,
    })
    observation.strength = float(np.clip(raw_strength * scale, 0.0, 1.0))
    return observation, outcome


def _apply_evidence_gates(
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    supported_policies: Optional[Set[Tuple[str, str]]] = None,
    policy_support: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
    gating_mode: str = "hard",
    gate_prior_strength: float = 0.25,
    gate_prior_pseudocount: float = 2.0,
) -> Dict[str, Any]:
    summary = Counter()
    policy_hits = Counter()

    for value_key, observations in value_observations.items():
        record = value_registry.get(value_key)
        if record is None:
            continue
        archetype = _column_archetype(metadata.get(record.column, {}))
        for observation in observations:
            _, outcome = _gate_observation(
                observation,
                record.column,
                metadata,
                supported_policies=supported_policies,
                policy_support=policy_support,
                gating_mode=gating_mode,
                gate_prior_strength=gate_prior_strength,
                gate_prior_pseudocount=gate_prior_pseudocount,
            )
            summary[outcome] += 1
            scale = float(observation.metadata.get("gate_scale", 1.0))
            if scale < 1.0:
                policy_hits[(archetype, observation.source_id, scale)] += 1

    for cell_key, observations in cell_observations.items():
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        archetype = _column_archetype(metadata.get(record.column, {}))
        for observation in observations:
            _, outcome = _gate_observation(
                observation,
                record.column,
                metadata,
                supported_policies=supported_policies,
                policy_support=policy_support,
                gating_mode=gating_mode,
                gate_prior_strength=gate_prior_strength,
                gate_prior_pseudocount=gate_prior_pseudocount,
            )
            summary[outcome] += 1
            scale = float(observation.metadata.get("gate_scale", 1.0))
            if scale < 1.0:
                policy_hits[(archetype, observation.source_id, scale)] += 1

    return {
        "counts": dict(summary),
        "policy_hits": [
            {
                "archetype": archetype,
                "source_id": source_id,
                "scale": scale,
                "count": count,
            }
            for (archetype, source_id, scale), count in sorted(policy_hits.items())
        ],
    }


def _scale_agent_pattern_rule_evidence(
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    scale: float,
) -> Dict[str, int]:
    """Scale only evidence backed by evaluated Agent-generated pattern rules."""
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("--agent_pattern_rule_scale must be finite and >= 0")

    counts = Counter()
    for scope, observations_by_key in (
        ("value", value_observations),
        ("cell", cell_observations),
    ):
        for observations in observations_by_key.values():
            for observation in observations:
                metadata = observation.metadata or {}
                if (
                    observation.family != "pattern"
                    or metadata.get("rule_pool_source")
                    != "agentic_clean_rule_pool"
                ):
                    continue
                before = float(observation.strength)
                observation.strength = before * float(scale)
                metadata["agent_pattern_rule_scale"] = float(scale)
                metadata["pre_agent_pattern_rule_scale_strength"] = before
                observation.metadata = metadata
                counts[scope] += 1
                counts[observation.polarity] += 1
    return dict(counts)


def _label_supported_gating_policies(
    selected_cell_keys: List[str],
    active_labels: Dict[str, int],
    metadata: Dict[str, Dict[str, Any]],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_registry: Dict[str, CellRecord],
    min_distinct_clean_values: int = 2,
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict[str, int]]]:
    """Enable attenuation only after observing clean counterexamples.

    A fixed gate is intended to suppress a dirty-polarity source that fires on
    legitimate values. Requiring distinct clean queried values prevents a pair
    of repeated missing cells from authorizing broad changes to an archetype.
    """
    clean_support: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    dirty_support: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for cell_key in selected_cell_keys:
        if cell_key not in active_labels:
            continue
        label = int(active_labels[cell_key])
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        archetype = _column_archetype(metadata.get(record.column, {}))
        policy = ARCHETYPE_EVIDENCE_GATES.get(archetype, {})
        if not policy:
            continue
        observations = list(cell_observations.get(cell_key, []))
        observations.extend(value_observations.get(record.value_key, []))
        for observation in observations:
            if (
                observation.polarity == "dirty"
                and observation.source_id in policy
                and float(policy[observation.source_id]) < 1.0
            ):
                target = clean_support if label == 0 else dirty_support
                target[(archetype, observation.source_id)].add(record.value_key)
                target[(f"column:{record.column}", observation.source_id)].add(
                    record.value_key
                )

    keys = set(clean_support) | set(dirty_support)
    support_counts = {
        key: {
            "clean": len(clean_support.get(key, set())),
            "dirty": len(dirty_support.get(key, set())),
        }
        for key in keys
    }
    supported = {
        key
        for key, counts in support_counts.items()
        if counts["clean"] >= max(1, int(min_distinct_clean_values))
    }
    return supported, support_counts


def _matches_regex_candidates(value: Any, regex_candidates: List[Dict[str, Any]]) -> bool:
    return _matching_regex_rate(value, regex_candidates) > 0.0


def _grammar_distance_score(text: str, metadata: Dict[str, Any]) -> float:
    grammar_variants = metadata.get("grammar_variants") or []
    accepted = {
        str(item.get("signature"))
        for item in grammar_variants
        if float(item.get("ratio", 0.0)) >= 0.05
    }
    if not accepted:
        return 0.0
    signature = _grammar_signature(text)
    return 0.0 if signature in accepted else 1.0


def _shape_distance_score(text: str, metadata: Dict[str, Any]) -> float:
    shape_distribution = metadata.get("shape_distribution") or []
    accepted = {
        str(item.get("shape"))
        for item in shape_distribution
        if float(item.get("ratio", 0.0)) >= 0.05
    }
    if not accepted:
        return 0.0
    shape = _shape_signature(text)
    return 0.0 if shape in accepted else 1.0


def _row_family_pass_ratio(
    outputs: np.ndarray,
    family_indices: Dict[str, List[int]],
    family: str,
    row_idx: int,
) -> Optional[float]:
    indices = family_indices.get(family, [])
    if outputs.size == 0 or not indices:
        return None
    return float(np.mean(outputs[row_idx, indices]))


def _agent_rule_provenance(
    rules: List[Dict[str, Any]],
    outputs: np.ndarray,
    family: str,
    row_idx: int,
) -> Dict[str, Any]:
    """Return auditable per-rule outcomes without changing evidence features."""
    if outputs.size == 0 or row_idx < 0 or row_idx >= outputs.shape[0]:
        return {}

    evaluated_rules: List[Dict[str, Any]] = []
    failed_rules: List[Dict[str, Any]] = []
    passed_rule_names: List[str] = []
    for rule_idx, rule in enumerate(rules):
        if str(rule.get("family") or "") != family:
            continue
        outcome = bool(outputs[row_idx, rule_idx])
        rule_record = {
            "rule_name": str(rule.get("rule_name") or f"{family}_{rule_idx}"),
            "agent": str(rule.get("agent") or ""),
            "family": family,
            "rule_str": str(rule.get("rule_str") or ""),
            "outcome": "pass" if outcome else "fail",
            "proposal_source": str(rule.get("proposal_source") or "llm"),
            "validation": dict(rule.get("validation") or {}),
        }
        evaluated_rules.append(rule_record)
        if outcome:
            passed_rule_names.append(rule_record["rule_name"])
        else:
            failed_rules.append(rule_record)

    if not evaluated_rules:
        return {}
    return {
        "rule_pool_source": "agentic_clean_rule_pool",
        "rule_family": family,
        "evaluated_rule_count": len(evaluated_rules),
        "passed_rule_count": len(evaluated_rules) - len(failed_rules),
        "failed_rule_count": len(failed_rules),
        "failed_rules": failed_rules,
        "passed_rule_names": passed_rule_names,
    }


def _prototype_distance_evidence(text: str, metadata: Dict[str, Any]) -> Tuple[float, float]:
    prototype_candidates = (metadata.get("prototype_candidates") or {}).get("candidates") or []
    if not text or not prototype_candidates:
        return 0.0, 0.0

    best_similarity = 0.0
    for candidate in prototype_candidates:
        candidate_value = str(candidate.get("value") or candidate.get("example") or "").strip()
        if not candidate_value:
            continue
        edit_similarity, ngram_similarity, token_similarity, _ = _prototype_similarity(text, candidate_value)
        similarity = 0.55 * edit_similarity + 0.25 * ngram_similarity + 0.20 * token_similarity
        best_similarity = max(best_similarity, similarity)

    if best_similarity >= 0.88:
        return min(1.0, 0.4 + 0.6 * best_similarity), 0.0
    if best_similarity <= 0.35:
        return 0.0, min(1.0, 0.5 + (0.35 - best_similarity))
    return 0.0, 0.0


def _get_closed_set_legal_values(metadata: Dict[str, Any]) -> set:
    for key in ("allowed_values", "valid_values", "enum_values"):
        values = metadata.get(key)
        if values and isinstance(values, (list, tuple, set)):
            return {_normalize_signature(value) for value in values}
    return set()


def _enum_mutation_signature(source: str, target: str) -> str:
    """Return a compact repeated-corruption signature for two enum surfaces.

    The signature intentionally ignores the position and target character,
    retaining the source-side mutation token (for example, ``x`` replacing
    different letters in several values). This lets the profile distinguish a
    repeated corruption family from an isolated, potentially legitimate rare
    category without relying on a column name or a fixed vocabulary.
    """
    if not source or not target:
        return ""
    changed_tokens: List[str] = []
    if len(source) == len(target):
        changed_tokens = [
            char.casefold()
            for char, target_char in zip(source, target)
            if char != target_char
        ]
    else:
        matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
        for tag, source_start, source_end, _, _ in matcher.get_opcodes():
            if tag in {"replace", "delete"}:
                changed_tokens.extend(
                    char.casefold()
                    for char in source[source_start:source_end]
                    if char.isalnum()
                )
            elif tag == "insert":
                changed_tokens.append("<insert>")
    if not changed_tokens:
        return ""
    counts = Counter(changed_tokens)
    token, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return token


def _infer_repeated_character_contracts(
    value_records: List[ValueRecord],
    metadata: Optional[Dict[str, Any]] = None,
    max_source_support: int = 3,
    min_group_support: int = 3,
    target_support_multiplier: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Find repeated character-substitution families from a value profile.

    The detector is deliberately label-free and column-name agnostic. It uses
    frequent observed surfaces as prototypes, then groups rare near-neighbors
    by the source-side mutation token. The returned contracts are discovery
    evidence only; callers must keep them out of inference until active labels
    validate the resulting mechanism cluster.
    """
    del metadata  # Reserved for future profile-aware thresholds.
    if not value_records:
        return {}

    normalized_records: List[Tuple[ValueRecord, str, int]] = []
    for record in value_records:
        value = record.representative_value
        if not isinstance(value, str):
            continue
        source = _normalize_signature(value)
        if source in {"<na>", "<empty>"} or not re.search(r"[a-z0-9]", source):
            continue
        if len(source) > 512:
            continue
        normalized_records.append((record, source, int(record.support_count)))
    if not normalized_records:
        return {}

    target_records = [
        (record, source, support)
        for record, source, support in normalized_records
        if support >= 5
    ]
    if not target_records:
        return {}

    # Keep the profile bounded on high-cardinality columns while retaining all
    # strong modes. Rare values are compared against prototypes, not against
    # one another, so this does not discard candidate corruption groups.
    target_records.sort(key=lambda item: (-item[2], item[1]))
    target_records = target_records[:256]
    pair_candidates: List[Tuple[str, str, int, str, int, int, str]] = []
    for record, source, source_support in normalized_records:
        if source_support > int(max_source_support):
            continue
        candidates: List[Tuple[int, int, str, str]] = []
        for target_record, target, target_support in target_records:
            if target == source or target_support < max(
                5, source_support * int(target_support_multiplier)
            ):
                continue
            max_length = max(len(source), len(target))
            max_distance = max(1, min(24, int(math.ceil(0.25 * max_length))))
            distance = _levenshtein_distance(
                source,
                target,
                max_distance=max_distance,
            )
            if distance < 1 or distance > max_distance:
                continue
            signature = _enum_mutation_signature(source, target)
            # Insertions/deletions and non-alphabetic substitutions are too
            # ambiguous to promote as a repeated character mechanism. Numeric
            # edits are handled by numeric/format contracts elsewhere.
            if not signature or not signature.isalpha() or len(signature) != 1:
                continue
            candidates.append((int(target_support), -distance, target, signature))
        if not candidates:
            continue
        target_support, negative_distance, target, signature = max(candidates)
        pair_candidates.append((
            record.key,
            source,
            source_support,
            target,
            target_support,
            -negative_distance,
            signature,
        ))

    signature_groups: Dict[str, Set[str]] = defaultdict(set)
    for record_key, _, _, _, _, _, signature in pair_candidates:
        signature_groups[signature].add(record_key)
    contracts: Dict[str, Dict[str, Any]] = {}
    for record_key, source, source_support, target, target_support, distance, signature in pair_candidates:
        group_support = len(signature_groups[signature])
        if group_support < max(1, int(min_group_support)):
            continue
        target_ratio = float(target_support / max(source_support, 1))
        distance_ratio = float(distance / max(len(source), 1))
        confidence = float(np.clip(
            0.55
            + 0.15 * min(1.0, target_ratio / max(float(target_support_multiplier), 1.0))
            + 0.20 * min(1.0, group_support / 6.0)
            + 0.10 * max(0.0, 1.0 - distance_ratio),
            0.0,
            1.0,
        ))
        # The identity is intentionally independent of the concrete column and
        # prototype value. Adaptive discovery will validate it per archetype.
        contracts[source] = {
            "candidate_value": target,
            "source_support": int(source_support),
            "target_support": int(target_support),
            "edit_distance": int(distance),
            "mutation_signature": signature,
            "candidate_group_support": int(group_support),
            "target_support_ratio": target_ratio,
            "confidence": confidence,
            "hard": False,
            "discovery_signature": f"repeated_character:{signature}",
            "mechanism_hint": "repeated_character_corruption",
        }
    return contracts


def _infer_unit_measure_contract(
    values: List[Any],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Infer a numeric-plus-unit contract from a unit-measure column."""
    if _column_archetype(metadata) != "unit_measure":
        return None
    unit_counts: Counter = Counter()
    numeric_count = 0
    non_missing_count = 0
    for value in values:
        if _is_missing_like(value, metadata):
            continue
        text = str(value).strip().casefold()
        if not text:
            continue
        non_missing_count += 1
        match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s+([a-z][a-z-]*)", text)
        if match:
            numeric_count += 1
            unit_counts[match.group(1)] += 1
        else:
            # Also learn a dominant unit from values with a missing/invalid
            # numeric prefix, provided the suffix is unambiguous.
            for unit, count in unit_counts.most_common(3):
                if text.endswith(unit):
                    unit_counts[unit] += 0
                    break
    if not unit_counts or non_missing_count == 0:
        return None
    dominant_unit, dominant_support = unit_counts.most_common(1)[0]
    return {
        "dominant_unit": dominant_unit,
        "dominant_support": int(dominant_support),
        "unit_counts": dict(unit_counts),
        "numeric_fraction": float(numeric_count / max(non_missing_count, 1)),
        "non_missing_count": int(non_missing_count),
    }


def _infer_unit_measure_anomaly(
    value: Any,
    value_record: Optional[ValueRecord],
    contract: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Check numeric/unit agreement without assuming a particular unit name."""
    if contract is None or value_record is None or _is_missing_like(value, metadata):
        return None
    text = str(value).strip().casefold()
    dominant_unit = str(contract.get("dominant_unit") or "")
    if not text or not dominant_unit:
        return None

    # Split at the learned dominant suffix first. This handles both a normal
    # separator and common token contamination such as ``6xpatients``.
    suffix_match = re.search(r"([a-z][a-z-]*)$", text)
    observed_unit = suffix_match.group(1) if suffix_match else ""
    prefix = text[: suffix_match.start()].strip() if suffix_match else text
    numeric_match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", prefix)
    unit_max_distance = max(1, min(2, int(round(0.25 * len(dominant_unit)))))
    unit_distance = (
        _levenshtein_distance(
            observed_unit,
            dominant_unit,
            max_distance=unit_max_distance,
        )
        if observed_unit
        else len(dominant_unit)
    )
    source_support = int(value_record.support_count)
    dominant_support = int(contract.get("dominant_support") or 0)
    rare_variant = source_support <= max(3, int(round(dominant_support * 0.10)))

    if observed_unit == dominant_unit and numeric_match:
        try:
            amount = float(prefix)
        except (TypeError, ValueError):
            amount = None
        # English plural agreement is a generic, high-precision contract for
        # count units. Only apply it when the profile has a dominant plural
        # surface and the amount is exactly one.
        if (
            amount == 1.0
            and dominant_unit.endswith("s")
            and len(dominant_unit) > 3
        ):
            return {
                "reason": "unit_plurality_mismatch",
                "observed_unit": observed_unit,
                "expected_unit": dominant_unit[:-1],
                "amount": amount,
                "source_support": source_support,
                "dominant_support": dominant_support,
                "hard": True,
                "confidence": 0.96,
            }
        return None

    if (
        observed_unit == dominant_unit
        and not numeric_match
        and contract.get("numeric_fraction", 0.0) >= 0.80
    ):
        return {
            "reason": "unit_numeric_prefix_mismatch",
            "observed_unit": observed_unit,
            "expected_unit": dominant_unit,
            "raw_prefix": prefix,
            "source_support": source_support,
            "dominant_support": dominant_support,
            "hard": True,
            "confidence": 0.93,
        }

    if (
        observed_unit
        and unit_distance <= unit_max_distance
        and rare_variant
        and dominant_support >= max(5, source_support * 5)
    ):
        return {
            "reason": "unit_token_mismatch",
            "observed_unit": observed_unit,
            "expected_unit": dominant_unit,
            "edit_distance": unit_distance,
            "source_support": source_support,
            "dominant_support": dominant_support,
            "hard": True,
            "confidence": float(np.clip(0.84 + 0.04 * (2 - unit_distance), 0.0, 1.0)),
        }
    return None


def _infer_whitespace_anomaly(
    value: Any,
    value_record: Optional[ValueRecord],
    profile: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Detect rare repeated internal whitespace from a column-level baseline."""
    if value_record is None or profile is None or _is_missing_like(value, metadata):
        return None
    text = str(value)
    if not re.search(r"[ \t]{2,}", text):
        return None
    if _column_archetype(metadata) not in {
        "open_entity", "geo_name", "closed_enum", "unit_measure"
    }:
        return None
    repeat_rate = float(profile.get("repeat_rate", 1.0))
    source_support = int(value_record.support_count)
    total_support = int(profile.get("non_missing_count", 0))
    source_rate = source_support / max(total_support, 1)
    # A low column-level rate and a low-support value indicate a formatting
    # outlier; high-rate columns may intentionally use aligned whitespace.
    if repeat_rate > 0.10 or source_rate > 0.10:
        return None
    return {
        "repeat_rate": repeat_rate,
        "source_support": source_support,
        "non_missing_count": total_support,
        # Formatting anomalies are informative but not universal corruption
        # contracts; active labels should validate them before promotion.
        "hard": False,
        "confidence": float(np.clip(0.75 + 0.20 * (1.0 - repeat_rate), 0.0, 1.0)),
    }


def _infer_closed_enum_contamination(
    value: Any,
    value_record: Optional[ValueRecord],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find a low-support near-neighbor of a concentrated enum mode.

    This is a profile-derived fallback for tables without an explicit enum
    vocabulary. It deliberately requires a strong mode, a large support
    imbalance, and a small edit distance so that legitimate multi-category
    enums are left untouched. The returned candidate is still auditable and
    can be promoted to a hard decision only under the strict confidence gate.
    """
    if value_record is None or _column_archetype(metadata) != "closed_enum":
        return None
    if _is_missing_like(value, metadata):
        return None
    source = _normalize_signature(value)
    if not source or source in {"<na>", "<empty>"}:
        return None

    semantics = metadata.get("semantics") or {}
    try:
        semantic_confidence = float(
            semantics.get("confidence", 0.0)
            if isinstance(semantics, dict)
            else getattr(semantics, "confidence", 0.0)
        )
    except (TypeError, ValueError):
        semantic_confidence = 0.0
    if semantic_confidence < 0.75:
        return None

    source_count = int(value_record.support_count)
    top_values = metadata.get("normalized_top_values") or []
    candidates: List[Tuple[Tuple[float, int, int], Dict[str, Any]]] = []
    for item in top_values:
        target = _normalize_signature(item.get("value"))
        if not target or target == source:
            continue
        try:
            target_count = int(item.get("count") or 0)
            target_ratio = float(item.get("ratio") or 0.0)
        except (TypeError, ValueError):
            continue
        # A concentrated dominant mode is needed; otherwise a rare category
        # is not distinguishable from a legitimate enum member.
        if target_count < 20 or target_ratio < 0.75:
            continue
        if source_count > max(20, int(round(target_count * 0.05))):
            continue
        if len(source) > 64 or len(target) > 64:
            continue
        max_distance = 1 if max(len(source), len(target)) <= 4 else max(
            1, min(3, int(round(0.15 * max(len(source), len(target)))))
        )
        distance = _levenshtein_distance(
            source,
            target,
            max_distance=max_distance,
        )
        if distance < 1 or distance > max_distance:
            continue
        # Preserve the coarse surface contract. This excludes accidental
        # matches between a code and a free-form phrase.
        source_surface = re.sub(r"[^a-z0-9]+", "", source)
        target_surface = re.sub(r"[^a-z0-9]+", "", target)
        if not source_surface or not target_surface:
            continue
        if abs(len(source_surface) - len(target_surface)) > max_distance:
            continue
        mutation_signature = _enum_mutation_signature(source, target)
        confidence = min(
            1.0,
            0.55
            + 0.30 * min(1.0, target_ratio)
            + 0.15 * min(1.0, target_count / max(source_count * 20, 1)),
        )
        payload = {
            "candidate_value": item.get("example", item.get("value")),
            "source_support": source_count,
            "target_support": target_count,
            "target_ratio": target_ratio,
            "edit_distance": distance,
            "confidence": confidence,
            "semantic_confidence": semantic_confidence,
            "mutation_signature": mutation_signature,
        }
        candidates.append(((target_ratio, target_count, -distance), payload))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best = candidates[0]
    # With a highly concentrated mode and a large support imbalance, this is
    # a deterministic contamination contract; otherwise keep it calibratable.
    strict_profile = bool(
        best["confidence"] >= 0.85
        and best["target_ratio"] >= 0.90
        and best["target_support"] >= max(50, best["source_support"] * 20)
    )
    best_target = _normalize_signature(best.get("candidate_value"))
    best_signature = str(best.get("mutation_signature") or "")
    # The first pass only evaluates the requested source value. Recompute the
    # support of the selected edit signature across all observed value groups
    # so repeated corruption is measured at the profile level.
    signature_support = 0
    if best_signature:
        for item in top_values:
            candidate_source = _normalize_signature(item.get("value"))
            if not candidate_source or candidate_source == best_target:
                continue
            try:
                candidate_count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                continue
            if candidate_count > max(20, int(round(best["target_support"] * 0.05))):
                continue
            if len(candidate_source) > 64 or len(best_target) > 64:
                continue
            candidate_max_distance = 1 if max(
                len(candidate_source), len(best_target)
            ) <= 4 else max(
                1,
                min(3, int(round(0.15 * max(len(candidate_source), len(best_target)))))
            )
            candidate_distance = _levenshtein_distance(
                candidate_source,
                best_target,
                max_distance=candidate_max_distance,
            )
            if candidate_distance < 1 or candidate_distance > candidate_max_distance:
                continue
            candidate_surface = re.sub(r"[^a-z0-9]+", "", candidate_source)
            target_surface = re.sub(r"[^a-z0-9]+", "", best_target)
            if (
                not candidate_surface
                or not target_surface
                or abs(len(candidate_surface) - len(target_surface))
                > candidate_max_distance
            ):
                continue
            if _enum_mutation_signature(candidate_source, best_target) == best_signature:
                signature_support += 1
    # A repeated edit signature is stronger than rarity alone. A singleton is
    # promoted only at an extreme support imbalance, leaving ambiguous rare
    # enum members as soft evidence for active-query validation.
    best["mutation_signature_support"] = int(signature_support)
    if (
        signature_support < 2
        and best["source_support"]
        > max(3, int(round(best["target_support"] * 0.01)))
    ):
        # One isolated near-neighbor is not enough to infer corruption from a
        # dirty table alone: it may be a legitimate minority category. Keep
        # the detector conservative and let explicit vocabularies, contextual
        # evidence, or a future active query resolve it.
        return None
    best["hard"] = bool(
        strict_profile
        and (
            signature_support >= 2
            or best["source_support"]
            <= max(3, int(round(best["target_support"] * 0.01)))
        )
    )
    return best


def _numeric_rarity_strength(numeric: float, metadata: Dict[str, Any]) -> float:
    quantiles = metadata.get("quantiles") or {}
    p05 = quantiles.get("p05")
    p95 = quantiles.get("p95")
    p01 = quantiles.get("p01")
    p99 = quantiles.get("p99")
    if p05 is None or p95 is None:
        return 0.0

    if p01 is not None and numeric < float(p01):
        return 1.0
    if p99 is not None and numeric > float(p99):
        return 1.0
    if numeric < float(p05):
        denom = max(float(p05) - float(p01 or p05), 1e-6)
        return float(min(1.0, 0.4 + (float(p05) - numeric) / denom))
    if numeric > float(p95):
        denom = max(float(p99 or p95) - float(p95), 1e-6)
        return float(min(1.0, 0.4 + (numeric - float(p95)) / denom))
    return 0.0


def _low_frequency_strength(value_record: ValueRecord, metadata: Dict[str, Any]) -> float:
    normalized = value_record.normalized_value
    low_frequency_values = {
        _normalize_signature(item.get("value"))
        for item in metadata.get("low_frequency_values") or []
        if item.get("value") is not None
    }
    if normalized in low_frequency_values:
        return 0.85
    if value_record.support_count <= 1:
        return 0.80
    if value_record.support_count == 2:
        return 0.65
    if value_record.support_count == 3:
        return 0.50
    return 0.0


def _allow_discrete_rarity_evidence(metadata: Dict[str, Any]) -> bool:
    """Whether singleton/category rarity is meaningful for this column."""
    semantics = metadata.get("semantics") or {}
    if isinstance(semantics, dict):
        archetype = str(semantics.get("archetype") or "")
        open_set_score = float(semantics.get("open_set_score") or 0.0)
    else:
        archetype = str(getattr(semantics, "archetype", "") or "")
        open_set_score = float(
            getattr(semantics, "open_set_score", 0.0) or 0.0
        )
    # Open measurement domains commonly contain unique legitimate values.
    # Format, range, unit, and canonicalization evidence remain active.
    if (
        archetype in {"numeric_measure", "unit_measure", "temporal_measure"}
        and open_set_score >= 0.20
    ):
        return False
    return True


def _update_structural_missing_from_labels(
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    active_labels: Dict[str, int],
) -> Dict[str, Any]:
    """Update the structural-missing prior using labels at archetype scope."""
    support: Dict[str, Counter] = defaultdict(Counter)
    for cell_key, label in active_labels.items():
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        observations = cell_observations.get(cell_key, [])
        if not any(
            item.reason_code == "structural_missing_token"
            for item in observations
        ):
            continue
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        archetype = (
            str(semantics.get("archetype") or "unknown")
            if isinstance(semantics, dict)
            else str(getattr(semantics, "archetype", "unknown"))
        )
        support[archetype]["dirty" if int(label) == 1 else "clean"] += 1

    promoted_archetypes = {
        archetype
        for archetype, counts in support.items()
        if counts["dirty"] > counts["clean"]
    }
    if not promoted_archetypes:
        return {"support": dict(support), "promoted_archetypes": [], "updated": 0}

    updated = 0
    for observations, registry in (
        (value_observations, value_registry),
        (cell_observations, cell_registry),
    ):
        for target_key, items in observations.items():
            record = registry.get(target_key)
            if record is None:
                continue
            column = record.column
            semantics = metadata.get(column, {}).get("semantics") or {}
            archetype = (
                str(semantics.get("archetype") or "unknown")
                if isinstance(semantics, dict)
                else str(getattr(semantics, "archetype", "unknown"))
            )
            if archetype not in promoted_archetypes:
                continue
            for item in items:
                if item.reason_code != "structural_missing_token":
                    continue
                item.strength = 1.0
                item.hard = True
                item.metadata["label_updated"] = True
                item.metadata["label_update_scope"] = archetype
                updated += 1
    return {
        "support": {
            archetype: dict(counts) for archetype, counts in support.items()
        },
        "promoted_archetypes": sorted(promoted_archetypes),
        "updated": updated,
    }


def _ensure_structural_missing_query_coverage(
    selected_cell_keys: List[str],
    budget: int,
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_registry: Dict[str, CellRecord],
    metadata: Dict[str, Dict[str, Any]],
    protected_cell_keys: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reserve at most one query per structural-missing archetype."""
    selected = list(dict.fromkeys(selected_cell_keys))[:max(int(budget), 0)]
    protected_cell_keys = set(protected_cell_keys or set())

    def archetype_for(cell_key: str) -> str:
        record = cell_registry.get(cell_key)
        if record is None:
            return "unknown"
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        return (
            str(semantics.get("archetype") or "unknown")
            if isinstance(semantics, dict)
            else str(getattr(semantics, "archetype", "unknown"))
        )

    structural_candidates: Dict[str, str] = {}
    structural_keys: Set[str] = set()
    for cell_key in sorted(cell_observations):
        if not any(
            item.reason_code == "structural_missing_token"
            for item in cell_observations[cell_key]
        ):
            continue
        structural_keys.add(cell_key)
        structural_candidates.setdefault(archetype_for(cell_key), cell_key)

    covered = {
        archetype_for(cell_key)
        for cell_key in selected
        if cell_key in structural_keys
    }
    inserted: List[str] = []
    replaced: List[str] = []
    for archetype, candidate in sorted(structural_candidates.items()):
        if archetype in covered or candidate in selected or budget <= 0:
            continue
        if len(selected) < budget:
            selected.append(candidate)
        else:
            counts = Counter(archetype_for(key) for key in selected)
            column_counts = Counter(
                cell_registry[key].column
                for key in selected
                if key in cell_registry
            )
            # Preserve semantic coverage: first replace a repeated query from
            # the same archetype, ideally from a duplicated column.
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in structural_keys
                    and selected[index] not in protected_cell_keys
                    and archetype_for(selected[index]) == archetype
                    and selected[index] in cell_registry
                    and column_counts[cell_registry[selected[index]].column] > 1
                ),
                None,
            )
            if replace_index is None:
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index] not in structural_keys
                        and selected[index] not in protected_cell_keys
                        and archetype_for(selected[index]) == archetype
                    ),
                    None,
                )
            if replace_index is None:
                replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in structural_keys
                    and selected[index] not in protected_cell_keys
                    and counts[archetype_for(selected[index])] > 1
                ),
                None,
            )
            if replace_index is None:
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index] not in structural_keys
                        and selected[index] not in protected_cell_keys
                    ),
                    None,
                )
            if replace_index is None:
                continue
            replaced.append(selected[replace_index])
            selected[replace_index] = candidate
        inserted.append(candidate)
        covered.add(archetype)
    return selected, {
        "structural_archetypes": sorted(structural_candidates),
        "inserted": inserted,
        "replaced": replaced,
    }


def _ensure_mechanism_contrast_query_coverage(
    selected_cell_keys: List[str],
    budget: int,
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_registry: Dict[str, CellRecord],
    metadata: Dict[str, Dict[str, Any]],
    max_archetypes: int = 2,
) -> Tuple[List[str], Set[str], Dict[str, Any]]:
    """Protect match/mismatch contrasts for informative structured mechanisms."""
    selected = list(dict.fromkeys(selected_cell_keys))[:max(int(budget), 0)]
    structured_archetypes = {
        "numeric_measure", "geo_name", "closed_enum", "unit_measure",
    }

    def archetype_for(cell_key: str) -> str:
        record = cell_registry.get(cell_key)
        if record is None:
            return "unknown"
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        return str(
            semantics.get("archetype", "unknown")
            if isinstance(semantics, dict)
            else getattr(semantics, "archetype", "unknown")
        )

    cells_by_archetype: Dict[str, Set[str]] = defaultdict(set)
    mismatch_by_archetype: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    match_by_archetype: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for cell_key in cell_registry:
        archetype = archetype_for(cell_key)
        cells_by_archetype[archetype].add(cell_key)
        observations = cell_observations.get(cell_key, [])
        mismatch_strength = max(
            (float(item.strength) for item in observations if item.source_id == "pattern_mismatch"),
            default=0.0,
        )
        match_strength = max(
            (float(item.strength) for item in observations if item.source_id == "pattern_match"),
            default=0.0,
        )
        if mismatch_strength > 0.0:
            mismatch_by_archetype[archetype].append((mismatch_strength, cell_key))
        if match_strength > 0.0:
            match_by_archetype[archetype].append((match_strength, cell_key))

    ranked: List[Tuple[float, float, str]] = []
    ratios: Dict[str, float] = {}
    for archetype in structured_archetypes:
        total = len(cells_by_archetype.get(archetype, set()))
        mismatch_count = len(mismatch_by_archetype.get(archetype, []))
        if total <= 0 or mismatch_count <= 0 or not match_by_archetype.get(archetype):
            continue
        ratio = float(mismatch_count / total)
        ratios[archetype] = ratio
        if ratio > 0.50:
            continue
        needs = []
        for info in metadata.values():
            semantics = info.get("semantics") or {}
            if str(semantics.get("archetype") or "unknown") == archetype:
                needs.append(float(semantics.get("canonicalization_need") or 0.0))
        ranked.append((max(needs or [0.0]), -ratio, archetype))
    target_archetypes = [
        item[2] for item in sorted(ranked, reverse=True)[:max_archetypes]
    ]

    protected: Set[str] = set()
    inserted: List[str] = []
    replaced: List[str] = []

    def has_source(cell_key: str, source_id: str) -> bool:
        return any(
            item.source_id == source_id
            for item in cell_observations.get(cell_key, [])
        )

    def ensure_candidate(
        archetype: str,
        source_id: str,
        candidates: List[Tuple[float, str]],
    ) -> None:
        existing = next((
            key for key in selected
            if archetype_for(key) == archetype and has_source(key, source_id)
        ), None)
        if existing is not None:
            protected.add(existing)
            return
        candidate = max(candidates)[1]
        if candidate in selected:
            protected.add(candidate)
            return
        if len(selected) < budget:
            selected.append(candidate)
        else:
            replace_index = next((
                index for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in protected
                and archetype_for(selected[index]) == archetype
            ), None)
            if replace_index is None:
                replace_index = next((
                    index for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in protected
                ), None)
            if replace_index is None:
                return
            replaced.append(selected[replace_index])
            selected[replace_index] = candidate
        inserted.append(candidate)
        protected.add(candidate)

    for archetype in target_archetypes:
        ensure_candidate(archetype, "pattern_mismatch", mismatch_by_archetype[archetype])
        ensure_candidate(archetype, "pattern_match", match_by_archetype[archetype])

    return selected, protected, {
        "target_archetypes": target_archetypes,
        "mismatch_ratios": {key: ratios[key] for key in target_archetypes},
        "inserted": inserted,
        "replaced": replaced,
        "protected": sorted(protected),
    }


def _ensure_evidence_source_query_coverage(
    selected_cell_keys: List[str],
    budget: int,
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_registry: Dict[str, CellRecord],
    protected_reason_codes: Optional[Set[str]] = None,
    required_source_ids: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reserve one label for each sparse, explicitly requested evidence source."""
    selected = list(dict.fromkeys(selected_cell_keys))[:max(int(budget), 0)]
    protected_reason_codes = set(protected_reason_codes or set())
    required_source_ids = set(required_source_ids or set())

    def has_source(cell_key: str, source_id: str) -> bool:
        return any(
            item.source_id == source_id
            for item in cell_observations.get(cell_key, [])
        )

    def protected(cell_key: str) -> bool:
        return any(
            item.reason_code in protected_reason_codes
            or item.source_id in required_source_ids
            for item in cell_observations.get(cell_key, [])
        )

    inserted: List[str] = []
    replaced: List[str] = []
    for source_id in sorted(required_source_ids):
        candidates = [
            (float(item.strength), cell_key)
            for cell_key, items in cell_observations.items()
            for item in items
            if item.source_id == source_id
        ]
        if not candidates or any(has_source(key, source_id) for key in selected):
            continue
        candidate = max(candidates)[1]
        if candidate in selected or budget <= 0:
            continue
        if len(selected) < budget:
            selected.append(candidate)
        else:
            candidate_record = cell_registry.get(candidate)
            column_counts = Counter(
                cell_registry[key].column
                for key in selected if key in cell_registry
            )
            replace_index = next(
                (
                    index for index in range(len(selected) - 1, -1, -1)
                    if not protected(selected[index])
                    and selected[index] in cell_registry
                    and candidate_record is not None
                    and cell_registry[selected[index]].column == candidate_record.column
                    and column_counts[candidate_record.column] > 1
                ),
                None,
            )
            if replace_index is None:
                replace_index = next(
                    (
                        index for index in range(len(selected) - 1, -1, -1)
                        if not protected(selected[index])
                    ),
                    None,
                )
            if replace_index is None:
                continue
            replaced.append(selected[replace_index])
            selected[replace_index] = candidate
        inserted.append(candidate)
    return selected, {"sources": sorted(required_source_ids), "inserted": inserted, "replaced": replaced}


def _ensure_standardization_query_coverage(
    selected_cell_keys: List[str],
    budget: int,
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_observations: Dict[str, List[EvidenceObservation]],
    protected_cell_keys: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reserve one query for a high-need format violation within a fixed budget."""
    selected = list(dict.fromkeys(selected_cell_keys))[:max(int(budget), 0)]
    protected_cell_keys = set(protected_cell_keys or set())
    candidate_columns = {
        column for column, info in metadata.items()
        if str((info.get("semantics") or {}).get("archetype") or "") == "unit_measure"
        or float((info.get("semantics") or {}).get("canonicalization_need") or 0.0) >= 0.8
    }
    candidates: List[Tuple[float, str]] = []
    for cell_key, observations in cell_observations.items():
        record = cell_registry.get(cell_key)
        if record is None or record.column not in candidate_columns:
            continue
        mismatch_strength = max(
            (float(item.strength) for item in observations if item.source_id == "pattern_mismatch"),
            default=0.0,
        )
        if mismatch_strength > 0.0:
            need = float(
                ((metadata.get(record.column, {}).get("semantics") or {}).get(
                    "canonicalization_need", 0.0
                ))
            )
            candidates.append((need + mismatch_strength, cell_key))
    if not candidates:
        return selected, {"inserted": [], "replaced": [], "reason": "no_format_candidates"}
    candidates.sort(reverse=True)
    candidate = candidates[0][1]
    candidate_column = cell_registry[candidate].column
    if any(
        cell_registry.get(key) is not None
        and cell_registry[key].column == candidate_column
        and any(item.source_id == "pattern_mismatch" for item in cell_observations.get(key, []))
        for key in selected
    ):
        return selected, {"inserted": [], "replaced": [], "covered_column": candidate_column}
    if candidate in selected or budget <= 0:
        return selected, {"inserted": [], "replaced": [], "covered_column": candidate_column}
    replaced: List[str] = []
    if len(selected) < budget:
        selected.append(candidate)
    else:
        replace_index = next(
            (
                index for index in range(len(selected) - 1, -1, -1)
                if not any(
                    item.reason_code == "structural_missing_token"
                    for item in cell_observations.get(selected[index], [])
                )
                and selected[index] not in protected_cell_keys
            ),
            None,
        )
        if replace_index is None:
            return selected, {"inserted": [], "replaced": [], "reason": "no_replaceable_query"}
        replaced.append(selected[replace_index])
        selected[replace_index] = candidate
    return selected, {
        "inserted": [candidate],
        "replaced": replaced,
        "covered_column": candidate_column,
    }


def _ensure_core_mechanism_query_coverage(
    selected_cell_keys: List[str],
    budget: int,
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_observations: Dict[str, List[EvidenceObservation]],
    protected_cell_keys: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Discover and cover high-value evidence mechanisms within a fixed budget.

    Mechanisms are inferred from observed evidence rather than a dataset-specific
    list. Profile mechanisms are only semantic priors; observed support and
    strength determine which representative receives a label.
    """
    selected = list(dict.fromkeys(selected_cell_keys))[:max(int(budget), 0)]
    protected: Set[str] = set(protected_cell_keys or set())
    inserted: List[str] = []
    replaced: List[str] = []

    def archetype_for(cell_key: str) -> str:
        record = cell_registry.get(cell_key)
        if record is None:
            return "unknown"
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        return str(semantics.get("archetype") or "unknown")

    def profile_mechanisms(cell_key: str) -> Set[str]:
        record = cell_registry.get(cell_key)
        if record is None:
            return set()
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        values = semantics.get("possible_error_mechanisms", []) if isinstance(semantics, dict) else []
        return {str(value).strip().lower() for value in values if str(value).strip()}

    def profile_contract(cell_key: str, mechanism: str) -> Dict[str, Any]:
        record = cell_registry.get(cell_key)
        semantics = (metadata.get(record.column, {}).get("semantics") or {}) if record else {}
        contracts = semantics.get("mechanism_contracts", []) if isinstance(semantics, dict) else []
        for contract in contracts if isinstance(contracts, list) else []:
            if isinstance(contract, dict) and str(contract.get("mechanism", "")).lower() == mechanism:
                return contract
        # Conservative compatibility defaults for legacy profile caches.
        return {
            "confidence": 0.5,
            "repairability": {
                "encoding_artifact": 0.65,
                "temporal_format_violation": 0.55,
                "geo_suffix_shift": 0.55,
                "format_violation": 0.15,
            }.get(mechanism, 0.20),
            "safe_to_propagate": mechanism in {"encoding_artifact", "geo_suffix_shift"},
        }

    def mechanism_for(item: EvidenceObservation) -> str:
        source = str(item.source_id or item.reason_code or "unknown").lower()
        reason = str(item.reason_code or "").lower()
        if str(item.polarity or "").lower() == "clean":
            return "unknown"
        # Positive consistency observations describe clean values, not query
        # candidates. Missingness is handled by the dedicated structural
        # coverage pass so the two selectors do not compete for the same slot.
        if any(token in source or token in reason for token in (
            "pass", "success", "agreement", "satisfied", "prototype_close",
            "missing",
        )):
            return "unknown"
        if "unicode" in source or "unicode" in reason or "encoding" in source:
            return "encoding_artifact"
        if "temporal" in source or "temporal" in reason:
            return "temporal_format_violation"
        if "pagination" in source or "pagination" in reason:
            return "format_violation"
        if "parse" in source or "parse" in reason or "type_conflict" in source:
            return "parse_failure"
        if "pattern_mismatch" in source or "format_contract" in reason or "format_violation" in source:
            return "format_violation"
        if "rarity" in source or "frequency" in reason:
            return "outlier"
        return source or "unknown"

    def replace_with(candidate: Optional[str], mechanism: str) -> None:
        if candidate is None or candidate in selected or budget <= 0:
            return
        if len(selected) < budget:
            selected.append(candidate)
            inserted.append(candidate)
            protected.add(candidate)
            return

        candidate_archetype = archetype_for(candidate)
        def is_clean_proxy(cell_key: str) -> bool:
            """Keep a clean-looking calibration representative in the budget."""
            observations = cell_observations.get(cell_key, [])
            clean_strength = sum(
                max(0.0, float(item.strength))
                for item in observations
                if str(item.polarity).lower() == "clean"
            )
            dirty_strength = sum(
                max(0.0, float(item.strength))
                for item in observations
                if str(item.polarity).lower() == "dirty"
            )
            return clean_strength > 0.0 and clean_strength > dirty_strength

        replace_index = next((
            index for index in range(len(selected) - 1, -1, -1)
            if selected[index] not in protected
            and archetype_for(selected[index]) == candidate_archetype
            and not is_clean_proxy(selected[index])
            and not cell_observations.get(selected[index])
        ), None)
        if replace_index is None:
            replace_index = next((
                index for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in protected
                and not is_clean_proxy(selected[index])
            ), None)
        if replace_index is None:
            return
        replaced.append(selected[replace_index])
        selected[replace_index] = candidate
        inserted.append(candidate)
        protected.add(candidate)

    clusters: Dict[Tuple[str, str], List[Tuple[str, EvidenceObservation]]] = defaultdict(list)
    for cell_key, observations in cell_observations.items():
        for item in observations:
            mechanism = mechanism_for(item)
            if mechanism != "unknown":
                clusters[(archetype_for(cell_key), mechanism)].append((cell_key, item))
    ranked = []
    for (archetype, mechanism), members in clusters.items():
        unique_cells = list(dict.fromkeys(key for key, _ in members))
        support = len(unique_cells)
        strength = sum(max(0.0, float(item.strength)) for _, item in members) / max(len(members), 1)
        prior = sum(1 for key in unique_cells if mechanism in profile_mechanisms(key)) / max(support, 1)
        # A high-volume detector is not automatically a discovered mechanism.
        # Require profile compatibility before allowing it to consume the
        # dedicated coverage slots; otherwise leave it to D-optimal sampling.
        if prior <= 0.0:
            continue
        hard_support = sum(1 for _, item in members if bool(item.hard) and str(item.polarity).lower() == "dirty")
        # Broad soft format mismatches are often normal variants or an overly
        # narrow inferred pattern. Admit them to core coverage only when a
        # hard dirty witness exists; small clusters remain eligible for
        # discovery because they are naturally local anomalies.
        if mechanism == "format_violation" and support > 100 and hard_support == 0:
            continue
        contracts = [profile_contract(key, mechanism) for key in unique_cells]
        repairability = sum(float(item.get("repairability", 0.0)) for item in contracts) / max(len(contracts), 1)
        confidence = sum(float(item.get("confidence", 0.0)) for item in contracts) / max(len(contracts), 1)
        safe_ratio = sum(bool(item.get("safe_to_propagate", False)) for item in contracts) / max(len(contracts), 1)
        if mechanism == "format_violation" and repairability < 0.35 and safe_ratio < 0.25:
            continue
        # Diminishing returns prevent a broad, weak format detector from
        # monopolizing the fixed query budget. A smaller but repairable
        # mechanism can now outrank a huge low-value cluster.
        score = math.log1p(support) * (0.5 + strength) * (1.0 + 0.25 * prior) * max(0.25, repairability) * max(0.5, confidence)
        representative = max(unique_cells, key=lambda key: max(float(item.strength) for item in cell_observations.get(key, [])))
        ranked.append((score, representative, f"{archetype}:{mechanism}", support, strength))
    # Deterministic surface detectors remain available when an observation
    # source was not attached (for example in cached/legacy profiles). They
    # contribute ordinary ranked candidates rather than mandatory mechanisms.
    raw_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for cell_key, record in cell_registry.items():
        value = record.raw_value
        if archetype_for(cell_key) == "temporal_measure" and value is not None:
            if re.fullmatch(r"\s*\d{1,2}/\d{1,2}/\d{1,2}\s*", str(value)):
                raw_groups[(archetype_for(cell_key), "temporal_format_violation")].append(cell_key)
        unicode_kind = _unicode_corruption_kind(value)
        # Character pollution is a text-semantic mechanism. A stray mark in a
        # numeric/enum surface is handled by the column's ordinary conflict
        # evidence and must not receive a text-repair calibration label.
        if (
            unicode_kind is not None
            and unicode_kind != "non_latin_mixed_script"
            and archetype_for(cell_key) in {
            "free_text", "open_entity", "geo_name",
            }
        ):
            raw_groups[(archetype_for(cell_key), f"encoding_artifact:{unicode_kind}")].append(cell_key)
    for (archetype, mechanism), cells in raw_groups.items():
        if any(item[2] == f"{archetype}:{mechanism}" for item in ranked):
            continue
        unicode_bonus = 0.1 if "unresolved_combining_mark" in mechanism else 0.0
        base_score = len(cells) * (2.0 if mechanism == "temporal_format_violation" else 1.5 + unicode_bonus)
        ranked.append((base_score, cells[0], f"{archetype}:{mechanism}", len(cells), 1.0))
    ranked.sort(key=lambda item: (-item[0], item[2], item[1]))
    covered: Set[str] = set()
    coverage = []
    # Mechanism discovery is a coverage supplement, not a replacement for the
    # D-optimal calibration set. Keep most of the fixed budget for general
    # uncertainty reduction and structural-missing coverage.
    max_discovered = min(len(ranked), max(3, int(budget) // 3))
    for score, candidate, mechanism, support, strength in ranked:
        if len(coverage) >= max_discovered:
            break
        if mechanism in covered or candidate in selected:
            continue
        before = len(inserted)
        replace_with(candidate, mechanism)
        if len(inserted) > before:
            covered.add(mechanism)
            coverage.append({"mechanism": mechanism, "candidate": candidate, "support": support, "strength": round(strength, 4), "score": round(score, 4)})
    return selected, {
        "inserted": inserted,
        "replaced": replaced,
        "discovered_clusters": len(ranked),
        "coverage": coverage,
        # Backward-compatible diagnostic view for callers and reports that
        # previously consumed the Unicode-specific coverage field.
        "unicode_coverage": [
            {
                "kind": item["mechanism"].rsplit(":", 1)[1],
                "candidate": item["candidate"],
                "support": item["support"],
            }
            for item in coverage
            if ":encoding_artifact:" in item["mechanism"]
        ],
        "protected_seed_count": len(protected_cell_keys or set()),
    }


def _build_table_relationship_summary(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    max_candidates: int = 24,
) -> Dict[str, Any]:
    """Summarize statistically viable FD pairs without exposing the full table."""
    signatures = {
        column: [_normalize_signature(value) for value in df[column].tolist()]
        for column in df.columns
    }
    representatives = {
        column: {
            signature: value
            for signature, value in zip(signatures[column], df[column].tolist())
        }
        for column in df.columns
    }
    determinant_archetypes = {"identifier", "closed_enum"}
    dependent_archetypes = {"identifier", "closed_enum", "temporal_measure"}
    columns = []
    for column in df.columns:
        semantics = metadata.get(column, {}).get("semantics") or {}
        columns.append({
            "name": column,
            "archetype": str(semantics.get("archetype") or "unknown"),
        })

    candidates: List[Dict[str, Any]] = []
    for determinant_column in df.columns:
        determinant_semantics = metadata.get(determinant_column, {}).get("semantics") or {}
        if determinant_semantics.get("archetype") not in determinant_archetypes:
            continue
        determinant_signatures = signatures[determinant_column]
        for dependent_column in df.columns:
            if dependent_column == determinant_column:
                continue
            dependent_semantics = metadata.get(dependent_column, {}).get("semantics") or {}
            if dependent_semantics.get("archetype") not in dependent_archetypes:
                continue
            dependent_signatures = signatures[dependent_column]
            grouped_counts: Dict[str, Counter] = defaultdict(Counter)
            grouped_rows: Dict[str, Dict[str, List[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for row_idx, (determinant_sig, dependent_sig) in enumerate(
                zip(determinant_signatures, dependent_signatures)
            ):
                if determinant_sig in {"<na>", "<empty>"} or dependent_sig in {"<na>", "<empty>"}:
                    continue
                grouped_counts[determinant_sig][dependent_sig] += 1
                grouped_rows[determinant_sig][dependent_sig].append(row_idx)

            eligible_groups = []
            for determinant_sig, counts in grouped_counts.items():
                group_total = int(sum(counts.values()))
                if group_total < LLM_FD_MIN_GROUP_SUPPORT:
                    continue
                expected_sig, expected_count = counts.most_common(1)[0]
                eligible_groups.append((
                    determinant_sig,
                    group_total,
                    int(expected_count),
                    float(expected_count / group_total),
                    expected_sig,
                ))
            if not eligible_groups:
                continue

            predictiveness = _context_predictiveness(
                dependent_signatures,
                determinant_signatures,
            )
            examples = []
            for determinant_sig, group_total, expected_count, expected_probability, expected_sig in sorted(
                eligible_groups,
                key=lambda item: (-item[1], -item[3], item[0]),
            )[:2]:
                examples.append({
                    "determinant_value": str(
                        representatives[determinant_column].get(
                            determinant_sig, determinant_sig
                        )
                    ),
                    "group_size": group_total,
                    "dominant_dependent_value": str(
                        representatives[dependent_column].get(expected_sig, expected_sig)
                    ),
                    "dominant_count": expected_count,
                    "dominant_ratio": round(expected_probability, 4),
                })
            candidates.append({
                "determinant_column": determinant_column,
                "dependent_column": dependent_column,
                "predictiveness": round(float(predictiveness), 4),
                "repeated_group_count": len(eligible_groups),
                "median_group_size": float(np.median([group[1] for group in eligible_groups])),
                "max_group_size": max(group[1] for group in eligible_groups),
                "median_dominant_ratio": round(
                    float(np.median([group[3] for group in eligible_groups])), 4
                ),
                "examples": examples,
            })

    candidates.sort(
        key=lambda item: (
            -item["predictiveness"],
            -item["max_group_size"],
            item["dependent_column"],
        )
    )
    # Preserve directional coverage: a highly predictive reverse relation must
    # not crowd out the semantically meaningful direction before LLM review.
    by_determinant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_determinant[candidate["determinant_column"]].append(candidate)
    selected_candidates: List[Dict[str, Any]] = []
    rank = 0
    while len(selected_candidates) < max_candidates:
        added = False
        for determinant_column in sorted(by_determinant):
            items = by_determinant[determinant_column]
            if rank < len(items) and len(selected_candidates) < max_candidates:
                selected_candidates.append(items[rank])
                added = True
        if not added:
            break
        rank += 1
    return {
        "row_count": int(len(df)),
        "columns": columns,
        "candidates": selected_candidates,
    }


def _validate_llm_relationship_rule_candidates(
    df: pd.DataFrame,
    raw_rules: List[Dict[str, Any]],
    table_summary: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate the limited FD schema and bind every accepted proposal to data."""
    candidate_by_pair = {
        (str(item.get("determinant_column")), str(item.get("dependent_column"))): item
        for item in table_summary.get("candidates") or []
        if isinstance(item, dict)
    }
    accepted: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    seen_rule_ids: Set[str] = set()
    available_columns = set(df.columns)

    for raw_rule in raw_rules:
        report = {"proposal": dict(raw_rule), "accepted": False, "reason": ""}
        determinant_columns = raw_rule.get("determinant_columns")
        dependent_column = raw_rule.get("dependent_column")
        determinant_granularity = raw_rule.get("determinant_granularity")
        dependent_variability = raw_rule.get("dependent_variability")
        if raw_rule.get("scope") != "cross_row":
            report["reason"] = "unsupported_scope"
        elif raw_rule.get("type") != "functional_dependency":
            report["reason"] = "unsupported_type"
        elif not isinstance(determinant_columns, list) or len(determinant_columns) != 1:
            report["reason"] = "requires_one_determinant_column"
        elif not isinstance(determinant_columns[0], str) or not isinstance(dependent_column, str):
            report["reason"] = "invalid_column_name"
        elif determinant_granularity not in {"instance", "recurring_entity", "unknown"}:
            report["reason"] = "invalid_determinant_granularity"
        elif dependent_variability not in {"invariant", "instance_varying", "unknown"}:
            report["reason"] = "invalid_dependent_variability"
        elif determinant_granularity == "unknown" or dependent_variability == "unknown":
            report["reason"] = "ambiguous_semantic_granularity"
        elif (
            determinant_granularity == "recurring_entity"
            and dependent_variability == "instance_varying"
        ):
            report["reason"] = "instance_varying_dependent_requires_instance_key"
        else:
            determinant_column = determinant_columns[0]
            if determinant_column not in available_columns or dependent_column not in available_columns:
                report["reason"] = "unknown_column"
            elif determinant_column == dependent_column:
                report["reason"] = "same_determinant_and_dependent"
            elif (determinant_column, dependent_column) not in candidate_by_pair:
                report["reason"] = "not_a_statistically_viable_candidate"
            else:
                rule_id = str(raw_rule.get("rule_id") or "").strip()
                if not rule_id:
                    rule_id = f"fd_{determinant_column}_{dependent_column}"
                if rule_id in seen_rule_ids:
                    report["reason"] = "duplicate_rule_id"
                else:
                    seen_rule_ids.add(rule_id)
                    validation = {
                        "schema": "cross_row_functional_dependency_v2",
                        "candidate_statistics": candidate_by_pair[
                            (determinant_column, dependent_column)
                        ],
                        "determinant_granularity": determinant_granularity,
                        "dependent_variability": dependent_variability,
                    }
                    accepted_rule = {
                        "rule_id": rule_id,
                        "scope": "cross_row",
                        "type": "functional_dependency",
                        "determinant_column": determinant_column,
                        "dependent_column": dependent_column,
                        "determinant_granularity": determinant_granularity,
                        "dependent_variability": dependent_variability,
                        "claim": str(raw_rule.get("claim") or (
                            f"Rows with the same '{determinant_column}' normally share "
                            f"'{dependent_column}'"
                        )),
                        "proposal_source": "llm",
                        "validation": validation,
                    }
                    accepted.append(accepted_rule)
                    report.update({
                        "accepted": True,
                        "reason": "accepted",
                        "validated_rule": accepted_rule,
                    })
        reports.append(report)
    return accepted, reports



def _entropy_from_counts(counts: Counter) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / total
        if probability > 0:
            entropy -= probability * np.log(probability)
    return float(entropy)


def _context_predictiveness(
    target_signatures: List[str],
    context_signatures: List[str],
) -> float:
    target_counts = Counter(target_signatures)
    target_entropy = _entropy_from_counts(target_counts)
    if target_entropy <= 1e-12:
        return 0.0

    grouped_counts: Dict[str, Counter] = defaultdict(Counter)
    for target_sig, context_sig in zip(target_signatures, context_signatures):
        if context_sig in {"<na>", "<empty>"}:
            continue
        grouped_counts[context_sig][target_sig] += 1

    total = float(len(target_signatures))
    conditional_entropy = 0.0
    covered = 0.0
    for counts in grouped_counts.values():
        group_total = float(sum(counts.values()))
        if group_total <= 0:
            continue
        covered += group_total
        conditional_entropy += (group_total / total) * _entropy_from_counts(counts)

    if covered <= 0:
        return 0.0
    return float(np.clip(1.0 - conditional_entropy / target_entropy, 0.0, 1.0))


def _column_name_tokens(column: str) -> Set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column))
    return {
        token for token in re.split(r"[^a-z0-9]+", expanded.lower()) if token
    }


def _parse_temporal_minutes(value: Any) -> float:
    """Parse a time-of-day without attaching dataset-specific semantics."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if text.lower() in {
        "", "nan", "na", "n/a", "none", "null", "unknown",
        "not available", "xxxxx",
    }:
        return float("nan")
    normalized = re.sub(
        r"(?i)(\d{1,2}):?(\d{2})\s*(a|p)\b", r"\1:\2 \3m", text
    )
    normalized = re.sub(
        r"(?i)(\d{1,2}):?(\d{2})\s*(am|pm)\b", r"\1:\2 \3", normalized
    )
    try:
        parsed = date_parser.parse(
            normalized,
            fuzzy=True,
            default=datetime(2000, 1, 1, 0, 0),
        )
        return float(parsed.hour * 60 + parsed.minute)
    except (TypeError, ValueError, OverflowError):
        match = re.fullmatch(r"\s*(\d{3,4})\s*", text)
        if match:
            hour, minute = divmod(int(match.group(1)), 100)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return float(hour * 60 + minute)
    return float("nan")


def _circular_minute_distance(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 720.0) % 1440.0) - 720.0)


def _infer_multisource_temporal_roles(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    min_sources: int,
) -> Dict[str, Any]:
    """Infer semantic roles, then validate them with repeated-source support."""
    source_terms = {"source", "src", "provenance", "provider", "feed"}
    observed_terms = {"actual", "act", "observed", "observation", "measured", "realized"}
    planned_terms = {"scheduled", "schedule", "sched", "planned", "expected"}

    source_candidates = [
        column for column in df.columns
        if _column_name_tokens(column) & source_terms
    ]
    entity_candidates = [
        column for column in df.columns
        if _column_archetype(metadata.get(column, {})) == "identifier"
    ]
    temporal_columns = [
        column for column in df.columns
        if _column_archetype(metadata.get(column, {})) == "temporal_measure"
    ]
    observed_columns = [
        column for column in temporal_columns
        if _column_name_tokens(column) & observed_terms
    ]
    planned_columns = [
        column for column in temporal_columns
        if _column_name_tokens(column) & planned_terms
    ]
    if not source_candidates or not entity_candidates or not observed_columns:
        return {"applicable": False, "reason": "required_semantic_roles_missing"}

    best_pair = None
    for entity_column in entity_candidates:
        entity_unique_ratio = float(df[entity_column].nunique(dropna=True) / max(len(df), 1))
        if entity_unique_ratio >= 0.95:
            continue
        for source_column in source_candidates:
            if source_column == entity_column:
                continue
            work = pd.DataFrame({
                "entity": df[entity_column],
                "source": df[source_column].fillna("<NA>").astype(str),
            }).dropna(subset=["entity"])
            supports = work.groupby("entity")["source"].nunique()
            supported_groups = int(supports.ge(max(2, int(min_sources))).sum())
            supported_entities = supports.index[
                supports.ge(max(2, int(min_sources)))
            ]
            supported_rows = int(work["entity"].isin(supported_entities).sum())
            score = (supported_groups, supported_rows, -entity_unique_ratio)
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, entity_column, source_column)
    if best_pair is None or best_pair[0][0] <= 0:
        return {"applicable": False, "reason": "insufficient_repeated_source_groups"}

    _, entity_column, source_column = best_pair
    return {
        "applicable": True,
        "entity_column": entity_column,
        "source_column": source_column,
        "observed_columns": observed_columns,
        "planned_columns": planned_columns,
        "supported_groups": int(best_pair[0][0]),
        "supported_rows": int(best_pair[0][1]),
        "role_inference": "semantic_archetype_name_tokens_plus_data_validation",
    }


def _learn_temporal_source_reliability(
    frame: pd.DataFrame,
    entity_column: str,
    source_column: str,
    parsed_values: pd.Series,
    min_other_sources: int = 3,
    consensus_ratio: float = 0.60,
    tolerance: float = 10.0,
) -> Dict[str, float]:
    sources = frame[source_column].fillna("<NA>").astype(str)
    stats = {source: [1.0, 2.0] for source in sources.unique()}
    work = pd.DataFrame({
        "entity": frame[entity_column], "source": sources, "value": parsed_values,
    })
    work["bucket"] = (work["value"] / 5.0).round() * 5.0
    for _, group in work.dropna(subset=["entity", "value"]).groupby("entity"):
        group = group.drop_duplicates("source", keep=False)
        for _, row in group.iterrows():
            others = group[group["source"].ne(row["source"])]
            if others["source"].nunique() < min_other_sources:
                continue
            counts = others["bucket"].value_counts()
            if counts.empty or float(counts.iloc[0] / len(others)) < consensus_ratio:
                continue
            expected = float(counts.index[0])
            agrees = _circular_minute_distance(float(row["value"]), expected) <= tolerance
            stats[str(row["source"])][0] += float(agrees)
            stats[str(row["source"])][1] += 1.0
    return {source: success / total for source, (success, total) in stats.items()}


def _add_multisource_temporal_consensus_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    min_sources: int = MULTISOURCE_MIN_DISTINCT_SOURCES,
    min_ratio: float = MULTISOURCE_MIN_DOMINANT_WEIGHT_RATIO,
    tolerance: float = MULTISOURCE_TIME_TOLERANCE_MINUTES,
    min_source_reliability: float = MULTISOURCE_MIN_SOURCE_RELIABILITY,
) -> Dict[str, Any]:
    roles = _infer_multisource_temporal_roles(df, metadata, min_sources)
    if not roles.get("applicable"):
        return roles
    entity_column = str(roles["entity_column"])
    source_column = str(roles["source_column"])
    source_values = df[source_column].fillna("<NA>").astype(str)
    emitted = Counter()
    reliability_summary: Dict[str, Dict[str, float]] = {}

    for target_column in roles["observed_columns"]:
        parsed = df[target_column].map(_parse_temporal_minutes)
        reliability = _learn_temporal_source_reliability(
            df, entity_column, source_column, parsed,
            min_other_sources=max(2, min_sources),
            consensus_ratio=max(0.50, min_ratio),
            tolerance=tolerance,
        )
        reliability_summary[target_column] = reliability
        work = pd.DataFrame({
            "entity": df[entity_column], "source": source_values,
            "value": parsed,
        })
        work["bucket"] = (work["value"] / 5.0).round() * 5.0
        for entity_value, group in work.dropna(subset=["entity", "value"]).groupby("entity"):
            group = group.drop_duplicates("source", keep=False)
            if group["source"].nunique() < min_sources:
                continue
            bucket_weights: Dict[float, float] = defaultdict(float)
            for row in group.itertuples():
                bucket_weights[float(row.bucket)] += reliability.get(str(row.source), 0.5)
            total_weight = float(sum(bucket_weights.values()))
            if total_weight <= 0.0:
                continue
            expected, expected_weight = max(bucket_weights.items(), key=lambda item: item[1])
            dominant_ratio = float(expected_weight / total_weight)
            if dominant_ratio < min_ratio:
                continue
            reference_rows = [
                int(idx) for idx, row in group.iterrows()
                if _circular_minute_distance(float(row["value"]), expected) <= tolerance
            ][:5]
            for idx, row in group.iterrows():
                source = str(row["source"])
                source_reliability = float(reliability.get(source, 0.5))
                distance = _circular_minute_distance(float(row["value"]), expected)
                if distance <= tolerance or source_reliability < min_source_reliability:
                    continue
                distance_strength = 0.5 + 0.5 * min(1.0, distance / max(4.0 * tolerance, 1.0))
                strength = float(np.clip(
                    dominant_ratio * source_reliability * distance_strength, 0.0, 1.0
                ))
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=f"{int(idx)}::{target_column}",
                    source_id="multisource_temporal_disagreement",
                    family="contextual",
                    polarity="dirty",
                    strength=strength,
                    hard=False,
                    reason_code="source_weighted_temporal_consensus_disagreement",
                    metadata={
                        "relation_type": "multisource_temporal_consensus",
                        "entity_column": entity_column,
                        "source_column": source_column,
                        "target_column": target_column,
                        "entity_value": entity_value,
                        "source_value": source,
                        "observed_minutes": float(row["value"]),
                        "expected_minutes": float(expected),
                        "distance_minutes": float(distance),
                        "distinct_sources": int(group["source"].nunique()),
                        "dominant_weight_ratio": dominant_ratio,
                        "source_reliability": source_reliability,
                        "minimum_source_reliability": float(min_source_reliability),
                        "minimum_dominant_ratio": float(min_ratio),
                        "tolerance_minutes": float(tolerance),
                        "reference_rows": reference_rows,
                        "role_inference": roles["role_inference"],
                        "planned_columns": list(roles.get("planned_columns") or []),
                    },
                ))
                emitted[target_column] += 1
    return {
        **roles,
        "emitted": dict(emitted),
        "source_reliability": reliability_summary,
        "configuration": {
            "min_sources": int(min_sources), "min_ratio": float(min_ratio),
            "tolerance": float(tolerance),
            "min_source_reliability": float(min_source_reliability),
        },
    }


def _add_contextual_consensus_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    validated_relationship_rules: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Add generic row-context evidence learned from conditional concentration.

    A context column is useful only to the extent that it reduces uncertainty
    about the target column across the table. This keeps the signal generic:
    it can represent entity consensus, functional dependencies, or repeated
    source agreement without hard-coding any dataset-specific relationship.
    """

    n_rows = len(df)
    if n_rows <= 1 or len(df.columns) <= 1:
        return

    llm_rules_by_pair = {
        (
            str(rule.get("determinant_column")),
            str(rule.get("dependent_column")),
        ): rule
        for rule in (validated_relationship_rules or [])
        if rule.get("scope") == "cross_row"
        and rule.get("type") == "functional_dependency"
    }

    signatures_by_column = {
        column: [_normalize_signature(value) for value in df[column].tolist()]
        for column in df.columns
    }
    representatives_by_column: Dict[str, Dict[str, Any]] = {}
    for column in df.columns:
        representatives: Dict[str, Any] = {}
        for signature, value in zip(signatures_by_column[column], df[column].tolist()):
            representatives.setdefault(signature, value)
        representatives_by_column[column] = representatives

    for target_column in df.columns:
        target_signatures = signatures_by_column[target_column]
        context_profiles: List[
            Tuple[
                str,
                float,
                Dict[str, Counter],
                Dict[str, Dict[str, List[int]]],
            ]
        ] = []

        for context_column in df.columns:
            if context_column == target_column:
                continue
            context_signatures = signatures_by_column[context_column]
            predictiveness = _context_predictiveness(target_signatures, context_signatures)
            if predictiveness <= 0.0:
                continue

            grouped_counts: Dict[str, Counter] = defaultdict(Counter)
            grouped_rows: Dict[str, Dict[str, List[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for candidate_idx, (target_sig, context_sig) in enumerate(
                zip(target_signatures, context_signatures)
            ):
                if context_sig in {"<na>", "<empty>"}:
                    continue
                grouped_counts[context_sig][target_sig] += 1
                grouped_rows[context_sig][target_sig].append(candidate_idx)

            context_profiles.append(
                (
                    context_column,
                    predictiveness,
                    grouped_counts,
                    grouped_rows,
                )
            )

        if not context_profiles:
            continue

        col_meta = metadata.get(target_column, {})
        for row_idx, target_sig in enumerate(target_signatures):
            value = df.iloc[row_idx][target_column]
            if _is_missing_like(value, col_meta):
                continue

            best_disagreement = 0.0
            best_agreement = 0.0
            best_disagreement_context = ""
            best_agreement_context = ""
            best_disagreement_metadata: Dict[str, Any] = {}
            best_agreement_metadata: Dict[str, Any] = {}

            for (
                context_column,
                predictiveness,
                grouped_counts,
                grouped_rows,
            ) in context_profiles:
                context_sig = signatures_by_column[context_column][row_idx]
                counts = grouped_counts.get(context_sig)
                if not counts:
                    continue
                group_total = float(sum(counts.values()))
                if group_total < CONTEXT_MIN_GROUP_SUPPORT:
                    continue

                support = float(counts.get(target_sig, 0))
                conditional_probability = support / group_total
                group_information = np.log1p(group_total - 1.0) / np.log1p(max(float(n_rows), 2.0))
                strength_base = float(np.clip(predictiveness * group_information, 0.0, 1.0))
                disagreement = strength_base * (1.0 - conditional_probability)
                agreement = strength_base * conditional_probability
                expected_sig, expected_count = counts.most_common(1)[0]
                expected_probability = float(expected_count / group_total)
                relation_is_reliable = bool(
                    predictiveness >= CONTEXT_MIN_PREDICTIVENESS
                    and expected_count >= CONTEXT_MIN_GROUP_SUPPORT
                    and expected_probability
                    >= CONTEXT_MIN_EXPECTED_PROBABILITY
                )
                target_archetype = str(
                    (col_meta.get("semantics") or {}).get("archetype") or ""
                )
                identifier_agreement_is_reliable = bool(
                    target_archetype == "identifier"
                    and group_total >= CONTEXT_IDENTIFIER_CLEAN_MIN_GROUP_SUPPORT
                    and predictiveness
                    >= CONTEXT_IDENTIFIER_CLEAN_MIN_PREDICTIVENESS
                )
                llm_rule = llm_rules_by_pair.get(
                    (context_column, target_column)
                )
                llm_low_support_disagreement_is_reliable = bool(
                    llm_rule
                    and target_sig != expected_sig
                    and group_total >= LLM_FD_MIN_GROUP_SUPPORT
                    and expected_count >= LLM_FD_MIN_EXPECTED_SUPPORT
                    and expected_count > support
                    and expected_probability >= LLM_FD_MIN_EXPECTED_PROBABILITY
                    and predictiveness >= CONTEXT_MIN_PREDICTIVENESS
                )
                disagreement_is_reliable = bool(
                    relation_is_reliable
                    or llm_low_support_disagreement_is_reliable
                )
                agreement_is_reliable = bool(
                    relation_is_reliable or identifier_agreement_is_reliable
                )
                if target_sig != expected_sig and not disagreement_is_reliable:
                    continue
                if target_sig == expected_sig and not agreement_is_reliable:
                    continue
                expected_value = representatives_by_column[target_column].get(
                    expected_sig,
                    expected_sig,
                )
                context_value = df.iloc[row_idx][context_column]
                reference_rows = [
                    int(reference_idx)
                    for reference_idx in (
                        grouped_rows.get(context_sig, {}).get(expected_sig, [])
                    )
                    if reference_idx != row_idx
                ][:5]
                relation_metadata = {
                    "relation_type": "conditional_consensus",
                    "determinant_column": context_column,
                    "dependent_column": target_column,
                    "left_column": context_column,
                    "right_column": target_column,
                    "related_columns": [context_column, target_column],
                    "context_column": context_column,
                    "determinant_value": context_value,
                    "observed_value": value,
                    "expected_value": expected_value,
                    "group_size": int(group_total),
                    "observed_support": int(support),
                    "expected_support": int(expected_count),
                    "conditional_probability": float(conditional_probability),
                    "expected_probability": expected_probability,
                    "predictiveness": float(predictiveness),
                    "validation_status": (
                        "data_validated"
                        if relation_is_reliable
                        else "identifier_semantic_agreement"
                        if identifier_agreement_is_reliable
                        else "llm_validated_functional_dependency"
                    ),
                    "agreement_reliability": (
                        "strict_relation"
                        if relation_is_reliable
                        else "identifier_semantic"
                        if identifier_agreement_is_reliable
                        else "llm_validated_low_support"
                    ),
                    "minimum_group_support": (
                        CONTEXT_MIN_GROUP_SUPPORT
                        if relation_is_reliable
                        else LLM_FD_MIN_GROUP_SUPPORT
                    ),
                    "minimum_expected_probability": (
                        CONTEXT_MIN_EXPECTED_PROBABILITY
                        if relation_is_reliable
                        else LLM_FD_MIN_EXPECTED_PROBABILITY
                    ),
                    "reference_rows": reference_rows,
                    "rule_violated": bool(target_sig != expected_sig),
                    "rule_text": (
                        llm_rule.get("claim")
                        if llm_rule
                        else f"Rows with the same '{context_column}' value should "
                        f"normally share the dominant '{target_column}' value"
                    ),
                    "proposal_source": (
                        llm_rule.get("proposal_source") if llm_rule else "data"
                    ),
                    "relationship_rule_id": (
                        llm_rule.get("rule_id") if llm_rule else None
                    ),
                    "relationship_claim": (
                        llm_rule.get("claim") if llm_rule else None
                    ),
                    "relationship_validation": (
                        llm_rule.get("validation") if llm_rule else None
                    ),
                }

                # A low probability is not itself a violation when the observed
                # value is still the group's dominant value. This guard prevents
                # diverse groups from creating self-contradictory dirty evidence.
                if target_sig != expected_sig and disagreement > best_disagreement:
                    best_disagreement = disagreement
                    best_disagreement_context = context_column
                    best_disagreement_metadata = dict(relation_metadata)
                if target_sig == expected_sig and agreement > best_agreement:
                    best_agreement = agreement
                    best_agreement_context = context_column
                    best_agreement_metadata = dict(relation_metadata)

            cell_key = f"{row_idx}::{target_column}"
            if best_disagreement > 0.0:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id=(
                        "llm_fd_disagreement"
                        if best_disagreement_metadata.get("validation_status")
                        == "llm_validated_functional_dependency"
                        else "contextual_disagreement"
                    ),
                    family=(
                        "relationship"
                        if best_disagreement_metadata.get("validation_status")
                        == "llm_validated_functional_dependency"
                        else "contextual"
                    ),
                    polarity="dirty",
                    strength=best_disagreement,
                    hard=False,
                    reason_code=(
                        "llm_validated_functional_dependency"
                        if best_disagreement_metadata.get("validation_status")
                        == "llm_validated_functional_dependency"
                        else "conditional_value_disagreement"
                    ),
                    metadata=best_disagreement_metadata or {
                        "context_column": best_disagreement_context,
                        "determinant_column": best_disagreement_context,
                        "dependent_column": target_column,
                        "related_columns": [
                            best_disagreement_context,
                            target_column,
                        ],
                    },
                ))
            if best_agreement > 0.0:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="contextual_agreement",
                    family="contextual",
                    polarity="clean",
                    strength=best_agreement,
                    hard=False,
                    reason_code="conditional_value_agreement",
                    metadata=best_agreement_metadata or {
                        "context_column": best_agreement_context,
                        "determinant_column": best_agreement_context,
                        "dependent_column": target_column,
                        "related_columns": [
                            best_agreement_context,
                            target_column,
                        ],
                    },
                ))


def _build_value_registry(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, ValueRecord]:
    registry: Dict[str, ValueRecord] = {}
    for column in df.columns:
        groups: Dict[str, ValueRecord] = {}
        for row_idx, value in enumerate(df[column].tolist()):
            normalized = _normalize_signature(value)
            value_key = f"{column}::{normalized}"
            record = groups.get(value_key)
            if record is None:
                record = ValueRecord(
                    key=value_key,
                    column=column,
                    normalized_value=normalized,
                    representative_value=value,
                    row_indices=[],
                    support_count=0,
                )
                groups[value_key] = record
            record.row_indices.append(row_idx)
            record.support_count += 1
        registry.update(groups)
    return registry


def _build_cell_registry(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
) -> Dict[str, CellRecord]:
    registry: Dict[str, CellRecord] = {}
    for row_idx in range(len(df)):
        for column in df.columns:
            value = df.iloc[row_idx][column]
            normalized = _normalize_signature(value)
            value_key = f"{column}::{normalized}"
            cell_key = f"{row_idx}::{column}"
            registry[cell_key] = CellRecord(
                key=cell_key,
                row_index=row_idx,
                column=column,
                raw_value=value,
                value_key=value_key,
                archetype=_column_archetype(metadata.get(column, {})),
            )
    return registry


def _infer_deterministic_surface_repairs(
    cell_registry: Dict[str, CellRecord],
) -> Tuple[Dict[str, List[RepairCandidate]], Dict[str, int]]:
    """Generate query-only repairs for observable structural transformations.

    These contracts use no column names or dataset vocabulary. They are kept
    out of inference until adaptive mechanism validation observes an active
    label, including corrected-value agreement for the proposed repair.
    """
    candidates: Dict[str, List[RepairCandidate]] = defaultdict(list)
    mechanism_counts: Counter = Counter()

    def add_candidate(
        record: CellRecord,
        mechanism: str,
        candidate_value: str,
        confidence: float,
        rationale: str,
    ) -> None:
        raw_value = record.raw_value
        if candidate_value == raw_value:
            return
        existing = candidates[record.key]
        if any(
            item.mechanism == mechanism
            and item.candidate_value == candidate_value
            for item in existing
        ):
            return
        existing.append(RepairCandidate(
            cell_key=record.key,
            row_index=record.row_index,
            column=record.column,
            mechanism=mechanism,
            candidate_value=candidate_value,
            detection_confidence=float(confidence),
            repairability=0.99,
            ambiguity=0.01,
            changed_fields={record.column: candidate_value},
            supporting_evidence=[rationale],
            metadata={
                "exclude_from_inference": True,
                "discovery_signature": f"repair:{mechanism}",
                "mechanism_hint": mechanism,
                "validation_contract": "active_label_and_corrected_value",
            },
        ))
        mechanism_counts[mechanism] += 1

    for record in cell_registry.values():
        if not isinstance(record.raw_value, str) or not record.raw_value:
            continue
        text = record.raw_value

        terminal_normalized = re.sub(r"([.!?])\1{1,}(\s*)$", r"\1\2", text)
        if terminal_normalized != text:
            add_candidate(
                record,
                "repeated_terminal_punctuation",
                terminal_normalized,
                0.98,
                "A terminal punctuation mark is repeated consecutively.",
            )

        # Repeated list segments such as A,A or A,B,A,B are distinguishable
        # from ordinary repeated words because a stable list delimiter is
        # present and the entire item sequence is periodic.
        for delimiter in (",", ";", "|"):
            if delimiter not in text:
                continue
            raw_items = text.split(delimiter)
            items = [item.strip() for item in raw_items]
            if len(items) < 2 or any(not item for item in items):
                continue
            period = next(
                (
                    width
                    for width in range(1, len(items) // 2 + 1)
                    if len(items) % width == 0
                    and items == items[:width] * (len(items) // width)
                ),
                None,
            )
            if period is None:
                continue
            delimiter_match = re.search(re.escape(delimiter) + r"\s*", text)
            joiner = delimiter_match.group(0) if delimiter_match else delimiter
            add_candidate(
                record,
                "repeated_list_segment",
                joiner.join(items[:period]),
                0.99,
                "The complete delimited item sequence repeats exactly.",
            )
            break

        # One doubled space can be intentional. Two or more independent runs
        # are a stronger anomaly; propose each one-run correction separately
        # and let corrected-value validation select the valid repair.
        whitespace_runs = list(re.finditer(r"[ \t]{2,}", text))
        if len(whitespace_runs) >= 2:
            for match in whitespace_runs:
                normalized = (
                    text[:match.start()] + " " + text[match.end():]
                )
                add_candidate(
                    record,
                    "excess_internal_whitespace",
                    normalized,
                    0.96,
                    "The value contains multiple independent runs of internal whitespace.",
                )

    return dict(candidates), dict(sorted(mechanism_counts.items()))


def _add_canonicalization_evidence(
    repair_candidates: Dict[str, List[RepairCandidate]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> None:
    """Translate auditable repair candidates into calibratable dirty evidence."""
    for cell_key, candidates in repair_candidates.items():
        for candidate in candidates:
            detection_strength = float(np.clip(
                candidate.detection_confidence * (1.0 - 0.10 * candidate.ambiguity),
                0.0,
                1.0,
            ))
            candidate_metadata = {
                # Canonicalization plans are column-specific. Keep the
                # evidence feature scoped to the profiled column so a label
                # for one field cannot recalibrate an unrelated field that
                # happens to use the same operation syntax.
                "feature_namespace": f"column:{candidate.column}",
                "column": candidate.column,
                "candidate_value": candidate.candidate_value,
                "repairability": candidate.repairability,
                "ambiguity": candidate.ambiguity,
                "changed_fields": candidate.changed_fields,
                "supporting_evidence": candidate.supporting_evidence,
                **dict(candidate.metadata or {}),
            }
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id=candidate.mechanism,
                family="canonicalization",
                polarity="dirty",
                strength=detection_strength,
                hard=False,
                reason_code=f"canonicalization_{candidate.mechanism}",
                metadata=candidate_metadata,
            ))


@lru_cache(maxsize=1)
def _load_optional_english_lexicon() -> frozenset:
    """Load a host word list when available; absence means lexical abstention."""
    candidates = [
        os.environ.get("CLEANEM_WORDLIST", ""),
        "/usr/share/dict/words",
        "/usr/share/dict/american-english",
        "/usr/share/dict/british-english",
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                words = {
                    line.strip().casefold()
                    for line in handle
                    if re.fullmatch(r"[A-Za-z]{2,32}", line.strip())
                }
            if words:
                return frozenset(words)
        except OSError:
            continue
    return frozenset()


def _infer_closed_enum_normalizations(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    lexicon: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Find conservative lexical conflicts inside closed vocabularies.

    The dirty table alone cannot generally determine the direction of two
    competing surfaces. Direction is admitted only for an exact reversal, a
    strongly supported ASCII target for Unicode contamination, or a short edit
    from a non-word to an observed dictionary word. The function otherwise
    abstains, including when no optional host lexicon is available.
    """
    dictionary = set(_load_optional_english_lexicon() if lexicon is None else lexicon)
    proposals: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for column in df.columns:
        if _column_archetype(metadata.get(column, {})) != "closed_enum":
            continue
        counts = Counter(
            str(value).strip().casefold()
            for value in df[column].tolist()
            if not _is_missing_like(value, metadata.get(column, {}))
            and str(value).strip()
        )
        targets = {
            value: count
            for value, count in counts.items()
            if count >= 3 and 2 <= len(value) <= 32
        }
        total_support = sum(counts.values())
        short_code_support = sum(
            count
            for value, count in counts.items()
            if re.fullmatch(r"[a-z]{2,3}", value)
        )
        code_like_column = (
            total_support > 0
            and short_code_support / total_support >= 0.70
        )
        column_proposals: Dict[str, Dict[str, Any]] = {}
        for source, source_count in counts.items():
            if not source or len(source) > 32:
                continue
            best: Optional[Tuple[Tuple[int, int, int], str, str]] = None
            source_ascii_word = bool(re.fullmatch(r"[a-z]+", source))
            for target, target_count in targets.items():
                if source == target:
                    continue
                distance = _levenshtein_distance(source, target)
                short_edit = distance <= 1 or (
                    min(len(source), len(target)) >= 8 and distance <= 2
                )
                reverse = (
                    not code_like_column
                    and
                    source[::-1] == target
                    and target_count >= max(3, 2 * source_count)
                )
                unicode_to_dominant = (
                    not source.isascii()
                    and bool(re.fullmatch(r"[a-z]+", target))
                    and short_edit
                    and target_count >= max(5, 2 * source_count)
                )
                lexical_typo = (
                    bool(dictionary)
                    and source_ascii_word
                    and bool(re.fullmatch(r"[a-z]+", target))
                    and min(len(source), len(target)) >= 4
                    and source not in dictionary
                    and target in dictionary
                    and short_edit
                    and target_count >= 5
                )
                if not (reverse or unicode_to_dominant or lexical_typo):
                    continue
                mode = (
                    "reversed_surface"
                    if reverse else
                    "unicode_contamination"
                    if unicode_to_dominant else
                    "dictionary_edit"
                )
                priority = 3 if reverse else 2 if unicode_to_dominant else 1
                rank = (priority, -distance, target_count)
                if best is None or rank > best[0]:
                    best = (rank, target, mode)
            if best is not None:
                _, target, mode = best
                column_proposals[source] = {
                    "candidate_value": target,
                    "mode": mode,
                    "source_support": int(source_count),
                    "target_support": int(counts[target]),
                    "edit_distance": int(_levenshtein_distance(source, target)),
                }
        if column_proposals:
            proposals[column] = column_proposals
    return proposals


def _add_closed_enum_normalization_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
    value_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, Any]:
    proposals = _infer_closed_enum_normalizations(df, metadata)
    emitted_values = 0
    emitted_cells = 0
    modes = Counter()
    for value_key, record in value_registry.items():
        source = str(record.representative_value).strip().casefold()
        proposal = proposals.get(record.column, {}).get(source)
        if proposal is None:
            continue
        _record_observation(value_observations, EvidenceObservation(
            target_scope="value",
            target_key=value_key,
            source_id="closed_enum_normalization",
            family="canonicalization",
            polarity="dirty",
            strength=1.0,
            hard=True,
            reason_code=f"closed_enum_{proposal['mode']}",
            metadata=dict(proposal),
        ))
        emitted_values += 1
        emitted_cells += int(record.support_count)
        modes[str(proposal["mode"])] += int(record.support_count)
    return {
        "columns": len(proposals),
        "value_groups": emitted_values,
        "cells": emitted_cells,
        "modes": dict(modes),
    }


def _extract_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    rule_pool: Dict[str, List[Dict[str, Any]]],
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    repair_candidates: Optional[Dict[str, List[RepairCandidate]]] = None,
    validated_relationship_rules: Optional[List[Dict[str, Any]]] = None,
    enable_canonicalization_evidence: bool = True,
    enable_closed_enum_normalization: bool = True,
    enable_specialized_mechanisms: bool = False,
    enable_evidence_gating: bool = True,
    defer_evidence_gating: bool = False,
    enable_multisource_temporal_consensus: bool = True,
    multisource_min_sources: int = MULTISOURCE_MIN_DISTINCT_SOURCES,
    multisource_min_ratio: float = MULTISOURCE_MIN_DOMINANT_WEIGHT_RATIO,
    multisource_time_tolerance: float = MULTISOURCE_TIME_TOLERANCE_MINUTES,
    multisource_min_source_reliability: float = MULTISOURCE_MIN_SOURCE_RELIABILITY,
) -> Tuple[Dict[str, List[EvidenceObservation]], Dict[str, List[EvidenceObservation]]]:
    value_observations: Dict[str, List[EvidenceObservation]] = {}
    cell_observations: Dict[str, List[EvidenceObservation]] = {}
    percent_decimal_contracts = (
        {
            column: contract
            for column in df.columns
            if (
                contract := _infer_percent_decimal_contract(
                    df[column], metadata.get(column, {})
                )
            ) is not None
        }
        if enable_specialized_mechanisms
        else {}
    )
    unit_measure_contracts = (
        {
            column: contract
            for column in df.columns
            if (
                contract := _infer_unit_measure_contract(
                    df[column].tolist(), metadata.get(column, {})
                )
            ) is not None
        }
        if enable_specialized_mechanisms
        else {}
    )
    value_records_by_column: Dict[str, List[ValueRecord]] = defaultdict(list)
    for value_record in value_registry.values():
        value_records_by_column[value_record.column].append(value_record)
    repeated_character_contracts = (
        {
            column: _infer_repeated_character_contracts(
                records,
                metadata.get(column, {}),
            )
            for column, records in value_records_by_column.items()
        }
        if enable_specialized_mechanisms
        else {}
    )
    repeated_character_summary = {
        column: len(contracts)
        for column, contracts in repeated_character_contracts.items()
        if contracts
    }
    cleanem_logger.info(
        "repeated_character_discovery: "
        f"columns={len(repeated_character_summary)}, "
        f"candidate_values={sum(repeated_character_summary.values())}, "
        f"by_column={repeated_character_summary}"
    )
    outputs_by_column: Dict[str, np.ndarray] = {}
    family_indices_by_column: Dict[str, Dict[str, List[int]]] = {}
    for column in df.columns:
        rules = rule_pool.get(column, [])
        outputs = _compute_rule_outputs_for_column(df, column, rules) if rules else np.zeros((len(df), 0), dtype=int)
        outputs_by_column[column] = outputs
        family_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, rule in enumerate(rules):
            family_indices[rule.get("family", "")].append(idx)
        family_indices_by_column[column] = dict(family_indices)

    for value_key, record in value_registry.items():
        col_meta = metadata.get(record.column, {})
        value = record.representative_value
        normalized = record.normalized_value
        missing_like = _is_missing_like(value, col_meta)
        closed_set_legal_values = _get_closed_set_legal_values(col_meta) if enable_specialized_mechanisms else set()
        representative_row_idx = record.row_indices[0]
        regex_match = _matches_regex_candidates(value, col_meta.get("regex_candidates") or [])
        regex_match_rate = _matching_regex_rate(value, col_meta.get("regex_candidates") or [])
        pattern_ratio = _row_family_pass_ratio(
            outputs_by_column.get(record.column, np.zeros((0, 0), dtype=int)),
            family_indices_by_column.get(record.column, {}),
            "pattern",
            representative_row_idx,
        )

        if missing_like:
            explicit_missing_marker = _is_explicit_missing_marker(value)
            structural_missing = _is_structural_missing_group(
                record.support_count, len(df)
            ) and not explicit_missing_marker
            missing_reason = (
                "explicit_missing_marker"
                if explicit_missing_marker
                else "structural_missing_token"
                if structural_missing else "missing_token"
            )
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="missing_token",
                family="missing",
                polarity="dirty",
                strength=0.15 if structural_missing else 1.0,
                hard=not structural_missing,
                reason_code=missing_reason,
                metadata={
                    "explicit_missing_marker": explicit_missing_marker,
                    "structural_missing": structural_missing,
                    "support": int(record.support_count),
                    "column_rate": float(record.support_count / max(len(df), 1)),
                },
            ))
            continue

        surface_conflict = _surface_role_conflict(value, col_meta)
        if surface_conflict is not None:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id=surface_conflict["source_id"],
                family="outlier",
                polarity="dirty",
                strength=float(surface_conflict["strength"]),
                hard=bool(surface_conflict["hard"]),
                reason_code=surface_conflict["reason_code"],
                metadata=dict(surface_conflict["metadata"]),
            ))

        temporal_conflict = _temporal_range_conflict(value, col_meta) if enable_specialized_mechanisms else None
        if temporal_conflict is not None:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id=temporal_conflict["source_id"],
                family="outlier",
                polarity="dirty",
                strength=float(temporal_conflict["strength"]),
                hard=bool(temporal_conflict["hard"]),
                reason_code=temporal_conflict["reason_code"],
                metadata=dict(temporal_conflict["metadata"]),
            ))

        if closed_set_legal_values and normalized not in closed_set_legal_values:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="closed_set_invalid",
                family="outlier",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="closed_set_invalid",
            ))
            continue

        # Profile-derived enum contamination: a rare near-neighbor of a
        # highly concentrated mode is a strong candidate for a corrupted
        # category, even when no explicit vocabulary was supplied. Keep the
        # mechanism generic and auditable; strict profiles may promote it to
        # hard evidence while ambiguous profiles remain calibratable.
        enum_contamination = (
            _infer_closed_enum_contamination(value, record, col_meta)
            if enable_specialized_mechanisms
            else None
        )
        if enum_contamination is not None:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="closed_enum_contamination",
                family="canonicalization",
                polarity="dirty",
                strength=float(enum_contamination["confidence"]),
                hard=bool(enum_contamination.get("hard", False)),
                reason_code="closed_enum_near_neighbor_contamination",
                metadata=dict(enum_contamination),
            ))

        repeated_character = repeated_character_contracts.get(record.column, {}).get(
            normalized
        )
        if repeated_character is not None:
            repeated_metadata = dict(repeated_character)
            repeated_metadata.update({
                # Discovery identity is stable across columns and datasets;
                # the archetype namespace is added by the discovery module.
                "exclude_from_inference": True,
                "exclude_from_query_design": False,
            })
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="repeated_character_corruption",
                family="canonicalization",
                polarity="dirty",
                strength=float(repeated_character["confidence"]),
                hard=False,
                reason_code="repeated_character_corruption",
                metadata=repeated_metadata,
            ))

        unit_anomaly = _infer_unit_measure_anomaly(
            value,
            record,
            unit_measure_contracts.get(record.column),
            col_meta,
        )
        if unit_anomaly is not None:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="unit_contract_violation",
                family="pattern",
                polarity="dirty",
                strength=float(unit_anomaly["confidence"]),
                # Keep value-level priors stable; cell-level hard evidence
                # below localizes the contract violation to affected rows.
                hard=False,
                reason_code=str(unit_anomaly["reason"]),
                metadata=dict(unit_anomaly),
            ))

        if col_meta.get("type") == "numeric":
            numeric = _coerce_numeric_with_metadata(value, col_meta)
            full_string_numeric = _is_full_string_numeric(value)
            hard_numeric_failure = _has_trusted_numeric_contract(col_meta)
            if numeric is None:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="parse_failure",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_parse_failure",
                ))
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="type_conflict",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_type_conflict",
                ))
                continue

            if regex_match_rate > 0:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="regex_pass",
                    family="pattern",
                    polarity="clean",
                    strength=min(1.0, max(0.6, regex_match_rate)),
                    hard=False,
                    reason_code="regex_pass",
                ))
            if full_string_numeric:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="parser_success",
                    family="outlier",
                    polarity="clean",
                    strength=1.0,
                    hard=False,
                    reason_code="numeric_full_string_parse_success",
                    metadata={"parse_mode": "full_string"},
                ))
            archetype = _column_archetype(col_meta)
            rarity_strength = (
                0.0
                if archetype in {"identifier", "temporal_measure"}
                else _numeric_rarity_strength(numeric, col_meta)
            )
            if rarity_strength > 0:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="rarity_high",
                    family="outlier",
                    polarity="dirty",
                    strength=rarity_strength,
                    hard=False,
                    reason_code="numeric_rarity_high",
                ))
            continue

        if regex_match:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="regex_pass",
                family="pattern",
                polarity="clean",
                strength=1.0,
                hard=False,
                reason_code="regex_pass",
            ))
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="parser_success",
                family="pattern",
                polarity="clean",
                strength=0.8,
                hard=False,
                reason_code="pattern_parser_success",
            ))
        elif pattern_ratio is not None and pattern_ratio >= 0.75:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="parser_success",
                family="pattern",
                polarity="clean",
                strength=pattern_ratio,
                hard=False,
                reason_code="pattern_rule_support",
            ))

        low_frequency_strength = (
            _low_frequency_strength(record, col_meta)
            if _allow_discrete_rarity_evidence(col_meta)
            else 0.0
        )
        if low_frequency_strength > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="rarity_high",
                family="pattern",
                polarity="dirty",
                strength=low_frequency_strength,
                hard=False,
                reason_code="low_frequency_value",
            ))

        text = str(value).strip()
        prototype_close, prototype_far = _prototype_distance_evidence(text, col_meta)
        if prototype_close > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="prototype_close",
                family="prototype_lexical",
                polarity="clean",
                strength=prototype_close,
                hard=False,
                reason_code="prototype_close",
            ))
        if prototype_far > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="prototype_far",
                family="prototype_lexical",
                polarity="dirty",
                strength=prototype_far,
                hard=False,
                reason_code="prototype_far",
            ))

    for cell_key, record in cell_registry.items():
        col_meta = metadata.get(record.column, {})
        value = record.raw_value
        normalized = _normalize_signature(value)
        row = df.iloc[record.row_index]
        missing_like = _is_missing_like(value, col_meta)
        closed_set_legal_values = _get_closed_set_legal_values(col_meta) if enable_specialized_mechanisms else set()
        regex_match = _matches_regex_candidates(value, col_meta.get("regex_candidates") or [])
        regex_match_rate = _matching_regex_rate(value, col_meta.get("regex_candidates") or [])
        pattern_ratio = _row_family_pass_ratio(
            outputs_by_column.get(record.column, np.zeros((0, 0), dtype=int)),
            family_indices_by_column.get(record.column, {}),
            "pattern",
            record.row_index,
        )
        pattern_rule_provenance = _agent_rule_provenance(
            rule_pool.get(record.column, []),
            outputs_by_column.get(record.column, np.zeros((0, 0), dtype=int)),
            "pattern",
            record.row_index,
        )

        emitted_hard_dirty = False
        parse_success = False
        principal_parser_success = False
        type_conflict = False
        closed_set_valid = True
        relationship_applicable = 0
        strong_relationship_violation = 0
        weak_relationship_violation = 0
        relationship_satisfied = 0
        violated_relationships: List[Dict[str, Any]] = []
        satisfied_relationships: List[Dict[str, Any]] = []

        if missing_like:
            explicit_missing_marker = _is_explicit_missing_marker(value)
            value_record = value_registry.get(record.value_key)
            missing_support = (
                int(value_record.support_count) if value_record is not None else 1
            )
            structural_missing = _is_structural_missing_group(
                missing_support, len(df)
            ) and not explicit_missing_marker
            missing_reason = (
                "explicit_missing_marker"
                if explicit_missing_marker
                else "structural_missing_token"
                if structural_missing else "missing_token"
            )
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="missing_token",
                family="missing",
                polarity="dirty",
                strength=0.15 if structural_missing else 1.0,
                hard=not structural_missing,
                reason_code=missing_reason,
                metadata={
                    "explicit_missing_marker": explicit_missing_marker,
                    "structural_missing": structural_missing,
                    "support": missing_support,
                    "column_rate": float(missing_support / max(len(df), 1)),
                },
            ))
            continue

        surface_conflict = _surface_role_conflict(value, col_meta)
        if surface_conflict is not None:
            emitted_hard_dirty = bool(
                emitted_hard_dirty or surface_conflict["hard"]
            )
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id=surface_conflict["source_id"],
                family="outlier",
                polarity="dirty",
                strength=float(surface_conflict["strength"]),
                hard=bool(surface_conflict["hard"]),
                reason_code=surface_conflict["reason_code"],
                metadata=dict(surface_conflict["metadata"]),
            ))

        temporal_conflict = _temporal_range_conflict(value, col_meta) if enable_specialized_mechanisms else None
        if temporal_conflict is not None:
            emitted_hard_dirty = bool(
                emitted_hard_dirty or temporal_conflict["hard"]
            )
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id=temporal_conflict["source_id"],
                family="outlier",
                polarity="dirty",
                strength=float(temporal_conflict["strength"]),
                hard=bool(temporal_conflict["hard"]),
                reason_code=temporal_conflict["reason_code"],
                metadata=dict(temporal_conflict["metadata"]),
            ))

        unicode_corruption = _unicode_corruption_kind(value) if enable_specialized_mechanisms else None
        if unicode_corruption is not None:
            explicit_replacement = unicode_corruption == "replacement_character"
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id=f"unicode_{unicode_corruption}",
                family="pattern",
                polarity="dirty",
                strength=1.0 if explicit_replacement else 0.85,
                # U+FFFD is an explicit decoder failure, unlike a merely
                # unusual combining mark; it is safe to decide directly.
                hard=explicit_replacement,
                reason_code=f"unicode_{unicode_corruption}",
                metadata={
                    # Mixed-script detection is useful evidence, but its
                    # high-volume clusters are poor calibration representatives.
                    # Keep it in inference while excluding it from query design;
                    # the flag is generic and applies to any column archetype.
                    "exclude_from_query_design": (
                        unicode_corruption == "non_latin_mixed_script"
                    ),
                },
            ))

        if enable_specialized_mechanisms and _has_pagination_contamination(record.column, value, col_meta):
            emitted_hard_dirty = True
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="pagination_contamination",
                family="pattern",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="pagination_suffix_or_delimiter_contamination",
            ))

        percent_contract = percent_decimal_contracts.get(record.column) if enable_specialized_mechanisms else None
        if (
            percent_contract is not None
            and re.fullmatch(
                r"\s*[-+]?\d+(?:\.\d+)?\s*%\s*", str(value)
            )
        ):
            emitted_hard_dirty = True
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="percent_decimal_mismatch",
                family="pattern",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="percent_decimal_mismatch",
                metadata=dict(percent_contract),
            ))

        if closed_set_legal_values:
            closed_set_valid = normalized in closed_set_legal_values
            if not closed_set_valid:
                emitted_hard_dirty = True
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="closed_set_invalid",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=True,
                    reason_code="closed_set_invalid",
                ))

        # Mirror the value-level mechanism at cell scope so the final cell
        enum_contamination = (
            _infer_closed_enum_contamination(value, value_record, col_meta)
            if enable_specialized_mechanisms
            else None
        )
        if enum_contamination is not None:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="closed_enum_contamination",
                family="canonicalization",
                polarity="dirty",
                strength=float(enum_contamination["confidence"]),
                hard=bool(enum_contamination.get("hard", False)),
                reason_code="closed_enum_near_neighbor_contamination",
                metadata=dict(enum_contamination),
            ))

        repeated_character = repeated_character_contracts.get(record.column, {}).get(
            normalized
        )
        if repeated_character is not None:
            repeated_metadata = dict(repeated_character)
            repeated_metadata.update({
                "exclude_from_inference": True,
                "exclude_from_query_design": False,
            })
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="repeated_character_corruption",
                family="canonicalization",
                polarity="dirty",
                strength=float(repeated_character["confidence"]),
                hard=False,
                reason_code="repeated_character_corruption",
                metadata=repeated_metadata,
            ))

        unit_anomaly = _infer_unit_measure_anomaly(
            value,
            value_record,
            unit_measure_contracts.get(record.column),
            col_meta,
        )
        if unit_anomaly is not None:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="unit_contract_violation",
                family="pattern",
                polarity="dirty",
                strength=float(unit_anomaly["confidence"]),
                hard=bool(unit_anomaly.get("hard", False)),
                reason_code=str(unit_anomaly["reason"]),
                metadata=dict(unit_anomaly),
            ))

        if col_meta.get("type") == "numeric":
            numeric = _coerce_numeric_with_metadata(value, col_meta)
            full_string_numeric = _is_full_string_numeric(value)
            hard_numeric_failure = _has_trusted_numeric_contract(col_meta)
            if numeric is None:
                emitted_hard_dirty = hard_numeric_failure
                type_conflict = True
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="cell_parse_failure",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_parse_failure",
                ))
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="cell_type_conflict",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_type_conflict",
                ))
            else:
                parse_success = full_string_numeric
                principal_parser_success = full_string_numeric
                if regex_match_rate > 0:
                    _record_observation(cell_observations, EvidenceObservation(
                        target_scope="cell",
                        target_key=cell_key,
                        source_id="pattern_match",
                        family="pattern",
                        polarity="clean",
                        strength=min(1.0, max(0.6, regex_match_rate)),
                        hard=False,
                        reason_code="regex_match",
                        metadata=pattern_rule_provenance,
                    ))
                elif _best_regex_contract_rate(col_meta) >= 0.60:
                    _record_observation(cell_observations, EvidenceObservation(
                        target_scope="cell",
                        target_key=cell_key,
                        source_id="pattern_mismatch",
                        family="pattern",
                        polarity="dirty",
                        strength=0.6,
                        hard=False,
                        reason_code="format_contract_mismatch",
                        metadata=pattern_rule_provenance,
                    ))
        else:
            text = str(value).strip()
            grammar_distance = _grammar_distance_score(text, col_meta)
            shape_distance = _shape_distance_score(text, col_meta)
            mismatch_candidates = [grammar_distance, shape_distance]
            if pattern_ratio is not None:
                mismatch_candidates.append(max(0.0, 1.0 - pattern_ratio))
            mismatch_strength = max(mismatch_candidates) if mismatch_candidates else 0.0
            parse_success = regex_match or (pattern_ratio is not None and pattern_ratio >= 0.75)

            if regex_match:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_match",
                    family="pattern",
                    polarity="clean",
                    strength=1.0,
                    hard=False,
                    reason_code="regex_match",
                ))
            elif pattern_ratio is not None and pattern_ratio >= 0.75:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_match",
                    family="pattern",
                    polarity="clean",
                    strength=pattern_ratio,
                    hard=False,
                    reason_code="pattern_rule_match",
                    metadata=pattern_rule_provenance,
                ))

            if not regex_match and mismatch_strength >= 0.55:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_mismatch",
                    family="pattern",
                    polarity="dirty",
                    strength=mismatch_strength,
                    hard=False,
                    reason_code="pattern_mismatch",
                    metadata=pattern_rule_provenance,
                ))

        for profile in col_meta.get("relationship_profiles") or []:
            other_column = profile.get("other_column")
            if not other_column or other_column not in df.columns:
                continue
            other_value = row[other_column]
            if _is_missing_like(value, col_meta) or _is_missing_like(other_value, metadata.get(other_column, {})):
                continue
            if str(value).strip() == "" or str(other_value).strip() == "":
                continue
            relationship_applicable += 1
            is_valid = _relationship_predicate(value, other_value, profile.get("type", ""))
            relation_detail = {
                "relation_type": str(profile.get("type") or "relationship"),
                "determinant_column": other_column,
                "dependent_column": record.column,
                "left_column": other_column,
                "right_column": record.column,
                "related_columns": [other_column, record.column],
                "determinant_value": other_value,
                "observed_value": value,
                "description": str(
                    profile.get("description")
                    or (
                        f"'{record.column}' must satisfy "
                        f"{profile.get('type', 'the relation')} with "
                        f"'{other_column}'"
                    )
                ),
                "rule_text": str(
                    profile.get("description")
                    or (
                        f"'{record.column}' must satisfy "
                        f"{profile.get('type', 'the relation')} with "
                        f"'{other_column}'"
                    )
                ),
                "applicable_count": int(profile.get("applicable_count", 0)),
                "violation_rate": float(profile.get("violation_rate", 1.0)),
                "rule_confidence": float(
                    np.clip(
                        1.0 - float(profile.get("violation_rate", 1.0)),
                        0.0,
                        1.0,
                    )
                ),
            }
            if is_valid:
                relationship_satisfied += 1
                satisfied_relationships.append(relation_detail)
            else:
                violated_relationships.append(relation_detail)
                if float(profile.get("violation_rate", 1.0)) <= 0.10:
                    strong_relationship_violation += 1
                else:
                    weak_relationship_violation += 1

        violated_relationships.sort(
            key=lambda item: (
                float(item.get("violation_rate", 1.0)),
                str(item.get("determinant_column", "")),
            )
        )
        satisfied_relationships.sort(
            key=lambda item: (
                float(item.get("violation_rate", 1.0)),
                str(item.get("determinant_column", "")),
            )
        )
        primary_violated_relationship = (
            dict(violated_relationships[0]) if violated_relationships else {}
        )
        primary_satisfied_relationship = (
            dict(satisfied_relationships[0]) if satisfied_relationships else {}
        )
        if primary_violated_relationship:
            primary_violated_relationship["violated_relationships"] = (
                violated_relationships
            )
        if primary_satisfied_relationship:
            primary_satisfied_relationship["satisfied_relationships"] = (
                satisfied_relationships
            )

        if strong_relationship_violation > 0:
            emitted_hard_dirty = True
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="strong_relationship_violation",
                family="relationship",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="strong_relationship_violation",
                metadata=primary_violated_relationship,
            ))
        elif weak_relationship_violation > 0:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="weak_relationship_violation",
                family="relationship",
                polarity="dirty",
                strength=min(1.0, 0.35 + weak_relationship_violation / max(relationship_applicable, 1)),
                hard=False,
                reason_code="weak_relationship_violation",
                metadata=primary_violated_relationship,
            ))
        elif relationship_applicable > 0:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="relationship_satisfied",
                family="relationship",
                polarity="clean",
                strength=min(1.0, 0.4 + relationship_satisfied / max(relationship_applicable, 1)),
                hard=False,
                reason_code="relationship_satisfied",
                metadata=primary_satisfied_relationship,
            ))

        if relationship_applicable >= 2:
            if relationship_satisfied == relationship_applicable:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="local_consistency_pass",
                    family="relationship",
                    polarity="clean",
                    strength=0.75,
                    hard=False,
                    reason_code="local_consistency_pass",
                    metadata={
                        **primary_satisfied_relationship,
                        "relationship_applicable": relationship_applicable,
                        "relationship_satisfied": relationship_satisfied,
                    },
                ))
            elif (strong_relationship_violation + weak_relationship_violation) / max(relationship_applicable, 1) >= 0.5:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="local_consistency_fail",
                    family="relationship",
                    polarity="dirty",
                    strength=0.75,
                    hard=False,
                    reason_code="local_consistency_fail",
                    metadata={
                        **primary_violated_relationship,
                        "relationship_applicable": relationship_applicable,
                        "relationship_satisfied": relationship_satisfied,
                        "strong_violation_count": strong_relationship_violation,
                        "weak_violation_count": weak_relationship_violation,
                    },
                ))

        if (
            not emitted_hard_dirty
            and principal_parser_success
            and not type_conflict
            and closed_set_valid
            and (relationship_applicable == 0 or strong_relationship_violation == 0 and weak_relationship_violation == 0)
        ):
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="cell_parser_success",
                family="outlier",
                polarity="clean",
                strength=1.0,
                hard=False,
                reason_code="parser_contract_satisfied",
            ))

    enum_normalization_summary = (
        _add_closed_enum_normalization_evidence(
            df,
            metadata,
            value_registry,
            value_observations,
        )
        if enable_closed_enum_normalization and enable_specialized_mechanisms
        else {"disabled": True, "columns": 0, "value_groups": 0, "cells": 0}
    )
    cleanem_logger.info(
        "closed_enum_normalization: "
        f"{json.dumps(enum_normalization_summary, ensure_ascii=False)}"
    )

    if enable_canonicalization_evidence:
        if repair_candidates:
            _add_canonicalization_evidence(repair_candidates, cell_observations)
        cleanem_logger.info(
            "canonicalization_evidence: enabled, "
            f"candidate_cells={len(repair_candidates or {})}, "
            f"candidates={sum(len(items) for items in (repair_candidates or {}).values())}"
        )
    else:
        cleanem_logger.info("canonicalization_evidence: disabled")

    _add_contextual_consensus_evidence(
        df,
        metadata,
        cell_observations,
        validated_relationship_rules=validated_relationship_rules,
    )

    if enable_specialized_mechanisms:
        geo_suffix_summary = _geo_suffix_shift_observations(
            df, metadata, cell_registry, cell_observations
        )
        cleanem_logger.info(
            "specialized_relationship_detectors: "
            f"{json.dumps(geo_suffix_summary, ensure_ascii=False, default=str)}"
        )
        representation_summary = _add_representation_role_conflict_evidence(
            df, metadata, cell_registry, cell_observations
        )
        cleanem_logger.info(
            "specialized_representation_detector: "
            f"{json.dumps(representation_summary, ensure_ascii=False, default=str)}"
        )
    else:
        cleanem_logger.info("specialized_relationship_detectors: disabled")

    if enable_multisource_temporal_consensus and enable_specialized_mechanisms:
        multisource_summary = _add_multisource_temporal_consensus_evidence(
            df,
            metadata,
            cell_observations,
            min_sources=multisource_min_sources,
            min_ratio=multisource_min_ratio,
            tolerance=multisource_time_tolerance,
            min_source_reliability=multisource_min_source_reliability,
        )
        cleanem_logger.info(
            "multisource_temporal_consensus: "
            f"{json.dumps(multisource_summary, ensure_ascii=False, default=str)}"
        )
    else:
        cleanem_logger.info("multisource_temporal_consensus: disabled")

    if enable_evidence_gating:
        gating_summary = _apply_evidence_gates(
            value_observations,
            cell_observations,
            metadata,
            value_registry,
            cell_registry,
        )
        cleanem_logger.info(f"evidence_gating_counts: {gating_summary['counts']}")
        for item in gating_summary["policy_hits"]:
            cleanem_logger.info(
                "  - Evidence gate: "
                f"archetype={item['archetype']}, source={item['source_id']}, "
                f"scale={item['scale']:.2f}, count={item['count']}"
            )
    elif defer_evidence_gating:
        cleanem_logger.info("evidence_gating: deferred_until_active_labels")
    else:
        cleanem_logger.info("evidence_gating: disabled")

    return value_observations, cell_observations


_ADAPTIVE_SPECIALIZED_SOURCE_IDS = frozenset({
    "temporal_range_violation",
    "percent_decimal_mismatch",
    "unicode_replacement_character",
    "unicode_unresolved_combining_mark",
    "unicode_non_latin_mixed_script",
    "pagination_contamination",
    "closed_set_invalid",
    "closed_enum_contamination",
    "closed_enum_normalization",
    "repeated_character_corruption",
    "unit_contract_violation",
    "geo_suffix_pollution",
    "displaced_region_missing",
    "representation_role_conflict",
    "multisource_temporal_disagreement",
    "strong_relationship_violation",
})


def _defer_specialized_mechanism_evidence(
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, int]:
    """Turn specialized detector output into query-only mechanism candidates.

    Detector code remains deterministic and auditable, but its output cannot
    affect inference until the adaptive mechanism validator enables the
    corresponding candidate. Explicit missingness, parser failures, and
    generic pattern evidence remain available as base evidence.
    """
    deferred = Counter()
    for observation_map in (value_observations, cell_observations):
        for observations in observation_map.values():
            for observation in observations:
                source_id = str(observation.source_id or "")
                reason_code = str(observation.reason_code or "")
                source_match = source_id in _ADAPTIVE_SPECIALIZED_SOURCE_IDS
                reason_match = any(
                    token in reason_code
                    for token in (
                        "temporal_range",
                        "percent_decimal",
                        "unicode_",
                        "unit_",
                        "geo_suffix",
                        "closed_enum_",
                        "representation_role",
                        "relationship_violation",
                        "pagination_",
                    )
                )
                if not (source_match or reason_match):
                    continue
                observation.metadata = dict(observation.metadata or {})
                observation.metadata.update({
                    "adaptive_candidate": True,
                    "original_hard": bool(observation.hard),
                    "exclude_from_inference": True,
                    "exclude_from_query_design": False,
                    "discovery_signature": "adaptive:{}:{}".format(
                        source_id or "unknown",
                        reason_code or source_id or "unknown",
                    ),
                    "mechanism_hint": "adaptive_{}".format(
                        source_id or reason_code or "observed_pattern",
                    ),
                })
                observation.hard = False
                deferred[source_id or reason_code or "unknown"] += 1
    return dict(sorted(deferred.items()))

def _build_hard_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, Dict[str, HardEvidenceLabel]]:
    hard_labels = {"value": {}, "cell": {}}

    for target_key, observations in value_observations.items():
        dirty_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "dirty"})
        clean_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "clean"})
        if dirty_reasons:
            hard_labels["value"][target_key] = HardEvidenceLabel(
                target_scope="value",
                target_key=target_key,
                label="dirty",
                reasons=dirty_reasons,
            )
        elif clean_reasons:
            hard_labels["value"][target_key] = HardEvidenceLabel(
                target_scope="value",
                target_key=target_key,
                label="clean",
                reasons=clean_reasons,
            )

    for target_key, observations in cell_observations.items():
        dirty_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "dirty"})
        clean_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "clean"})
        if dirty_reasons:
            hard_labels["cell"][target_key] = HardEvidenceLabel(
                target_scope="cell",
                target_key=target_key,
                label="dirty",
                reasons=dirty_reasons,
            )
        elif clean_reasons:
            hard_labels["cell"][target_key] = HardEvidenceLabel(
                target_scope="cell",
                target_key=target_key,
                label="clean",
                reasons=clean_reasons,
            )

    return hard_labels


def _oracle_label_from_clean(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    cell_record: CellRecord,
) -> Optional[int]:
    if cell_record.row_index >= len(clean_df) or cell_record.column not in clean_df.columns:
        return None
    dirty_val = dirty_df.iloc[cell_record.row_index][cell_record.column]
    clean_val = clean_df.iloc[cell_record.row_index][cell_record.column]
    return int(not Judge.ground_truth_values_equal(dirty_val, clean_val))


def _query_active_oracle_labels(
    dirty_df: pd.DataFrame,
    clean_df: Optional[pd.DataFrame],
    cell_registry: Dict[str, CellRecord],
    selected_cell_keys: List[str],
) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    if clean_df is None:
        return {}, []

    labels: Dict[str, int] = {}
    queried_cells: List[Dict[str, object]] = []
    for cell_key in selected_cell_keys:
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        label = _oracle_label_from_clean(dirty_df, clean_df, record)
        if label is None:
            continue
        labels[cell_key] = label
        queried_cells.append(
            {
                "cell_key": cell_key,
                "row_index": int(record.row_index),
                "column": record.column,
                "value": record.raw_value,
                "label": "dirty" if label else "clean",
            }
        )
    return labels, queried_cells


def _build_corrected_value_standardization_candidates(
    dirty_df: pd.DataFrame,
    clean_df: Optional[pd.DataFrame],
    cell_registry: Dict[str, CellRecord],
    selected_cell_keys: List[str],
    active_labels: Dict[str, int],
    detection_observations: Optional[Dict[str, List[EvidenceObservation]]] = None,
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, List[RepairCandidate]], Dict[str, Any]]:
    """Learn standardization mappings and format evidence from corrected labels.

    D-optimal selection happens before this function. Direct annotations are
    retained, propagation requires consistent anchors, and learned contracts
    may contribute evidence only to the second detection pass.
    """
    if clean_df is None or len(clean_df) != len(dirty_df):
        return {}, {"applicable": False, "reason": "corrected_values_unavailable"}

    def semantic_key(value: Any) -> Optional[Tuple[str, Any]]:
        if value is None or _normalize_token(value) in {"", "nan", "none", "null"}:
            return None
        text = str(value).strip()
        numeric_text = text.replace(",", "")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric_text):
            try:
                return ("numeric", round(float(numeric_text), 12))
            except ValueError:
                pass
        temporal = _parse_temporal_minutes(value)
        if temporal is not None:
            return ("temporal", round(float(temporal), 6))
        lexical = re.sub(r"[^a-z0-9]+", "", text.lower())
        return ("lexical", lexical) if lexical else None

    def numeric_unit(value: Any) -> Optional[Tuple[float, str]]:
        if value is None or _normalize_token(value) in {"", "nan", "none", "null"}:
            return None
        match = re.fullmatch(
            r"\s*([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\.?\s*",
            str(value),
        )
        if not match:
            return None
        unit = (match.group(2) or "").lower()
        unit = {"ounce": "oz", "ounces": "oz"}.get(unit, unit)
        return round(float(match.group(1)), 12), unit

    def semantically_equivalent(raw_value: Any, corrected_value: Any) -> bool:
        raw_measure = numeric_unit(raw_value)
        corrected_measure = numeric_unit(corrected_value)
        if raw_measure is not None and corrected_measure is not None:
            raw_number, raw_unit = raw_measure
            corrected_number, corrected_unit = corrected_measure
            return raw_number == corrected_number and (
                raw_unit == corrected_unit or not raw_unit or not corrected_unit
            )
        raw_key = semantic_key(raw_value)
        return raw_key is not None and raw_key == semantic_key(corrected_value)

    annotations: List[Tuple[CellRecord, Any]] = []
    mapping_targets: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)
    mapping_anchor_rows: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    excluded_semantic_corrections = 0
    excluded_semantic_cells: Set[str] = set()
    component_permutation_keys: Set[Tuple[str, str]] = set()

    def presentation_key(value: Any) -> str:
        if value is None or pd.isna(value):
            return "<na>"
        return re.sub(r"\s+", " ", str(value).strip())

    def slash_components(value: Any) -> Optional[Tuple[int, int, int]]:
        if value is None or pd.isna(value):
            return None
        match = re.fullmatch(
            r"\s*(\d{1,2})/(\d{1,2})/(\d{1,2})\s*", str(value)
        )
        return tuple(map(int, match.groups())) if match else None

    def is_labeled_component_permutation(
        record: CellRecord,
        corrected_value: Any,
    ) -> bool:
        semantics = ((metadata or {}).get(record.column, {}).get("semantics") or {})
        if str(semantics.get("archetype") or "") != "temporal_measure":
            return False
        raw_components = slash_components(record.raw_value)
        corrected_components = slash_components(corrected_value)
        return bool(
            raw_components is not None
            and corrected_components is not None
            and raw_components != corrected_components
            and sorted(raw_components) == sorted(corrected_components)
        )

    for cell_key in selected_cell_keys:
        record = cell_registry.get(cell_key)
        if record is None or int(active_labels.get(cell_key, 0)) != 1:
            continue
        corrected_value = clean_df.iloc[record.row_index][record.column]
        component_permutation = is_labeled_component_permutation(
            record, corrected_value
        )
        if not component_permutation and not semantically_equivalent(
            record.raw_value, corrected_value
        ):
            excluded_semantic_corrections += 1
            excluded_semantic_cells.add(cell_key)
            continue
        annotations.append((record, corrected_value))
        mapping_key = (record.column, presentation_key(record.raw_value))
        if component_permutation:
            component_permutation_keys.add(mapping_key)
        corrected_key = presentation_key(corrected_value)
        mapping_targets[mapping_key][corrected_key] = corrected_value
        mapping_anchor_rows[mapping_key].append(int(record.row_index))

    candidates: Dict[str, List[RepairCandidate]] = defaultdict(list)
    directly_annotated: Set[str] = set()
    for record, corrected_value in annotations:
        directly_annotated.add(record.key)
        candidates[record.key].append(RepairCandidate(
            cell_key=record.key,
            row_index=int(record.row_index),
            column=record.column,
            mechanism="corrected_value_direct_standardization",
            candidate_value=corrected_value,
            detection_confidence=1.0,
            repairability=1.0,
            ambiguity=0.0,
            changed_fields={record.column: corrected_value},
            supporting_evidence=["direct corrected value returned by the active annotation"],
        ))

    conflicting_mappings = 0
    propagated = 0
    component_permutation_detection = 0
    for mapping_key, corrected_values in mapping_targets.items():
        if len(corrected_values) != 1:
            conflicting_mappings += 1
            continue
        column, normalized_raw = mapping_key
        corrected_value = next(iter(corrected_values.values()))
        anchor_rows = mapping_anchor_rows[mapping_key]
        confidence = min(0.95, 0.75 + 0.05 * len(anchor_rows))
        for row_index, raw_value in dirty_df[column].items():
            cell_key = f"{int(row_index)}::{column}"
            if presentation_key(raw_value) != normalized_raw:
                continue
            if presentation_key(raw_value) == presentation_key(corrected_value):
                continue
            if (
                detection_observations is not None
                and mapping_key in component_permutation_keys
            ):
                _record_observation(detection_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="labeled_component_permutation",
                    family="corrected_standardization",
                    polarity="dirty",
                    strength=float(confidence),
                    hard=True,
                    reason_code="labeled_exact_component_permutation",
                    metadata={
                        "annotation_anchor_rows": anchor_rows,
                        "corrected_value": corrected_value,
                    },
                ))
                component_permutation_detection += 1
            if cell_key in directly_annotated:
                continue
            candidates[cell_key].append(RepairCandidate(
                cell_key=cell_key,
                row_index=int(row_index),
                column=column,
                mechanism="corrected_value_exact_mapping_standardization",
                candidate_value=corrected_value,
                detection_confidence=float(confidence),
                repairability=float(confidence),
                ambiguity=float(1.0 - confidence),
                changed_fields={column: corrected_value},
                supporting_evidence=[
                    "consistent corrected-value mapping for the exact normalized raw value",
                    f"annotation_anchor_rows={anchor_rows}",
                ],
            ))
            propagated += 1

    # Learn a conservative column-level numeric+unit presentation template
    # from corrected values.  This generalizes formatting corrections across
    # different numeric payloads while preserving the payload itself.
    def strict_measure_format(value: Any) -> Optional[Dict[str, Any]]:
        if value is None or pd.isna(value):
            return None
        match = re.fullmatch(
            r"\s*([-+]?\d+(?:\.(\d+))?)(?:(\s*)([A-Za-z]+)([.]?))?\s*",
            str(value),
        )
        if not match:
            return None
        unit = (match.group(4) or "").lower()
        unit = {"ounce": "oz", "ounces": "oz"}.get(unit, unit)
        return {
            "unit": unit,
            "decimal_places": len(match.group(2) or ""),
            "separator": match.group(3) or "",
            "terminal": match.group(5) or "",
        }

    def loose_measure_payload(value: Any) -> Optional[Tuple[float, str]]:
        if value is None or pd.isna(value):
            return None
        match = re.match(
            r"^\s*([-+]?\d+(?:\.\d+)?)(?:\s*([A-Za-z]+)\b)?",
            str(value),
        )
        if not match:
            return None
        unit = (match.group(2) or "").lower()
        unit = {"ounce": "oz", "ounces": "oz"}.get(unit, unit)
        return float(match.group(1)), unit

    template_votes: Dict[str, Counter] = defaultdict(Counter)
    template_anchor_rows: Dict[Tuple[str, Tuple[Any, ...]], List[int]] = defaultdict(list)
    template_source_units: Dict[Tuple[str, Tuple[Any, ...]], Set[str]] = defaultdict(set)
    for record, corrected_value in annotations:
        parsed = strict_measure_format(corrected_value)
        if parsed is None:
            continue
        signature = (
            parsed["unit"],
            parsed["decimal_places"],
            parsed["separator"],
            parsed["terminal"],
        )
        template_votes[record.column][signature] += 1
        template_anchor_rows[(record.column, signature)].append(int(record.row_index))
        raw_payload = loose_measure_payload(record.raw_value)
        if raw_payload is not None:
            template_source_units[(record.column, signature)].add(raw_payload[1])

    learned_templates = 0
    conflicting_templates = 0
    generalized = 0
    already_covered = set(candidates)
    for column, votes in template_votes.items():
        if len(votes) != 1:
            conflicting_templates += 1
            continue
        signature, support = next(iter(votes.items()))
        unit, decimal_places, separator, terminal = signature
        anchor_rows = template_anchor_rows[(column, signature)]
        allowed_source_units = template_source_units[(column, signature)]
        learned_templates += 1
        confidence = min(0.95, 0.78 + 0.05 * int(support))
        for row_index, raw_value in dirty_df[column].items():
            cell_key = f"{int(row_index)}::{column}"
            if cell_key in excluded_semantic_cells:
                continue
            if cell_key in already_covered:
                continue
            payload = loose_measure_payload(raw_value)
            if payload is None:
                continue
            number, raw_unit = payload
            if raw_unit != unit and raw_unit not in allowed_source_units:
                continue
            rounded = round(number, int(decimal_places))
            if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-12):
                continue
            number_text = f"{rounded:.{int(decimal_places)}f}"
            candidate_value = f"{number_text}{separator}{unit}{terminal}"
            if str(raw_value).strip() == candidate_value:
                continue
            candidates[cell_key].append(RepairCandidate(
                cell_key=cell_key,
                row_index=int(row_index),
                column=column,
                mechanism="corrected_value_format_template_standardization",
                candidate_value=candidate_value,
                detection_confidence=float(confidence),
                repairability=float(confidence),
                ambiguity=float(1.0 - confidence),
                changed_fields={column: candidate_value},
                supporting_evidence=[
                    "column format template learned from corrected-value anchors",
                    f"template=numeric({decimal_places}){separator}{unit}{terminal}",
                    f"annotation_anchor_rows={anchor_rows}",
                    "numeric payload preserved",
                ],
            ))
            already_covered.add(cell_key)
            generalized += 1

    detection_dirty = 0
    detection_clean = 0
    learned_detection_contracts = 0
    conflicting_detection_contracts = 0
    if detection_observations is not None:
        def coarse_format_signature(value: Any) -> Optional[Tuple[Any, ...]]:
            if value is None or pd.isna(value):
                return None
            text = str(value).strip()
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
                return ("numeric_only",)
            match = re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?(\s*)([A-Za-z]+)([.]?)",
                text,
            )
            if not match:
                return None
            return (
                "numeric_unit",
                match.group(2),
                bool(match.group(1)),
                match.group(3),
            )

        contract_votes: Dict[str, Counter] = defaultdict(Counter)
        contract_anchor_rows: Dict[Tuple[str, Tuple[Any, ...]], List[int]] = defaultdict(list)
        for cell_key in selected_cell_keys:
            record = cell_registry.get(cell_key)
            if record is None:
                continue
            label = active_labels.get(cell_key)
            anchor_value: Any = None
            if label == 0:
                anchor_value = record.raw_value
            elif label == 1 and cell_key not in excluded_semantic_cells:
                anchor_value = clean_df.iloc[record.row_index][record.column]
            else:
                continue
            signature = coarse_format_signature(anchor_value)
            if signature is None:
                continue
            contract_votes[record.column][signature] += 1
            contract_anchor_rows[(record.column, signature)].append(int(record.row_index))

        for column, votes in contract_votes.items():
            if len(votes) != 1:
                conflicting_detection_contracts += 1
                continue
            signature, support = next(iter(votes.items()))
            learned_detection_contracts += 1
            confidence = min(0.95, 0.78 + 0.05 * int(support))
            anchor_rows = contract_anchor_rows[(column, signature)]
            for row_index, raw_value in dirty_df[column].items():
                cell_key = f"{int(row_index)}::{column}"
                if cell_key in excluded_semantic_cells:
                    continue
                raw_signature = coarse_format_signature(raw_value)
                if raw_signature == signature:
                    polarity = "clean"
                    strength = 0.90
                    reason_code = "labeled_format_contract_match"
                    detection_clean += 1
                elif re.match(r"^\s*[-+]?\d", str(raw_value)):
                    polarity = "dirty"
                    strength = float(confidence)
                    reason_code = "labeled_format_contract_mismatch"
                    detection_dirty += 1
                else:
                    continue
                _record_observation(detection_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="labeled_format_contract",
                    family="corrected_standardization",
                    polarity=polarity,
                    strength=strength,
                    hard=bool(polarity == "dirty" and int(support) >= 2),
                    reason_code=reason_code,
                    metadata={
                        "annotation_support": int(support),
                        "annotation_anchor_rows": anchor_rows,
                        "format_signature": list(signature),
                    },
                ))
    return dict(candidates), {
        "applicable": bool(annotations) or learned_detection_contracts > 0,
        "dirty_annotations_with_corrected_values": len(annotations),
        "direct_candidates": len(directly_annotated),
        "propagated_candidates": propagated,
        "format_template_candidates": generalized,
        "candidate_cells": len(candidates),
        "consistent_exact_mappings": sum(
            len(values) == 1 for values in mapping_targets.values()
        ),
        "conflicting_exact_mappings_abstained": conflicting_mappings,
        "learned_format_templates": learned_templates,
        "conflicting_format_templates_abstained": conflicting_templates,
        "semantic_corrections_excluded": excluded_semantic_corrections,
        "component_permutation_contracts": len(component_permutation_keys),
        "component_permutation_detection": component_permutation_detection,
        "detection_dirty_evidence": detection_dirty,
        "detection_clean_evidence": detection_clean,
        "learned_detection_contracts": learned_detection_contracts,
        "conflicting_detection_contracts_abstained": conflicting_detection_contracts,
        "used_in_detection": bool(
            detection_observations is not None
            and (
                learned_detection_contracts > 0
                or component_permutation_detection > 0
            )
        ),
    }


def _summarize_observations(observations: List[EvidenceObservation], limit: int = 6) -> str:
    if not observations:
        return ""
    parts = []
    for obs in sorted(observations, key=lambda item: (item.hard, item.strength), reverse=True)[:limit]:
        hard_marker = "hard" if obs.hard else "soft"
        parts.append(f"{obs.source_id}:{obs.polarity}:{obs.strength:.2f}:{hard_marker}")
    return ";".join(parts)


def _summarize_hard_label(label: Optional[HardEvidenceLabel]) -> str:
    if not label:
        return ""
    return ";".join(label.reasons)


def _evidence_source_coverage(
    observation_map: Dict[str, List[EvidenceObservation]],
) -> Dict[str, int]:
    coverage: Dict[str, set] = defaultdict(set)
    for target_key, observations in observation_map.items():
        for obs in observations:
            coverage[obs.source_id].add(target_key)
    return {source_id: len(targets) for source_id, targets in sorted(coverage.items())}


EVIDENCE_REASON_TEMPLATES = {
    "missing": "The value is missing or is a missing-value placeholder",
    "missing_token": "The value matches a missing-value placeholder",
    "structural_missing_token": (
        "The value is a repeated structural missing-value placeholder"
    ),
    "rarity_high": "The value is rare in this column",
    "prototype_far": "The value differs substantially from typical values in this column",
    "prototype_close": "The value is close to typical values in this column",
    "pattern_mismatch": "The value does not match the dominant format of this column",
    "pattern_match": "The value matches the dominant format of this column",
    "regex_pass": "The value satisfies the column's regular-expression constraint",
    "outlier": "The value deviates from the numeric distribution of this column",
    "parser_success": "The entire value matches the expected column type",
    "parse_failure": "The value cannot be parsed as the expected column type",
    "type_conflict": "The value conflicts with the expected column type",
    "type_role_conflict": (
        "The value's surface type conflicts with the learned semantic role of this column"
    ),
    "surface_shape_conflict": (
        "The value's surface shape conflicts with the learned shape contract of this column"
    ),
    "cell_parser_success": "The entire cell matches the expected column type",
    "cell_parse_failure": "The cell cannot be parsed as the expected column type",
    "cell_type_conflict": "The cell conflicts with the expected column type",
    "unicode_replacement_character": "The text contains a Unicode replacement character",
    "unicode_unresolved_combining_mark": "The text contains an unresolved combining mark",
    "unicode_non_latin_mixed_script": (
        "The text contains a small non-Latin script insertion in an otherwise Latin surface"
    ),
    "pagination_contamination": "The page value contains suffix or delimiter contamination",
    "labeled_component_permutation": "The value matches a labeled temporal component permutation",
    "relationship": "The value violates a relation involving this row or related records",
    "relationship_satisfied": "The value satisfies a discovered cross-column relation",
    "strong_relationship_violation": "The value violates a high-confidence cross-column relation",
    "local_consistency_pass": "The value passes the local relational-consistency check",
    "local_consistency_fail": "The value fails the local relational-consistency check",
    "contextual_agreement": "The value agrees with the dominant value in a similar context",
    "contextual_disagreement": "The value disagrees with the dominant value in a similar context",
    "multisource_temporal_disagreement": (
        "The temporal value disagrees with a source-weighted consensus for the same entity"
    ),
    "unit_variant": "The value uses a non-dominant unit or unit expression",
    "suffix_pollution": "The value contains a suspected contaminating suffix",
    "percent_decimal_mismatch": "The value is inconsistent with the column's percent/decimal convention",
    "geo_suffix_pollution": "The geographic name contains a suspected extraneous region suffix",
    "encoding_artifact": "The value contains a suspected encoding artifact",
    "token_contamination": "The value contains additional content inconsistent with the column semantics",
    "case_variant": "The value's letter case differs from the dominant convention",
    "whitespace_variant": "The value's whitespace differs from the dominant convention",
    "repeated_internal_whitespace": "The value contains rare repeated internal whitespace",
    "excess_internal_whitespace": "The value contains multiple independent runs of internal whitespace",
    "repeated_terminal_punctuation": "The terminal punctuation mark is repeated consecutively",
    "repeated_list_segment": "The complete delimited item sequence repeats exactly",
    "unit_plurality_mismatch": "The numeric amount disagrees with the observed unit plurality",
    "unit_numeric_prefix_mismatch": "The value has a non-numeric prefix where a numeric count is expected",
    "unit_token_mismatch": "The value uses a near-match of the dominant measurement unit",
    "other_canonicalization": "A high-confidence canonicalization candidate is available for this value",
    "repeated_character_corruption": (
        "The value is a rare near-neighbor of a frequent value under a repeated character substitution"
    ),
}


def _json_safe(value: Any) -> Any:
    """Convert nested trace data to strict RFC-compatible JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _public_explanation_payload(trace: ExplanationTrace) -> Dict[str, Any]:
    """Return a user-facing trace without oracle-derived audit fields."""
    payload = asdict(trace)
    payload.pop("active_oracle_label", None)
    # ``confidence`` is retained internally for backward compatibility, but it
    # is entropy-based decision sharpness rather than epistemic confidence.
    payload.pop("confidence", None)
    return _json_safe(payload)


def _display_rule_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _relationship_reason_text(evidence: EvidenceContribution) -> str:
    metadata = evidence.metadata or {}
    determinant_column = str(
        metadata.get("determinant_column")
        or metadata.get("context_column")
        or metadata.get("left_column")
        or ""
    ).strip()
    dependent_column = str(
        metadata.get("dependent_column")
        or metadata.get("right_column")
        or ""
    ).strip()
    rule_text = str(
        metadata.get("rule_text")
        or metadata.get("description")
        or ""
    ).strip()
    determinant_value = _display_rule_value(
        metadata.get("determinant_value")
    )
    observed_value = _display_rule_value(metadata.get("observed_value"))
    expected_value = _display_rule_value(metadata.get("expected_value"))
    reference_rows = [
        int(row_index) + 1
        for row_index in (metadata.get("reference_rows") or [])
        if isinstance(row_index, (int, np.integer))
    ]

    if evidence.source_id in {"contextual_agreement", "contextual_disagreement"}:
        if not determinant_column:
            return ""
        relation = (
            rule_text
            or (
                f"rows with the same '{determinant_column}' value should "
                f"normally share the dominant '{dependent_column}' value"
            )
        )
        if evidence.source_id == "contextual_disagreement":
            if metadata.get("rule_violated") and dependent_column:
                detail = (
                    f"The value violates the learned conditional rule "
                    f"\"{relation}\": for '{determinant_column}'="
                    f"'{determinant_value}', the expected "
                    f"'{dependent_column}' value is '{expected_value}' "
                    f"({int(metadata.get('expected_support', 0))}/"
                    f"{int(metadata.get('group_size', 0))} supporting rows), "
                    f"but the observed value is '{observed_value}'"
                )
            else:
                detail = (
                    f"The observed '{dependent_column}' value has limited "
                    f"support under the learned conditional rule "
                    f"\"{relation}\" for '{determinant_column}'="
                    f"'{determinant_value}'"
                )
        else:
            detail = (
                f"The value supports the learned conditional rule "
                f"\"{relation}\" for '{determinant_column}'="
                f"'{determinant_value}'"
            )
        if reference_rows:
            detail += (
                "; representative supporting rows: "
                + ", ".join(str(row) for row in reference_rows)
            )
        return detail

    if evidence.family != "relationship":
        return ""
    if not rule_text:
        return ""
    columns = (
        f" between '{determinant_column}' and '{dependent_column}'"
        if determinant_column and dependent_column
        else ""
    )
    values = (
        f"; observed '{determinant_column}'='{determinant_value}' and "
        f"'{dependent_column}'='{observed_value}'"
        if determinant_column and dependent_column
        else ""
    )
    if evidence.polarity == "dirty":
        return (
            f"The value violates the relation \"{rule_text}\"{columns}"
            f"{values}"
        )
    return (
        f"The value satisfies the relation \"{rule_text}\"{columns}"
        f"{values}"
    )


def _explanation_reason_text(evidence: EvidenceContribution) -> str:
    relationship_text = _relationship_reason_text(evidence)
    if relationship_text:
        return relationship_text
    if evidence.source_id == "pattern_mismatch":
        failed_rules = (evidence.metadata or {}).get("failed_rules") or []
        if failed_rules:
            failed_rule_names = [
                str(rule.get("rule_name") or "unnamed_rule")
                for rule in failed_rules
            ]
            evaluated_count = int(
                (evidence.metadata or {}).get(
                    "evaluated_rule_count",
                    len(failed_rules),
                )
            )
            return (
                "The value does not match the dominant format of this column; "
                f"it failed {len(failed_rules)} of {evaluated_count} "
                "Agent-generated pattern rules: "
                + ", ".join(failed_rule_names)
            )
    if evidence.source_id in EVIDENCE_REASON_TEMPLATES:
        return EVIDENCE_REASON_TEMPLATES[evidence.source_id]
    if evidence.family in EVIDENCE_REASON_TEMPLATES:
        return EVIDENCE_REASON_TEMPLATES[evidence.family]
    readable = evidence.reason_code.replace("_", " ").strip()
    return f"Evidence source '{evidence.source_id}' triggered {readable or evidence.family}"


def _format_contribution_reason(evidence: EvidenceContribution) -> str:
    direction = "increases" if evidence.posterior_contribution >= 0 else "decreases"
    return (
        f"{_explanation_reason_text(evidence)}"
        f" (posterior without this evidence: {evidence.posterior_without:.4f}; "
        f"contribution to the error posterior: {evidence.posterior_contribution:+.4f}, "
        f"which {direction} the error likelihood)"
    )


def _generate_template_explanation(trace: ExplanationTrace) -> str:
    location = (
        f"Row {trace.row_index + 1} (internal index {trace.row_index}), "
        f"column '{trace.column}'"
    )
    if trace.decision == "suspicious" and trace.posterior < trace.threshold:
        opening = (
            f"{location} has an error posterior of {trace.posterior:.4f}, "
            f"which is below the threshold of {trace.threshold:.4f}; however, "
            "a validated strong relationship rule classifies the cell as suspicious."
        )
    elif trace.decision == "suspicious":
        opening = (
            f"{location} has an error posterior of {trace.posterior:.4f}, "
            f"which meets the threshold of {trace.threshold:.4f}; therefore, "
            f"the cell is classified as suspicious."
        )
    else:
        opening = (
            f"{location} has an error posterior of {trace.posterior:.4f}, "
            f"which is below the threshold of {trace.threshold:.4f}; therefore, "
            f"the cell is not classified as erroneous."
        )

    parts = [opening]
    if trace.primary_reason:
        parts.append(f" Primary evidence: {trace.primary_reason}.")
    elif not trace.calibration_fitted:
        parts.append(" No calibrated evidence contribution is available; the result is primarily determined by the base prior.")
    else:
        parts.append(" No evidence source has a measurable independent contribution to the posterior.")
    if trace.supporting_reasons:
        parts.append(" Additional supporting evidence: " + "; ".join(trace.supporting_reasons) + ".")
    if trace.counter_evidence:
        parts.append(" Counter-evidence: " + "; ".join(trace.counter_evidence) + ".")
    if (
        trace.repair_recommended
        and trace.repair_candidate is not None
        and str(trace.repair_candidate) != ""
    ):
        parts.append(
            f" Recommended repair: '{trace.repair_candidate}' "
            f"(mechanism: {trace.repair_mechanism or 'canonicalization'}; "
            f"repairability: {trace.repairability:.3f})."
        )
    return "".join(parts)


def _learn_column_thresholds(
    cell_registry: Dict[str, CellRecord],
    cell_posteriors: Dict[str, Dict[str, float]],
    active_labels: Dict[str, int],
    base_threshold: float,
    min_labels: int = 2,
) -> Dict[str, float]:
    """Learn conservative column thresholds with global-threshold shrinkage."""
    labeled_by_column: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for cell_key, label in active_labels.items():
        record = cell_registry.get(cell_key)
        bundle = cell_posteriors.get(cell_key)
        if record is None or bundle is None:
            continue
        labeled_by_column[record.column].append(
            (float(bundle.get("posterior", 0.5)), int(label))
        )

    thresholds: Dict[str, float] = {}
    for column, examples in labeled_by_column.items():
        if len(examples) < max(2, int(min_labels)):
            continue
        clean_scores = [score for score, label in examples if label == 0]
        dirty_scores = [score for score, label in examples if label == 1]
        if clean_scores and dirty_scores and max(clean_scores) < min(dirty_scores):
            empirical = 0.5 * (max(clean_scores) + min(dirty_scores))
        elif clean_scores:
            empirical = max(float(base_threshold), max(clean_scores) + 0.05)
        elif dirty_scores:
            empirical = min(float(base_threshold), min(dirty_scores) - 0.05)
        else:
            continue
        # Small label sets should only make a small adjustment.
        shrinkage = len(examples) / (len(examples) + 5.0)
        learned = (
            (1.0 - shrinkage) * float(base_threshold)
            + shrinkage * empirical
        )
        thresholds[column] = float(np.clip(learned, 0.05, 0.95))
    return thresholds


def _build_explanation_traces(
    metadata: Dict[str, Dict[str, Any]],
    cell_registry: Dict[str, CellRecord],
    cell_posteriors: Dict[str, Dict[str, float]],
    hard_labels: Dict[str, Dict[str, HardEvidenceLabel]],
    active_labels: Dict[str, int],
    repair_candidates: Dict[str, List[RepairCandidate]],
    loeo_contributions: Dict[str, List[EvidenceContribution]],
    calibration_trace: Any,
    threshold: float,
    column_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, ExplanationTrace]:
    traces: Dict[str, ExplanationTrace] = {}
    epsilon = 1e-12

    for cell_key, record in cell_registry.items():
        cell_threshold = float(
            (column_thresholds or {}).get(record.column, threshold)
        )
        posterior_bundle = cell_posteriors.get(
            cell_key, {"posterior": 0.5, "confidence": 0.0}
        )
        posterior = float(posterior_bundle["posterior"])
        evidence = loeo_contributions.get(cell_key, [])
        positive = [item for item in evidence if item.posterior_contribution > epsilon]
        negative = [item for item in evidence if item.posterior_contribution < -epsilon]
        positive.sort(key=lambda item: item.posterior_contribution, reverse=True)
        negative.sort(key=lambda item: item.posterior_contribution)

        cell_hard_label = hard_labels["cell"].get(cell_key)
        value_hard_label = hard_labels["value"].get(record.value_key)
        effective_hard_label = cell_hard_label or value_hard_label
        if effective_hard_label is not None:
            is_suspicious = effective_hard_label.label == "dirty"
        else:
            is_suspicious = posterior >= cell_threshold
        if is_suspicious:
            primary = positive[0] if positive else None
            supporting = positive[1:4] if primary else []
            counter = negative[:3]
        else:
            # For a clean decision, evidence that lowers the dirty posterior is
            # decision support; dirty-directed evidence is counter-evidence.
            primary = negative[0] if negative else None
            supporting = negative[1:4] if primary else []
            counter = positive[:3]
        repairs = repair_candidates.get(cell_key) or []
        primary_repair = repairs[0] if repairs else None
        canonicalization_support = [
            item
            for item in positive
            if item.family == "canonicalization"
            and (
                primary_repair is None
                or item.source_id == primary_repair.mechanism
            )
        ]
        repair_used_in_inference = bool(canonicalization_support)
        repair_recommended = bool(
            is_suspicious and primary_repair and repair_used_in_inference
        )
        semantics = metadata.get(record.column, {}).get("semantics") or {}
        archetype = (
            str(semantics.get("archetype") or "unknown")
            if isinstance(semantics, dict)
            else str(getattr(semantics, "archetype", "unknown"))
        )

        trace = ExplanationTrace(
            cell_key=cell_key,
            row_index=int(record.row_index),
            column=record.column,
            raw_value=record.raw_value,
            archetype=archetype,
            posterior=posterior,
            confidence=float(posterior_bundle["confidence"]),
            decision_certainty=float(posterior_bundle["confidence"]),
            full_logit=float(posterior_bundle.get("logit", 0.0)),
            threshold=cell_threshold,
            decision="suspicious" if is_suspicious else "not_suspicious",
            hard_label=effective_hard_label.label if effective_hard_label else "",
            active_oracle_label=(
                "dirty" if active_labels.get(cell_key) == 1
                else "clean" if active_labels.get(cell_key) == 0
                else ""
            ),
            calibration_fitted=bool(calibration_trace.fitted),
            calibration_reason=str(calibration_trace.reason),
            primary_reason=_format_contribution_reason(primary) if primary else "",
            supporting_reasons=[_format_contribution_reason(item) for item in supporting],
            counter_evidence=[_format_contribution_reason(item) for item in counter],
            evidence=list(evidence),
            repair_candidate=primary_repair.candidate_value if primary_repair else None,
            repair_mechanism=primary_repair.mechanism if primary_repair else "",
            repairability=primary_repair.repairability if primary_repair else 0.0,
            repair_used_in_inference=repair_used_in_inference,
            repair_recommended=repair_recommended,
        )
        trace.natural_language_explanation = _generate_template_explanation(trace)
        traces[cell_key] = trace

    return traces


def _export_cleanem_results(
    df: pd.DataFrame,
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    value_priors: Dict[str, Dict[str, float]],
    cell_posteriors: Dict[str, Dict[str, float]],
    hard_labels: Dict[str, Dict[str, HardEvidenceLabel]],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    repair_candidates: Optional[Dict[str, List[RepairCandidate]]] = None,
    active_labels: Optional[Dict[str, int]] = None,
    explanation_traces: Optional[Dict[str, ExplanationTrace]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    active_labels = active_labels or {}
    repair_candidates = repair_candidates or {}
    explanation_traces = explanation_traces or {}
    for cell_key, cell_record in cell_registry.items():
        value_record = value_registry[cell_record.value_key]
        prior_bundle = value_priors.get(cell_record.value_key, {"prior": 0.5, "confidence": 0.0})
        posterior_bundle = cell_posteriors.get(cell_key, {"posterior": 0.5, "confidence": 0.0})
        cell_label = hard_labels["cell"].get(cell_key)
        active_label = active_labels.get(cell_key)
        primary_repair = (
            repair_candidates.get(cell_key, [None])[0]
            if repair_candidates.get(cell_key)
            else None
        )
        explanation_trace = explanation_traces.get(cell_key)
        explanation_payload = (
            _public_explanation_payload(explanation_trace)
            if explanation_trace else {}
        )

        results.append({
            "cell_key": cell_key,
            "row_index": cell_record.row_index,
            "column": cell_record.column,
            "value": cell_record.raw_value,
            "value_key": cell_record.value_key,
            "value_prior": prior_bundle["prior"],
            "value_prior_confidence": prior_bundle["confidence"],
            "cell_posterior": posterior_bundle["posterior"],
            "cell_confidence": posterior_bundle["confidence"],
            "decision_certainty": posterior_bundle["confidence"],
            "hard_label": cell_label.label if cell_label else "",
            "hard_reason_summary": _summarize_hard_label(cell_label),
            "queried_for_calibration": cell_key in active_labels,
            "active_oracle_label": "dirty" if active_label == 1 else "clean" if active_label == 0 else "",
            "value_evidence_summary": _summarize_observations(value_observations.get(cell_record.value_key, [])),
            "cell_evidence_summary": _summarize_observations(cell_observations.get(cell_key, [])),
            "repair_mechanism": primary_repair.mechanism if primary_repair else "",
            "repair_candidate": primary_repair.candidate_value if primary_repair else "",
            "repair_detection_confidence": (
                primary_repair.detection_confidence if primary_repair else 0.0
            ),
            "repairability_score": primary_repair.repairability if primary_repair else 0.0,
            "repair_ambiguity": primary_repair.ambiguity if primary_repair else 0.0,
            "repair_used_in_inference": (
                explanation_trace.repair_used_in_inference
                if explanation_trace else False
            ),
            "repair_recommended": (
                explanation_trace.repair_recommended
                if explanation_trace else False
            ),
            "repair_changed_fields": (
                json.dumps(
                    primary_repair.changed_fields,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if primary_repair else ""
            ),
            "primary_reason": explanation_trace.primary_reason if explanation_trace else "",
            "supporting_reasons": (
                "; ".join(explanation_trace.supporting_reasons)
                if explanation_trace else ""
            ),
            "counter_evidence": (
                "; ".join(explanation_trace.counter_evidence)
                if explanation_trace else ""
            ),
            "natural_language_explanation": (
                explanation_trace.natural_language_explanation
                if explanation_trace else ""
            ),
            "explanation_trace": json.dumps(
                explanation_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        })

    results.sort(key=lambda item: (item["cell_posterior"], item["value_prior"]), reverse=True)
    return results


def run_clean_em_mode(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.rule_min_pass_rate) <= 1.0:
        raise ValueError("--rule_min_pass_rate must be between 0 and 1")
    if not 0.0 <= float(args.broad_evidence_coverage_threshold) <= 1.0:
        raise ValueError(
            "--broad_evidence_coverage_threshold must be between 0 and 1"
        )
    if not 0.0 <= float(args.unvalidated_evidence_coverage_threshold) <= 1.0:
        raise ValueError(
            "--unvalidated_evidence_coverage_threshold must be between 0 and 1"
        )
    mechanism_free_mode = args.mechanism_mode in {
        "adaptive_agent",
        "agent_rules_only",
    } or args.enable_adaptive_all_mechanisms
    adaptive_mode = (
        args.mechanism_mode != "agent_rules_only"
        and args.enable_adaptive_mechanism_discovery
        and not args.disable_adaptive_mechanism_discovery
        and (
            args.mechanism_mode == "adaptive_agent"
            or args.enable_adaptive_all_mechanisms
        )
    )
    # This build has no legacy specialized-detector execution path. The mode
    # switch controls only whether discovered mechanism candidates are formed
    # and validated; generic evidence remains available in both modes.
    args.enable_adaptive_mechanism_discovery = bool(adaptive_mode)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]

    logger = _setup_cleanem_logger(args.output_dir, dataset_name)
    logger.info("=" * 60)
    logger.info("CleanEM Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"  - mechanism_mode: {args.mechanism_mode}")
    logger.info(f"  - adaptive_mechanism_discovery: {adaptive_mode}")
    logger.info("  - specialized_mechanism_detectors: disabled")
    logger.info(f"[1/4] Profiling dirty dataset: {args.dirty_csv}")
    logger.info("Configuration:")
    logger.info(f"  - clean_seed_percent (legacy/unused): {args.clean_seed_percent}")
    logger.info(f"  - synthetic_per_family: {args.synthetic_per_family}")
    logger.info(f"  - em_max_iters (legacy/unused): {args.em_max_iters}")
    logger.info(f"  - em_prior_dirty (base prior): {args.em_prior_dirty}")
    logger.info(f"  - calib_min_clean_pass (legacy/unused): {args.calib_min_clean_pass}")
    logger.info(f"  - calib_max_dirty_pass (legacy/unused): {args.calib_max_dirty_pass}")
    logger.info(f"  - score_threshold (cell_posterior): {args.score_threshold}")
    logger.info(f"  - rule_min_pass_rate: {args.rule_min_pass_rate}")
    logger.info(f"  - active_label_budget: {args.active_label_budget}")
    logger.info(
        "  - adaptive_mechanism_query_budget: "
        f"{args.adaptive_mechanism_query_budget}"
    )
    logger.info(f"  - active_query_strategy: {args.active_query_strategy}")
    logger.info(
        "  - active_query_clean_proxy_fraction: "
        f"{args.active_query_clean_proxy_fraction}"
    )
    logger.info(f"  - evidence_gating: {not args.disable_evidence_gating}")
    logger.info(
        f"  - evidence_weight_learning: {not args.disable_evidence_weight_learning}"
    )
    logger.info(f"  - evidence_gating_mode: {args.evidence_gating_mode}")
    logger.info(
        "  - broad_evidence_guard: "
        f"{not args.disable_broad_evidence_guard}"
    )
    logger.info(
        "  - broad_evidence_coverage_threshold: "
        f"{args.broad_evidence_coverage_threshold}"
    )
    logger.info(
        "  - unvalidated_evidence_coverage_threshold: "
        f"{args.unvalidated_evidence_coverage_threshold}"
    )
    logger.info(
        f"  - hard_gate_min_clean_support: {args.hard_gate_min_clean_support}"
    )
    logger.info(f"  - gate_prior_strength: {args.gate_prior_strength}")
    logger.info(f"  - gate_prior_pseudocount: {args.gate_prior_pseudocount}")
    logger.info(
        f"  - canonicalization_evidence: {not args.disable_canonicalization_evidence}"
    )
    logger.info(
        "  - closed_enum_normalization: "
        f"{not args.disable_closed_enum_normalization}"
    )
    logger.info(f"  - agent_pattern_rule_scale: {args.agent_pattern_rule_scale}")
    logger.info(
        "  - adaptive_mechanism_allow_query_replacement: "
        f"{args.adaptive_mechanism_allow_query_replacement}"
    )
    logger.info(f"  - rule_pool_cache: {args.rule_pool_cache or 'disabled'}")
    logger.info(f"  - active_query_cache: {args.active_query_cache or 'disabled'}")
    logger.info(
        f"  - active_query_seed_cache: {args.active_query_seed_cache or 'disabled'}"
    )
    logger.info(
        "  - active_query_trajectory_cache: "
        f"{args.active_query_trajectory_cache or 'disabled'}"
    )
    logger.info(
        f"  - active_query_checkpoints: {args.active_query_checkpoints}"
    )
    logger.info(
        f"  - canonicalization_plan_cache: "
        f"{args.canonicalization_plan_cache or 'disabled'}"
    )
    logger.info(f"  - top_k: {args.top_k}")

    profiler = PandasProfiler(args.dirty_csv)
    df = profiler.df
    metadata = profiler.get_metadata()
    n_rows = len(df)

    logger.info("Dataset profile complete:")
    logger.info(f"  - Total rows: {n_rows}")
    logger.info(f"  - Total columns: {len(metadata)}")
    logger.info(f"  - Columns: {list(metadata.keys())}")
    logger.debug("Per-column metadata summary:")
    for col, meta in metadata.items():
        logger.debug(
            f"  - {col}: type={meta.get('type', 'unknown')}, nulls={meta.get('null_count', 0)}, "
            f"unique={meta.get('unique_count', 0)}, families={meta.get('available_families', [])}"
        )

    logger.info("[2/4] Inferring column semantics and generating evidence sources")
    factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)
    factory._disable_agent_rules = bool(args.disable_agent_rules)
    rule_pool_cache_hit = bool(
        args.rule_pool_cache and os.path.exists(args.rule_pool_cache)
    )
    if rule_pool_cache_hit:
        rule_pool, cached_semantics = _load_rule_pool_cache(
            args.rule_pool_cache,
            df,
            rule_min_pass_rate=args.rule_min_pass_rate,
        )
        for column in df.columns:
            metadata[column]["semantics"] = dict(cached_semantics[column])
        if args.disable_agent_rules:
            rule_pool = {}
            logger.info("Agent clean-rule pool: disabled")
        logger.info(f"Loaded column semantics and rule pool from: {args.rule_pool_cache}")
    else:
        column_semantics = factory.infer_column_semantics(metadata)
        for column, semantics in column_semantics.items():
            metadata[column]["semantics"] = asdict(semantics)
        rule_pool, raw_rule_proposals = _build_clean_rule_pool(
            df,
            metadata,
            factory,
            rule_min_pass_rate=args.rule_min_pass_rate,
        )
        raw_rule_path = os.path.join(args.output_dir, f"{dataset_name}_raw_rule_proposals.json")
        _write_json_atomic(raw_rule_path, {"version": 1, "dataframe_fingerprint": _dataframe_fingerprint(df), "columns": list(df.columns), "proposals": raw_rule_proposals})
        logger.info(f"Saved raw rule proposals to: {raw_rule_path}")
        if args.rule_pool_cache:
            _save_rule_pool_cache(
                args.rule_pool_cache,
                df,
                metadata,
                rule_pool,
            )
            logger.info(f"Saved column semantics and rule pool to: {args.rule_pool_cache}")

    for column in df.columns:
        semantics = metadata[column]["semantics"]
        logger.info(
            f"  - Column semantics '{column}': archetype={semantics.get('archetype')}, "
            f"confidence={float(semantics.get('confidence', 0.0)):.3f}, "
            f"open_set={float(semantics.get('open_set_score', 0.0)):.3f}, "
            f"structure={float(semantics.get('structure_strength', 0.0)):.3f}, "
            f"canonicalization_need={float(semantics.get('canonicalization_need', 0.0)):.3f}, "
            f"mechanisms={semantics.get('possible_error_mechanisms', [])}"
        )

    total_rules = sum(len(rules) for rules in rule_pool.values())
    logger.info(
        "Rule pool loaded successfully:"
        if rule_pool_cache_hit else
        "Rule pool built successfully:"
    )
    logger.info(f"  - Total columns with rules: {len(rule_pool)}")
    logger.info(f"  - Total rules: {total_rules}")

    family_counts = {family: 0 for family in FAMILIES if family != "prototype_lexical"}
    for column in metadata.keys():
        rules = rule_pool.get(column, [])
        per_family = {family: 0 for family in family_counts}
        for rule in rules:
            fam = rule.get("family")
            if fam in per_family:
                per_family[fam] += 1
                family_counts[fam] += 1
        logger.info(
            f"  - Column '{column}': {len(rules)} rules "
            f"(missing={per_family.get('missing', 0)}, outlier={per_family.get('outlier', 0)}, "
            f"pattern={per_family.get('pattern', 0)}, relationship={per_family.get('relationship', 0)})"
        )
        for rule in rules:
            logger.info(
                f"    - Rule '{rule.get('rule_name', 'unknown')}' [{rule.get('family', 'unknown')}]: {rule.get('rule_str', '')}"
            )
    logger.info(
        "  - Rule family distribution: "
        + ", ".join(f"{family}={count}" for family, count in family_counts.items())
    )

    relationship_rules: List[Dict[str, Any]] = []
    if args.disable_agent_rules or args.disable_llm_relationship_rules:
        logger.info("LLM relationship rules: disabled")
    else:
        cache_hit = bool(
            args.relationship_rule_cache
            and os.path.exists(args.relationship_rule_cache)
        )
        if cache_hit:
            relationship_payload = _load_relationship_rule_cache(
                args.relationship_rule_cache,
                df,
            )
            logger.info(
                "Loaded LLM relationship rules from: "
                f"{args.relationship_rule_cache}"
            )
        else:
            relationship_summary = _build_table_relationship_summary(df, metadata)
            raw_relationship_rules: List[Dict[str, Any]] = []
            prompt_info: Dict[str, str] = {}
            if relationship_summary["candidates"]:
                raw_relationship_rules, prompt_info = (
                    factory.infer_table_relationship_candidates(relationship_summary)
                )
            relationship_rules, validation_reports = (
                _validate_llm_relationship_rule_candidates(
                    df,
                    raw_relationship_rules,
                    relationship_summary,
                )
            )
            relationship_payload = {
                "table_summary": relationship_summary,
                "prompt_info": prompt_info,
                "raw_proposals": raw_relationship_rules,
                "validation_reports": validation_reports,
                "validated_rules": relationship_rules,
            }
            if args.relationship_rule_cache:
                _save_relationship_rule_cache(
                    args.relationship_rule_cache,
                    df,
                    relationship_payload,
                )
                logger.info(
                    "Saved LLM relationship rules to: "
                    f"{args.relationship_rule_cache}"
                )

        relationship_rules = list(
            relationship_payload.get("validated_rules") or []
        )
        raw_relationship_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_raw_relationship_proposals.json",
        )
        validated_relationship_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_validated_relationship_rules.json",
        )
        _write_json_atomic(raw_relationship_path, {
            "version": 1,
            "dataframe_fingerprint": _dataframe_fingerprint(df),
            "columns": list(df.columns),
            "table_summary": relationship_payload.get("table_summary") or {},
            "prompt_info": relationship_payload.get("prompt_info") or {},
            "raw_proposals": relationship_payload.get("raw_proposals") or [],
        })
        _write_json_atomic(validated_relationship_path, {
            "version": 1,
            "dataframe_fingerprint": _dataframe_fingerprint(df),
            "columns": list(df.columns),
            "validation_reports": relationship_payload.get("validation_reports") or [],
            "validated_rules": relationship_rules,
        })
        logger.info(
            "LLM relationship rules: "
            f"accepted={len(relationship_rules)}, "
            f"proposed={len(relationship_payload.get('raw_proposals') or [])}"
        )
        for rule in relationship_rules:
            logger.info(
                "  - Validated FD: "
                f"{rule['determinant_column']} -> {rule['dependent_column']} "
                f"({rule['rule_id']})"
            )

    logger.info("[3/4] Running active-learning calibrated evidence inference")
    value_registry = _build_value_registry(df, metadata)
    cell_registry = _build_cell_registry(df, metadata, value_registry)
    logger.info(f"Value registry size: {len(value_registry)}")
    logger.info(f"Cell registry size: {len(cell_registry)}")

    canonicalization_profiles = build_canonicalization_profiles(df, metadata)
    plan_cache_hit = bool(
        args.canonicalization_plan_cache
        and os.path.exists(args.canonicalization_plan_cache)
    )
    if plan_cache_hit:
        canonicalization_plans = _load_canonicalization_plan_cache(
            args.canonicalization_plan_cache,
            df,
        )
        logger.info(
            "Loaded canonicalization plans from: "
            f"{args.canonicalization_plan_cache}"
        )
    else:
        canonicalization_plans = factory.infer_canonicalization_plans(
            canonicalization_profiles
        )
        if args.canonicalization_plan_cache:
            _save_canonicalization_plan_cache(
                args.canonicalization_plan_cache,
                df,
                canonicalization_plans,
            )
            logger.info(
                "Saved canonicalization plans to: "
                f"{args.canonicalization_plan_cache}"
            )

    repair_agent = PlanDrivenCanonicalizationAgent(
        df,
        metadata,
        cell_registry,
        canonicalization_plans,
    )
    plan_diagnostics = repair_agent.validate_plans()
    for column in df.columns:
        plan = canonicalization_plans.get(column) or {}
        diagnostics = plan_diagnostics.get(column) or []
        accepted = [item for item in diagnostics if item.get("accepted")]
        rejected = [item for item in diagnostics if not item.get("accepted")]
        logger.info(
            f"  - Canonicalization plan '{column}': "
            f"applicable={bool(plan.get('applicable'))}, "
            f"confidence={float(plan.get('confidence') or 0.0):.3f}, "
            f"proposed={len(plan.get('operations') or [])}, "
            f"accepted={len(accepted)}, rejected={len(rejected)}, "
            f"rejection_reasons={[item.get('reason') for item in rejected]}"
        )
    repair_candidates = repair_agent.generate()
    if mechanism_free_mode:
        deterministic_repairs, deterministic_repair_counts = {}, {}
    else:
        deterministic_repairs, deterministic_repair_counts = (
            _infer_deterministic_surface_repairs(cell_registry)
        )
    for cell_key, candidates in deterministic_repairs.items():
        target = repair_candidates.setdefault(cell_key, [])
        for candidate in candidates:
            if any(
                existing.mechanism == candidate.mechanism
                and existing.candidate_value == candidate.candidate_value
                for existing in target
            ):
                continue
            target.append(candidate)
        target.sort(
            key=lambda item: (
                item.detection_confidence
                * item.repairability
                * (1.0 - item.ambiguity)
            ),
            reverse=True,
        )
    repair_mechanism_counts = Counter(
        candidate.mechanism
        for candidates in repair_candidates.values()
        for candidate in candidates
    )
    logger.info(
        "Canonicalization/Repair Agent: "
        f"candidate_cells={len(repair_candidates)}, "
        f"candidates={sum(len(items) for items in repair_candidates.values())}, "
        f"mechanisms={dict(sorted(repair_mechanism_counts.items()))}"
    )
    logger.info(
        "deterministic_surface_repair_discovery: "
        f"candidate_cells={len(deterministic_repairs)}, "
        f"candidates={sum(len(items) for items in deterministic_repairs.values())}, "
        f"mechanisms={deterministic_repair_counts}, query_only=True"
    )

    value_observations, cell_observations = _extract_evidence(
        df,
        metadata,
        rule_pool,
        value_registry,
        cell_registry,
        # The standalone build never executes the legacy specialized
        # detectors; adaptive_agent differs from agent_rules_only only in
        # candidate discovery and validation.
        enable_specialized_mechanisms=False,
        repair_candidates=repair_candidates,
        validated_relationship_rules=relationship_rules,
        enable_canonicalization_evidence=not args.disable_canonicalization_evidence,
        enable_closed_enum_normalization=(
            not args.disable_closed_enum_normalization
        ),
        enable_evidence_gating=False,
        defer_evidence_gating=not args.disable_evidence_gating,
        enable_multisource_temporal_consensus=(
            not args.disable_multisource_temporal_consensus
        ),
        multisource_min_sources=args.multisource_min_sources,
        multisource_min_ratio=args.multisource_min_ratio,
        multisource_time_tolerance=args.multisource_time_tolerance,
        multisource_min_source_reliability=(
            args.multisource_min_source_reliability
        ),
    )
    if args.enable_adaptive_all_mechanisms:
        deferred_mechanisms = _defer_specialized_mechanism_evidence(
            value_observations,
            cell_observations,
        )
        logger.info(
            "adaptive_all_mechanisms: deferred_specialized_evidence=%s",
            deferred_mechanisms,
        )
    hard_labels = _build_hard_evidence(df, metadata, value_observations, cell_observations)
    adaptive_report = None
    if args.enable_adaptive_mechanism_discovery:
        adaptive_report = discover_candidate_mechanisms(
            metadata,
            cell_registry,
            value_observations,
            cell_observations,
            repair_candidates=repair_candidates,
            min_support=args.adaptive_mechanism_min_support,
        )
        if not args.disable_agent_mechanism_naming:
            try:
                adaptive_report.apply_names(
                    factory.name_discovered_mechanisms(
                        adaptive_report.agent_payload(cell_registry=cell_registry)
                    )
                )
            except Exception as exc:
                logger.warning(
                    "discovered_mechanism_naming_failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        annotated_count = annotate_discovered_observations(
            adaptive_report,
            value_observations,
            cell_observations,
        )
        discovery_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_adaptive_mechanism_discovery.json",
        )
        _write_json_atomic(
            discovery_path,
            {
                "dataframe_fingerprint": _dataframe_fingerprint(df),
                "columns": list(df.columns),
                "annotated_observations": annotated_count,
                **adaptive_report.to_dict(),
            },
        )
        logger.info(
            "adaptive_mechanism_discovery: "
            f"candidates={len(adaptive_report.candidates)}, "
            f"novel={len(adaptive_report.novel_candidates())}, "
            f"annotated={annotated_count}, output={discovery_path}"
        )

    evidence_matrix = build_cell_evidence_matrix(
        cell_registry,
        value_observations,
        cell_observations,
    )
    logger.info(
        f"Evidence matrix built: cells={len(evidence_matrix.cell_keys)}, "
        f"features={len(evidence_matrix.feature_names)}"
    )

    clean_df_for_oracle: Optional[pd.DataFrame] = None
    if args.clean_csv and os.path.exists(args.clean_csv):
        clean_df_for_oracle = pd.read_csv(args.clean_csv)
    else:
        logger.info("Active learning oracle unavailable: clean_csv not found")

    selected_cell_keys: List[str] = []
    calibration_cell_keys: List[str] = []
    active_labels: Dict[str, int] = {}
    queried_cells: List[Dict[str, object]] = []
    query_rounds: List[List[str]] = []
    labels_resolved_during_selection = False
    adaptive_eligible_mechanisms: Set[str] = set()
    trajectory_cache_enabled = bool(args.active_query_trajectory_cache)
    if clean_df_for_oracle is not None and (
        args.active_label_budget > 0 or trajectory_cache_enabled
    ):
        active_query_cache_hit = bool(
            trajectory_cache_enabled
            or (args.active_query_cache and os.path.exists(args.active_query_cache))
        )
        mechanism_budget = max(0, int(args.adaptive_mechanism_query_budget))
        separate_mechanism_budget = (
            adaptive_report is not None and mechanism_budget > 0
        )
        allocation_budget = (
            args.active_label_budget + mechanism_budget
            if separate_mechanism_budget
            else args.active_label_budget
        )
        if active_query_cache_hit:
            if trajectory_cache_enabled:
                checkpoints = _parse_active_query_checkpoints(
                    args.active_query_checkpoints
                )
                if int(args.active_label_budget) not in checkpoints:
                    raise ValueError(
                        f"active_label_budget={args.active_label_budget} is not "
                        f"in trajectory checkpoints {checkpoints}"
                    )
                if os.path.exists(args.active_query_trajectory_cache):
                    trajectory_payload = _load_active_query_trajectory_cache(
                        args.active_query_trajectory_cache,
                        df,
                        evidence_matrix,
                        adaptive_report,
                        args,
                    )
                    trajectory_action = "loaded"
                else:
                    trajectory_payload = _build_nested_active_query_trajectory(
                        evidence_matrix,
                        adaptive_report,
                        checkpoints,
                        args,
                        metadata,
                        cell_registry,
                        cell_observations,
                    )
                    _save_active_query_trajectory_cache(
                        args.active_query_trajectory_cache,
                        df,
                        trajectory_payload,
                    )
                    trajectory_action = "created"
                checkpoint_key = str(int(args.active_label_budget))
                selected_cell_keys = list(
                    trajectory_payload["checkpoint_cell_keys"][checkpoint_key]
                )
                calibration_cell_keys = list(selected_cell_keys)
                adaptive_eligible_mechanisms = set(
                    trajectory_payload["checkpoint_eligible_mechanisms"][
                        checkpoint_key
                    ]
                )
                logger.info(
                    "nested_active_query_trajectory: "
                    f"action={trajectory_action}, "
                    f"checkpoint={args.active_label_budget}, "
                    f"summary={trajectory_payload['checkpoint_summaries'][checkpoint_key]}, "
                    f"evidence_fingerprint={trajectory_payload['evidence_matrix_fingerprint']}, "
                    f"mechanism_fingerprint={trajectory_payload['mechanism_pool_fingerprint']}, "
                    f"path={args.active_query_trajectory_cache}"
                )
            else:
                selected_cell_keys = _load_active_query_cache(
                    args.active_query_cache,
                    df,
                    args.active_label_budget,
                    set(evidence_matrix.cell_keys),
                    args.active_query_strategy,
                    args.active_query_rounds,
                    args.active_query_batch_size,
                )
                calibration_cell_keys = list(selected_cell_keys)
                logger.info(f"Loaded active queries from: {args.active_query_cache}")
        else:
            base_query_cell_keys = select_active_queries(
                evidence_matrix,
                args.active_label_budget,
                clean_proxy_fraction=args.active_query_clean_proxy_fraction,
                # The base calibration design must remain stable when a new
                # discovery is present. Discovery-only evidence is consumed
                # by allocate_mechanism_queries below as a separate validation
                # reserve, never as an implicit replacement for calibration
                # diversity.
                include_query_features=False,
                strategy=args.active_query_strategy,
            )
            adaptive_protected_cell_keys: Set[str] = set()
            proxy_target = int(round(
                len(base_query_cell_keys)
                * max(0.0, min(0.5, args.active_query_clean_proxy_fraction))
            )) if args.active_query_strategy == "d_optimal" else 0
            if proxy_target:
                matrix_row = {
                    cell_key: index
                    for index, cell_key in enumerate(evidence_matrix.cell_keys)
                }
                # A clean proxy must be *purely* clean-looking: no positive
                # (dirty) evidence may coexist with its negative evidence.
                # Protecting arbitrary low net-score cells can reserve rows
                # that are actually ambiguous and was the source of the
                # recent Rayyan query drift.
                def proxy_rank(cell_key: str) -> Tuple[float, str]:
                    row = evidence_matrix.values[matrix_row[cell_key]]
                    clean_magnitude = -float(np.sum(np.minimum(row, 0.0)))
                    return (-clean_magnitude, cell_key)

                pure_clean_proxies = [
                    cell_key
                    for cell_key in base_query_cell_keys
                    if (
                        float(np.max(evidence_matrix.values[matrix_row[cell_key]]))
                        <= 1e-9
                        and float(np.min(evidence_matrix.values[matrix_row[cell_key]]))
                        < -1e-9
                    )
                ]
                pure_clean_proxies.sort(key=proxy_rank)
                adaptive_protected_cell_keys.update(
                    pure_clean_proxies[:proxy_target]
                )
            if args.active_query_seed_cache:
                seed_cell_keys = _load_active_query_seed_cache(
                    args.active_query_seed_cache,
                    df,
                    args.active_label_budget,
                    set(evidence_matrix.cell_keys),
                    args.active_query_strategy,
                    args.active_query_rounds,
                    args.active_query_batch_size,
                )
                selected_cell_keys = list(seed_cell_keys)
                seed_key_set = set(seed_cell_keys)
                selected_cell_keys.extend(
                    key for key in base_query_cell_keys
                    if key not in seed_key_set
                )
                selected_cell_keys = selected_cell_keys[:args.active_label_budget]
                logger.info(
                    "Loaded active-query seed prefix from: "
                    f"{args.active_query_seed_cache} "
                    f"(seed={len(seed_cell_keys)}, target={args.active_label_budget})"
                )
            else:
                selected_cell_keys = base_query_cell_keys
            if (
                args.enable_core_mechanism_query_coverage
                and not mechanism_free_mode
                and args.active_query_strategy != "random"
            ):

                selected_cell_keys, core_coverage_summary = (
                    _ensure_core_mechanism_query_coverage(
                        selected_cell_keys,
                        args.active_label_budget,
                        metadata,
                        cell_registry,
                        cell_observations,
                    )
                )
                logger.info(
                    "core_mechanism_query_coverage: "
                    f"{core_coverage_summary}"
                )
                adaptive_protected_cell_keys.update(
                    core_coverage_summary.get("inserted", [])
                )
            if (
                args.enable_structural_missing_query_coverage
                and args.active_query_strategy != "random"
            ):
                selected_cell_keys, coverage_summary = (
                    _ensure_structural_missing_query_coverage(
                        selected_cell_keys,
                        args.active_label_budget,
                        cell_observations,
                        cell_registry,
                        metadata,
                        protected_cell_keys=adaptive_protected_cell_keys,
                    )
                )
                logger.info(
                    "structural_missing_query_coverage: "
                    f"{coverage_summary}"
                )
                adaptive_protected_cell_keys.update(
                    coverage_summary.get("inserted", [])
                )

            # Keep the D-optimal/coverage set separate from any mechanism
            # validation queries appended below when an explicit *extra*
            # mechanism budget is requested. In the normal fixed-budget mode,
            # mechanism representatives replace cells inside the same ten
            # labels, so the final selected set (after allocation) is exactly
            # the set that must calibrate the model. Capturing the
            # pre-replacement list here would silently drop replacement labels
            # from calibration and make the historical Rayyan path
            # impossible to reproduce.
            calibration_cell_keys = list(selected_cell_keys)

            # Run this reservation last so later coverage policies cannot
            # evict one of the labels required to validate a candidate.
            if adaptive_report is not None:
                selected_cell_keys, adaptive_query_summary = allocate_mechanism_queries(
                    selected_cell_keys,
                    allocation_budget,
                    adaptive_report,
                    protected_cell_keys=(
                        adaptive_protected_cell_keys.intersection(selected_cell_keys)
                    ),
                    fraction=args.adaptive_mechanism_query_fraction,
                    verification_samples=args.adaptive_mechanism_min_labeled_support,
                    max_candidates_per_round=(
                        args.adaptive_mechanism_max_candidates_per_round
                    ),
                    allow_replacement=(
                        args.adaptive_mechanism_allow_query_replacement
                    ),
                    mechanism_budget=(
                        mechanism_budget if separate_mechanism_budget else None
                    ),
                )
                logger.info(
                    "adaptive_mechanism_query_allocation: "
                    f"{adaptive_query_summary}"
                )
                adaptive_eligible_mechanisms = set(
                    adaptive_query_summary.get("covered_mechanisms", [])
                )

                # With no explicit extra budget, allocation is a fixed-size
                # replacement operation. All final labels (including
                # mechanism representatives) participate in calibration.
                if not separate_mechanism_budget:
                    calibration_cell_keys = list(selected_cell_keys)

            selected_cell_keys = list(dict.fromkeys(selected_cell_keys))[
                :allocation_budget
            ]
            logger.info(
                "active_query_budget: "
                f"selected={len(selected_cell_keys)}, "
                f"calibration={len(calibration_cell_keys)}, "
                f"mechanism_extra={max(0, len(selected_cell_keys) - len(calibration_cell_keys))}, "
                f"limit={allocation_budget}, fixed_limit=True"
            )

        if (
            active_query_cache_hit
            and not trajectory_cache_enabled
            and adaptive_report is not None
            and separate_mechanism_budget
        ):
            selected_cell_keys, adaptive_query_summary = allocate_mechanism_queries(
                selected_cell_keys,
                allocation_budget,
                adaptive_report,
                fraction=args.adaptive_mechanism_query_fraction,
                verification_samples=args.adaptive_mechanism_min_labeled_support,
                max_candidates_per_round=args.adaptive_mechanism_max_candidates_per_round,
                allow_replacement=args.adaptive_mechanism_allow_query_replacement,
                mechanism_budget=mechanism_budget,
            )
            adaptive_eligible_mechanisms = set(
                adaptive_query_summary.get("covered_mechanisms", [])
            )
            selected_cell_keys = list(dict.fromkeys(selected_cell_keys))[:allocation_budget]
            logger.info(
                "adaptive_mechanism_query_allocation: "
                f"{adaptive_query_summary}"
            )
            logger.info(
                "active_query_budget: "
                f"selected={len(selected_cell_keys)}, "
                f"calibration={len(calibration_cell_keys)}, "
                f"mechanism_extra={max(0, len(selected_cell_keys) - len(calibration_cell_keys))}, "
                f"limit={allocation_budget}, fixed_limit=True"
            )

        query_rounds = [list(calibration_cell_keys)] if calibration_cell_keys else []

        if args.active_query_cache and not active_query_cache_hit:
            _save_active_query_cache(
                args.active_query_cache,
                df,
                args.active_label_budget,
                calibration_cell_keys,
                args.active_query_strategy,
                args.active_query_rounds,
                args.active_query_batch_size,
                query_rounds=query_rounds,
            )
            logger.info(f"Saved active queries to: {args.active_query_cache}")

    if not labels_resolved_during_selection:
        active_labels, queried_cells = _query_active_oracle_labels(
            df,
            clean_df_for_oracle,
            cell_registry,
            selected_cell_keys,
        )
    calibration_key_set = set(calibration_cell_keys)
    calibration_labels = {
        key: label
        for key, label in active_labels.items()
        if key in calibration_key_set
    }
    calibration_queried_cells = [
        item for item in queried_cells
        if str(item.get("cell_key")) in calibration_key_set
    ]

    if adaptive_report is not None:
        adaptive_corrected_values = {
            cell_key: clean_df_for_oracle.iloc[record.row_index][record.column]
            for cell_key, label in active_labels.items()
            for record in [cell_registry.get(cell_key)]
            if (
                int(label) == 1
                and record is not None
                and clean_df_for_oracle is not None
                and record.row_index < len(clean_df_for_oracle)
                and record.column in clean_df_for_oracle.columns
            )
        }
        validate_mechanism_candidates(
            adaptive_report,
            active_labels,
            eligible_mechanisms=adaptive_eligible_mechanisms,
            corrected_values=adaptive_corrected_values,
            values_equal=Judge.ground_truth_values_equal,
            min_labeled_support=args.adaptive_mechanism_min_labeled_support,
            min_label_purity=args.adaptive_mechanism_min_label_purity,
            max_false_positive_risk=args.adaptive_mechanism_max_fp_risk,
            min_archetype_consistency=(
                args.adaptive_mechanism_min_archetype_consistency
            ),
            min_repair_success_rate=(
                args.adaptive_mechanism_min_repair_success_rate
            ),
        )
        activation_summary = enable_validated_mechanisms(
            adaptive_report,
            value_observations,
            cell_observations,
        )
        validation_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_adaptive_mechanism_validation.json",
        )
        _write_json_atomic(validation_path, adaptive_report.to_dict())
        logger.info(
            "adaptive_mechanism_validation: "
            f"validated={len(adaptive_report.validated_candidates())}, "
            f"activation={activation_summary}, output={validation_path}"
        )
        if activation_summary.get("added", 0):
            evidence_matrix = build_cell_evidence_matrix(
                cell_registry, value_observations, cell_observations
            )

    corrected_standardization_candidates: Dict[str, List[RepairCandidate]] = {}
    if args.enable_corrected_value_standardization:
        (
            corrected_standardization_candidates,
            corrected_standardization_summary,
        ) = _build_corrected_value_standardization_candidates(
            df,
            clean_df_for_oracle,
            cell_registry,
            calibration_cell_keys,
            calibration_labels,
            detection_observations=cell_observations,
            metadata=metadata,
        )
        logger.info(
            "corrected_value_standardization: "
            f"{corrected_standardization_summary}"
        )
        if corrected_standardization_summary.get("used_in_detection"):
            hard_labels = _build_hard_evidence(
                df,
                metadata,
                value_observations,
                cell_observations,
            )
            evidence_matrix = build_cell_evidence_matrix(
                cell_registry,
                value_observations,
                cell_observations,
            )
            logger.info(
                "Evidence matrix rebuilt with corrected-format contract: "
                f"cells={len(evidence_matrix.cell_keys)}, "
                f"features={len(evidence_matrix.feature_names)}"
            )

    structural_missing_update = _update_structural_missing_from_labels(
        value_observations,
        cell_observations,
        metadata,
        value_registry,
        cell_registry,
        calibration_labels,
    )
    logger.info(
        f"structural_missing_label_update: {structural_missing_update}"
    )
    if structural_missing_update["updated"]:
        # Structural labels are promoted after the first active-query pass.
        # Rebuild hard evidence as well as the matrix so audit counts and
        # downstream decisions reflect the newly learned mechanism.
        hard_labels = _build_hard_evidence(
            df,
            metadata,
            value_observations,
            cell_observations,
        )
        evidence_matrix = build_cell_evidence_matrix(
            cell_registry,
            value_observations,
            cell_observations,
        )

    broad_feature_guard = (
        {}
        if args.disable_broad_evidence_guard
        else identify_unvalidated_broad_features(
            evidence_matrix,
            cell_registry,
            calibration_labels,
            broad_unstructured_coverage=(
                args.broad_evidence_coverage_threshold
            ),
            repair_candidate_coverage=(
                args.unvalidated_evidence_coverage_threshold
            ),
            distributional_coverage=(
                args.unvalidated_evidence_coverage_threshold
            ),
        )
    )
    # Broad unvalidated evidence remains available as a weak, mechanism-neutral
    # prior instead of being removed from inference. Active labels can still
    # drive its weight to zero through the learned cap.
    feature_weight_caps = {
        feature_name: math.log(max(1.0, float(args.unvalidated_broad_feature_cap)))
        for feature_name in broad_feature_guard
    }
    # A discovered feature exists in the inference matrix only after label and
    # repair-value validation. With one fixed-budget label, unconstrained
    # logistic shrinkage can still leave that isolated feature below the
    # decision threshold. A 4:1 minimum likelihood ratio is the Laplace-
    # smoothed evidence level for one positive, zero negative validations;
    # broad generic features remain governed by the independent safety caps.
    feature_weight_floors = {
        feature_name: math.log(4.0)
        for feature_name, descriptor in evidence_matrix.feature_metadata.items()
        if descriptor.get("family") == "discovered_mechanism"
    }
    logger.info(
        "broad_evidence_guard: "
        f"guarded={len(broad_feature_guard)}, "
        f"disabled={bool(args.disable_broad_evidence_guard)}, "
        f"weak_cap={args.unvalidated_broad_feature_cap}"
    )
    for feature_name, report in sorted(broad_feature_guard.items()):
        logger.info(
            "  - Broad evidence weakly capped: "
            f"feature={feature_name}, reason={report['reason']}, "
            f"coverage={report['max_column_coverage']:.4f}, "
            f"positive_support={report['positive_support']}, "
            f"clean_labels={report['clean_label_support']}, "
            f"dirty_labels={report['dirty_label_support']}"
        )

    fixed_calibration = None
    unsupported_feature_weight_priors: Dict[str, float] = {}

    if not args.disable_evidence_gating:
        # A validated canonicalization candidate may miss the tiny active
        # label sample. Give that isolated feature a conservative prior rather
        # than silently assigning weight zero. The feature names include the
        # target column namespace, so this cannot transfer evidence between
        # columns; active labels can still override the prior when observed.
        def generic_feature_prior(feature_name: str) -> Optional[float]:
            """Return a conservative prior for an uncovered evidence source.

            Archetype namespaces deliberately prevent a label from one semantic
            role from calibrating another.  A fixed zero for every uncovered
            namespace would, however, make an unqueried column invisible. Use
            small source-level priors as a hierarchical fallback; active
            labels still replace these values whenever the feature is observed
            in the calibration sample.
            """
            parts = str(feature_name).split(":", 3)
            if len(parts) < 3 or ":archetype:" not in str(feature_name):
                return None
            source = parts[2]
            archetype = str(feature_name).rsplit(":archetype:", 1)[-1]
            multipliers = {
                # Direct parser/type failures are stronger than distributional
                # rarity, but remain below hard deterministic contracts.
                "parse_failure": 2.0,
                "cell_parse_failure": 2.0,
                "type_conflict": 2.0,
                "cell_type_conflict": 2.0,
                "pattern_mismatch": 1.8,
                "unicode_unresolved_combining_mark": 1.8,
                "unicode_non_latin_mixed_script": 2.0,
                "temporal_range_violation": 1.8,
                "surface_shape_conflict": 1.8,
                # Clean passes get a weak positive reliability prior so they
                # lower scores without dominating when a column is unqueried.
                "parser_success": 1.15,
                "cell_parser_success": 1.15,
                "regex_pass": 1.10,
                "pattern_match": 1.10,
                "prototype_close": 1.10,
                "contextual_agreement": 1.10,
            }
            multiplier = multipliers.get(source)
            if multiplier is None or multiplier <= 1.0:
                return None
            # Format mismatch is meaningful for structured scalar roles and
            # for geography, but is too broad for free-form entities and
            # closed taxonomies where legitimate variants are common.
            if source == "pattern_mismatch" and archetype not in {
                "identifier",
                "temporal_measure",
                "numeric_measure",
                "unit_measure",
                "geo_name",
            }:
                return None
            if source == "temporal_range_violation" and archetype != "temporal_measure":
                return None
            if source == "surface_shape_conflict" and archetype not in {
                "open_entity",
                "geo_name",
                "closed_enum",
            }:
                return None
            if source == "unicode_unresolved_combining_mark" and archetype not in {
                "open_entity",
                "geo_name",
                "closed_enum",
                "free_text",
            }:
                return None
            return math.log(multiplier)

        unsupported_feature_weight_priors = {}
        for feature_name in evidence_matrix.feature_names:
            if ":canonicalization:" in feature_name:
                unsupported_feature_weight_priors[feature_name] = math.log(2.5)
                continue
            prior = generic_feature_prior(feature_name)
            if prior is not None:
                unsupported_feature_weight_priors[feature_name] = prior
        unsupported_feature_weight_priors[
            "cell:contextual:multisource_temporal_disagreement"
        ] = math.log(3.0)
        # Learn globally comparable evidence weights from the raw observations.
        # Gating is a post-calibration attenuation step: refitting after scaling
        # one archetype would otherwise change weights for every other column.
        _, _, fixed_calibration = estimate_calibrated_posteriors(
            value_registry,
            cell_registry,
            evidence_matrix,
            calibration_cell_keys,
            calibration_labels,
            args.em_prior_dirty,
            queried_cells=calibration_queried_cells,
            unsupported_feature_weight_priors=unsupported_feature_weight_priors,
            feature_weight_floors=feature_weight_floors,
            feature_weight_caps=feature_weight_caps,
            learn_evidence_weights=not args.disable_evidence_weight_learning,
        )
        supported_policies, policy_support = _label_supported_gating_policies(
            calibration_cell_keys,
            calibration_labels,
            metadata,
            value_observations,
            cell_observations,
            cell_registry,
            min_distinct_clean_values=max(1, args.hard_gate_min_clean_support),
        )
        logger.info(
            "label_supported_gating: "
            f"mode={args.evidence_gating_mode}, "
            f"prior_strength={args.gate_prior_strength}, "
            f"prior_pseudocount={args.gate_prior_pseudocount}, "
            f"supported={sorted(supported_policies)}, "
            f"support_counts={dict(sorted(policy_support.items()))}"
        )
        gating_summary = _apply_evidence_gates(
            value_observations,
            cell_observations,
            metadata,
            value_registry,
            cell_registry,
            supported_policies=supported_policies,
            policy_support=policy_support,
            gating_mode=args.evidence_gating_mode,
            gate_prior_strength=args.gate_prior_strength,
            gate_prior_pseudocount=args.gate_prior_pseudocount,
        )
        logger.info(f"evidence_gating_counts: {gating_summary['counts']}")
        for item in gating_summary["policy_hits"]:
            logger.info(
                "  - Evidence gate: "
                f"archetype={item['archetype']}, source={item['source_id']}, "
                f"scale={item['scale']:.2f}, count={item['count']}"
            )
        evidence_matrix = build_cell_evidence_matrix(
            cell_registry,
            value_observations,
            cell_observations,
        )
        logger.info(
            "Evidence matrix rebuilt after label-supported gating: "
            f"cells={len(evidence_matrix.cell_keys)}, "
            f"features={len(evidence_matrix.feature_names)}"
        )

    pattern_scale_counts = _scale_agent_pattern_rule_evidence(
        value_observations,
        cell_observations,
        args.agent_pattern_rule_scale,
    )
    logger.info(
        "agent_pattern_rule_scaling: "
        f"scale={args.agent_pattern_rule_scale}, counts={pattern_scale_counts}"
    )
    if args.agent_pattern_rule_scale != 1.0:
        evidence_matrix = build_cell_evidence_matrix(
            cell_registry,
            value_observations,
            cell_observations,
        )
        logger.info(
            "Evidence matrix rebuilt after Agent pattern-rule scaling: "
            f"cells={len(evidence_matrix.cell_keys)}, "
            f"features={len(evidence_matrix.feature_names)}"
        )

    value_priors, cell_posteriors, calibration_trace = estimate_calibrated_posteriors(
        value_registry,
        cell_registry,
        evidence_matrix,
        calibration_cell_keys,
        calibration_labels,
        args.em_prior_dirty,
        queried_cells=calibration_queried_cells,
        fixed_calibration=fixed_calibration,
        unsupported_feature_weight_priors=unsupported_feature_weight_priors,
        feature_weight_floors=feature_weight_floors,
        feature_weight_caps=feature_weight_caps,
        learn_evidence_weights=not args.disable_evidence_weight_learning,
    )
    loeo_contributions = compute_leave_one_evidence_out_contributions(
        cell_registry,
        value_observations,
        cell_observations,
        cell_posteriors,
        calibration_trace,
    )
    column_thresholds = (
        {}
        if args.disable_column_thresholds
        else _learn_column_thresholds(
            cell_registry,
            cell_posteriors,
            calibration_labels,
            args.score_threshold,
            min_labels=args.column_threshold_min_labels,
        )
    )
    logger.info(
        "column_decision_thresholds: "
        f"base={args.score_threshold}, learned={dict(sorted(column_thresholds.items()))}"
    )
    explanation_traces = _build_explanation_traces(
        metadata,
        cell_registry,
        cell_posteriors,
        hard_labels,
        active_labels,
        repair_candidates,
        loeo_contributions,
        calibration_trace,
        args.score_threshold,
        column_thresholds=column_thresholds,
    )
    logger.info(
        "Explanation traces built: "
        f"cells={len(explanation_traces)}, "
        f"evidence_observations={sum(len(items) for items in loeo_contributions.values())}, "
        f"nonzero_loeo={sum(1 for items in loeo_contributions.values() for item in items if abs(item.posterior_contribution) > 1e-12)}"
    )
    results = _export_cleanem_results(
        df,
        value_registry,
        cell_registry,
        value_priors,
        cell_posteriors,
        hard_labels,
        value_observations,
        cell_observations,
        repair_candidates=repair_candidates,
        active_labels=active_labels,
        explanation_traces=explanation_traces,
    )

    for value_key, record in value_registry.items():
        bundle = value_priors.get(value_key, {"prior": 0.5, "confidence": 0.0})
        record.prior = bundle["prior"]
        record.confidence = bundle["confidence"]
    for cell_key, record in cell_registry.items():
        bundle = cell_posteriors.get(cell_key, {"posterior": 0.5, "confidence": 0.0})
        record.posterior = bundle["posterior"]
        record.confidence = bundle["confidence"]

    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"total_cells_exported: {len(results)}")
    logger.info(f"value_records: {len(value_registry)}")
    logger.info(f"cell_records: {len(cell_registry)}")
    logger.info(
        f"hard_evidence_counts: "
        f"value_dirty={sum(1 for label in hard_labels['value'].values() if label.label == 'dirty')}, "
        f"value_clean={sum(1 for label in hard_labels['value'].values() if label.label == 'clean')}, "
        f"cell_dirty={sum(1 for label in hard_labels['cell'].values() if label.label == 'dirty')}, "
        f"cell_clean={sum(1 for label in hard_labels['cell'].values() if label.label == 'clean')}"
    )
    label_counts = Counter(active_labels.values())
    calibration_label_counts = Counter(calibration_labels.values())
    logger.info(
        f"active_calibration: fitted={calibration_trace.fitted}, reason={calibration_trace.reason}, "
        f"queried={len(active_labels)}/{len(selected_cell_keys)}, "
        f"calibration_queried={len(calibration_labels)}/{len(calibration_cell_keys)}, "
        f"dirty_labels={label_counts.get(1, 0)}, clean_labels={label_counts.get(0, 0)}, "
        f"calibration_dirty={calibration_label_counts.get(1, 0)}, "
        f"calibration_clean={calibration_label_counts.get(0, 0)}, "
        f"features={calibration_trace.feature_count}"
    )
    for query in calibration_trace.queried_cells:
        logger.info(
            "  - Active query: "
            f"row={query.get('row_index')}, column={query.get('column')}, "
            f"value={query.get('value')!r}, label={query.get('label')}"
        )
    if calibration_trace.feature_weights:
        top_weights = sorted(
            calibration_trace.feature_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:20]
        logger.info("learned_evidence_weights:")
        for feature_name, weight in top_weights:
            logger.info(f"  - {feature_name}: {weight:.4f}")
    logger.info(f"value_evidence_source_coverage: {_evidence_source_coverage(value_observations)}")
    logger.info(f"cell_evidence_source_coverage: {_evidence_source_coverage(cell_observations)}")

    value_prior_values = [item["value_prior"] for item in results if pd.notna(item["value_prior"])]
    cell_posterior_values = [item["cell_posterior"] for item in results if pd.notna(item["cell_posterior"])]

    logger.info("value_prior distribution:")
    if value_prior_values:
        logger.info(f"  - Mean: {sum(value_prior_values)/len(value_prior_values):.4f}")
        logger.info(f"  - Min: {min(value_prior_values):.4f}")
        logger.info(f"  - Max: {max(value_prior_values):.4f}")
    else:
        logger.info("  - No value priors")

    logger.info("cell_posterior distribution:")
    if cell_posterior_values:
        logger.info(f"  - Mean: {sum(cell_posterior_values)/len(cell_posterior_values):.4f}")
        logger.info(f"  - Min: {min(cell_posterior_values):.4f}")
        logger.info(f"  - Max: {max(cell_posterior_values):.4f}")
        above_threshold = [score for score in cell_posterior_values if score >= args.score_threshold]
        logger.info(
            f"  - Above threshold ({args.score_threshold}): {len(above_threshold)}/{len(cell_posterior_values)} "
            f"({len(above_threshold)/len(cell_posterior_values)*100:.1f}%)"
        )
    else:
        logger.info("  - No cell posteriors")

    top_k = min(args.top_k, len(results))
    logger.info(f"\nTop {top_k} highest scoring cells by cell_posterior:")
    for item in results[:top_k]:
        log_msg = (
            f"Row {item['row_index']}, Column '{item['column']}': value={item['value']!r}, "
            f"value_prior={item['value_prior']:.4f}, cell_posterior={item['cell_posterior']:.4f}, "
            f"hard_label={item['hard_label'] or 'none'}"
        )
        logger.info(f"  {log_msg}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{timestamp}_{dataset_name}_clean_em_scores.csv")
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"\nSaved all cell scores to: {output_path}")

    explanation_output_path = os.path.join(
        args.output_dir,
        f"{timestamp}_{dataset_name}_explanation_traces.jsonl",
    )
    with open(explanation_output_path, "w", encoding="utf-8") as explanation_file:
        for cell_key in cell_registry:
            trace = explanation_traces[cell_key]
            explanation_file.write(json.dumps(
                _public_explanation_payload(trace),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ))
            explanation_file.write("\n")
    logger.info(
        f"Saved {len(explanation_traces)} unified explanation traces to: "
        f"{explanation_output_path}"
    )

    explanation_audit_path = os.path.join(
        args.output_dir,
        f"{timestamp}_{dataset_name}_explanation_audit.jsonl",
    )
    with open(explanation_audit_path, "w", encoding="utf-8") as audit_file:
        for cell_key in cell_registry:
            trace = explanation_traces[cell_key]
            if not trace.active_oracle_label:
                continue
            audit_file.write(json.dumps(
                _json_safe(asdict(trace)),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ))
            audit_file.write("\n")
    logger.info(
        "Saved active-label explanation audit traces to: "
        f"{explanation_audit_path}"
    )

    repair_output_path = os.path.join(
        args.output_dir,
        f"{timestamp}_{dataset_name}_repair_candidates.csv",
    )
    repair_rows = flatten_repair_candidates(repair_candidates)
    pd.DataFrame(repair_rows).to_csv(
        repair_output_path,
        index=False,
        encoding="utf-8",
    )
    logger.info(
        f"Saved {len(repair_rows)} repair candidates to: {repair_output_path}"
    )

    if args.enable_corrected_value_standardization:
        corrected_standardization_path = os.path.join(
            args.output_dir,
            f"{timestamp}_{dataset_name}_corrected_value_standardization.csv",
        )
        corrected_standardization_rows = flatten_repair_candidates(
            corrected_standardization_candidates
        )
        pd.DataFrame(corrected_standardization_rows).to_csv(
            corrected_standardization_path,
            index=False,
            encoding="utf-8",
        )
        logger.info(
            "Saved corrected-value standardization candidates: "
            f"count={len(corrected_standardization_rows)}, "
            f"path={corrected_standardization_path}, "
            f"used_in_detection={bool(corrected_standardization_summary.get('used_in_detection'))}"
        )

    if args.clean_csv and os.path.exists(args.clean_csv):
        logger.info("\nEvaluating clean_em scores against ground truth:")
        clean_df = clean_df_for_oracle if clean_df_for_oracle is not None else pd.read_csv(args.clean_csv)
        judge = Judge()
        detected_errors = [
            {"row_index": item["row_index"], "column": item["column"]}
            for item in results
            if (
                item.get("cell_key") in explanation_traces
                and explanation_traces[item["cell_key"]].decision == "suspicious"
            )
        ]
        logger.info(
            f"Detected {len(detected_errors)} cells using global/column thresholds"
        )
        metrics = judge.evaluate_with_ground_truth(df, clean_df, detected_errors)
        judge.print_evaluation_summary(metrics)
        if hasattr(metrics, "get"):
            logger.info("Evaluation Metrics:")
            for key, value in metrics.items() if isinstance(metrics, dict) else []:
                logger.info(f"  - {key}: {value}")

    logger.info("\n[4/4] clean_em pipeline complete.")
    logger.info("=" * 60)


def run_clean_rule_refinement(df, metadata, base_rules, clean_rules, factory, judge, args,
                              clean_prompts=None, dirty_prompts=None, clean_df=None):
    """
    Run clean rule-level refinement for all columns.

    Args:
        df: DataFrame
        metadata: Column metadata
        base_rules: Dict[column] -> List[(agent, rule_str)]
        clean_rules: Dict[column] -> List[(clean_rule, rule_str)]
        factory: AgentFactory
        judge: Judge instance
        args: CLI arguments
        clean_prompts: Dict[column] -> Dict[clean_rule] -> {'prompt': ..., 'response': ...}
        dirty_prompts: Dict[column] -> Dict[agent] -> {'prompt': ..., 'response': ...}

    Returns:
        Tuple of (best_rules, refinement_history)
    """
    if clean_prompts is None:
        clean_prompts = {}
    if dirty_prompts is None:
        dirty_prompts = {}
    from dual_types import CleanRule, CleanRuleSet
    from modification_memory import start_logger, stop_logger
    import re
    import pandas as pd
    import numpy as np
    import os
    try:
        from dateutil.parser import parse  # type: ignore
    except Exception:
        parse = None

    best_rules: Dict[str, Any] = {}
    all_history: Dict[str, Any] = {}
    clean_rule_sets: Dict[str, Any] = {}
    initial_rules: Dict[str, Any] = {}

    # Extract dataset name from dirty_csv path
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]

    # Start a single logger for all columns
    print(f"\nRefinement log started: {dataset_name}")
    logger = start_logger(args.output_dir, dataset_name)

    # Log initial rule generation
    logger.log_initial_rules(clean_rules, base_rules, clean_prompts, dirty_prompts)

    for column in metadata.keys():
        clean_rules_dict = {}
        for rule_name, rule_str in clean_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                clean_rules_dict[rule_name] = CleanRule(
                    name=rule_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile clean rule {rule_name}: {e}")

        dirty_rules = {}
        for agent_name, rule_str in base_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                dirty_rules[agent_name] = CleanRule(
                    name=agent_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile dirty rule {agent_name}: {e}")

        if not clean_rules_dict and not dirty_rules:
            continue

        rule_set = CleanRuleSet(
            column=column,
            clean_rules=clean_rules_dict,
            dirty_rules=dirty_rules
        )
        clean_rule_sets[column] = rule_set

    for column, rule_set in clean_rule_sets.items():
        dual_rule = rule_set.to_dual_rule(agent_factory=factory)
        initial_rules[column] = dual_rule

    if clean_df is not None and initial_rules:
        print("\nInitial performance (before refinement):")
        initial_detected_errors = judge.get_detected_dirty_values(initial_rules, df)
        initial_metrics_summary = judge.evaluate_with_ground_truth(
            df,
            clean_df,
            initial_detected_errors
        )
        judge.print_evaluation_summary(initial_metrics_summary)

    def _make_console_buffer():
        lines: List[str] = []

        def console(msg: str = "") -> None:
            lines.append(msg)

        return console, lines

    def _refine_single_column(column: str, rule_set: Any):
        console, log_lines = _make_console_buffer()
        col_metadata = metadata.get(column, {})
        refined_set, history = judge.refine_clean_rules(
            df,
            column,
            rule_set,
            factory=factory,
            max_rounds=args.max_rounds,
            conflict_tolerance=0.01,
            metadata=col_metadata,
            output_dir=args.output_dir,
            logger=logger,
            console=console,
        )
        dual_rule = refined_set.to_dual_rule(agent_factory=factory)
        return column, dual_rule, history, log_lines

    max_workers = max(1, int(getattr(args, "max_workers", 1) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_column = {
            executor.submit(_refine_single_column, column, rule_set): column
            for column, rule_set in clean_rule_sets.items()
        }
        for future in as_completed(future_to_column):
            column, dual_rule, history, log_lines = future.result()
            print(f"\nProcessing column: {column}")
            for line in log_lines:
                print(line)
            best_rules[column] = dual_rule
            all_history[column] = history

    # Stop the shared logger
    final_summary = {
        'dataset': dataset_name,
        'total_columns': len(metadata.keys()),
        'processed_columns': list(best_rules.keys()),
    }
    stop_logger(final_summary)
    print(f"\nRefinement log saved: {dataset_name}")

    return best_rules, all_history


def run_dual_mode(args: argparse.Namespace) -> None:
    # Create console log file with the same timestamp format
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]
    console_log_path = os.path.join(args.output_dir, f"{timestamp}_{dataset_name}_running_log.log")

    # Custom stdout that writes to both terminal and file
    class DualOutput:
        def __init__(self, filepath, dataset_name):
            self.filepath = filepath
            self.file = open(filepath, 'w', encoding='utf-8')
            self.write_header(dataset_name)

        def write_header(self, dataset_name):
            self.file.write("=" * 60 + "\n")
            self.file.write("CONSOLE OUTPUT LOG\n")
            self.file.write("=" * 60 + "\n")
            self.file.write(f"Dataset: {dataset_name}\n")
            self.file.write(f"Started: {datetime.now().isoformat()}\n")
            self.file.write("=" * 60 + "\n\n")
            self.file.flush()

        def write(self, text):
            import sys
            sys.__stdout__.write(text)
            ts = datetime.now().strftime("%H:%M:%S")
            self.file.write(f"[{ts}] {text}")
            self.file.flush()

        def flush(self):
            import sys
            sys.__stdout__.flush()
            self.file.flush()

        def close(self):
            self.file.write("\n" + "=" * 60 + "\n")
            self.file.write(f"Finished: {datetime.now().isoformat()}\n")
            self.file.write("=" * 60 + "\n")
            self.file.close()

    dual_output = DualOutput(console_log_path, dataset_name)

    # Redirect stdout to both terminal and file
    import sys
    old_stdout = sys.stdout
    sys.stdout = dual_output

    try:
        print(f"[1/7] Profiling dirty dataset: {args.dirty_csv}")
        profiler = PandasProfiler(args.dirty_csv)
        metadata = profiler.get_metadata()

        factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)

        print("[2/7] Generating base agent rules (Missing/Typo/Pattern/Outlier)")
        base_rules, dirty_prompts = factory.generate_rules_per_column(metadata)

        print("[3/7] Generating clean rules (Completeness/Accuracy/Pattern/Relationship)")
        clean_rules, clean_prompts = factory.generate_clean_rules_per_column(metadata)

        judge = Judge(threshold=args.vr_threshold, violation_threshold=args.vr_threshold)

        print("[4/7] Standalone rule-based error detection")
        base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
        accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)
        judge.print_summary(accepted_base_rules)
        # judge.print_detected_errors(base_detected_errors)

        # Evaluate standalone clean rules (Completeness/Accuracy/Pattern/Relationship)
        print("\nStandalone clean rules evaluation:")
        clean_evaluation_results = judge.evaluate_rules(profiler.df, clean_rules, rule_type="clean")
        accepted_clean_rules = judge.get_accepted_rules(clean_evaluation_results)
        judge.print_summary(accepted_clean_rules, rule_type="clean")

        # Combined error detection using AND/OR logic
        # Clean Rule (AND): All clean rules must be satisfied
        # Dirty Rule (OR): Violating any dirty rule marks as potentially dirty
        # Error = (NOT all clean rules satisfied) AND (at least one dirty rule violated)
        print("\nCombined AND/OR error detection:")
        base_detected_errors = judge.get_detected_errors(
            accepted_base_rules,      # dirty rules (OR logic)
            accepted_clean_rules      # clean rules (AND logic)
        )
        print(f"Detected {len(base_detected_errors)} errors using combined AND/OR logic")

        # Evaluate standalone base rules against ground truth if clean CSV is provided
        base_metrics_summary = None
        clean_df = None
        if args.clean_csv:
            print("\nGround truth evaluation (base rules):")
            clean_df = pd.read_csv(args.clean_csv)
            base_metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                base_detected_errors
            )
            judge.print_evaluation_summary(base_metrics_summary)

        print("\n[5/7] Clean rule-level refinement")
        best_rules, refinement_history = run_clean_rule_refinement(
            profiler.df,
            metadata,
            {
                column: [(r['agent'], r['rule_string']) for r in accepted_base_rules.get(column, [])]
                for column in accepted_base_rules.keys()
            },
            {
                column: [(r['agent'], r['rule_string']) for r in accepted_clean_rules.get(column, [])]
                for column in accepted_clean_rules.keys()
            },
            factory,
            judge,
            args,
            clean_prompts,
            dirty_prompts,
            clean_df
        )

        if not best_rules:
            print("✗ No acceptable dual rules were produced. See refinement logs for details.")
            return

        print("\n[6/7] Validating disjointness of refined rules")
        validator = DisjointnessValidator(gap_tolerance=args.grey_tolerance)
        validation_result = validator.validate_batch(profiler.df, best_rules)
        print(validator.report_violations(validation_result))

        coverage_gaps = sorted(set(metadata.keys()) - set(best_rules.keys()))

        print("[7/7] Evaluating final dual rules on the dataset")
        evaluation_payload = _materialize_rule_payload(best_rules)
        evaluation_results = judge.evaluate_dual_rules(
            profiler.df,
            evaluation_payload,
            grey_tolerance=args.grey_tolerance,
            metadata=metadata
        )
        detected_dirty_values = judge.get_detected_dirty_values(best_rules, profiler.df)
        judge.print_dual_summary(best_rules, evaluation_results)

        refined_metrics_summary = None
        # Evaluate refined dual rules against ground truth if clean CSV is provided
        if args.clean_csv and clean_df is not None:
            print("\nGround truth evaluation (refined rules):")
            refined_metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                detected_dirty_values
            )
            judge.print_evaluation_summary(refined_metrics_summary)

        print("\n✓ Dual verification complete.")

    finally:
        # Restore stdout and close the log file
        sys.stdout = old_stdout
        dual_output.close()


def _materialize_rule_payload(best_rules) -> Dict[str, List[tuple]]:
    """Convert DualRule map into judge.evaluate_dual_rules input."""
    payload: Dict[str, List[tuple]] = {}
    for column, rule in best_rules.items():
        payload[column] = [(rule.agent_name, rule.clean_rule_str, rule.dirty_rule_str)]
    return payload


def main() -> None:
    args = parse_args()
    os.environ.pop("CLEANEM_LLM_TERMINAL_FAILURE", None)
    usage_log = (
        args.llm_usage_log
        or os.environ.get("CLEANEM_USAGE_LOG")
        or os.path.join(args.output_dir, "llm_usage.jsonl")
    )
    os.environ["CLEANEM_USAGE_LOG"] = usage_log
    os.makedirs(os.path.dirname(os.path.abspath(usage_log)), exist_ok=True)
    seed_text = os.environ.get("CLEANEM_LLM_SEED", "").strip()
    if seed_text:
        try:
            seed = int(seed_text)
            random.seed(seed)
            np.random.seed(seed)
            cleanem_logger.info("seeded_run: seed=%s", seed)
        except ValueError:
            cleanem_logger.warning("Ignoring invalid CLEANEM_LLM_SEED=%r", seed_text)
    if args.enable_adaptive_all_mechanisms:
        args.enable_adaptive_mechanism_discovery = True
    if args.mode == "dual":
        run_dual_mode(args)
    else:
        run_clean_em_mode(args)
    terminal_failure = os.environ.get("CLEANEM_LLM_TERMINAL_FAILURE", "").strip()
    if terminal_failure:
        raise RuntimeError(
            "CleanEM rejected an incomplete run because an LLM request "
            f"exhausted its retries: {terminal_failure}"
        )


if __name__ == "__main__":
    main()
