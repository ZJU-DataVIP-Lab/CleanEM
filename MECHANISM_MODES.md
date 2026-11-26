# CleanEM mechanism modes

This directory is the standalone CleanEM build without the legacy specialized
mechanism detectors. The old `legacy_specialized` mode is intentionally not
available here.

## Full adaptive mode

Use `--mechanism_mode adaptive_agent` (the default). The pipeline keeps generic
missing, parser, rule, distribution, context, and repair evidence. It also
forms evidence-signature candidates, allocates validation queries, validates
candidate mechanisms with active labels, and adds only validated mechanisms to
the evidence matrix.

## No adaptive mechanism discovery

Use:

```text
--mechanism_mode agent_rules_only
```

or, with the default mode selected elsewhere:

```text
--disable_adaptive_mechanism_discovery
```

This disables candidate clustering, mechanism-query allocation, and discovered
mechanism activation. It does not disable the generic evidence matrix or the
Agent-generated rule pool. Since the legacy specialized detectors are disabled
in this build, this is the appropriate ablation for:

```text
Agent rules + mechanism-independent evidence
```

With no active-query cache or seed cache, the remaining active labels are the
base D-optimal calibration set, subject to the default structural-missing
coverage policy. For a pure D-optimal query ablation, also pass:

```text
--disable_structural_missing_query_coverage
```

Keep the same model, seed, total label budget, prompt, and output protocol when
comparing this mode with `adaptive_agent`.

## Active-query and evidence-calibration ablations

The default active-query selector is D-optimal. Use:

```text
--active_query_strategy random
```

for a seeded uniform baseline over the same evidence-signature candidate pool.
Random mode skips clean-proxy and structural/core coverage rewrites. Do not
provide an active-query cache when measuring this ablation. To prevent adaptive
mechanism validation from replacing random base queries, combine it with:

```text
--disable_adaptive_mechanism_query_replacement
```

The replacement policy can also be ablated independently with the same flag
while retaining the default D-optimal selector.

To retain evidence extraction but remove label-fitted source weights, use:

```text
--disable_evidence_weight_learning
```

This uses fixed equal evidence weights and keeps the extracted evidence matrix,
static safety constraints, gating, and field-threshold calibration unchanged.
It is distinct from `--disable_evidence_gating`, which disables gating but does
not by itself disable weight fitting.
