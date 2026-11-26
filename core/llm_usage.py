"""Small, best-effort accounting sink for OpenAI-compatible LLM calls."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_USAGE_LOG_LOCK = threading.Lock()


def _usage_field(usage: Any, name: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def record_llm_usage(
    *,
    response: Any = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: Optional[int] = None,
    requested_max_tokens: Optional[int] = None,
    attempt: int = 1,
    agent: Optional[str] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    """Append one request's usage without ever affecting the experiment.

    The destination is selected by ``CLEANEM_USAGE_LOG``. Prompts and responses
    are intentionally excluded so the accounting file remains compact and does
    not duplicate potentially sensitive experiment data.
    """
    log_path = os.environ.get("CLEANEM_USAGE_LOG", "").strip()
    if not log_path:
        return

    usage = getattr(response, "usage", None) if response is not None else None
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": model,
        "base_url": base_url,
        "seed": seed,
        "agent": agent,
        "attempt": int(attempt),
        "requested_max_tokens": requested_max_tokens,
        "prompt_tokens": _usage_field(usage, "prompt_tokens"),
        "completion_tokens": _usage_field(usage, "completion_tokens"),
        "total_tokens": _usage_field(usage, "total_tokens"),
        "usage_available": usage is not None,
    }
    if error:
        record["error"] = str(error)[:1000]

    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _USAGE_LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Accounting must never turn a successful model call into a failed run.
        print(f"  Warning: could not write LLM usage log: {exc}")
