"""Resolve named dispatch rules to the engine's lower-is-higher priorities.

Holding-cost inputs are used as dispatch indices only. This module does not
reinterpret them as inventory backorders or claim an optimal scheduling policy.
Exact ties in the two new cost-based rules remain in the same priority class.
The legacy fastest_service_first rule retains its distinct skill-index tie ranks.
"""
from __future__ import annotations

from typing import Any
import numpy as np

PRIORITY_RULES_VERSION = "1.0"
NAMED_PRIORITY_RULES = (
    "fastest_service_first",
    "holding_cost_first",
    "holding_cost_times_service_rate",
)


def _finite_nonnegative(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite, nonnegative values.")
    return array.copy()


def _descending_dense_ranks(scores: np.ndarray) -> np.ndarray:
    """Largest distinct score -> 1; exact equal scores -> equal integer ranks."""
    unique, inverse = np.unique(scores, return_inverse=True)
    return (len(unique) - inverse).astype(np.int64)


def resolve_priorities(
    priorities: Any,
    *,
    holding_costs: Any,
    service_rates: Any,
    skill_server_assignment: Any,
) -> dict[str, Any]:
    """Return the numeric priorities plus metadata for saving or inspection.

    Parameters
    ----------
    priorities:
        None, a supported named rule, or finite nonnegative numeric priorities
        broadcastable to (servers, skills). Smaller *numeric* priorities win.
    holding_costs:
        Nonnegative finite per-skill vector or scalar.
    service_rates:
        Nonnegative finite per-skill vector or (servers, skills) matrix.
    skill_server_assignment:
        Binary (servers, skills) matrix. Ineligible pairs are never served.

    New rules
    ---------
    holding_cost_first: descending h[k], independent of the service rate.
    holding_cost_times_service_rate: descending h[k] * mu[k]. With a rate
        matrix, rank eligible skills separately within each server using
        h[k] * mu[server, k]. Server selection still follows the existing
        engine; this is not a global maximum-index matching algorithm.

    The legacy fastest_service_first rule requires a per-skill rate vector
    and retains distinct ranks for equal rates (skill-index tie breaking).
    New cost rules preserve exact ties; no hidden rounding or tolerance is
    applied. Positive products that underflow to zero or overflow are rejected.
    """
    assignment = np.asarray(skill_server_assignment)
    if (assignment.ndim != 2 or 0 in assignment.shape
            or not np.all((assignment == 0) | (assignment == 1))):
        raise ValueError("skill_server_assignment must be a nonempty binary (servers, skills) matrix.")
    n, m = assignment.shape
    service = _finite_nonnegative(service_rates, "service_rates")
    if service.shape not in ((m,), (n, m)):
        raise ValueError("service_rates must be per-skill or a (servers, skills) matrix.")
    holding = _finite_nonnegative(holding_costs, "holding_costs")
    try:
        holding = np.broadcast_to(holding, (m,)).copy()
    except ValueError as exc:
        raise ValueError("holding_costs must be a scalar or a length-m_skills vector.") from exc
    eligible = (assignment != 0) & (service > 0)

    base = {
        "priority_resolution_version": PRIORITY_RULES_VERSION,
        "priority_metric_values": None,
        "priority_metric_name": None,
        "priority_metric_larger_is_higher": None,
    }
    if priorities is None:
        return dict(base, priorities=None, priority_rule_name="equal_priority",
                    priority_rule="equal priority (FIFO across eligible queue heads)",
                    priority_scope="all eligible server-skill pairs",
                    priority_tie_policy="FIFO among eligible waiting jobs of equal priority")
    if not isinstance(priorities, str):
        values = _finite_nonnegative(priorities, "priorities")
        try:
            np.broadcast_to(values, (n, m))
        except ValueError as exc:
            raise ValueError("priorities must be scalar, per-skill, or server-by-skill.") from exc
        return dict(base, priorities=values, priority_rule_name="custom",
                    priority_rule="caller supplied; smaller value has higher priority",
                    priority_scope="numeric input broadcast to (servers, skills)",
                    priority_tie_policy="existing engine score ties use FIFO; scores round to 6 decimals")
    if priorities not in NAMED_PRIORITY_RULES:
        raise ValueError(
            f"Unknown priority rule {priorities!r}. Use None, numbers, or one of "
            f"{', '.join(NAMED_PRIORITY_RULES)}."
        )

    if priorities == "fastest_service_first":
        if service.ndim != 1:
            raise ValueError("fastest_service_first requires per-skill service rates; use explicit priorities for matrix rates.")
        order = np.argsort(-service, kind="stable")
        values = np.empty(m, dtype=np.int64)
        values[order] = np.arange(1, m + 1)
        return dict(base, priorities=values, priority_rule_name=priorities,
                    priority_rule="fastest_service_first; equal-rate ties broken by skill index",
                    priority_metric_values=service.copy(), priority_metric_name="service_rates",
                    priority_metric_larger_is_higher=True, priority_scope="per skill",
                    priority_tie_policy="distinct ranks; equal rates ordered by skill index (legacy behavior)")

    if priorities == "holding_cost_first":
        scores = holding
        metric_name = "holding_costs"
        scope = "per skill"
        values = _descending_dense_ranks(scores)
        rule = "holding_cost_first; larger holding cost has higher priority; exact ties share a rank"
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scores = holding * service
        if not np.all(np.isfinite(scores)):
            raise ValueError("holding_costs * service_rates overflows; rescale the inputs consistently.")
        if np.any((holding > 0) & (service > 0) & (scores == 0)):
            raise ValueError("holding_costs * service_rates underflows to zero; rescale the inputs consistently.")
        metric_name = "holding_costs * service_rates"
        if scores.ndim == 1:
            values = _descending_dense_ranks(scores)
            scope = "per skill"
        else:
            # Finite placeholder for ineligible pairs; the engine excludes them.
            values = np.full((n, m), m + 1, dtype=np.int64)
            for server in range(n):
                mask = eligible[server]
                if mask.any():
                    values[server, mask] = _descending_dense_ranks(scores[server, mask])
            scope = "per server among its eligible skills"
        rule = ("holding_cost_times_service_rate; larger holding-cost-times-rate "
                "has higher priority; exact ties share a rank")

    return dict(base, priorities=values, priority_rule_name=priorities,
                priority_rule=rule, priority_metric_values=scores.copy(),
                priority_metric_name=metric_name,
                priority_metric_larger_is_higher=True, priority_scope=scope,
                priority_tie_policy="exact equal scores share a rank; FIFO within each eligible class")
