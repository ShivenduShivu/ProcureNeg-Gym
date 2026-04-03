from typing import Any

from server.models import ContractClauses, IPOwnershipType


def normalize(
    value: float,
    min_val: float,
    max_val: float,
    higher_is_better: bool = True,
) -> float:
    if higher_is_better:
        return (value - min_val) / (max_val - min_val)
    return (max_val - value) / (max_val - min_val)


def score_clauses(contract: ContractClauses) -> float:
    scores = []

    scores.append(normalize(contract.annual_fee, 100000, 2000000, False))
    scores.append(normalize(contract.payment_terms, 15, 90, True))
    scores.append(normalize(contract.duration_years, 1, 5, True))
    scores.append(normalize(contract.sla_uptime, 99.0, 99.999, True))
    scores.append(normalize(contract.sla_penalty_rate, 0.01, 0.30, True))
    scores.append(normalize(contract.liability_cap, 0.25, 3.0, True))

    ip_score_map = {
        IPOwnershipType.CLIENT: 1.0,
        IPOwnershipType.JOINT: 0.5,
        IPOwnershipType.VENDOR: 0.0,
    }
    scores.append(ip_score_map[contract.ip_ownership])

    scores.append(normalize(contract.termination_days, 14, 180, False))

    return sum(scores) / len(scores)


def compute_score(
    contract: ContractClauses,
    steps_used: int,
    max_steps: int,
    deal_closed: bool,
) -> dict[str, Any]:
    clause_score = score_clauses(contract)
    efficiency_score = 1 - (steps_used / max_steps)
    completion_bonus = 1.0 if deal_closed else 0.0

    final_score = (
        0.6 * clause_score
        + 0.2 * efficiency_score
        + 0.2 * completion_bonus
    )

    return {
        "final_score": round(final_score, 4),
        "clause_score": round(clause_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "completion_bonus": completion_bonus,
    }
