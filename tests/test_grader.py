from server.grader import compute_score
from server.models import ContractClauses


def test_grader_runs() -> None:
    contract = ContractClauses(
        annual_fee=100000,
        payment_terms=30,
        duration_years=2,
        sla_uptime=99.5,
        sla_penalty_rate=0.05,
        liability_cap=1.0,
        ip_ownership="client",
        termination_days=30,
    )

    result = compute_score(contract, 1, 10, True)

    assert "final_score" in result
    assert result["final_score"] >= 0
