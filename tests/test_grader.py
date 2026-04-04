from server.grader import compute_score
from server.environment import ProcureNegEnv
from server.models import Action, ActionType, ContractClauses


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


def test_reward_stays_within_declared_range() -> None:
    env = ProcureNegEnv()
    env.reset("medium")

    first = env.step(
        Action(
            action_type=ActionType.ANCHOR,
            offer=ContractClauses(
                annual_fee=700000,
                payment_terms=45,
                duration_years=3,
                sla_uptime=99.5,
                sla_penalty_rate=0.05,
                liability_cap=1.0,
                ip_ownership="joint",
                termination_days=60,
            ),
        )
    )
    second = env.step(
        Action(
            action_type=ActionType.ANCHOR,
            offer=ContractClauses(
                annual_fee=720000,
                payment_terms=40,
                duration_years=3,
                sla_uptime=99.4,
                sla_penalty_rate=0.05,
                liability_cap=1.0,
                ip_ownership="joint",
                termination_days=60,
            ),
        )
    )

    assert -0.1 <= first.reward <= 1.0
    assert -0.1 <= second.reward <= 1.0


def test_hard_is_harder_than_easy() -> None:
    offer = ContractClauses(
        annual_fee=800000,
        payment_terms=40,
        duration_years=3,
        sla_uptime=99.5,
        sla_penalty_rate=0.05,
        liability_cap=1.0,
        ip_ownership="joint",
        termination_days=60,
    )

    easy_env = ProcureNegEnv()
    hard_env = ProcureNegEnv()
    easy_env.reset("easy")
    hard_env.reset("hard")

    easy_result = easy_env.step(Action(action_type=ActionType.PROPOSE, offer=offer))
    hard_result = hard_env.step(Action(action_type=ActionType.PROPOSE, offer=offer))

    easy_counter = easy_result.observation.counterparty_offer
    hard_counter = hard_result.observation.counterparty_offer

    assert easy_counter is not None
    assert hard_counter is not None
    assert hard_counter.annual_fee > easy_counter.annual_fee
