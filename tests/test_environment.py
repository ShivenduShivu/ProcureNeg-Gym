from server.environment import ProcureNegEnv
from server.models import Action, ActionType, ContractClauses


def make_offer(
    *,
    annual_fee: float = 700000,
    payment_terms: int = 45,
    duration_years: int = 3,
    sla_uptime: float = 99.5,
    sla_penalty_rate: float = 0.05,
    liability_cap: float = 1.0,
    ip_ownership: str = "joint",
    termination_days: int = 60,
) -> ContractClauses:
    return ContractClauses(
        annual_fee=annual_fee,
        payment_terms=payment_terms,
        duration_years=duration_years,
        sla_uptime=sla_uptime,
        sla_penalty_rate=sla_penalty_rate,
        liability_cap=liability_cap,
        ip_ownership=ip_ownership,
        termination_days=termination_days,
    )


def test_reset() -> None:
    env = ProcureNegEnv()
    obs = env.reset("easy")

    assert obs.step_count == 0
    assert obs.max_steps > 0
    assert obs.constraints["max_steps"] == obs.max_steps


def test_repeat_offer_reduces_flexibility() -> None:
    env = ProcureNegEnv()
    env.reset("medium")
    offer = make_offer()

    env.step(Action(action_type=ActionType.PROPOSE, offer=offer))
    first_flexibility = env.counterparty.flexibility

    env.step(Action(action_type=ActionType.PROPOSE, offer=offer))

    assert env.counterparty.flexibility < first_flexibility


def test_deterministic_episode_same_rewards() -> None:
    actions = [
        Action(action_type=ActionType.ANCHOR, offer=make_offer(annual_fee=650000, payment_terms=50)),
        Action(action_type=ActionType.PACKAGE_TRADE, offer=make_offer(annual_fee=620000, payment_terms=55, sla_uptime=99.6)),
        Action(action_type=ActionType.CONCEDE, offer=make_offer(annual_fee=780000, payment_terms=35, ip_ownership="vendor")),
    ]

    env_one = ProcureNegEnv()
    env_two = ProcureNegEnv()
    env_one.reset("medium")
    env_two.reset("medium")

    rewards_one = [env_one.step(action).reward for action in actions]
    rewards_two = [env_two.step(action).reward for action in actions]

    assert rewards_one == rewards_two


def test_reset_clears_state() -> None:
    env = ProcureNegEnv()
    env.reset("hard")
    env.step(Action(action_type=ActionType.PROPOSE, offer=make_offer(annual_fee=1300000, payment_terms=20)))

    obs = env.reset("easy")

    assert obs.step_count == 0
    assert obs.negotiation_history == []
    assert obs.current_offer is None
    assert obs.counterparty_offer is None


def test_concede_improves_counterparty_offer() -> None:
    baseline_env = ProcureNegEnv()
    concede_env = ProcureNegEnv()
    baseline_env.reset("medium")
    concede_env.reset("medium")

    offer = make_offer(annual_fee=760000, payment_terms=40, duration_years=3, sla_uptime=99.5)

    baseline_result = baseline_env.step(
        Action(action_type=ActionType.PROPOSE, offer=offer)
    )
    concede_result = concede_env.step(
        Action(action_type=ActionType.CONCEDE, offer=offer)
    )

    baseline_counter = baseline_result.observation.counterparty_offer
    concede_counter = concede_result.observation.counterparty_offer

    assert baseline_counter is not None
    assert concede_counter is not None
    assert concede_counter.annual_fee <= baseline_counter.annual_fee
