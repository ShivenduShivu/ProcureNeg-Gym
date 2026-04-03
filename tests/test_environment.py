from server.environment import ProcureNegEnv


def test_reset() -> None:
    env = ProcureNegEnv()
    obs = env.reset("easy")

    assert obs.step_count == 0
    assert obs.max_steps > 0
    assert obs.constraints["max_steps"] == obs.max_steps
