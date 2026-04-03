from server.models import Action, ActionType


def test_action_validation() -> None:
    action = Action(action_type=ActionType.ACCEPT)

    assert action.action_type == ActionType.ACCEPT
