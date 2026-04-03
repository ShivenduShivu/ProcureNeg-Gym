from pathlib import Path
from typing import Any, Optional

import yaml

from server.counterparty import Counterparty
from server.grader import compute_score
from server.models import (
    Action,
    ActionType,
    ContractClauses,
    Observation,
    StepResult,
)


class ProcureNegEnv:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.reset()

    def reset(self, task_name: str = "medium") -> Observation:
        task = load_task(task_name)

        self.max_steps = task["max_steps"]
        self.step_count = 0
        self.done = False
        self.history: list[Action] = []
        self.current_offer: Optional[ContractClauses] = None
        self.counterparty_offer: Optional[ContractClauses] = None

        reservation = ContractClauses(**task["counterparty"]["reservation"])
        flexibility = task["counterparty"]["flexibility"]

        self.counterparty = Counterparty(reservation, flexibility)
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode already completed")

        self.step_count += 1
        self.history.append(action)
        reward = 0.0

        if action.action_type in {
            ActionType.PROPOSE,
            ActionType.COUNTER,
            ActionType.CONCEDE,
            ActionType.PACKAGE_TRADE,
            ActionType.ANCHOR,
        }:
            self.current_offer = action.offer

            cp_action, cp_offer = self.counterparty.respond(
                action.action_type,
                action.offer,
            )
            self.counterparty_offer = cp_offer

            if cp_action == ActionType.ACCEPT:
                self.done = True
                reward = self._compute_reward(cp_offer)

        elif action.action_type == ActionType.ACCEPT:
            if self.counterparty_offer is None:
                raise ValueError("No counterparty offer available to accept")

            self.done = True
            reward = self._compute_reward(self.counterparty_offer)

        elif action.action_type == ActionType.WALKAWAY:
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info={"step": self.step_count},
        )

    def state(self) -> dict[str, int | bool]:
        return {
            "step_count": self.step_count,
            "done": self.done,
        }

    def _get_observation(self) -> Observation:
        return Observation(
            current_offer=self.current_offer,
            counterparty_offer=self.counterparty_offer,
            negotiation_history=self.history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            constraints={"max_steps": self.max_steps},
        )

    def _compute_reward(self, contract: ContractClauses) -> float:
        result = compute_score(
            contract,
            steps_used=self.step_count,
            max_steps=self.max_steps,
            deal_closed=True,
        )
        return result["final_score"]


def load_task(task_name: str) -> dict[str, Any]:
    path = Path(__file__).parent / "tasks" / f"{task_name}.yaml"

    with path.open("r", encoding="utf-8") as task_file:
        return yaml.safe_load(task_file)
