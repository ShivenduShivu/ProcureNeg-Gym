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
        self.step_count = 0
        self.done = False
        self.history: list[Action] = []
        self.current_offer: Optional[ContractClauses] = None
        self.counterparty_offer: Optional[ContractClauses] = None
        self.counterparty: Optional[Counterparty] = None

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
        if self.counterparty is None:
            raise RuntimeError("Environment not initialized. Call reset first")

        if self.done:
            raise RuntimeError("Episode already completed")

        previous_offer = self.current_offer
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
            if action.offer is not None and previous_offer is not None:
                if self._is_better_offer(action.offer, previous_offer):
                    reward += 0.05
                else:
                    reward -= 0.02

            if action.action_type == ActionType.CONCEDE and self.counterparty is not None:
                self.counterparty.flexibility = min(1.0, self.counterparty.flexibility * 1.15)

            if action.action_type == ActionType.COUNTER and self.counterparty is not None:
                self.counterparty.flexibility = min(1.0, self.counterparty.flexibility * 1.05)

            if action.action_type == ActionType.ANCHOR:
                reward -= 0.01

            if (
                action.action_type == ActionType.PACKAGE_TRADE
                and action.offer is not None
                and previous_offer is not None
            ):
                improvements = 0

                if action.offer.annual_fee < previous_offer.annual_fee:
                    improvements += 1
                if action.offer.sla_uptime > previous_offer.sla_uptime:
                    improvements += 1
                if action.offer.sla_penalty_rate > previous_offer.sla_penalty_rate:
                    improvements += 1
                if action.offer.payment_terms > previous_offer.payment_terms:
                    improvements += 1

                if improvements >= 2:
                    reward += 0.04
                else:
                    reward -= 0.01

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

        reward = max(-0.1, min(1.0, reward))

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

    def _is_better_offer(
        self,
        new_offer: ContractClauses,
        previous_offer: ContractClauses,
    ) -> bool:
        lower_fee = new_offer.annual_fee < previous_offer.annual_fee
        better_sla = (
            new_offer.sla_uptime > previous_offer.sla_uptime
            or new_offer.sla_penalty_rate > previous_offer.sla_penalty_rate
        )
        return lower_fee or better_sla


def load_task(task_name: str) -> dict[str, Any]:
    path = Path(__file__).parent / "tasks" / f"{task_name}.yaml"

    with path.open("r", encoding="utf-8") as task_file:
        return yaml.safe_load(task_file)
