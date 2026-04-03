from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    PROPOSE = "propose"
    COUNTER = "counter"
    ACCEPT = "accept"
    ACCEPT_PACKAGE = "accept_package"
    PROBE = "probe"
    ANCHOR = "anchor"
    CONCEDE = "concede"
    PACKAGE_TRADE = "package_trade"
    WALKAWAY = "walkaway"


class IPOwnershipType(str, Enum):
    VENDOR = "vendor"
    JOINT = "joint"
    CLIENT = "client"


class ContractClauses(BaseModel):
    annual_fee: float = Field(..., ge=100000, le=2000000)
    payment_terms: int = Field(..., ge=15, le=90)
    duration_years: int = Field(..., ge=1, le=5)
    sla_uptime: float = Field(..., ge=99.0, le=99.999)
    sla_penalty_rate: float = Field(..., ge=0.01, le=0.30)
    liability_cap: float = Field(..., ge=0.25, le=3.0)
    ip_ownership: IPOwnershipType
    termination_days: int = Field(..., ge=14, le=180)


class Action(BaseModel):
    action_type: ActionType
    offer: Optional[ContractClauses] = None
    message: Optional[str] = None

    @model_validator(mode="after")
    def validate_offer(self) -> "Action":
        offer_required = {
            ActionType.PROPOSE,
            ActionType.COUNTER,
            ActionType.CONCEDE,
            ActionType.PACKAGE_TRADE,
            ActionType.ANCHOR,
        }

        if self.action_type in offer_required and self.offer is None:
            raise ValueError(f"{self.action_type.value} requires an offer")

        return self


class Observation(BaseModel):
    current_offer: Optional[ContractClauses] = None
    counterparty_offer: Optional[ContractClauses] = None
    negotiation_history: list[Action] = Field(default_factory=list)
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    constraints: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
