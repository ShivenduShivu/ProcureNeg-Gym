import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from server.environment import ProcureNegEnv
from server.models import Action, Observation, StepResult


app = FastAPI(title="ProcureNeg-Gym OpenEnv API")

env = ProcureNegEnv()


class ResetRequest(BaseModel):
    task_name: Literal["easy", "medium", "hard"] = "medium"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest | None = None) -> Observation:
    task_name = "medium" if request is None else request.task_name
    return env.reset(task_name)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict[str, int | bool]:
    return env.state()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, str | bool]:
    return {
        "name": "procureneg-gym",
        "version": "1.0",
        "description": "Deterministic procurement negotiation RL environment",
        "author": "your-team-name",
        "task_type": "negotiation",
        "deterministic": True,
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": {
            "type": "structured",
            "actions": [
                "propose",
                "counter",
                "accept",
                "anchor",
                "concede",
                "package_trade",
                "walkaway",
            ],
        },
        "observation": {
            "type": "partial",
            "fields": [
                "current_offer",
                "counterparty_offer",
                "negotiation_history",
                "step_count",
                "constraints",
            ],
        },
        "state": {
            "type": "public",
            "fields": [
                "step_count",
                "done",
            ],
        },
        "reward": {
            "type": "deterministic",
            "range": [0.0, 1.0],
        },
    }


@app.post("/mcp")
def mcp() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "status": "not_implemented",
            "note": "MCP interface placeholder",
        },
    }


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
