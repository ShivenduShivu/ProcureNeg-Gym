import os
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from server.environment import ProcureNegEnv
from server.models import Action, Observation, StepResult


app = FastAPI(title="ProcureNeg-Gym OpenEnv API")

env: ProcureNegEnv | None = None
env_lock = Lock()


class ResetRequest(BaseModel):
    task_name: Literal["easy", "medium", "hard"] = "medium"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    global env
    with env_lock:
        env = ProcureNegEnv()
        return env.reset(request.task_name)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    with env_lock:
        if env is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )

        if env.done:
            raise HTTPException(
                status_code=400,
                detail="Episode finished. Call /reset.",
            )

        try:
            return env.step(action)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict[str, int | bool]:
    with env_lock:
        if env is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized.",
            )

        return env.state()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "name": "procureneg-gym",
        "version": "1.0.0",
        "description": "Enterprise procurement contract negotiation RL environment",
        "author": "Shivendu Shivu",
        "tags": ["procurement", "negotiation", "enterprise", "openenv"],
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
            "range": [-0.1, 1.0],
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
