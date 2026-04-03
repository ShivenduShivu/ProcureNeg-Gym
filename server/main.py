from fastapi import FastAPI, HTTPException

from server.environment import ProcureNegEnv
from server.models import Action, Observation, StepResult


app = FastAPI(title="ProcureNeg-Gym API")

env = ProcureNegEnv()


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    """
    Start a new negotiation episode.
    """
    return env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """
    Take one step in the environment.
    """
    try:
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict[str, int | bool]:
    """
    Get public environment state.
    """
    return env.state()
