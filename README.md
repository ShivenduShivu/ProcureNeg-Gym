---
title: ProcureNeg-Gym
emoji: "🤝"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /docs
fullWidth: true
pinned: false
short_description: OpenEnv-compatible procurement negotiation API.
---

# ProcureNeg-Gym

ProcureNeg-Gym is a deterministic procurement negotiation environment built for agent benchmarking and OpenEnv-style evaluation.

## What It Does

The environment simulates buyer-vs-seller procurement negotiation with:

- structured contract clauses
- deterministic counterparty behavior
- deterministic grading
- YAML-configured difficulty levels
- a deployable FastAPI/OpenEnv-style API

## Action Space

Agent actions are structured and validated. Core actions used in the live API flow are:

- `propose`
- `counter`
- `accept`
- `walkaway`

Each proposal carries a contract offer with these clauses:

- `annual_fee`
- `payment_terms`
- `duration_years`
- `sla_uptime`
- `sla_penalty_rate`
- `liability_cap`
- `ip_ownership`
- `termination_days`

## Observation Space

Each environment step returns only public state:

- `current_offer`
- `counterparty_offer`
- `negotiation_history`
- `step_count`
- `max_steps`
- `constraints`

Hidden counterparty reservation values are never exposed through the API.

## Tasks

Difficulty is loaded from YAML task files in `server/tasks/`:

- `easy`: more flexible counterparty, `max_steps = 12`
- `medium`: balanced negotiation, `max_steps = 10`
- `hard`: tougher counterparty, `max_steps = 8`

## API

The deployed service exposes these primary endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

Example reset request:

```json
{
  "task_name": "easy"
}
```

Supported tasks are `easy`, `medium`, and `hard`.

## Local Run

Run the API locally with:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then open `/docs` to explore and test the API.

## Deployment

The repository is configured for Docker and Hugging Face Spaces.

Local Docker build:

```bash
docker build -t procureneg-gym .
docker run -p 7860:7860 procureneg-gym
```

## Hugging Face Space

This repository is configured for a Docker Space. The Space should listen on port `7860`, which matches the README metadata and Dockerfile.

After pushing to the Hugging Face Space remote, verify the deployment with:

```bash
curl -X POST https://YOUR-SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```
