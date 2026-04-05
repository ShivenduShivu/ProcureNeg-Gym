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

Agent actions are structured and validated. Supported actions are:

- `propose`
- `counter`
- `accept`
- `anchor`
- `concede`
- `package_trade`
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

## Reward Design

Final reward is deterministic and computed by the grader from:

- clause quality
- negotiation efficiency
- deal completion

The environment also adds light intermediate shaping during negotiation:

- small positive reward for improved offers
- small penalty for worse offers

This keeps the system deterministic while making step-by-step behavior less sparse.

The declared reward range for the API is `[-0.1, 1.0]`:

- terminal rewards come from the deterministic grader
- intermediate shaping can be slightly negative for strategically weak actions

### Grader Formula
final_score = (0.6 × clause_score)
+ (0.2 × efficiency_score)
+ (0.2 × completion_bonus)
clause_score     = weighted sum of 8 normalized clauses
efficiency_score = 1 - (steps_used / max_steps)
completion_bonus = 1.0 if deal closed, else 0.0

Clause weights:

| Clause            | Weight |
|-------------------|-------:|
| annual_fee        |   0.28 |
| payment_terms     |   0.14 |
| sla_uptime        |   0.14 |
| sla_penalty_rate  |   0.14 |
| liability_cap     |   0.10 |
| duration_years    |   0.10 |
| termination_days  |   0.06 |
| ip_ownership      |   0.04 |

Weights reflect business importance.
annual_fee carries the highest weight as the primary
financial outcome. ip_ownership carries the lowest
as it is typically non-negotiable in most scenarios.

## Determinism

The environment is designed to be fully deterministic:

- task configuration comes from static YAML files
- the counterparty uses fixed rules, not randomness
- the grader is a pure scoring function
- the fallback inference policy follows deterministic observation-based rules

Given the same task and the same action sequence, the environment produces the same trajectory and score.

## Counterparty Behavior

The counterparty is rule-based and deterministic.

It evaluates each incoming offer against hidden reservation values and generates structured counteroffers. It also adjusts its concession flexibility based on:

- `anchor` actions
- `concede` actions
- repeated offers
- `counter` actions

There is no randomness in counterparty behavior.

## Environment Design Note

The API is intentionally single-session for hackathon evaluation:

- one active environment is created on each `POST /reset`
- `POST /step` operates on that active episode
- hidden seller state is kept server-side

This keeps the evaluation lifecycle simple and predictable for external validation.

## Reproducibility

The included `inference.py` runner is API-driven and includes a deterministic fallback policy.

This means the system can still produce repeatable demonstrations even when no model API key is configured.

## Baseline Results

### Deterministic Fallback (no LLM required)
Rule-based policy. Reproducible with `temperature=0.0`, `seed=42`.

| Task   | Steps | Score  | Deal |
|--------|------:|-------:|------|
| easy   |    10 | 0.5007 | Yes  |
| medium |     8 | 0.4756 | Yes  |
| hard   |     8 | 0.0500 | No   |

### LLM Agent Baseline (`gpt-4o-mini`)
Strategic LLM agent. Shows environment headroom above fallback.

| Task   | Steps | Score  | Deal |
|--------|------:|-------:|------|
| easy   |     3 | 0.6090 | Yes  |
| medium |     2 | 0.6089 | Yes  |
| hard   |     3 | 0.5154 | Yes  |

Key observations:
- LLM scores 20%+ higher than fallback on easy/medium.
- LLM closes hard task; fallback cannot, showing strategic reasoning matters.
- Hard remains the lowest LLM score, confirming difficulty progression.
- Environment rewards genuine negotiation strategy.

### Deterministic Inference

We enforce reproducibility in the model call path with:

- `temperature = 0.0`
- `seed = 42`

Fallback policy ensures a fully deterministic baseline even without LLM access.

## What Makes This Challenging

The negotiation problem is intentionally structured to create non-trivial decision pressure:

- hidden reservation values mean the agent must infer seller limits indirectly
- multiple contract clauses create real tradeoffs instead of a single optimization target
- buyer-friendly improvements can conflict with deal completion
- action sequencing matters because anchors, concessions, and package trades affect negotiation dynamics

## What Makes This Different

This environment is built for evaluation, not free-form roleplay:

- fully deterministic grading
- structured action space instead of open-ended chat
- real contract clauses with bounded values
- reproducible benchmark behavior across tasks

## Why This Matters

This project is designed to be useful beyond a toy simulation:

- deterministic evaluation makes runs comparable
- procurement-style contract negotiation is enterprise-relevant
- reproducibility supports benchmarking and debugging
- structured scoring makes it easier to evaluate AI agent quality

## Known Limitations

- Single-session design: one active episode per server
  instance. Not suitable for concurrent multi-user
  evaluation without modification.
- Counterparty preferences are fully active for
  annual_fee, payment_terms, and sla_uptime. Other
  clauses use generic concession rules.
- LLM inference reproducibility depends on provider
  determinism at temperature=0.0 and seed=42.
- Models procurement negotiation at benchmark level.
  Does not capture full legal/commercial nuance of
  real enterprise contracts.

## API

The deployed service exposes these primary endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`
- `GET /schema`

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

## Running Inference

### Fallback only (no LLM, fully deterministic)
```bash
export API_BASE_URL="http://127.0.0.1:7860"
python inference.py
```

No API key required. Uses deterministic fallback policy.
Produces reproducible baseline scores.

### Local environment with LLM
```bash
export API_BASE_URL="http://127.0.0.1:7860"
export LLM_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

### Against live HF Space with LLM
```bash
export API_BASE_URL="https://starwarrior24x7-procureneg-gym.hf.space"
export LLM_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

Note: API_BASE_URL controls where the environment is running
(local or HF Space). LLM baseline results in this README
were produced with MODEL_NAME=gpt-4o-mini against the
local API at http://127.0.0.1:7860.

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
curl -X POST https://starwarrior24x7-procureneg-gym.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```
