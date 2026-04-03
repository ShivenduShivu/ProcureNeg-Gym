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
short_description: Deterministic procurement negotiation environment with OpenEnv-compatible API endpoints.
---

# ProcureNeg-Gym

ProcureNeg-Gym is a deterministic procurement negotiation environment built for agent benchmarking and OpenEnv-style evaluation.

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

## Hugging Face Space

This repository is configured for a Docker Space. The Space should listen on port `7860`, which matches the README metadata and Dockerfile.

After pushing to the Hugging Face Space remote, verify the deployment with:

```bash
curl -X POST https://YOUR-SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```
