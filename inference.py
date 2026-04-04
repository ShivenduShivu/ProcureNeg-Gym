import json
import os
from typing import Any

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_API_BASE = os.getenv("API_BASE_URL_LLM", "https://router.huggingface.co/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
INFERENCE_SEED = int(os.getenv("INFERENCE_SEED", "42"))

llm_client = OpenAI(
    api_key=API_KEY,
    base_url=MODEL_API_BASE,
) if API_KEY else None


def build_prompt(observation: dict[str, Any]) -> str:
    return (
        "You are negotiating a procurement contract.\n"
        "Choose the next best structured action for the buyer.\n"
        "You must use strategic actions:\n"
        '- Use "anchor" at the beginning to set a strong position\n'
        '- Use "concede" when negotiation stalls\n'
        '- Use "package_trade" when improving multiple terms together\n'
        "Do not always use the same action.\n"
        "Balance between aggressive and cooperative strategies.\n"
        "Return ONLY valid JSON.\n\n"
        f"Observation:\n{json.dumps(observation, indent=2, sort_keys=True)}\n\n"
        "Allowed actions: propose, counter, accept, anchor, concede, package_trade, walkaway.\n"
        "If you return propose, counter, anchor, concede, or package_trade, "
        "include a full offer object with these fields:\n"
        "annual_fee, payment_terms, duration_years, sla_uptime, sla_penalty_rate, "
        "liability_cap, ip_ownership, termination_days.\n"
        'Valid ip_ownership values: "vendor", "joint", "client".\n'
        "Prefer efficient deals. Avoid invalid values.\n"
        'Examples:\n{"action":"accept"}\n'
        '{"action":"propose","offer":{"annual_fee":700000,"payment_terms":45,'
        '"duration_years":3,"sla_uptime":99.5,"sla_penalty_rate":0.05,'
        '"liability_cap":1.0,"ip_ownership":"joint","termination_days":60}}'
    )


def call_model(prompt: str) -> str:
    if llm_client is None:
        raise RuntimeError("No API key configured for the OpenAI client")

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful negotiation agent. "
                    "Always return strict JSON with no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        seed=INFERENCE_SEED,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return response.choices[0].message.content or ""


def extract_json_object(response_text: str) -> dict[str, Any]:
    response_text = response_text.strip()
    if not response_text:
        raise ValueError("Empty model response")

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")

    return json.loads(response_text[start : end + 1])


def fallback_policy(step: int, observation: dict[str, Any]) -> dict[str, Any]:
    counterparty_offer = observation.get("counterparty_offer")
    max_steps = observation.get("max_steps", MAX_STEPS)

    if counterparty_offer and step >= max_steps - 2:
        return {"action": "accept"}

    ip_ownership = "client"
    if step >= 4:
        ip_ownership = "vendor"
    elif step >= 2:
        ip_ownership = "joint"

    if step == 0:
        action_type = "anchor"
    elif step == 2:
        action_type = "package_trade"
    elif step >= 4:
        action_type = "concede"
    else:
        action_type = "propose"

    return {
        "action": action_type,
        "offer": {
            "annual_fee": min(2000000, 400000 + step * 120000),
            "payment_terms": max(15, 60 - step * 6),
            "duration_years": min(5, 1 + (step // 2)),
            "sla_uptime": max(99.0, round(99.95 - step * 0.165, 3)),
            "sla_penalty_rate": max(0.01, round(0.12 - step * 0.015, 3)),
            "liability_cap": max(0.25, round(1.8 - step * 0.15, 2)),
            "ip_ownership": ip_ownership,
            "termination_days": min(180, 30 + step * 10),
        },
    }


def normalize_action(action: dict[str, Any], step: int, observation: dict[str, Any]) -> dict[str, Any]:
    chosen_action = str(action.get("action", "")).strip().lower()
    allowed_actions = {
        "propose",
        "counter",
        "accept",
        "anchor",
        "concede",
        "package_trade",
        "walkaway",
    }
    if chosen_action not in allowed_actions:
        return fallback_policy(step, observation)

    normalized: dict[str, Any] = {"action": chosen_action}
    if chosen_action in {"propose", "counter", "anchor", "concede", "package_trade"}:
        offer = action.get("offer")
        if not isinstance(offer, dict):
            return fallback_policy(step, observation)
        normalized["offer"] = offer

    return normalized


def post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def reset_env(task: str) -> dict[str, Any]:
    return post_json("/reset", {"task_name": task})


def send_action(action: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "action_type": action["action"],
        "offer": action.get("offer"),
    }
    return post_json("/step", payload)


def execute_action(
    action: dict[str, Any],
    step: int,
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return action, send_action(action)
    except requests.RequestException:
        fallback_action = fallback_policy(step, observation)
        return fallback_action, send_action(fallback_action)


def run_episode(task: str) -> dict[str, Any]:
    observation = reset_env(task)
    done = False
    step = 0
    final_reward = 0.0

    while not done and step < MAX_STEPS:
        prompt = build_prompt(observation)
        try:
            model_output = call_model(prompt)
            parsed_action = extract_json_object(model_output)
            action = normalize_action(parsed_action, step, observation)
        except Exception:
            action = fallback_policy(step, observation)

        action, result = execute_action(action, step, observation)
        observation = result["observation"]
        done = result["done"]
        final_reward = result["reward"]

        print(
            f"[{task}] step={step} action={action['action']} "
            f"reward={final_reward:.4f}"
        )
        step += 1

    final_contract = observation.get("counterparty_offer") or observation.get("current_offer")
    return {
        "task": task,
        "steps": step,
        "reward": final_reward,
        "final_contract": final_contract,
    }


if __name__ == "__main__":
    results = [run_episode(task) for task in ["easy", "medium", "hard"]]
    print("\nFINAL RESULTS:")
    for result in results:
        print(result)
