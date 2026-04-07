import json
import os
from typing import Any

import requests
from openai import OpenAI


# Mandatory env vars
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# API_KEY chain for flexibility
API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN or os.getenv("API_KEY")

# Internal constants (not env vars)
TEMPERATURE = 0.0
MAX_TOKENS = 500
INFERENCE_SEED = 42
REQUEST_TIMEOUT = 30
INFERENCE_MAX_STEPS = 10
END_SCORE_DECIMALS = 3

llm_client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
) if API_KEY else None


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=procureneg model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None = None,
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.{END_SCORE_DECIMALS}f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(observation: dict[str, Any]) -> str:
    step_count = int(observation.get("step_count", 0))
    max_steps = int(observation.get("max_steps", 0))
    steps_remaining = max(max_steps - step_count, 0)
    current_offer = observation.get("current_offer")
    counterparty_offer = observation.get("counterparty_offer")
    counterparty_has_offer = counterparty_offer is not None

    current_fee = (
        current_offer.get("annual_fee")
        if isinstance(current_offer, dict)
        else None
    )
    counterparty_fee = (
        counterparty_offer.get("annual_fee")
        if isinstance(counterparty_offer, dict)
        else None
    )
    fee_difference = (
        abs(current_fee - counterparty_fee)
        if isinstance(current_fee, (int, float)) and isinstance(counterparty_fee, (int, float))
        else None
    )

    return (
        "You are negotiating a procurement contract.\n"
        "Choose the next best structured action for the buyer.\n"
        "ACCEPTANCE RULE:\n"
        "If counterparty_offer exists AND step >= max_steps * 0.6 AND\n"
        "counterparty annual_fee is within 25% of your current_offer annual_fee,\n"
        "you SHOULD call accept to close the deal.\n"
        "If you are in the final 2 steps and any counterparty offer exists,\n"
        "you MUST call accept.\n\n"
        "WHEN TO USE EACH ACTION:\n"
        '- anchor: ONLY on step 0 as opening move\n'
        '- propose/counter: when you want to move toward agreement\n'
        '- concede: when you want to signal flexibility\n'
        '- package_trade: when offers are close on most clauses\n'
        '- accept: when counterparty offer is acceptable OR you are running out of steps\n'
        '- walkaway: NEVER use this unless deal is clearly impossible\n\n'
        f"CURRENT STEP: {step_count}\n"
        f"MAX STEPS: {max_steps}\n"
        f"STEPS REMAINING: {steps_remaining}\n"
        f"COUNTERPARTY OFFER EXISTS: {counterparty_has_offer}\n\n"
        "YOUR CURRENT OFFER:\n"
        f"  annual_fee: {current_fee}\n\n"
        "COUNTERPARTY OFFER:\n"
        f"  annual_fee: {counterparty_fee}\n\n"
        f"DIFFERENCE: {fee_difference}\n"
        f"STEPS REMAINING: {steps_remaining}\n\n"
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


def is_close(
    current_offer: dict[str, Any] | None,
    counterparty_offer: dict[str, Any] | None,
    max_steps: int,
) -> bool:
    if current_offer is None or counterparty_offer is None:
        return False

    fee_diff = abs(
        current_offer["annual_fee"] -
        counterparty_offer["annual_fee"]
    )

    fee_relative = fee_diff / counterparty_offer["annual_fee"]

    tight = max_steps <= 8
    threshold = 0.25 if not tight else 0.12

    return fee_relative < threshold


def is_stuck(history: list[dict[str, Any]]) -> bool:
    if len(history) < 2:
        return False

    last = history[-1].get("offer", {})
    prev = history[-2].get("offer", {})

    return last.get("annual_fee") == prev.get("annual_fee")


def build_offer(
    counterparty_offer: dict[str, Any] | None,
    aggressiveness: float,
    max_steps: int = 10,
) -> dict[str, Any] | None:
    if not counterparty_offer:
        return None

    tight = max_steps <= 8
    min_aggressiveness = 0.35 if tight else 0.25
    effective_aggressiveness = max(aggressiveness, min_aggressiveness)
    factor = 1.0 - (0.6 * effective_aggressiveness)

    return {
        "annual_fee": round(counterparty_offer["annual_fee"] * factor),
        "payment_terms": counterparty_offer["payment_terms"],
        "duration_years": counterparty_offer["duration_years"],
        "sla_uptime": counterparty_offer["sla_uptime"],
        "sla_penalty_rate": counterparty_offer["sla_penalty_rate"],
        "liability_cap": counterparty_offer["liability_cap"],
        "ip_ownership": counterparty_offer["ip_ownership"],
        "termination_days": counterparty_offer["termination_days"],
    }


def bootstrap_offer() -> dict[str, Any]:
    return {
        "annual_fee": 500000,
        "payment_terms": 45,
        "duration_years": 3,
        "sla_uptime": 99.5,
        "sla_penalty_rate": 0.08,
        "liability_cap": 1.2,
        "ip_ownership": "joint",
        "termination_days": 60,
    }


def fallback_policy(step: int, observation: dict[str, Any]) -> dict[str, Any]:
    progress = step / observation["max_steps"]
    aggressiveness = 1.0 - progress
    max_steps = observation["max_steps"]
    current_offer = observation.get("current_offer")
    counterparty_offer = observation.get("counterparty_offer")
    history = observation.get("negotiation_history", [])

    if progress < 0.3:
        action_type = "anchor" if step == 0 else "propose"
        offer = build_offer(
            counterparty_offer,
            aggressiveness,
            max_steps=observation["max_steps"],
        )
    elif progress < 0.7:
        if is_close(current_offer, counterparty_offer, max_steps):
            action_type = "accept"
            offer = None
        elif is_stuck(history):
            action_type = "concede"
            offer = build_offer(
                counterparty_offer,
                aggressiveness,
                max_steps=observation["max_steps"],
            )
        else:
            action_type = "propose"
            offer = build_offer(
                counterparty_offer,
                aggressiveness,
                max_steps=observation["max_steps"],
            )
    else:
        if is_close(current_offer, counterparty_offer, max_steps):
            action_type = "accept"
            offer = None
        else:
            action_type = "concede"
            offer = build_offer(
                counterparty_offer,
                aggressiveness,
                max_steps=observation["max_steps"],
            )

    if offer is None and action_type != "accept":
        offer = current_offer or bootstrap_offer()

    if action_type == "accept":
        return {"action": action_type}

    return {
        "action": action_type,
        "offer": offer,
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
    log_start(task=task, model=MODEL_NAME)

    rewards_list: list[float] = []
    steps_taken = 0
    last_observation: dict[str, Any] | None = None
    final_reward = 0.0
    episode_error: str | None = None

    observation = reset_env(task)
    done = False
    step = 0

    try:
        while not done and step < INFERENCE_MAX_STEPS:
            prompt = build_prompt(observation)
            try:
                model_output = call_model(prompt)
                parsed_action = extract_json_object(model_output)
                action = normalize_action(parsed_action, step, observation)
            except Exception:
                action = fallback_policy(step, observation)

            try:
                action, result = execute_action(action, step, observation)
                observation = result["observation"]
                last_observation = observation
                done = result["done"]
                final_reward = result["reward"]
                rewards_list.append(final_reward)
                steps_taken = step + 1
                log_step(
                    step=steps_taken,
                    action=action["action"],
                    reward=final_reward,
                    done=done,
                    error=None,
                )
                step += 1
            except Exception as exc:
                episode_error = str(exc)
                done = True
                steps_taken = step + 1
                log_step(
                    step=steps_taken,
                    action=action.get("action", "unknown"),
                    reward=final_reward,
                    done=True,
                    error=episode_error,
                )
                break
    finally:
        raw_score = rewards_list[-1] if rewards_list else 0.0
        clamped_score = max(0.0, min(1.0, raw_score))
        margin = 10 ** (-END_SCORE_DECIMALS)
        final_score = min(1.0 - margin, max(margin, clamped_score))
        success = final_score > 0.1
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards_list,
        )

    final_contract = None
    if last_observation is not None:
        final_contract = (
            last_observation.get("counterparty_offer")
            or last_observation.get("current_offer")
        )
    return {
        "task": task,
        "steps": steps_taken,
        "reward": final_reward,
        "final_contract": final_contract,
        "error": episode_error,
    }


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)
