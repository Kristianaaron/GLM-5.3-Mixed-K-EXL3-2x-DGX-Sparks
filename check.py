#!/usr/bin/env python3
"""Small OpenAI-compatible API smoke test for the public serving recipe."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


TIMEOUT_SECONDS = 180
NEEDLE = "LANTERN-ORCHID-7319"


class CheckError(Exception):
    pass


def endpoint(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise CheckError(f"HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        reason = str(exc.reason).lower()
        if "refused" in reason:
            raise CheckError("connection refused") from None
        if "timed out" in reason or "timeout" in reason:
            raise CheckError("connection timed out") from None
        raise CheckError("connection failed or dropped") from None
    except (TimeoutError, ConnectionError, OSError):
        raise CheckError("connection failed or dropped") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckError("invalid JSON response") from None
    if not isinstance(result, dict):
        raise CheckError("response was not a JSON object")
    return result


def stream_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Read SSE events, returning deltas, usage, and timing without echoing data."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_delta: float | None = None
    last_event: float | None = None
    last_delta: float | None = None
    longest_gap = 0.0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, str]] = {}
    first_tool_name: float | None = None
    last_tool_arguments: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    fingerprint: str | None = None

    try:
        response = urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        raise CheckError(f"HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        reason = str(exc.reason).lower()
        if "refused" in reason:
            raise CheckError("connection refused") from None
        if "timed out" in reason or "timeout" in reason:
            raise CheckError("connection timed out") from None
        raise CheckError("connection failed or dropped") from None
    except (TimeoutError, ConnectionError, OSError):
        raise CheckError("connection failed or dropped") from None

    try:
        with response:
            for raw_line in response:
                now = time.monotonic()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    raise CheckError("invalid streaming JSON event") from None
                if not isinstance(event, dict):
                    continue
                if last_event is not None:
                    longest_gap = max(longest_gap, now - last_event)
                last_event = now
                if "system_fingerprint" in event:
                    fp = event.get("system_fingerprint")
                    fingerprint = None if fp is None else str(fp)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                textual = False
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    textual = True
                reasoning = delta.get("reasoning")
                if reasoning is None:
                    reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    textual = True
                calls = delta.get("tool_calls") or []
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    index = int(call.get("index", 0))
                    slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] += str(call["id"])
                    function = call.get("function") or {}
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        slot["name"] += name
                        first_tool_name = first_tool_name or now
                        textual = True
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        slot["arguments"] += arguments
                        last_tool_arguments = now
                        textual = True
                if textual and first_delta is None:
                    first_delta = now
                if textual:
                    last_delta = now
    except urllib.error.HTTPError as exc:
        raise CheckError(f"HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise CheckError("connection failed or dropped while streaming") from None
    except (TimeoutError, ConnectionError, OSError):
        raise CheckError("connection failed or dropped while streaming") from None

    elapsed = time.monotonic() - started
    # Stream event arrival timing is the closest standard-library measure of chunk gaps.
    # Events with no delta (such as usage) do not count as model output chunks.
    # First delta and finish are used to estimate output decode speed.
    text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    prompt_tokens, completion_tokens, cached_tokens = extract_usage(usage)
    ttft = None if first_delta is None else first_delta - started
    decode_seconds = None if first_delta is None or last_delta is None else max(0.0, last_delta - first_delta)
    metrics = {
        "elapsed": elapsed,
        "ttft": ttft,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "prefill_tps": rate(prompt_tokens, ttft),
        "decode_tps": rate(completion_tokens, decode_seconds),
        "finish_reason": finish_reason,
        "content": text,
        "reasoning": reasoning_text,
        "tool_calls": tool_calls,
        "first_tool_name": first_tool_name,
        "last_tool_arguments": last_tool_arguments,
        "fingerprint": fingerprint,
        "longest_gap": longest_gap,
    }
    return metrics


def extract_usage(usage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
        cached if isinstance(cached, int) else None,
    )


def rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def metrics_text(metrics: dict[str, Any], include_decode: bool = True) -> str:
    fields = [
        f"prefill={fmt(metrics.get('prefill_tps'), ' tok/s')}",
        f"TTFT={fmt(metrics.get('ttft'), ' s')}",
    ]
    if include_decode:
        fields.append(f"decode={fmt(metrics.get('decode_tps'), ' tok/s')}")
    return " ".join(fields)


def short_chat(model: str, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
    }
    payload.update(extra)
    return stream_json(endpoint(BASE_URL, "/chat/completions"), payload)


def check_models(base_url: str) -> str:
    response = request_json(endpoint(base_url, "/models"))
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise CheckError("expected exactly one model id")
    model = data[0].get("id") if isinstance(data[0], dict) else None
    if not isinstance(model, str) or not model.strip():
        raise CheckError("model id was missing")
    return model


def tool_spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise CheckError("tool arguments were not valid JSON") from None
    if not isinstance(value, dict):
        raise CheckError("tool arguments were not a JSON object")
    return value


def main() -> int:
    global BASE_URL, TIMEOUT_SECONDS
    parser = argparse.ArgumentParser(description="Smoke test the public chat completion recipe.")
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1", help="API base URL")
    parser.add_argument("--long", action="store_true", help="run the long-context check")
    parser.add_argument("--timeout", type=int, help="client timeout in seconds (default 180, or 900 with --long)")
    parser.add_argument("--write", action="store_true", help="run the large write_file tool check")
    parser.add_argument("--ablate", action="store_true", help="run the ablation-hook check (requires --allow-ablate server)")
    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")
    TIMEOUT_SECONDS = args.timeout or (900 if args.long else 180)
    results: list[dict[str, Any]] = []
    model: str | None = None

    def record(name: str, action: Any) -> Any:
        try:
            detail = action()
            results.append({"check": name, "passed": True, "detail": detail})
            print(f"PASS {name}: {detail}")
            return detail
        except CheckError as exc:
            results.append({"check": name, "passed": False, "error": str(exc)})
            print(f"FAIL {name}: {exc}")
        except Exception as exc:
            results.append({"check": name, "passed": False, "error": f"unexpected {type(exc).__name__}"})
            print(f"FAIL {name}: unexpected {type(exc).__name__}")
        return None

    def models_action() -> dict[str, Any]:
        nonlocal model
        model = check_models(BASE_URL)
        return {"model_id": model}

    record("models", models_action)

    def require_model() -> str:
        if model is None:
            raise CheckError("skipped because /models did not provide one model id")
        return model

    messages = [
        {"role": "system", "content": "Think carefully, then answer briefly."},
        {"role": "user", "content": "In one short sentence, describe a clear blue sky."},
    ]
    first_response: dict[str, Any] | None = None

    def short_action() -> dict[str, Any]:
        nonlocal first_response
        result = short_chat(require_model(), messages, max_tokens=160)
        if not result["reasoning"]:
            raise CheckError("no reasoning text arrived")
        if not result["content"]:
            raise CheckError("no content arrived")
        if result["finish_reason"] != "stop":
            raise CheckError(f"finish reason was {result['finish_reason'] or 'missing'}, expected stop")
        first_response = result
        return {"finish": result["finish_reason"], **metrics_text_obj(result)}

    first = record("short chat (thinking on)", short_action)

    def warm_action() -> dict[str, Any]:
        if first is None or first_response is None:
            raise CheckError("skipped because the first short chat failed")
        next_turn = messages + [
            {"role": "assistant", "content": first_response["content"]},
            {"role": "user", "content": "Now give one short sentence about a gentle breeze."},
        ]
        result = short_chat(require_model(), next_turn, max_tokens=160)
        if not result["content"]:
            raise CheckError("no content arrived")
        if not result["cached_tokens"] or result["cached_tokens"] <= 0:
            raise CheckError("cached prompt tokens were not above zero")
        return {"cached_tokens": result["cached_tokens"], **metrics_text_obj(result)}

    record("same chat cached turn", warm_action)

    weather_tool = tool_spec(
        "get_weather",
        "Get the weather forecast for a city.",
        {"city": {"type": "string"}, "days": {"type": "integer"}},
        ["city", "days"],
    )

    def weather_action() -> dict[str, Any]:
        result = stream_json(
            endpoint(BASE_URL, "/chat/completions"),
            {
                "model": require_model(),
                "messages": [{"role": "user", "content": "Use get_weather for Paris for 3 days."}],
                "tools": [weather_tool],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "stream": True,
                "stream_options": {"include_usage": True},
                "enable_thinking": True,
            },
        )
        calls = result["tool_calls"]
        if result["finish_reason"] != "tool_calls" or not calls:
            raise CheckError("expected a streamed tool call with finish tool_calls")
        call = calls[min(calls)]
        args_obj = parse_arguments(call["arguments"])
        if call["name"] != "get_weather":
            raise CheckError("tool name was not get_weather")
        if not isinstance(args_obj.get("city"), str) or type(args_obj.get("days")) is not int:
            raise CheckError("arguments did not contain city:string and days:integer")
        if result["first_tool_name"] is None or result["last_tool_arguments"] is None:
            raise CheckError("tool name or arguments did not arrive as deltas")
        if result["first_tool_name"] >= result["last_tool_arguments"]:
            raise CheckError("name delta did not arrive before the final arguments delta")
        return {"finish": result["finish_reason"], "name_before_arguments_end": True, **metrics_text_obj(result)}

    record("streamed get_weather tool", weather_action)

    def strict_action() -> dict[str, Any]:
        strict_tool = json.loads(json.dumps(weather_tool))
        strict_tool["function"]["strict"] = True
        result = stream_json(
            endpoint(BASE_URL, "/chat/completions"),
            {
                "model": require_model(),
                "messages": [{"role": "user", "content": "Use get_weather for Oslo for 2 days."}],
                "tools": [strict_tool],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "stream": True,
                "stream_options": {"include_usage": True},
                "enable_thinking": True,
            },
        )
        calls = result["tool_calls"]
        if result["finish_reason"] != "tool_calls" or not calls:
            raise CheckError("expected a strict tool call with finish tool_calls")
        call = calls[min(calls)]
        args_obj = parse_arguments(call["arguments"])
        if call["name"] != "get_weather":
            raise CheckError("tool name was not get_weather")
        if set(args_obj) != {"city", "days"} or not isinstance(args_obj["city"], str) or type(args_obj["days"]) is not int:
            raise CheckError("strict arguments did not match the schema")
        return {"finish": result["finish_reason"], **metrics_text_obj(result)}

    record("strict get_weather tool", strict_action)

    def seeded_action() -> dict[str, Any]:
        prompt = [{"role": "user", "content": "Give one concise sentence about an autumn leaf."}]
        one = short_chat(require_model(), prompt, temperature=1.0, seed=7, max_tokens=200)
        two = short_chat(require_model(), prompt, temperature=1.0, seed=7, max_tokens=200)
        text_one = (one["content"], one["reasoning"])
        text_two = (two["content"], two["reasoning"])
        if text_one != text_two:
            raise CheckError("seeded content plus reasoning differed")
        if one["fingerprint"] != two["fingerprint"]:
            raise CheckError("system_fingerprint differed between repeats")
        return {"text_equal": True, "fingerprint_equal": True, "fingerprint": one["fingerprint"]}

    record("seeded repeat (T=1.0, seed=7)", seeded_action)

    if args.ablate:
        def ablate_action() -> dict[str, Any]:
            # Auto-ablate: server with vectors ablates by default; plain request still answers.
            result = short_chat(require_model(), messages, max_tokens=160)
            if not result["content"]:
                raise CheckError("no content arrived with auto-ablate server")
            return {"finish": result["finish_reason"], **metrics_text_obj(result)}

        def ablate_off_action() -> dict[str, Any]:
            # Explicit opt-out with "ablate": false must also be accepted.
            result = short_chat(require_model(), messages, max_tokens=160, ablate=False)
            if not result["content"]:
                raise CheckError("no content arrived with ablate=false opt-out")
            return {"finish": result["finish_reason"], **metrics_text_obj(result)}

        record("ablated chat (auto-ablate default)", ablate_action)
        record("ablate=false opt-out", ablate_off_action)

    if args.long:
        def long_prompt() -> str:
            # A short numbered line is about 9.5 tokens; this is about 203k tokens.
            lines = [f"Row {i:05d} records routine sample data." for i in range(21500)]
            middle = len(lines) // 2
            lines[middle] = f"Record {middle:05d}: the unique needle is {NEEDLE}. Remember it exactly."
            return "Unique head: LONG-CONTEXT-SMOKE-2026.\n" + "\n".join(lines)

        long_text = long_prompt()

        def long_action() -> dict[str, Any]:
            model_id = require_model()
            cold = short_chat(
                model_id,
                [{"role": "user", "content": long_text + "\nWhat is the unique needle?"}],
                max_tokens=2048,
            )
            if NEEDLE not in cold["content"]:
                raise CheckError("needle was not present in the reply")
            warm = short_chat(
                model_id,
                [
                    {"role": "user", "content": long_text + "\nWhat is the unique needle?"},
                    {"role": "assistant", "content": cold["content"]},
                    {"role": "user", "content": "Repeat the unique needle exactly."},
                ],
                max_tokens=2048,
            )
            return {
                "needle_found": True,
                "cold": metrics_text_obj(cold),
                "warm": metrics_text_obj(warm, label="warm TTFT"),
            }

        record("long context (~203k-token prompt)", long_action)

    if args.write:
        def write_action() -> dict[str, Any]:
            html_tool = tool_spec(
                "write_file",
                "Write a file with the supplied path and complete contents.",
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            )
            started = time.monotonic()
            result = stream_json(
                endpoint(BASE_URL, "/chat/completions"),
                {
                    "model": require_model(),
                    "messages": [{"role": "user", "content": "Call write_file to write a complete HTML page with exactly 10 substantial sections, semantic markup, inline CSS, responsive layout, and enough useful detail to exceed 15 KB. Use path public-recipe.html."}],
                    "tools": [html_tool],
                    "tool_choice": {"type": "function", "function": {"name": "write_file"}},
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "enable_thinking": True,
                    # Deliberately omit max_tokens.
                },
            )
            elapsed = time.monotonic() - started
            calls = result["tool_calls"]
            if result["finish_reason"] != "tool_calls" or not calls:
                raise CheckError("expected finish tool_calls")
            call = calls[min(calls)]
            obj = parse_arguments(call["arguments"])
            if call["name"] != "write_file":
                raise CheckError("tool name was not write_file")
            content = obj.get("content")
            if not isinstance(content, str) or "</html>" not in content.lower():
                raise CheckError("HTML content did not contain </html>")
            if len(content.encode("utf-8")) < 15_000:
                raise CheckError("HTML content was under 15 KB")
            gap = result["longest_gap"]
            if gap >= 120:
                raise CheckError("chunk gap was at least 120 seconds")
            return {
                "finish": result["finish_reason"],
                "bytes": len(content.encode("utf-8")),
                "longest_chunk_gap_s": round(gap, 2),
                "elapsed_s": round(elapsed, 2),
                **metrics_text_obj(result),
            }

        record("write_file large HTML tool", write_action)

    summary = {"passed": all(item["passed"] for item in results), "checks": results}
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0 if summary["passed"] else 1


def metrics_text_obj(metrics: dict[str, Any], label: str = "TTFT") -> dict[str, Any]:
    # Structured companion to the human-readable timed line.
    return {
        "prefill_tok_s": rounded(metrics.get("prefill_tps")),
        label.lower().replace(" ", "_") + "_s": rounded(metrics.get("ttft")),
        "decode_tok_s": rounded(metrics.get("decode_tps")),
    }


def rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 2)


BASE_URL = "http://127.0.0.1:8888/v1"


if __name__ == "__main__":
    sys.exit(main())
