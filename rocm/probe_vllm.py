"""Probe a running vLLM planner server with the three t1 requests; prints one JSON line per request and PROBE_OK/PROBE_FAIL.

usage: python rocm/probe_vllm.py --url http://127.0.0.1:18100 --model /data/zhengjli/arl/ckpt/rl-v006
Standard library only, so it runs in any container on the node.
"""

import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib

TOOL = {"type": "function", "function": {
    "name": "move_to", "description": "Move the robot gripper to a world position (metres).",
    "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in "xyz"},
                   "required": ["x", "y", "z"]}}}
ASK = "Move the gripper to (0.1, 0.2, 0.3) by calling the tool."
# The eval protocol of harnesses/rpent_libero/harness.yaml, as RPent's api planner sends it.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 3407,
            "reasoning_effort": "low", "max_tokens": 8192}


def png(size: int = 1024) -> bytes:
    lo, hi = size * 3 // 8, size * 5 // 8
    rows = b"".join(b"\x00" + b"".join(b"\xd0\x20\x20" if lo <= x < hi and lo <= y < hi else b"\x80\x80\x80"
                                       for x in range(size)) for y in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def post(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r), time.monotonic() - t0


def check(resp: dict, *, want_color: bool, want_ids: bool) -> dict:
    choice = resp["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    args = None
    if len(calls) == 1:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, json.JSONDecodeError):
            args = None
    ok = {
        "tool_call": len(calls) == 1 and calls[0]["function"]["name"] == "move_to"
                     and isinstance(args, dict) and [args.get(k) for k in "xyz"] == [0.1, 0.2, 0.3],
        "reasoning_separated": bool(reasoning.strip()) and "<think>" not in content and "</think>" not in content,
    }
    if want_color:
        ok["says_red"] = "red" in (reasoning + content).lower()
    if want_ids:
        ids = choice.get("token_ids") or []
        lps = (choice.get("logprobs") or {}).get("content") or []
        ok["ids_match_logprobs"] = bool(ids) and len(ids) == len(lps)
    return {"ok": ok, "finish_reason": choice.get("finish_reason"), "tool_calls": calls, "content": content[:300],
            "reasoning_head": reasoning[:300], "usage": resp.get("usage")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    img = "data:image/png;base64," + base64.b64encode(png()).decode()
    cases = [
        ("text_tool", [{"role": "user", "content": ASK}], {}, False, False),
        ("image_tool", [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text", "text": "What colour is the square in the image? Then: " + ASK}]}], {}, True, False),
        ("logprobs", [{"role": "user", "content": ASK}], {"logprobs": True, "return_token_ids": True}, False, True),
    ]
    passed = 0
    for name, messages, extra, want_color, want_ids in cases:
        body = {"model": a.model, "messages": messages, "tools": [TOOL], **SAMPLING, **extra}
        try:
            resp, dt = post(a.url, body)
            row = {"case": name, "seconds": round(dt, 2), **check(resp, want_color=want_color, want_ids=want_ids)}
        except Exception as exc:  # noqa: BLE001
            row = {"case": name, "ok": {"request": False}, "error": repr(exc)[:500]}
        passed += all(row["ok"].values())
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"PROBE_{'OK' if passed == len(cases) else 'FAIL'} {passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
