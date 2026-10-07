#!/usr/bin/env python3
"""Send a fixed set of greedy prompts to a vLLM server, and compare two such runs.

    scripts/compare.py run --url http://127.0.0.1:8000 --out results/openreg.json
    scripts/compare.py run --url http://127.0.0.1:8000 --out results/cuda.json
    scripts/compare.py diff results/openreg.json results/cuda.json

`run --concurrent` sends all prompts at once, so the server batches them.
Standard library only.
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MAX_TOKENS = 20

LONG_PROMPT = (
    "The following is a list of the planets in the solar system, in order from the "
    "sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. The following "
    "is a list of the days of the week, in order: Monday, Tuesday, Wednesday, Thursday, "
    "Friday, Saturday, Sunday. The following is a list of the months of the year, in "
    "order: January,"
)

# (name, kind, prompt). "completion" prompts are sent as raw text; the "chat" one
# goes through the model's chat template with thinking switched off.
PROMPTS = [
    ("capital", "completion", "The capital of France is"),
    ("code", "completion", "def fibonacci(n):\n    "),
    ("zh", "completion", "台灣最高的山是"),
    ("count", "completion", "1, 2, 3, 4, 5, 6, 7,"),
    ("story", "completion", "Once upon a time, in a small village by the sea,"),
    ("long", "completion", LONG_PROMPT),
    ("chat", "chat", "What is the capital of France? Answer in one short sentence."),
]


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def token_id(token: str) -> int:
    # return_tokens_as_token_ids makes the server render tokens as "token_id:<id>".
    return int(token.removeprefix("token_id:"))


def generate(base_url: str, model: str, kind: str, prompt: str) -> dict:
    common = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "return_tokens_as_token_ids": True,
    }
    if kind == "completion":
        choice = post(
            f"{base_url}/v1/completions", {**common, "prompt": prompt, "logprobs": 1}
        )["choices"][0]
        return {
            "text": choice["text"],
            "token_ids": [token_id(t) for t in choice["logprobs"]["tokens"]],
            "logprobs": choice["logprobs"]["token_logprobs"],
        }
    choice = post(
        f"{base_url}/v1/chat/completions",
        {
            **common,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )["choices"][0]
    content = choice["logprobs"]["content"]
    return {
        "text": choice["message"]["content"],
        "token_ids": [token_id(c["token"]) for c in content],
        "logprobs": [c["logprob"] for c in content],
    }


def cmd_run(args: argparse.Namespace) -> int:
    base_url = args.url.rstrip("/")

    def one(item: tuple[str, str, str]) -> dict:
        name, kind, prompt = item
        return {"name": name, "kind": kind, "prompt": prompt,
                **generate(base_url, args.model, kind, prompt)}

    with ThreadPoolExecutor(max_workers=len(PROMPTS) if args.concurrent else 1) as pool:
        results = list(pool.map(one, PROMPTS))

    for r in results:
        print(f"[{r['name']:8s}] {len(r['token_ids']):2d} tokens: {r['text']!r}")
    with open(args.out, "w") as f:
        json.dump({"url": base_url, "concurrent": args.concurrent, "results": results},
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    with open(args.a) as fa, open(args.b) as fb:
        a, b = json.load(fa)["results"], json.load(fb)["results"]
    assert [r["name"] for r in a] == [r["name"] for r in b], "different prompt sets"

    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'prompt':8s}  {'tokens':>6s}  {'same':>5s}  {'first diff':>10s}  max |dlogprob|")
    all_same = True
    for ra, rb in zip(a, b):
        ta, tb = ra["token_ids"][:MAX_TOKENS], rb["token_ids"][:MAX_TOKENS]
        same = sum(1 for x, y in zip(ta, tb) if x == y)
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        identical = ta == tb and len(ta) > 0
        all_same &= identical
        # Only positions before the first divergence are comparable.
        upto = len(ta) if first is None else first
        dlogprob = max(
            (abs(x - y) for x, y in zip(ra["logprobs"][:upto], rb["logprobs"][:upto])),
            default=float("nan"),
        )
        print(f"{ra['name']:8s}  {len(ta):6d}  {same:2d}/{len(ta):<2d}  "
              f"{'-' if first is None else first:>10}  {dlogprob:.2e}")
        if not identical:
            print(f"    A: {ra['text']!r}\n    B: {rb['text']!r}")
    print("\nall first-%d-token sequences identical: %s" % (MAX_TOKENS, all_same))
    return 0 if all_same else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate with every prompt and save the tokens")
    run.add_argument("--url", default="http://127.0.0.1:8000")
    run.add_argument("--model", default="model", help="served model name")
    run.add_argument("--out", required=True)
    run.add_argument("--concurrent", action="store_true")
    run.set_defaults(func=cmd_run)
    diff = sub.add_parser("diff", help="compare two saved runs token by token")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.set_defaults(func=cmd_diff)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
