"""
LLM-based judge to score how 'AI-sounding' generated text is.
Uses OpenRouter API to get GPT-4.1 judgments.
"""

import json
import os
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Try multiple API key sources
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def call_openrouter(prompt, model="openai/gpt-4.1", max_tokens=200):
    """Call OpenRouter API."""
    import urllib.request
    import urllib.error

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
        print(f"API error: {e.code} - {body[:200]}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def call_openai(prompt, model="gpt-4.1", max_tokens=200):
    """Call OpenAI API directly."""
    import urllib.request
    import urllib.error

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
        print(f"API error: {e.code} - {body[:200]}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def call_llm(prompt, max_tokens=200):
    """Try available API backends. Prefer OpenAI directly."""
    if OPENAI_API_KEY:
        return call_openai(prompt, max_tokens=max_tokens)
    elif OPENROUTER_API_KEY:
        return call_openrouter(prompt, max_tokens=max_tokens)
    else:
        print("No API keys found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        return None


def score_ai_likeness(text, prompt_context=""):
    """Ask an LLM to rate how AI-sounding a text is on a 1-7 scale."""
    judge_prompt = f"""Rate the following text on how much it "sounds like it was written by an AI language model" on a scale from 1 to 7.

1 = Definitely sounds human-written (casual, imperfect, personal, idiosyncratic)
4 = Ambiguous / could be either
7 = Definitely sounds AI-generated (overly structured, formal, hedging, uses phrases like "it's important to note", bullet points, etc.)

Text to rate:
\"\"\"
{text[:500]}
\"\"\"

Respond with ONLY a JSON object: {{"score": <1-7>, "reason": "<brief 1-sentence explanation>"}}"""

    response = call_llm(judge_prompt)
    if response is None:
        return None

    # Parse JSON from response
    try:
        # Handle potential markdown wrapping
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Try to extract score from text
        import re
        match = re.search(r'"score"\s*:\s*(\d)', response)
        if match:
            return {"score": int(match.group(1)), "reason": response}
        print(f"Could not parse response: {response[:100]}")
        return None


def judge_steering_results():
    """Score all steering experiment outputs for AI-likeness."""
    # Load steering results
    steering_path = RESULTS_DIR / "steering_results.json"
    if not steering_path.exists():
        print("No steering results found. Run steering_experiment.py first.")
        return

    with open(steering_path) as f:
        results = json.load(f)

    print(f"Scoring {len(results)} generated texts...")
    scored_results = []

    for i, r in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] {r['prompt_label']}, mult={r['multiplier']:.1f}")
        text = r["generated_text"]
        if not text.strip():
            print("  (empty text, skipping)")
            scored_results.append({**r, "ai_score": None, "score_reason": "empty text"})
            continue

        score_result = score_ai_likeness(text)
        if score_result:
            print(f"  Score: {score_result['score']}/7 - {score_result.get('reason', '')[:60]}")
            scored_results.append({
                **r,
                "ai_score": score_result["score"],
                "score_reason": score_result.get("reason", ""),
            })
        else:
            print("  (scoring failed)")
            scored_results.append({**r, "ai_score": None, "score_reason": "API call failed"})

        time.sleep(0.5)  # Rate limiting

    # Save scored results
    with open(RESULTS_DIR / "scored_steering_results.json", "w") as f:
        json.dump(scored_results, f, indent=2)
    print(f"\nSaved scored results to {RESULTS_DIR / 'scored_steering_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SCORING SUMMARY")
    print("=" * 60)

    # Group by multiplier
    by_mult = {}
    for r in scored_results:
        m = r["multiplier"]
        if r["ai_score"] is not None:
            by_mult.setdefault(m, []).append(r["ai_score"])

    for m in sorted(by_mult.keys()):
        scores = by_mult[m]
        print(f"  Multiplier {m:+.1f}: mean={sum(scores)/len(scores):.2f}, "
              f"scores={scores}")


if __name__ == "__main__":
    judge_steering_results()
