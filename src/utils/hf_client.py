"""HuggingFace Inference API wrapper.

Uses requests directly rather than huggingface_hub — fewer dependencies,
more transparent request/response handling, easier to debug API issues.
"""

import json
import time

import requests


# Mock response for --dry-run testing. Mirrors the structure of a real
# Prompt 1a response so downstream JSON parsing can be tested without
# burning API calls.
MOCK_RESPONSE = {
    "paper_id": "mock_paper",
    "definitions_found": [
        {
            "quote": "This is a mock extracted quote for testing purposes.",
            "term": "loss of control",
            "type": "EXPLICIT",
            "grounding_note": "Mock grounding note — this is a dry-run response.",
            "new_term": False,
            "page_or_section": "Section 1",
        }
    ],
    "flagged_new_terms": [],
}


def query_llm(
    prompt: str,
    model: str,
    hf_token: str,
    temperature: float = 0.1,
    max_new_tokens: int = 4096,
    return_full_response: bool = True,
    dry_run: bool = False,
) -> dict:
    """Send a prompt to the HuggingFace Inference API.

    Args:
        prompt: The full prompt string to send.
        model: HF model ID (e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct").
        hf_token: HuggingFace API token.
        temperature: Sampling temperature. Default 0.1 for high-fidelity extraction.
        max_new_tokens: Maximum tokens to generate.
        return_full_response: If True, include raw API response in output.
        dry_run: If True, return a mock response without calling the API.

    Returns:
        Dict with keys:
          - generated_text: the model's response string
          - raw_response: full API response (if return_full_response=True)
          - usage: token counts if available
          - latency_seconds: round-trip time (0.0 for dry runs)

    Raises:
        requests.HTTPError: After 3 retries on recoverable errors.
    """
    if dry_run:
        mock_text = json.dumps(MOCK_RESPONSE, indent=2)
        return {
            "generated_text": mock_text,
            "raw_response": {"mock": True},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "latency_seconds": 0.0,
        }

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,  # Only return generated text, not the prompt echo
        },
    }

    # Retry on 503 (model loading) and 429 (rate limit) with exponential backoff.
    # 3 attempts total: wait 10s, then 30s before giving up.
    max_retries = 3
    backoff_seconds = [10, 30]

    for attempt in range(max_retries):
        start = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        latency = time.time() - start

        if response.status_code == 200:
            break

        # Recoverable errors: model loading (503) or rate limit (429)
        if response.status_code in (503, 429) and attempt < max_retries - 1:
            wait = backoff_seconds[attempt]
            print(f"  [RETRY] {response.status_code} — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue

        # Non-recoverable or final attempt — raise
        response.raise_for_status()

    raw = response.json()

    # HF Inference API returns a list of generated outputs
    if isinstance(raw, list) and len(raw) > 0:
        generated_text = raw[0].get("generated_text", "")
    elif isinstance(raw, dict):
        generated_text = raw.get("generated_text", "")
    else:
        generated_text = str(raw)

    result = {
        "generated_text": generated_text,
        "latency_seconds": round(latency, 2),
    }

    if return_full_response:
        result["raw_response"] = raw

    # Usage info may not always be present in the HF response
    if isinstance(raw, dict) and "usage" in raw:
        result["usage"] = raw["usage"]
    else:
        result["usage"] = None

    return result


if __name__ == "__main__":
    import sys

    # Usage: uv run python src/utils/hf_client.py [--dry-run] [HF_TOKEN]
    #
    # --dry-run: return mock response (no API call)
    # HF_TOKEN:  your HuggingFace token (required unless --dry-run)

    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]

    test_prompt = "What is 2 + 2? Answer in one word."
    test_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    if dry_run:
        print("=== DRY RUN ===")
        result = query_llm(test_prompt, test_model, hf_token="", dry_run=True)
    else:
        if not args:
            print("Usage: python src/utils/hf_client.py [--dry-run] [HF_TOKEN]")
            print("  Pass --dry-run to test without an API token.")
            sys.exit(1)
        hf_token = args[0]
        print(f"=== LIVE API CALL to {test_model} ===")
        result = query_llm(test_prompt, test_model, hf_token)

    print(f"Latency: {result['latency_seconds']}s")
    print(f"Usage:   {result['usage']}")
    print(f"\n--- Generated text ---\n{result['generated_text']}")
