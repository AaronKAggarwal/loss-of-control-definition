"""HuggingFace Inference API wrapper.

Uses huggingface_hub's InferenceClient with a provider parameter.
This routes requests to the provider directly (e.g. nscale), giving
full 128K+ context — unlike the raw router endpoint which caps at 8K.
"""

import json
import time

from huggingface_hub import InferenceClient


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
    provider: str = "nscale",
    return_full_response: bool = True,
    dry_run: bool = False,
) -> dict:
    """Send a prompt to the HuggingFace Inference API via InferenceClient.

    Args:
        prompt: The full prompt string to send.
        model: HF model ID (e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct").
        hf_token: HuggingFace API token.
        temperature: Sampling temperature. Default 0.1 for high-fidelity extraction.
        max_new_tokens: Maximum tokens to generate.
        provider: Inference provider (e.g. "nscale"). Routes to the provider
                  directly for full context length support.
        return_full_response: If True, include raw API response in output.
        dry_run: If True, return a mock response without calling the API.

    Returns:
        Dict with keys:
          - generated_text: the model's response string
          - raw_response: full API response (if return_full_response=True)
          - usage: token counts if available
          - latency_seconds: round-trip time (0.0 for dry runs)

    Raises:
        HfHubHTTPError: After 3 retries on recoverable errors.
    """
    if dry_run:
        mock_text = json.dumps(MOCK_RESPONSE, indent=2)
        return {
            "generated_text": mock_text,
            "raw_response": {"mock": True},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "latency_seconds": 0.0,
        }

    client = InferenceClient(model=model, provider=provider, token=hf_token)

    # Retry on 503 (model loading) and 429 (rate limit) with exponential backoff.
    # 3 attempts total: wait 10s, then 30s before giving up.
    max_retries = 3
    backoff_seconds = [10, 30]

    for attempt in range(max_retries):
        start = time.time()
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            latency = time.time() - start
            break
        except Exception as e:
            latency = time.time() - start
            error_str = str(e)

            # Check for recoverable HTTP status codes in the error message
            is_recoverable = any(code in error_str for code in ("503", "429"))
            if is_recoverable and attempt < max_retries - 1:
                wait = backoff_seconds[attempt]
                print(f"  [RETRY] {error_str[:100]} — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            # Non-recoverable or final attempt
            print(f"  [ERROR] {error_str[:300]}")
            raise

    # Extract generated text from OpenAI-compatible response
    try:
        generated_text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        generated_text = str(response)

    result = {
        "generated_text": generated_text,
        "latency_seconds": round(latency, 2),
    }

    if return_full_response:
        # Serialize the response object for logging
        try:
            raw = json.loads(response.to_json())
        except (AttributeError, TypeError, json.JSONDecodeError):
            raw = str(response)
        result["raw_response"] = raw

    # Extract usage if available
    try:
        usage = response.usage
        result["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        } if usage else None
    except AttributeError:
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
