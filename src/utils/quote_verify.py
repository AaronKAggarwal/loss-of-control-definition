"""Fuzzy quote verification against source text.

Uses difflib.SequenceMatcher to check whether extracted quotes actually
appear in the source paper. This catches hallucinated quotes before
human review — low cost, high value.

The approach: slide a window of the quote's length across the source
text and find the best match. This is O(n*m) but papers are small
enough (< 500K chars) that it runs in under a second per quote.
"""

import difflib
import re


def _get_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based splitter — good enough
    for academic prose where sentences end with period/question/exclamation
    followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _get_surrounding_context(text: str, start: int, end: int, n_sentences: int = 3) -> str:
    """Extract ±n sentences around a character range in the source text.

    Finds the sentence boundaries around [start, end] and expands outward
    by n_sentences in each direction.
    """
    sentences = _get_sentences(text)
    if not sentences:
        return text[max(0, start - 200):end + 200]

    # Find which sentences overlap with [start, end]
    char_pos = 0
    sentence_spans = []
    for sent in sentences:
        idx = text.find(sent, char_pos)
        if idx == -1:
            idx = char_pos
        sentence_spans.append((idx, idx + len(sent), sent))
        char_pos = idx + len(sent)

    # Find first and last sentence that overlaps the match
    first_idx = 0
    last_idx = len(sentence_spans) - 1
    for i, (s_start, s_end, _) in enumerate(sentence_spans):
        if s_end >= start:
            first_idx = i
            break
    for i, (s_start, s_end, _) in enumerate(sentence_spans):
        if s_start <= end:
            last_idx = i

    # Expand by n_sentences in each direction
    ctx_start = max(0, first_idx - n_sentences)
    ctx_end = min(len(sentence_spans), last_idx + n_sentences + 1)

    return " ".join(span[2] for span in sentence_spans[ctx_start:ctx_end])


def verify_quote(quote: str, source_text: str, threshold: float = 0.85) -> dict:
    """Fuzzy match a quote against the source text.

    Slides a window of the quote's length across the source text
    and returns the position with the highest SequenceMatcher ratio.

    Args:
        quote: The extracted quote to verify.
        source_text: The full text of the source paper.
        threshold: Minimum match ratio to consider the quote verified.

    Returns:
        Dict with keys:
          - found: bool (match ratio >= threshold)
          - match_ratio: float (best ratio found)
          - best_match_context: str (±3 sentences around best match)
          - match_start_char: int or None
    """
    if not quote or not source_text:
        return {
            "found": False,
            "match_ratio": 0.0,
            "best_match_context": "",
            "match_start_char": None,
        }

    quote_len = len(quote)
    best_ratio = 0.0
    best_start = 0

    # Step size trades off accuracy vs speed. Step of quote_len//4
    # means we check ~4 positions per quote-length window. For a
    # 200-char quote in a 100K-char paper, that's ~2000 comparisons.
    step = max(1, quote_len // 4)

    for i in range(0, len(source_text) - quote_len + 1, step):
        window = source_text[i:i + quote_len]
        ratio = difflib.SequenceMatcher(None, quote, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    # Refine: search char-by-char in the neighborhood of the best match
    # to find the true optimum. Search ±step chars around best_start.
    refine_start = max(0, best_start - step)
    refine_end = min(len(source_text) - quote_len + 1, best_start + step + 1)
    for i in range(refine_start, refine_end):
        window = source_text[i:i + quote_len]
        ratio = difflib.SequenceMatcher(None, quote, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    best_ratio = round(best_ratio, 4)
    found = best_ratio >= threshold

    context = _get_surrounding_context(
        source_text, best_start, best_start + quote_len
    ) if found else ""

    return {
        "found": found,
        "match_ratio": best_ratio,
        "best_match_context": context,
        "match_start_char": best_start if found else None,
    }


def verify_all_quotes(extractions: dict, raw_text: str) -> dict:
    """Verify all quotes in a paper's extraction output.

    For each extraction in definitions_found, runs verify_quote and
    adds verification fields to the extraction dict.

    Args:
        extractions: The parsed JSON output from run_1a for one paper.
                     Expected to have a "definitions_found" list.
        raw_text: The full source text of the paper.

    Returns:
        The extractions dict with added verification fields on each
        definition: quote_verified, quote_match_ratio, quote_context.
    """
    for defn in extractions.get("definitions_found", []):
        result = verify_quote(defn.get("quote", ""), raw_text)
        defn["quote_verified"] = result["found"]
        defn["quote_match_ratio"] = result["match_ratio"]
        defn["quote_context"] = result["best_match_context"]
        defn["quote_match_start_char"] = result["match_start_char"]

    return extractions


if __name__ == "__main__":
    # Self-test with three cases: exact match, slightly modified, fabricated.

    source = (
        "Artificial intelligence poses unique challenges to governance. "
        "We define loss of control as a situation in which humans can no "
        "longer direct, correct, or shut down an AI system that is pursuing "
        "objectives misaligned with human values. This is distinct from mere "
        "automation, where humans retain override capability. Furthermore, "
        "the risk of catastrophic outcomes increases when AI systems acquire "
        "resources or influence beyond what their operators intended."
    )

    # Test 1: exact quote
    exact_quote = (
        "We define loss of control as a situation in which humans can no "
        "longer direct, correct, or shut down an AI system that is pursuing "
        "objectives misaligned with human values."
    )
    r1 = verify_quote(exact_quote, source)
    print(f"Test 1 (exact):      found={r1['found']}, ratio={r1['match_ratio']}")
    assert r1["found"], "Exact quote should be found"
    assert r1["match_ratio"] >= 0.99, f"Exact match ratio should be ~1.0, got {r1['match_ratio']}"

    # Test 2: slightly modified (typo, word change) — should still match
    modified_quote = (
        "We define loss of control as a situation in which humans can no "
        "longer direct, correct, or shut down an AI system that pursues "  # "pursuing" → "pursues"
        "objectives misaligned with human values."
    )
    r2 = verify_quote(modified_quote, source)
    print(f"Test 2 (modified):   found={r2['found']}, ratio={r2['match_ratio']}")
    assert r2["found"], f"Modified quote should still match (ratio={r2['match_ratio']})"

    # Test 3: completely fabricated — should fail
    fake_quote = (
        "The alignment tax represents the additional computational cost "
        "incurred when training AI systems with reinforcement learning from "
        "human feedback compared to standard unsupervised pretraining."
    )
    r3 = verify_quote(fake_quote, source)
    print(f"Test 3 (fabricated): found={r3['found']}, ratio={r3['match_ratio']}")
    assert not r3["found"], f"Fabricated quote should NOT match (ratio={r3['match_ratio']})"

    # Test 4: verify_all_quotes integration
    extractions = {
        "paper_id": "test",
        "definitions_found": [
            {"quote": exact_quote, "term": "loss of control"},
            {"quote": fake_quote, "term": "alignment tax"},
        ],
    }
    result = verify_all_quotes(extractions, source)
    assert result["definitions_found"][0]["quote_verified"] is True
    assert result["definitions_found"][1]["quote_verified"] is False
    print(f"Test 4 (all_quotes): verified=[{result['definitions_found'][0]['quote_verified']}, {result['definitions_found'][1]['quote_verified']}]")

    print("\n[OK] All quote_verify.py tests passed.")
