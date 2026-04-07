"""Prompt template loading and placeholder filling.

Each fill function does simple string replacement on known placeholder text
from Swaptik's prompts. The prompts themselves are never modified — only
placeholders are replaced at runtime.
"""


# Exact placeholder strings from the prompt files. Defined as constants
# so a typo here fails loudly rather than silently producing a prompt
# with an unfilled placeholder.
PLACEHOLDER_1A_SEED_TERMS = (
    '[INSERT YOUR SEED TERM LIST HERE - e.g., "loss of control",\n'
    '"losing control", "uncontrollable", "control failure",\n'
    '"control problem", "controllability", etc.]'
)
PLACEHOLDER_1A_PAPER_TEXT = "[INSERT PAPER TEXT HERE]"
PLACEHOLDER_1B_DEFINITIONS = "[INSERT ALL COMPILED DEFINITIONS FROM PROMPT 1a HERE]"
PLACEHOLDER_1D_ATTRIBUTES = "[INSERT HUMAN-VERIFIED ATTRIBUTES FROM PHASE 3 HERE]"


def load_prompt(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _replace_placeholder(template: str, placeholder: str, replacement: str, label: str) -> str:
    """Replace a placeholder in a template, raising if not found.

    Args:
        template: The prompt template string.
        placeholder: The exact placeholder text to find.
        replacement: The text to substitute in.
        label: Human-readable name for error messages (e.g. "seed terms").

    Raises:
        ValueError: If the placeholder is not found in the template.
    """
    if placeholder not in template:
        raise ValueError(
            f"Placeholder for {label} not found in template. "
            f"Expected to find: {placeholder!r}"
        )
    return template.replace(placeholder, replacement)


def fill_prompt_1a(template: str, seed_terms: list[str], paper_text: str) -> str:
    """Fill Prompt 1a with seed terms and paper text.

    Seed terms are formatted as a comma-separated quoted list, matching
    the style shown in the placeholder example text.
    """
    # Format: "term1", "term2", "term3"
    formatted_terms = ", ".join(f'"{term}"' for term in seed_terms)

    result = _replace_placeholder(template, PLACEHOLDER_1A_SEED_TERMS, formatted_terms, "seed terms")
    result = _replace_placeholder(result, PLACEHOLDER_1A_PAPER_TEXT, paper_text, "paper text")
    return result


def fill_prompt_1b(template: str, all_definitions_json: str) -> str:
    """Fill Prompt 1b with compiled definitions JSON from verified 1a outputs."""
    return _replace_placeholder(template, PLACEHOLDER_1B_DEFINITIONS, all_definitions_json, "compiled definitions")


def fill_prompt_1d(template: str, verified_attributes_json: str) -> str:
    """Fill Prompt 1d with verified attributes JSON from reviewed 1b output."""
    return _replace_placeholder(template, PLACEHOLDER_1D_ATTRIBUTES, verified_attributes_json, "verified attributes")


if __name__ == "__main__":
    import os

    # Test: load prompt_1a_v1.txt, fill with dummy data, verify placeholders replaced.
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "prompt_1a_v1.txt")
    template = load_prompt(prompt_path)

    dummy_terms = ["loss of control", "control failure", "corrigibility"]
    dummy_text = "This is dummy paper text for testing placeholder replacement."

    filled = fill_prompt_1a(template, dummy_terms, dummy_text)

    # Verify placeholders are gone
    assert PLACEHOLDER_1A_SEED_TERMS not in filled, "Seed terms placeholder was NOT replaced!"
    assert PLACEHOLDER_1A_PAPER_TEXT not in filled, "Paper text placeholder was NOT replaced!"

    # Verify replacements are present
    assert '"loss of control", "control failure", "corrigibility"' in filled, "Seed terms not found in output!"
    assert dummy_text in filled, "Paper text not found in output!"

    lines = filled.splitlines()
    print(f"Total lines: {len(lines)}")
    print("\n--- First 30 lines ---")
    print("\n".join(lines[:30]))
    print("\n--- Last 10 lines ---")
    print("\n".join(lines[-10:]))

    # Also test that missing placeholder raises ValueError
    try:
        fill_prompt_1a(filled, dummy_terms, dummy_text)
        print("\n[FAIL] Should have raised ValueError on double-fill!")
    except ValueError as e:
        print(f"\n[OK] Double-fill correctly raised: {e}")

    print("\n[OK] All prompt.py tests passed.")
