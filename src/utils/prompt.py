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

    Format depends on prompt version:
      - v1 (header "SEED TERMS TO SEARCH FOR:"): comma-separated quoted list
        to match the placeholder example style.
      - v2 (header "SEED TERMS:"): bulleted list ("- term") to match
        Swaptik's intended format.
    """
    # Detect v2 by its header. v1 has "SEED TERMS TO SEARCH FOR:",
    # v2 has just "SEED TERMS:". Check for the v1 header specifically
    # so any future version defaults to bulleted format.
    is_v1 = "SEED TERMS TO SEARCH FOR:" in template

    if is_v1:
        # Format: "term1", "term2", "term3"
        formatted_terms = ", ".join(f'"{term}"' for term in seed_terms)
    else:
        # Format: - term1\n- term2\n- term3
        formatted_terms = "\n".join(f"- {term}" for term in seed_terms)

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

    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
    dummy_text = "This is dummy paper text for testing placeholder replacement."

    # --- Test v1 ---
    print("=== V1 TEST ===\n")
    template_v1 = load_prompt(os.path.join(prompts_dir, "prompt_1a_v1.txt"))
    dummy_terms_v1 = ["loss of control", "control failure", "corrigibility"]

    filled_v1 = fill_prompt_1a(template_v1, dummy_terms_v1, dummy_text)

    assert PLACEHOLDER_1A_SEED_TERMS not in filled_v1, "Seed terms placeholder was NOT replaced!"
    assert PLACEHOLDER_1A_PAPER_TEXT not in filled_v1, "Paper text placeholder was NOT replaced!"
    assert '"loss of control", "control failure", "corrigibility"' in filled_v1, "V1 terms should be comma-separated!"

    lines_v1 = filled_v1.splitlines()
    # Show the seed terms line
    for line in lines_v1:
        if "loss of control" in line and "control failure" in line:
            print(f"V1 format: {line.strip()}")
            break

    # --- Test v2 ---
    print("\n=== V2 TEST ===\n")
    template_v2 = load_prompt(os.path.join(prompts_dir, "prompt_1a_v2.txt"))
    dummy_terms_v2 = ["loss of control", "losing control", "uncontrollable"]

    filled_v2 = fill_prompt_1a(template_v2, dummy_terms_v2, dummy_text)

    assert PLACEHOLDER_1A_SEED_TERMS not in filled_v2, "Seed terms placeholder was NOT replaced!"
    assert PLACEHOLDER_1A_PAPER_TEXT not in filled_v2, "Paper text placeholder was NOT replaced!"
    assert "- loss of control" in filled_v2, "V2 terms should be bulleted!"
    assert "- losing control" in filled_v2, "V2 terms should be bulleted!"
    assert '"loss of control"' not in filled_v2, "V2 terms should NOT be quoted!"

    lines_v2 = filled_v2.splitlines()
    print("V2 format:")
    for line in lines_v2:
        if line.strip().startswith("- ") and "control" in line.lower():
            print(f"  {line}")

    # Show first 15 lines of v2 filled prompt
    print("\n--- V2 first 15 lines ---")
    print("\n".join(lines_v2[:15]))

    # --- Test double-fill raises ---
    try:
        fill_prompt_1a(filled_v1, dummy_terms_v1, dummy_text)
        print("\n[FAIL] Should have raised ValueError on double-fill!")
    except ValueError as e:
        print(f"\n[OK] Double-fill correctly raised ValueError")

    print("\n[OK] All prompt.py tests passed (v1 + v2).")
