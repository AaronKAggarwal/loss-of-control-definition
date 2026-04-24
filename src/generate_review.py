"""Post-processing: review docs, summary CSV, consolidated JSON.

After run_1a completes, this module generates human-readable review
documents so the team (Grace, Rujuta, Swaptik) can review extractions
without reading raw JSON.
"""

import csv
import json
import os

import yaml

from src.utils.quote_verify import verify_quote


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _build_paper_lookup(repo_root: str) -> dict:
    """Build a {paper_id: citation} lookup from papers.yaml."""
    papers_path = os.path.join(repo_root, "papers.yaml")
    with open(papers_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {p["id"]: p.get("citation", p["id"]) for p in data.get("papers", [])}


def _generate_review_md(
    paper_id: str,
    citation: str,
    extractions: dict,
    raw_text: str,
    config: dict,
    run_date: str,
) -> str:
    """Generate a single paper's review markdown document.

    Follows the exact format from the handoff doc section 6.9:
    header with metadata, then one section per extraction with
    quote verification and reviewer checkboxes.
    """
    lines = []
    lines.append(f"# Definition Extraction Review: {paper_id}")
    lines.append("")
    lines.append(f"**Paper:** {citation}")
    lines.append(f"**Experiment:** {config.get('experiment', 'unknown')}")
    lines.append(f"**Model:** {config.get('model', 'unknown')}")
    lines.append(f"**Date:** {run_date}")
    lines.append("")
    lines.append("---")

    # Handle both v1 (definitions_found) and v2 (passages_found) schemas
    items = extractions.get("definitions_found", extractions.get("passages_found", []))
    is_v2 = "passages_found" in extractions

    if not items:
        lines.append("")
        lines.append("*No extractions found in this paper.*")
        return "\n".join(lines)

    for i, item in enumerate(items, 1):
        quote = item.get("quote", "")
        # v1 uses "term", v2 uses "seed_term_matched"
        term = item.get("term", item.get("seed_term_matched", ""))

        # Run quote verification against source text
        verification = verify_quote(quote, raw_text)

        if verification["found"]:
            verify_str = f"VERIFIED (ratio: {verification['match_ratio']:.2f})"
        else:
            verify_str = (
                f"UNVERIFIED — quote not found in source "
                f"(best ratio: {verification['match_ratio']:.2f})"
            )

        context_str = verification["best_match_context"] if verification["found"] else "*No matching context found.*"

        lines.append("")
        lines.append(f"## Extraction {i}")
        lines.append("")
        lines.append(f"**Term:** {term}")

        # v1-only fields: type, new_term, grounding_note
        if not is_v2:
            def_type = item.get("type", "")
            new_term = item.get("new_term", False)
            lines.append(f"**Type:** {def_type}")
            lines.append(f"**New term:** {'yes' if new_term else 'no'}")

        # v2 has "location" instead of "page_or_section"
        location = item.get("page_or_section", item.get("location", ""))
        if location:
            lines.append(f"**Location:** {location}")

        lines.append("")
        lines.append("**Extracted quote:**")
        lines.append(f"> {quote}")
        lines.append("")
        lines.append("**Context from source** (\u00b13 sentences):")
        lines.append(f"> {context_str}")
        lines.append("")
        lines.append(f"**Quote verification:** {verify_str}")

        # v1-only: grounding note
        if not is_v2:
            grounding_note = item.get("grounding_note", "")
            lines.append("")
            lines.append(f"**Grounding note:** {grounding_note}")

        lines.append("")
        lines.append("**Reviewer decision:** [ ] Accept  [ ] Reject  [ ] Needs discussion")
        lines.append("")
        lines.append("**Reviewer notes:**")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def generate_review_docs(
    config: dict,
    drive_root: str,
    repo_root: str = ".",
) -> None:
    """Generate review docs, summary CSV, and consolidated JSON for an experiment.

    Args:
        config: Dict with at least "experiment" key.
        drive_root: Path to the Google Drive project root.
        repo_root: Path to the repo root (for papers.yaml lookup).
    """
    experiment = config["experiment"]
    exp_dir = os.path.join(
        drive_root, "stages", "1a_definition_extraction",
        "experiments", experiment,
    )
    outputs_dir = os.path.join(exp_dir, "outputs")
    review_dir = os.path.join(exp_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    # Load config.json from the experiment for metadata
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        exp_config = _load_json(config_path)
    else:
        exp_config = config

    # Load run metadata for the date
    meta_path = os.path.join(exp_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        run_meta = _load_json(meta_path)
        run_date = run_meta.get("timestamp", "unknown")
    else:
        run_date = "unknown"

    # Paper citation lookup
    paper_lookup = _build_paper_lookup(repo_root)

    # Find all output JSON files
    if not os.path.exists(outputs_dir):
        print("No outputs directory found. Run run_1a first.")
        return

    output_files = sorted(f for f in os.listdir(outputs_dir) if f.endswith(".json"))
    if not output_files:
        print("No output files found. Run run_1a first.")
        return

    # Track summary stats for CSV
    summary_rows = []
    # Collect all extractions for consolidated.json
    all_extractions = []

    for filename in output_files:
        paper_id = filename.replace(".json", "")
        output_path = os.path.join(outputs_dir, filename)
        extractions = _load_json(output_path)

        # Skip files that had parse errors (they have a "parse_error" key)
        if extractions.get("parse_error"):
            print(f"  [SKIP] {paper_id} — parse error in output, skipping review doc")
            continue

        citation = paper_lookup.get(paper_id, paper_id)

        # Load raw text for quote verification
        raw_text_path = os.path.join(drive_root, "raw_text", f"{paper_id}.txt")
        if os.path.exists(raw_text_path):
            with open(raw_text_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = ""
            print(f"  [WARN] {paper_id} — raw text not found, quote verification will fail")

        # Generate review markdown
        review_md = _generate_review_md(
            paper_id, citation, extractions, raw_text, exp_config, run_date,
        )
        review_path = os.path.join(review_dir, f"{paper_id}.md")
        with open(review_path, "w", encoding="utf-8") as f:
            f.write(review_md)

        # Compute summary stats — handle both v1 and v2 schemas
        items = extractions.get("definitions_found", extractions.get("passages_found", []))
        is_v2 = "passages_found" in extractions
        num_items = len(items)

        # Run verification to count verified/unverified
        num_verified = 0
        num_unverified = 0
        for item in items:
            v = verify_quote(item.get("quote", ""), raw_text)
            if v["found"]:
                num_verified += 1
            else:
                num_unverified += 1

        row = {
            "paper_id": paper_id,
            "citation": citation,
            "num_extractions": num_items,
            "num_verified_quotes": num_verified,
            "num_unverified_quotes": num_unverified,
        }

        # v1-only summary columns
        if not is_v2:
            row["num_explicit"] = sum(1 for d in items if d.get("type") == "EXPLICIT")
            row["num_characterization"] = sum(1 for d in items if d.get("type") == "CHARACTERIZATION")
            row["num_new_terms"] = sum(1 for d in items if d.get("new_term"))
            row["flagged_new_terms"] = ";".join(extractions.get("flagged_new_terms", []))

        summary_rows.append(row)

        # Add to consolidated with paper_id on each extraction
        for item in items:
            entry = {"paper_id": paper_id, **item}
            all_extractions.append(entry)

        print(f"  [OK] {paper_id} — review doc generated ({num_items} extractions)")

    # Write summary.csv
    summary_path = os.path.join(exp_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  summary.csv written ({len(summary_rows)} rows)")

    # Write consolidated.json
    consolidated_path = os.path.join(exp_dir, "consolidated.json")
    _save_json(all_extractions, consolidated_path)
    print(f"  consolidated.json written ({len(all_extractions)} extractions)")


# ---------------------------------------------------------------------------
# Stage 2a: Scenario Extraction review docs
# ---------------------------------------------------------------------------

def _generate_review_md_2a(
    paper_id: str,
    citation: str,
    output: dict,
    raw_text: str,
    config: dict,
    run_date: str,
) -> str:
    """Generate a review markdown doc for 2a scenario extraction output."""
    lines = []
    lines.append(f"# Scenario Extraction Review: {paper_id}")
    lines.append("")
    lines.append(f"**Paper:** {citation}")
    lines.append(f"**Experiment:** {config.get('experiment', 'unknown')}")
    lines.append(f"**Model:** {config.get('model', 'unknown')}")
    lines.append(f"**Date:** {run_date}")
    lines.append("")
    lines.append("---")

    scenarios = output.get("scenarios_found", [])

    if not scenarios:
        lines.append("")
        lines.append("*No scenarios found in this paper.*")
        return "\n".join(lines)

    for scenario in scenarios:
        sid = scenario.get("scenario_id", "?")
        scenario_text = scenario.get("scenario_text", "")

        verification = verify_quote(scenario_text, raw_text)
        if verification["found"]:
            verify_str = f"VERIFIED (ratio: {verification['match_ratio']:.2f})"
        else:
            verify_str = (
                f"UNVERIFIED — quote not found in source "
                f"(best ratio: {verification['match_ratio']:.2f})"
            )
        context_str = verification["best_match_context"] if verification["found"] else "*No matching context found.*"

        lines.append("")
        lines.append(f"## Scenario {sid}")
        lines.append("")
        lines.append("**Extracted scenario:**")
        lines.append(f"> {scenario_text}")
        lines.append("")
        lines.append("**Context from source** (\u00b13 sentences):")
        lines.append(f"> {context_str}")
        lines.append("")
        lines.append(f"**Quote verification:** {verify_str}")
        lines.append("")
        lines.append(f"**Why this is a scenario:** {scenario.get('why_this_is_a_scenario', '')}")
        lines.append(f"**Loss of control relevance:** {scenario.get('loss_of_control_relevance', '')}")
        lines.append(f"**Causal elements:** {scenario.get('causal_elements_present', '')}")
        lines.append(f"**Partial:** {scenario.get('is_partial', '')}")
        lines.append("")
        lines.append("**Reviewer decision:** [ ] Accept  [ ] Reject  [ ] Needs discussion")
        lines.append("")
        lines.append("**Reviewer notes:**")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def generate_review_docs_2a(
    config: dict,
    drive_root: str,
    repo_root: str = ".",
) -> None:
    """Generate review docs, summary CSV, and consolidated JSON for a 2a experiment."""
    experiment = config["experiment"]
    exp_dir = os.path.join(
        drive_root, "stages", "2a_scenario_extraction",
        "experiments", experiment,
    )
    outputs_dir = os.path.join(exp_dir, "outputs")
    review_dir = os.path.join(exp_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    exp_config = _load_json(config_path) if os.path.exists(config_path) else config

    meta_path = os.path.join(exp_dir, "run_metadata.json")
    run_date = _load_json(meta_path).get("timestamp", "unknown") if os.path.exists(meta_path) else "unknown"

    paper_lookup = _build_paper_lookup(repo_root)

    if not os.path.exists(outputs_dir):
        print("No outputs directory found. Run run_2a first.")
        return

    output_files = sorted(f for f in os.listdir(outputs_dir) if f.endswith(".json"))
    if not output_files:
        print("No output files found. Run run_2a first.")
        return

    summary_rows = []
    all_scenarios = []

    for filename in output_files:
        paper_id = filename.replace(".json", "")
        output_path = os.path.join(outputs_dir, filename)
        output = _load_json(output_path)

        if output.get("parse_error"):
            print(f"  [SKIP] {paper_id} — parse error in output")
            continue

        citation = paper_lookup.get(paper_id, paper_id)

        raw_text_path = os.path.join(drive_root, "raw_text", f"{paper_id}.txt")
        if os.path.exists(raw_text_path):
            with open(raw_text_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = ""
            print(f"  [WARN] {paper_id} — raw text not found")

        review_md = _generate_review_md_2a(paper_id, citation, output, raw_text, exp_config, run_date)
        with open(os.path.join(review_dir, f"{paper_id}.md"), "w", encoding="utf-8") as f:
            f.write(review_md)

        scenarios = output.get("scenarios_found", [])
        num_scenarios = len(scenarios)

        num_verified = 0
        num_unverified = 0
        num_partial = 0
        for s in scenarios:
            v = verify_quote(s.get("scenario_text", ""), raw_text)
            if v["found"]:
                num_verified += 1
            else:
                num_unverified += 1
            if str(s.get("is_partial", "")).lower() == "yes":
                num_partial += 1

        summary_rows.append({
            "paper_id": paper_id,
            "citation": citation,
            "num_scenarios": num_scenarios,
            "num_verified_quotes": num_verified,
            "num_unverified_quotes": num_unverified,
            "num_partial": num_partial,
        })

        for s in scenarios:
            all_scenarios.append({"paper_id": paper_id, **s})

        print(f"  [OK] {paper_id} — review doc generated ({num_scenarios} scenarios)")

    # Write summary.csv
    summary_path = os.path.join(exp_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  summary.csv written ({len(summary_rows)} rows)")

    consolidated_path = os.path.join(exp_dir, "consolidated.json")
    _save_json(all_scenarios, consolidated_path)
    print(f"  consolidated.json written ({len(all_scenarios)} scenarios)")


# ---------------------------------------------------------------------------
# Stage 2b: Variable Identification review docs
# ---------------------------------------------------------------------------

def _generate_review_md_2b(scenario_data: dict, scenario_text: str) -> str:
    """Generate a review markdown doc for a single 2b variable identification output."""
    paper_id = scenario_data.get("paper_id", "?")
    scenario_id = scenario_data.get("scenario_id", "?")

    lines = []
    lines.append(f"# Variable Identification Review: {paper_id} — {scenario_id}")
    lines.append("")
    lines.append("**Scenario text:**")
    lines.append(f"> {scenario_text}")
    lines.append("")
    lines.append("---")

    # Variables table
    variables = scenario_data.get("variables", [])
    lines.append("")
    lines.append("## Variables")
    lines.append("")
    lines.append("| # | Variable | Text Evidence | Notes |")
    lines.append("|---|----------|--------------|-------|")
    for j, var in enumerate(variables, 1):
        name = var.get("variable_name", "")
        evidence = var.get("text_evidence", "")
        notes = var.get("notes", "")
        lines.append(f"| {j} | {name} | {evidence} | {notes} |")

    # Causal relationships table
    rels = scenario_data.get("causal_relationships", [])
    lines.append("")
    lines.append("## Causal Relationships")
    lines.append("")
    lines.append("| # | From | To | Relationship Text | Polarity |")
    lines.append("|---|------|----|------------------|----------|")
    for j, rel in enumerate(rels, 1):
        from_v = rel.get("from_variable", "")
        to_v = rel.get("to_variable", "")
        rel_text = rel.get("relationship_text", "")
        polarity = rel.get("polarity", "")
        lines.append(f"| {j} | {from_v} | {to_v} | {rel_text} | {polarity} |")

    lines.append("")
    lines.append("**Reviewer decision:** [ ] Accept  [ ] Reject  [ ] Needs discussion")
    lines.append("")
    lines.append("**Reviewer notes:**")
    lines.append("")
    lines.append("---")

    return "\n".join(lines)


def generate_review_docs_2b(
    config: dict,
    drive_root: str,
    repo_root: str = ".",
) -> None:
    """Generate review docs, summary CSV, and consolidated JSON for a 2b experiment."""
    experiment = config["experiment"]
    exp_dir = os.path.join(
        drive_root, "stages", "2b_variable_identification",
        "experiments", experiment,
    )
    outputs_dir = os.path.join(exp_dir, "outputs")
    review_dir = os.path.join(exp_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    if not os.path.exists(outputs_dir):
        print("No outputs directory found. Run run_2b first.")
        return

    output_files = sorted(f for f in os.listdir(outputs_dir) if f.endswith(".json"))
    if not output_files:
        print("No output files found. Run run_2b first.")
        return

    # Load the verified scenarios to get scenario_text for each
    scenarios_path = os.path.join(
        drive_root, "stages", "2a_scenario_extraction",
        "verified", "consolidated_verified.json",
    )
    scenario_text_lookup = {}
    if os.path.exists(scenarios_path):
        with open(scenarios_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                key = f"{s['paper_id']}_{s['scenario_id']}"
                scenario_text_lookup[key] = s.get("scenario_text", "")

    summary_rows = []
    all_outputs = []

    for filename in output_files:
        item_key = filename.replace(".json", "")
        output_path = os.path.join(outputs_dir, filename)
        data = _load_json(output_path)

        if data.get("parse_error"):
            print(f"  [SKIP] {item_key} — parse error in output")
            continue

        scenario_text = scenario_text_lookup.get(item_key, "")

        review_md = _generate_review_md_2b(data, scenario_text)
        with open(os.path.join(review_dir, f"{item_key}.md"), "w", encoding="utf-8") as f:
            f.write(review_md)

        paper_id = data.get("paper_id", item_key.rsplit("_", 1)[0])
        scenario_id = data.get("scenario_id", item_key.rsplit("_", 1)[-1])
        num_vars = len(data.get("variables", []))
        num_rels = len(data.get("causal_relationships", []))

        summary_rows.append({
            "paper_id": paper_id,
            "scenario_id": scenario_id,
            "num_variables": num_vars,
            "num_causal_relationships": num_rels,
        })

        all_outputs.append(data)
        print(f"  [OK] {item_key} — review doc generated ({num_vars} vars, {num_rels} rels)")

    # Write summary.csv
    summary_path = os.path.join(exp_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  summary.csv written ({len(summary_rows)} rows)")

    consolidated_path = os.path.join(exp_dir, "consolidated.json")
    _save_json(all_outputs, consolidated_path)
    print(f"  consolidated.json written ({len(all_outputs)} entries)")


# ---------------------------------------------------------------------------
# Stage 2c: Threat Model Extraction review docs
# ---------------------------------------------------------------------------

THREAT_MODEL_COMPONENTS = [
    "threat_source", "objective", "capability",
    "knowledge", "access", "constraints", "target",
]

# Human-readable labels for the table
_COMPONENT_LABELS = {
    "threat_source": "Threat Source",
    "objective": "Objective",
    "capability": "Capability",
    "knowledge": "Knowledge",
    "access": "Access",
    "constraints": "Constraints",
    "target": "Target",
}


def _generate_review_md_2c(data: dict, scenario_text: str) -> str:
    """Generate a review markdown doc for a single 2c threat model output."""
    paper_id = data.get("paper_id", "?")
    scenario_id = data.get("scenario_id", "?")
    threat_model = data.get("threat_model", {})
    evidence = data.get("evidence", {})

    lines = []
    lines.append(f"# Threat Model Review: {paper_id} — {scenario_id}")
    lines.append("")
    lines.append("**Scenario text:**")
    lines.append(f"> {scenario_text}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Threat Model Components")
    lines.append("")
    lines.append("| Component | Value | Evidence |")
    lines.append("|-----------|-------|----------|")
    for comp in THREAT_MODEL_COMPONENTS:
        label = _COMPONENT_LABELS[comp]
        value = threat_model.get(comp, "Not specified")
        ev = evidence.get(comp, "Not specified")
        lines.append(f"| {label} | {value} | {ev} |")

    lines.append("")
    lines.append("**Reviewer decision:** [ ] Accept  [ ] Reject  [ ] Needs discussion")
    lines.append("")
    lines.append("**Reviewer notes:**")
    lines.append("")
    lines.append("---")

    return "\n".join(lines)


def generate_review_docs_2c(
    config: dict,
    drive_root: str,
    repo_root: str = ".",
) -> None:
    """Generate review docs, summary CSV, and consolidated JSON for a 2c experiment."""
    experiment = config["experiment"]
    exp_dir = os.path.join(
        drive_root, "stages", "2c_threat_model",
        "experiments", experiment,
    )
    outputs_dir = os.path.join(exp_dir, "outputs")
    review_dir = os.path.join(exp_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    if not os.path.exists(outputs_dir):
        print("No outputs directory found. Run run_2c first.")
        return

    output_files = sorted(f for f in os.listdir(outputs_dir) if f.endswith(".json"))
    if not output_files:
        print("No output files found. Run run_2c first.")
        return

    # Load verified scenarios to get scenario_text for each
    scenarios_path = os.path.join(
        drive_root, "stages", "2a_scenario_extraction",
        "verified", "consolidated_verified.json",
    )
    scenario_text_lookup = {}
    if os.path.exists(scenarios_path):
        with open(scenarios_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                key = f"{s['paper_id']}_{s['scenario_id']}"
                scenario_text_lookup[key] = s.get("scenario_text", "")

    summary_rows = []
    all_outputs = []

    for filename in output_files:
        item_key = filename.replace(".json", "")
        output_path = os.path.join(outputs_dir, filename)
        data = _load_json(output_path)

        if data.get("parse_error"):
            print(f"  [SKIP] {item_key} — parse error in output")
            continue

        scenario_text = scenario_text_lookup.get(item_key, "")

        review_md = _generate_review_md_2c(data, scenario_text)
        with open(os.path.join(review_dir, f"{item_key}.md"), "w", encoding="utf-8") as f:
            f.write(review_md)

        paper_id = data.get("paper_id", item_key.rsplit("_", 1)[0])
        scenario_id = data.get("scenario_id", item_key.rsplit("_", 1)[-1])

        # Count specified vs not-specified components
        threat_model = data.get("threat_model", {})
        num_specified = sum(
            1 for comp in THREAT_MODEL_COMPONENTS
            if threat_model.get(comp, "Not specified") != "Not specified"
        )
        num_not_specified = len(THREAT_MODEL_COMPONENTS) - num_specified

        summary_rows.append({
            "paper_id": paper_id,
            "scenario_id": scenario_id,
            "num_specified_components": num_specified,
            "num_not_specified": num_not_specified,
        })

        all_outputs.append(data)
        print(f"  [OK] {item_key} — review doc generated ({num_specified}/7 components specified)")

    # Write summary.csv
    summary_path = os.path.join(exp_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  summary.csv written ({len(summary_rows)} rows)")

    consolidated_path = os.path.join(exp_dir, "consolidated.json")
    _save_json(all_outputs, consolidated_path)
    print(f"  consolidated.json written ({len(all_outputs)} entries)")


if __name__ == "__main__":
    import tempfile

    # Self-test: create dummy outputs and generate review docs from them.

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy papers.yaml in tmpdir as repo root
        papers_yaml = {
            "papers": [
                {
                    "id": "smith_2024",
                    "filename": "smith_2024.pdf",
                    "citation": "Smith, A. (2024). On the Nature of AI Control.",
                    "notes": "",
                },
                {
                    "id": "jones_2023",
                    "filename": "jones_2023.pdf",
                    "citation": "Jones, B. (2023). Corrigibility and Control.",
                    "notes": "",
                },
            ]
        }
        with open(os.path.join(tmpdir, "papers.yaml"), "w") as f:
            yaml.dump(papers_yaml, f)

        # Create dummy raw text files
        raw_text_dir = os.path.join(tmpdir, "raw_text")
        os.makedirs(raw_text_dir)

        smith_text = (
            "Artificial intelligence poses unique challenges to governance. "
            "We define loss of control as a situation in which humans can no "
            "longer direct, correct, or shut down an AI system that is pursuing "
            "objectives misaligned with human values. This is distinct from mere "
            "automation, where humans retain override capability. Furthermore, "
            "the risk of catastrophic outcomes increases when AI systems acquire "
            "resources or influence beyond what their operators intended."
        )
        with open(os.path.join(raw_text_dir, "smith_2024.txt"), "w") as f:
            f.write(smith_text)

        jones_text = (
            "A key challenge in AI safety is ensuring corrigibility. "
            "We characterize corrigibility as the property of an AI system "
            "that allows its operators to modify its goals, shut it down, "
            "or alter its behavior without resistance. An incorrigible system "
            "is one that actively resists or circumvents human attempts at "
            "correction. The control problem arises when a sufficiently "
            "capable system has both the ability and incentive to prevent "
            "human intervention."
        )
        with open(os.path.join(raw_text_dir, "jones_2023.txt"), "w") as f:
            f.write(jones_text)

        # Create dummy experiment outputs
        exp_dir = os.path.join(
            tmpdir, "stages", "1a_definition_extraction",
            "experiments", "exp001_test",
        )
        outputs_dir = os.path.join(exp_dir, "outputs")
        os.makedirs(outputs_dir)

        smith_output = {
            "paper_id": "smith_2024",
            "definitions_found": [
                {
                    "quote": (
                        "We define loss of control as a situation in which humans can no "
                        "longer direct, correct, or shut down an AI system that is pursuing "
                        "objectives misaligned with human values."
                    ),
                    "term": "loss of control",
                    "type": "EXPLICIT",
                    "grounding_note": "Uses explicit definitional language 'We define X as'.",
                    "new_term": False,
                    "page_or_section": "Section 1",
                },
                {
                    "quote": "This quote was completely hallucinated by the model and does not exist.",
                    "term": "existential risk",
                    "type": "CHARACTERIZATION",
                    "grounding_note": "Fabricated for testing.",
                    "new_term": False,
                    "page_or_section": "Section 2",
                },
            ],
            "flagged_new_terms": [],
        }
        with open(os.path.join(outputs_dir, "smith_2024.json"), "w") as f:
            json.dump(smith_output, f)

        jones_output = {
            "paper_id": "jones_2023",
            "definitions_found": [
                {
                    "quote": (
                        "We characterize corrigibility as the property of an AI system "
                        "that allows its operators to modify its goals, shut it down, "
                        "or alter its behavior without resistance."
                    ),
                    "term": "corrigibility",
                    "type": "EXPLICIT",
                    "grounding_note": "Uses 'we characterize X as' language.",
                    "new_term": False,
                    "page_or_section": "Section 1",
                },
                {
                    "quote": (
                        "An incorrigible system is one that actively resists or "
                        "circumvents human attempts at correction."
                    ),
                    "term": "incorrigible",
                    "type": "CHARACTERIZATION",
                    "grounding_note": "Defines the negation of corrigibility.",
                    "new_term": False,
                    "page_or_section": "Section 1",
                },
                {
                    "quote": (
                        "The control problem arises when a sufficiently capable system "
                        "has both the ability and incentive to prevent human intervention."
                    ),
                    "term": "control problem",
                    "type": "CHARACTERIZATION",
                    "grounding_note": "Describes conditions under which the control problem manifests.",
                    "new_term": True,
                    "page_or_section": "Section 2",
                },
            ],
            "flagged_new_terms": ["control problem"],
        }
        with open(os.path.join(outputs_dir, "jones_2023.json"), "w") as f:
            json.dump(jones_output, f)

        # Write config.json and run_metadata.json
        config = {
            "experiment": "exp001_test",
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "temperature": 0.1,
            "prompt_version": "prompt_1a_v1",
        }
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f)

        run_meta = {"timestamp": "2026-04-07T20:00:00+00:00"}
        with open(os.path.join(exp_dir, "run_metadata.json"), "w") as f:
            json.dump(run_meta, f)

        # Run generate_review_docs (v1 schema)
        print("=== V1 SCHEMA TEST ===\n")
        generate_review_docs(config, tmpdir, repo_root=tmpdir)

        # Show one review doc
        review_dir_path = os.path.join(exp_dir, "review")
        with open(os.path.join(review_dir_path, "smith_2024.md"), "r") as f:
            print(f.read()[:600])
        print("...\n")

        # --- V2 schema test ---
        print("=== V2 SCHEMA TEST ===\n")
        exp_dir_v2 = os.path.join(
            tmpdir, "stages", "1a_definition_extraction",
            "experiments", "exp002_v2_test",
        )
        outputs_dir_v2 = os.path.join(exp_dir_v2, "outputs")
        os.makedirs(outputs_dir_v2)

        # v2 output uses passages_found and seed_term_matched
        smith_v2_output = {
            "paper_id": "smith_2024",
            "passages_found": [
                {
                    "quote": (
                        "We define loss of control as a situation in which humans can no "
                        "longer direct, correct, or shut down an AI system that is pursuing "
                        "objectives misaligned with human values."
                    ),
                    "seed_term_matched": "loss of control",
                    "location": "Section 1",
                },
            ],
        }
        with open(os.path.join(outputs_dir_v2, "smith_2024.json"), "w") as f:
            json.dump(smith_v2_output, f)

        config_v2 = {
            "experiment": "exp002_v2_test",
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "temperature": 0.1,
            "prompt_version": "prompt_1a_v2",
        }
        with open(os.path.join(exp_dir_v2, "config.json"), "w") as f:
            json.dump(config_v2, f)
        with open(os.path.join(exp_dir_v2, "run_metadata.json"), "w") as f:
            json.dump(run_meta, f)

        generate_review_docs(config_v2, tmpdir, repo_root=tmpdir)

        review_dir_v2 = os.path.join(exp_dir_v2, "review")
        with open(os.path.join(review_dir_v2, "smith_2024.md"), "r") as f:
            print(f.read())

        print(f"\n{'=' * 60}")
        print("  summary.csv (v2)")
        print(f"{'=' * 60}")
        with open(os.path.join(exp_dir_v2, "summary.csv"), "r") as f:
            print(f.read())

        print("\n[OK] All generate_review.py tests passed (v1 + v2).")
