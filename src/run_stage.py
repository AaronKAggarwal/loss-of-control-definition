"""Main entry points for each pipeline stage: run_1a(), run_1b(), run_1d()."""

import json
import os
import re
import time
from datetime import datetime, timezone

import yaml

from src.utils.hf_client import query_llm
from src.utils.prompt import fill_prompt_1a, load_prompt


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(data, path: str) -> None:
    """Write data as formatted JSON. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_json_response(text: str) -> dict | None:
    """Try to parse JSON from an LLM response.

    Strategy:
      1. Try json.loads on the full text (model returned clean JSON).
      2. Try to extract the outermost {...} via regex (model wrapped
         JSON in markdown fences or added preamble text).
      3. Return None if both fail — caller should log the raw response.
    """
    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: extract outermost braces. The regex is non-greedy from
    # the inside but we anchor to the first { and last } in the string,
    # which handles nested objects correctly.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def run_1a(
    config: dict,
    drive_root: str,
    hf_token: str,
    repo_root: str = ".",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Run Prompt 1a on all papers in the registry.

    Args:
        config: Dict with keys: experiment, model, temperature,
                prompt_version, seed_terms_version.
        drive_root: Path to the Google Drive project root
                    (e.g. /content/drive/MyDrive/loc_definition_project).
        hf_token: HuggingFace API token.
        repo_root: Path to the repo root (for loading prompts, configs).
        force: If True, re-run papers that already have output.
        dry_run: If True, use mock API responses instead of real calls.
    """
    run_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Load configs ---
    papers_path = os.path.join(repo_root, "papers.yaml")
    papers_config = _load_yaml(papers_path)["papers"]

    seed_terms_path = os.path.join(repo_root, "seed_terms.yaml")
    seed_data = _load_yaml(seed_terms_path)
    seed_terms = seed_data["terms"]

    prompt_path = os.path.join(repo_root, "prompts", f"{config['prompt_version']}.txt")
    template = load_prompt(prompt_path)

    # --- Set up experiment directory ---
    exp_dir = os.path.join(
        drive_root, "stages", "1a_definition_extraction",
        "experiments", config["experiment"],
    )
    outputs_dir = os.path.join(exp_dir, "outputs")
    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- Process each paper ---
    total = len(papers_config)
    per_paper_meta = []
    errors = []

    for i, paper in enumerate(papers_config, 1):
        paper_id = paper["id"]
        output_path = os.path.join(outputs_dir, f"{paper_id}.json")

        # Resume-safe: skip if output exists unless force=True
        if os.path.exists(output_path) and not force:
            print(f"  [{i}/{total}] SKIP {paper_id} — output already exists")
            per_paper_meta.append({"paper_id": paper_id, "skipped": True})
            continue

        print(f"  [{i}/{total}] Processing {paper_id}...")

        # Load raw text
        raw_text_path = os.path.join(drive_root, "raw_text", f"{paper_id}.txt")
        if not os.path.exists(raw_text_path):
            msg = f"Raw text not found: {raw_text_path}"
            print(f"  [{i}/{total}] ERROR {paper_id} — {msg}")
            errors.append({"paper_id": paper_id, "error": msg})
            per_paper_meta.append({"paper_id": paper_id, "skipped": False, "error": msg})
            continue

        with open(raw_text_path, "r", encoding="utf-8") as f:
            paper_text = f.read()

        # Fill prompt
        filled_prompt = fill_prompt_1a(template, seed_terms, paper_text)

        # Call LLM — catch HTTP errors per-paper so one failure doesn't kill the run
        paper_start = time.time()
        try:
            result = query_llm(
                prompt=filled_prompt,
                model=config["model"],
                hf_token=hf_token,
                temperature=config.get("temperature", 0.1),
                dry_run=dry_run,
            )
        except Exception as e:
            paper_latency = round(time.time() - paper_start, 2)
            msg = f"API error: {e}"
            print(f"  [{i}/{total}] ERROR {paper_id} — {msg}")
            errors.append({"paper_id": paper_id, "error": msg})
            per_paper_meta.append({
                "paper_id": paper_id, "skipped": False,
                "error": msg, "latency_seconds": paper_latency,
            })
            continue
        paper_latency = round(time.time() - paper_start, 2)

        generated_text = result["generated_text"]

        # Save raw API response to logs
        _save_json(result.get("raw_response", {}), os.path.join(logs_dir, f"{paper_id}_raw.json"))

        # Parse JSON from response
        parsed = _parse_json_response(generated_text)
        if parsed is None:
            msg = "Failed to parse JSON from model response"
            print(f"  [{i}/{total}] WARN {paper_id} — {msg}. Raw response saved to logs.")
            # Save raw text so it can be manually inspected
            _save_json({"raw_text": generated_text, "parse_error": True}, output_path)
            errors.append({"paper_id": paper_id, "error": msg})
            per_paper_meta.append({
                "paper_id": paper_id,
                "skipped": False,
                "error": msg,
                "latency_seconds": paper_latency,
                "usage": result.get("usage"),
            })
            continue

        # Save parsed output
        _save_json(parsed, output_path)

        num_defs = len(parsed.get("definitions_found", []))
        print(f"  [{i}/{total}] OK {paper_id} — {num_defs} definitions, {paper_latency}s")

        per_paper_meta.append({
            "paper_id": paper_id,
            "skipped": False,
            "definitions_found": num_defs,
            "latency_seconds": paper_latency,
            "usage": result.get("usage"),
        })

    # --- Save config snapshot ---
    config_snapshot = {
        **config,
        "seed_terms_version": config.get("seed_terms_version", seed_data.get("version")),
        "seed_terms": seed_terms,
        "dry_run": dry_run,
        "force": force,
    }
    _save_json(config_snapshot, os.path.join(exp_dir, "config.json"))

    # --- Save run metadata ---
    total_runtime = round(time.time() - run_start, 2)
    metadata = {
        "timestamp": timestamp,
        "total_runtime_seconds": total_runtime,
        "total_papers": total,
        "papers_processed": sum(1 for m in per_paper_meta if not m.get("skipped")),
        "papers_skipped": sum(1 for m in per_paper_meta if m.get("skipped")),
        "errors": errors,
        "per_paper": per_paper_meta,
    }
    _save_json(metadata, os.path.join(exp_dir, "run_metadata.json"))

    print(f"\nDone. {metadata['papers_processed']}/{total} papers processed in {total_runtime}s.")
    if errors:
        print(f"  {len(errors)} error(s) — see run_metadata.json for details.")


def run_1b(config: dict, drive_root: str, hf_token: str) -> None:
    """Run Prompt 1b on consolidated verified extractions. (Not yet implemented.)"""
    raise NotImplementedError("run_1b is blocked on human review of 1a outputs")


def run_1d(config: dict, drive_root: str, hf_token: str) -> None:
    """Run Prompt 1d on verified attributes. (Not yet implemented.)"""
    raise NotImplementedError("run_1d is blocked on human review of 1b outputs")


if __name__ == "__main__":
    import sys
    import tempfile

    # Self-test: run_1a in dry-run mode on a single dummy paper.
    # Creates a temp directory structure mimicking Google Drive layout,
    # writes a fake paper text file, and runs the pipeline.
    #
    # Usage: uv run python src/run_stage.py
    #    or: uv run python -m src.run_stage

    # Determine repo root (this file is at src/run_stage.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # We need at least one paper in papers.yaml to test. Check if there
    # are any uncommented entries; if not, use a temporary override.
    papers_data = _load_yaml(os.path.join(repo_root, "papers.yaml"))
    papers_list = papers_data.get("papers") or []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up a fake Drive structure
        raw_text_dir = os.path.join(tmpdir, "raw_text")
        os.makedirs(raw_text_dir)

        if not papers_list:
            # No papers in registry yet — create a temporary papers.yaml
            # with one dummy entry for testing
            test_papers = {
                "papers": [{
                    "id": "test_paper",
                    "filename": "test.pdf",
                    "citation": "Test Paper (2024). A Test.",
                    "notes": "dummy entry for dry-run testing",
                }]
            }
            test_papers_path = os.path.join(tmpdir, "papers.yaml")
            with open(test_papers_path, "w") as f:
                yaml.dump(test_papers, f)

            # Point repo_root at tmpdir for papers.yaml, but keep real
            # prompts and seed_terms from the actual repo
            import shutil
            shutil.copy(os.path.join(repo_root, "seed_terms.yaml"), tmpdir)
            shutil.copytree(os.path.join(repo_root, "prompts"), os.path.join(tmpdir, "prompts"))
            test_repo_root = tmpdir
            paper_id = "test_paper"
        else:
            test_repo_root = repo_root
            paper_id = papers_list[0]["id"]

        # Write dummy raw text
        with open(os.path.join(raw_text_dir, f"{paper_id}.txt"), "w") as f:
            f.write(
                "This paper argues that loss of control over advanced AI systems "
                "constitutes an existential risk. We define loss of control as a "
                "situation in which humans can no longer direct, correct, or shut "
                "down an AI system that is pursuing objectives misaligned with "
                "human values. This is distinct from mere automation, where humans "
                "retain override capability."
            )

        config = {
            "experiment": "test_dry_run",
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "temperature": 0.1,
            "prompt_version": "prompt_1a_v1",
            "seed_terms_version": "v1",
        }

        print("=== DRY RUN TEST ===\n")
        run_1a(config, tmpdir, hf_token="", repo_root=test_repo_root, dry_run=True)

        # Show what was written
        exp_dir = os.path.join(
            tmpdir, "stages", "1a_definition_extraction",
            "experiments", "test_dry_run",
        )
        print("\n--- config.json ---")
        with open(os.path.join(exp_dir, "config.json")) as f:
            print(json.dumps(json.load(f), indent=2)[:500])

        print("\n--- run_metadata.json ---")
        with open(os.path.join(exp_dir, "run_metadata.json")) as f:
            print(json.dumps(json.load(f), indent=2))

        output_path = os.path.join(exp_dir, "outputs", f"{paper_id}.json")
        if os.path.exists(output_path):
            print(f"\n--- outputs/{paper_id}.json ---")
            with open(output_path) as f:
                print(json.dumps(json.load(f), indent=2))

        print("\n[OK] Dry run test passed.")
