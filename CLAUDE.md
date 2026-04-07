# Instructions for Claude Code

## Project

You are building a research pipeline for extracting and synthesising definitions of "loss of control" from AI safety literature. Read `CLAUDE_CODE_HANDOFF.md` for the full spec before doing anything.

## Working Principles

1. **Swaptik's prompts are sacred.** Store them verbatim in `prompts/`. Never modify prompt text in code. The code fills placeholders only. If you think a prompt could be improved, say so but do not change it.

2. **Explain every decision.** When you make a choice (library, default value, error handling strategy, JSON parsing approach), add a brief comment explaining why. If there are multiple reasonable options, tell me what they are and which you picked before writing the code.

3. **Build incrementally, test at each step.** Do not write the entire pipeline in one go. Build in this order, committing after each:
   - Step 1: Repo skeleton (folder structure, pyproject.toml via `uv init`, empty modules, papers.yaml template, seed_terms.yaml). Generate requirements.txt from lockfile.
   - Step 2: `extract_text.py` — test it on one PDF
   - Step 3: `hf_client.py` — test with a simple prompt to verify API connectivity
   - Step 4: `prompt.py` — test placeholder filling, assert placeholders are replaced
   - Step 5: `run_stage.py` (run_1a only) — test on one paper, verify JSON output
   - Step 6: `quote_verify.py` — test on a known quote from the test paper's output
   - Step 7: `generate_review.py` — test on the single-paper output, verify review doc and summary
   - Step 8: Full run on all papers (only after all components verified individually)
   - Step 9: Notebooks
   - Step 10: run_1b, run_1d (can be deferred; blocked on human review anyway)

4. **Git commits at meaningful boundaries.** Commit after each step above with a clear message. Do not batch the whole project into one commit.

5. **Keep it simple.** No unnecessary abstractions, no class hierarchies, no frameworks. Plain functions, clear names, minimal dependencies. This codebase will be read by non-developers (Grace, Rujuta) and needs to be legible.

6. **Colab compatibility matters.** Everything must work when cloned into a Colab environment. No local paths, no OS-specific behaviour. Test imports work when the repo is added to sys.path.

7. **Ask me before proceeding if:**
   - You encounter a design decision not covered in the handoff doc
   - A library doesn't work as expected
   - The HF API behaves differently than expected (auth, rate limits, response format)
   - You want to deviate from Swaptik's prompt structure for any reason
   - You're unsure about the papers.yaml entries (I need to populate these)

## Testing

- After each component, write a brief test (can be a `if __name__ == "__main__"` block or a separate test script) that verifies the component works in isolation.
- For API-dependent code (hf_client, run_stage), include a `--dry-run` mode that uses a mock response so I can test the full flow without burning API calls.
- When testing the full pipeline, run on 2-3 papers first, not all 22.

## Environment and Tooling

- **Do not modify the system Python.** No `pip install` without a virtual environment.
- **Use `uv`** for dependency management. `uv init`, `uv add`, `uv lock`. The repo should have a `pyproject.toml` and `uv.lock`, not a bare `requirements.txt`.
- **Reproducibility:** Anyone cloning the repo should be able to `uv sync` and have an identical environment. Pin all dependency versions via the lockfile.
- **Colab compatibility:** Colab doesn't have `uv`. The notebooks should install from `pyproject.toml` using `!pip install .` or `!pip install -r <(uv export --format requirements-txt)`. Generate a `requirements.txt` from the lockfile as a Colab convenience artifact, but `pyproject.toml` + `uv.lock` are the source of truth.
- **Python version:** Use 3.11+ (Colab's default).

## What Not To Do

- Do not build tasks 1B or 1D until I ask. They're blocked on human review.
- Do not modify Swaptik's prompts.
- Do not add chunking or RAG. The baseline is one paper per request, no chunking.
- Do not over-engineer. No async, no parallel processing, no caching layers. Sequential is fine.
- Do not install large ML packages (transformers, torch). Inference is API-based.
