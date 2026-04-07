# Loss-of-Control Definition Extraction Pipeline

## Handoff Document for Claude Code

---

## 1. Project Context

We are part of a research fellowship building a paper titled "Mapping Alternative Pathways to Catastrophic AI Outcomes." Our mentor is Swaptik Chowdhury (RAND Corporation). The project applies Robust Decision Making to AI risk scenarios.

Before we can extract loss-of-control *scenarios* from the literature (a later task), we need a rigorous, literature-grounded operational definition of what "loss of control" means. That definition becomes the downstream filter for scenario extraction — when an LLM later asks "does this passage describe a loss-of-control scenario?", it checks against operationally precise conditions, not vague intuition.

## 2. What We Are Building

A pipeline that:

1. **Task 1A — Definition Extraction:** Runs an LLM prompt against each paper in our 22-paper corpus individually. For each paper, the model extracts every passage where the authors define or characterise "loss of control" or related concepts. Output: structured JSON per paper with exact quotes, the term being defined, definition type (explicit vs characterisation), a grounding note, and flagged new terms.

2. **Task 1B — Attribute Identification:** After humans have reviewed the 1A outputs, compiles all verified extractions into a single input and runs a second prompt once. The model identifies which defining attributes recur across multiple definitions. Output: ranked candidate attributes with traceability to source definitions.

3. **Task 1D — Definition Synthesis:** After humans have reviewed the 1B outputs, feeds the verified attribute list into a third prompt. The model synthesises attributes into an operational definition structured as verifiable conditions ("Loss of control occurs when one or more of the following conditions are met: (a)... (b)... (c)...").

There is a mandatory human review step between each stage. The pipeline must not chain stages automatically.

## 3. Constraints (Non-Negotiable)

- **Model:** Llama 3.1 Instruct via HuggingFace (size to be determined; default to 70B-Instruct)
- **Prompts:** Swaptik provided prompts verbatim. They are stored in `prompts/` and used as-is. The code fills two placeholders per prompt: seed terms and input text.
- **One paper at a time for 1A.** No chunking. Each paper's full text is sent in a single request. Llama 3.1 has 128K context; papers fit.
- **Outputs go to Google Drive** so the full team (Grace, Rujuta, Swaptik) can access them.
- **Code lives in a GitHub repo.** Colab notebooks are thin execution wrappers that clone the repo and import from `src/`.
- **Experiment-based structure.** Each run is logged as a named experiment with full config, approach description, and metadata. Multiple experiments per stage are expected.

## 4. Repo Structure

```
loss-of-control-definition/
├── README.md
├── papers.yaml                    # paper registry: id → PDF filename, citation
├── seed_terms.yaml                # seed term list, versioned
├── prompts/
│   ├── prompt_1a_v1.txt           # Swaptik's prompt for definition extraction
│   ├── prompt_1b_v1.txt           # Swaptik's prompt for attribute identification
│   └── prompt_1d_v1.txt           # Swaptik's prompt for definition synthesis
├── src/
│   ├── __init__.py
│   ├── extract_text.py            # PDF → plain text
│   ├── run_stage.py               # main entry point: run_1a(), run_1b(), run_1d()
│   ├── generate_review.py         # post-processing: review docs, summary, consolidated
│   └── utils/
│       ├── __init__.py
│       ├── hf_client.py           # HuggingFace Inference API wrapper
│       ├── prompt.py              # load and fill prompt templates
│       └── quote_verify.py        # fuzzy match extracted quotes against source text
├── notebooks/
│   ├── 01_extract_text.ipynb
│   ├── 02_task_1a.ipynb
│   ├── 03_task_1b.ipynb
│   └── 04_task_1d.ipynb
├── pyproject.toml                 # dependency source of truth (managed by uv)
├── uv.lock                        # pinned lockfile
└── requirements.txt               # generated from lockfile for Colab convenience
```

## 5. Google Drive Structure (Created by Pipeline)

The notebooks read PDFs from and write all outputs to a shared Google Drive folder. The pipeline creates this structure:

```
Drive/MyDrive/loc_definition_project/
├── papers/                              # raw PDFs (pre-existing)
├── raw_text/                            # extracted plain text per paper
│   └── {paper_id}.txt
└── stages/
    ├── 1a_definition_extraction/
    │   ├── experiments/
    │   │   └── exp001_baseline/
    │   │       ├── APPROACH.md          # what was done and why (written by researcher)
    │   │       ├── config.json          # exact parameters: model, temp, seed terms, prompt version
    │   │       ├── outputs/             # raw LLM output per paper
    │   │       │   └── {paper_id}.json
    │   │       ├── review/              # human-readable review docs
    │   │       │   └── {paper_id}.md
    │   │       ├── consolidated.json    # all papers' extractions merged
    │   │       ├── summary.csv          # one row per paper: counts, flags
    │   │       ├── run_metadata.json    # timestamp, runtime, token counts, errors
    │   │       └── logs/                # raw API responses per paper
    │   │           └── {paper_id}_raw.json
    │   └── verified/                    # human-reviewed output → becomes input to 1B
    │       └── consolidated_verified.json
    ├── 1b_attribute_identification/
    │   ├── experiments/
    │   │   └── exp001_baseline/
    │   │       ├── APPROACH.md
    │   │       ├── config.json
    │   │       ├── output.json          # single output (1B runs once, not per-paper)
    │   │       ├── review.md            # human-readable review doc
    │   │       ├── run_metadata.json
    │   │       └── logs/
    │   └── verified/                    # human-verified attributes → becomes input to 1D
    │       └── verified_attributes.json
    └── 1d_definition_synthesis/
        └── experiments/
            └── exp001_baseline/
                ├── APPROACH.md
                ├── config.json
                ├── output.json          # the operational definition
                ├── run_metadata.json
                └── logs/
```

## 6. Component Specifications

### 6.1 `papers.yaml`

Paper registry. Single source of truth for the corpus. Example:

```yaml
papers:
  - id: carlsmith_2023
    filename: "Carlsmith_2023_Is_Power_Seeking_AI_An_Existential_Risk.pdf"
    citation: "Carlsmith, J. (2023). Is Power-Seeking AI an Existential Risk?"
    notes: ""
  - id: kulveit_2024
    filename: "Kulveit_2024.pdf"
    citation: "Kulveit, J. (2024). ..."
    notes: ""
```

This will be populated manually. The pipeline iterates over this list, not over whatever PDFs happen to exist in a folder.

### 6.2 `seed_terms.yaml`

Versioned seed term list. Example:

```yaml
version: v1
terms:
  - "loss of control"
  - "losing control"
  - "uncontrollable"
  - "uncontrolled"
  - "control failure"
  - "control problem"
  - "controllability"
  - "disempowerment"
  - "human disempowerment"
  - "existential risk"
  - "existential catastrophe"
  - "x-risk"
  - "x-catastrophe"
  - "takeover"
  - "AI takeover"
  - "power-seeking"
  - "irreversible"
  - "irreversibly"
  - "extinction"
  - "civilisational collapse"
  - "civilisational catastrophe"
  - "loss of human influence"
  - "loss of human agency"
  - "human obsolescence"
  - "beyond human control"
  - "inability to control"
  - "corrigibility"
  - "incorrigible"
  - "shutdownability"
  - "override"
  - "human oversight failure"
```

NOTE: This list is a draft. Final list should be empirically grounded by scanning the "How the paper defines Loss of Control" column from the existing literature review spreadsheet.

### 6.3 `prompts/prompt_1a_v1.txt`

Swaptik's Prompt 1a stored verbatim. Contains two placeholders:
- `[INSERT YOUR SEED TERM LIST HERE - e.g., ...]` → replaced with formatted seed terms from `seed_terms.yaml`
- `[INSERT PAPER TEXT HERE]` → replaced with extracted paper text

The code must replace these placeholders exactly. Do not modify any other part of the prompt.

### 6.4 `src/extract_text.py`

**Purpose:** Extract plain text from PDFs.

**Library:** PyMuPDF (`fitz`).

**Interface:**
```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF. Returns full text as a single string."""

def extract_all_papers(papers_config: list, papers_dir: str, output_dir: str) -> dict:
    """
    For each paper in papers_config, extract text and save to output_dir/{paper_id}.txt.
    Returns dict of {paper_id: text_length} for logging.
    Skips papers whose .txt already exists (idempotent).
    """
```

**Behaviour:**
- Extracts text page by page, joins with double newlines between pages.
- Saves to `raw_text/{paper_id}.txt`.
- If output file already exists, skips (idempotent). Override with `force=True` param.
- Returns basic stats: character count, page count per paper.
- If extraction yields very little text relative to page count (suggesting scanned PDF), logs a warning.

### 6.5 `src/utils/hf_client.py`

**Purpose:** Thin wrapper around HuggingFace Inference API.

**Interface:**
```python
def query_llm(
    prompt: str,
    model: str,
    hf_token: str,
    temperature: float = 0.1,
    max_new_tokens: int = 4096,
    return_full_response: bool = True
) -> dict:
    """
    Send prompt to HF Inference API.
    Returns dict with keys:
      - 'generated_text': the model's response string
      - 'raw_response': full API response for logging
      - 'usage': token counts if available
      - 'latency_seconds': round-trip time
    Raises on API errors after 3 retries with exponential backoff.
    """
```

**Behaviour:**
- Uses `requests` library directly (not `huggingface_hub` — simpler, more transparent).
- Endpoint: `https://api-inference.huggingface.co/models/{model}`
- Retries up to 3 times on 503 (model loading) and 429 (rate limit) with exponential backoff.
- Logs all errors.
- Temperature default 0.1 (we want deterministic-ish extraction, not creative generation).

### 6.6 `src/utils/prompt.py`

**Purpose:** Load prompt templates and fill placeholders.

**Interface:**
```python
def load_prompt(prompt_path: str) -> str:
    """Load prompt template from file."""

def fill_prompt_1a(template: str, seed_terms: list[str], paper_text: str) -> str:
    """
    Replace seed term placeholder and paper text placeholder.
    Seed terms formatted as a comma-separated quoted list.
    """

def fill_prompt_1b(template: str, all_definitions_json: str) -> str:
    """Replace definitions placeholder with compiled JSON."""

def fill_prompt_1d(template: str, verified_attributes_json: str) -> str:
    """Replace attributes placeholder with verified attributes."""
```

**Behaviour:**
- String replacement on the known placeholder text.
- Raises an error if placeholder is not found in template (catches prompt version mismatches).

### 6.7 `src/utils/quote_verify.py`

**Purpose:** Check whether extracted quotes actually appear in the source text.

**Interface:**
```python
def verify_quote(quote: str, source_text: str, threshold: float = 0.85) -> dict:
    """
    Fuzzy match quote against source_text.
    Returns:
      - 'found': bool (match ratio >= threshold)
      - 'match_ratio': float
      - 'best_match_context': str (surrounding ±3 sentences of best match location)
      - 'match_start_char': int or None
    """

def verify_all_quotes(extractions: dict, raw_text: str) -> dict:
    """
    For each extraction in the paper's output, verify its quote.
    Returns the extractions dict with added verification fields.
    """
```

**Library:** `difflib.SequenceMatcher` for fuzzy matching. For each quote, slide a window of the quote's length across the source text and find the best match. This is O(n*m) but papers are small enough that it's fine.

### 6.8 `src/run_stage.py`

**Purpose:** Main execution logic for each stage.

**Interface:**
```python
def run_1a(config: dict, drive_root: str, hf_token: str) -> None:
    """
    Run Prompt 1a on all papers in the registry.

    config keys:
      - experiment: str (e.g., "exp001_baseline")
      - model: str (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct")
      - temperature: float
      - prompt_version: str (e.g., "prompt_1a_v1")
      - seed_terms_version: str (e.g., "v1")

    For each paper:
      1. Load raw text from drive_root/raw_text/{paper_id}.txt
      2. Load and fill prompt template
      3. Call HF API
      4. Parse JSON from response
      5. Save raw output to experiment outputs/
      6. Save raw API response to experiment logs/
    After all papers:
      7. Save config.json snapshot to experiment dir
      8. Save run_metadata.json (timestamp, total runtime, per-paper stats, errors)
    """

def run_1b(config: dict, drive_root: str, hf_token: str) -> None:
    """
    Run Prompt 1b on the consolidated verified extractions.

    Reads from: stages/1a_.../verified/consolidated_verified.json
    Writes to: stages/1b_.../experiments/{experiment}/output.json
    """

def run_1d(config: dict, drive_root: str, hf_token: str) -> None:
    """
    Run Prompt 2 on the verified attributes.

    Reads from: stages/1b_.../verified/verified_attributes.json
    Writes to: stages/1d_.../experiments/{experiment}/output.json
    """
```

**Behaviour for `run_1a`:**
- Iterates over `papers.yaml` registry.
- If an output file already exists for a paper in this experiment, skips it (resume-safe). Override with `force=True`.
- JSON parsing from the model response: attempt `json.loads()` on the response. If it fails (model returned text around the JSON), try to extract JSON from the response using a simple regex for the outermost `{...}`. If that also fails, save the raw response and log an error for that paper.
- Prints progress: `[3/22] Processing carlsmith_2023...`
- Collects per-paper metadata: tokens used, latency, number of definitions found, any errors.

### 6.9 `src/generate_review.py`

**Purpose:** Post-process LLM outputs into human-readable review documents.

**Interface:**
```python
def generate_review_docs(config: dict, drive_root: str) -> None:
    """
    For a given experiment, generate:
      1. review/{paper_id}.md for each paper
      2. summary.csv across all papers
      3. consolidated.json merging all papers
    """
```

**Review doc format (`review/{paper_id}.md`):**

```markdown
# Definition Extraction Review: {paper_id}

**Paper:** {full citation}
**Experiment:** {experiment name}
**Model:** {model}
**Date:** {run date}

---

## Extraction 1

**Term:** {term}
**Type:** {EXPLICIT / CHARACTERIZATION}
**New term:** {yes/no}

**Extracted quote:**
> {quote}

**Context from source** (±3 sentences):
> {surrounding text from raw_text file}

**Quote verification:** {VERIFIED (ratio: 0.97) / UNVERIFIED — quote not found in source (best ratio: 0.43)}

**Grounding note:** {grounding_note}

**Reviewer decision:** [ ] Accept  [ ] Reject  [ ] Needs discussion

**Reviewer notes:**

---

## Extraction 2
...
```

**`summary.csv` columns:**
- paper_id
- citation
- num_definitions_found
- num_explicit
- num_characterization
- num_new_terms
- num_verified_quotes
- num_unverified_quotes
- flagged_new_terms (semicolon-separated)

**`consolidated.json`:**
All papers' output JSONs merged into a single array, with paper_id on each extraction. This becomes the input to Prompt 1b after human review.

### 6.10 Notebooks

Each notebook is a thin wrapper. All logic lives in `src/`. Notebooks contain:

1. **Markdown cell:** What this notebook does, which stage it runs.
2. **Mount Drive cell.**
3. **Clone/pull repo cell.**
4. **Install requirements cell.**
5. **Path setup cell** (add repo to sys.path, define DRIVE_ROOT).
6. **HF token cell** (`from google.colab import userdata; HF_TOKEN = userdata.get('HF_TOKEN')`).
7. **Config cell** (dict with experiment name, model, temperature, prompt version, seed terms version).
8. **Execution cell** (one function call, e.g., `run_1a(config, DRIVE_ROOT, HF_TOKEN)`).
9. **Generate review docs cell** (for 1A only: `generate_review_docs(config, DRIVE_ROOT)`).
10. **Summary cell:** Load and display `summary.csv` in a pandas table.

### 6.11 Dependencies

Managed via `uv` with `pyproject.toml` and `uv.lock`. Core dependencies:

- `PyMuPDF>=1.24.0`
- `requests>=2.31.0`
- `pyyaml>=6.0`
- `pandas>=2.0.0`

A `requirements.txt` is generated from the lockfile (`uv export --format requirements-txt > requirements.txt`) as a convenience for Colab, which doesn't have `uv`.

## 7. Execution Flow

### First-time setup:
1. Researcher populates `papers.yaml` with all 22 papers.
2. Researcher finalises `seed_terms.yaml` based on literature review spreadsheet.
3. Researcher stores Swaptik's prompts verbatim in `prompts/`.
4. Researcher adds HF API token to Colab secrets.

### Running Task 1A:
1. Open `01_extract_text.ipynb`. Run all. PDFs → `raw_text/{paper_id}.txt` on Drive.
2. Open `02_task_1a.ipynb`. Set config (experiment name, model, etc.). Run all.
3. Pipeline processes all 22 papers, writes outputs, logs, review docs, summary, consolidated JSON to Drive.
4. Researcher writes `APPROACH.md` for the experiment (or it can be a markdown cell in the notebook that also gets saved to Drive).
5. Researcher shares the `review/` folder link with Grace, Rujuta, Swaptik for human review.

### Human review (1C):
6. Reviewers read review docs, mark each extraction as Accept/Reject/Discuss.
7. Researcher incorporates review decisions into `verified/consolidated_verified.json` (manual step — copy consolidated.json, remove rejected extractions, add any missed ones flagged by reviewers).

### Running Task 1B:
8. Open `03_task_1b.ipynb`. Run all. Reads verified input, runs Prompt 1b once, outputs candidate attributes.
9. Human review of attributes.
10. Researcher produces `verified/verified_attributes.json`.

### Running Task 1D:
11. Open `04_task_1d.ipynb`. Run all. Reads verified attributes, runs Prompt 2, outputs operational definition.
12. Final human review and sign-off on the definition.

## 8. Design Decisions and Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Model | Llama 3.1 Instruct (70B default) | Swaptik specified Llama 3.1 from HuggingFace. 70B balances capability with free-tier feasibility. |
| Temperature | 0.1 | Extraction task — want high fidelity, low creativity. |
| No chunking | Full paper per request | Swaptik's design. Papers fit in 128K context. Chunking risks splitting definitions. |
| Fuzzy quote verification | difflib.SequenceMatcher, threshold 0.85 | Catches hallucinated quotes before human review. Low cost, high value. |
| JSON parsing with fallback | Try json.loads, then regex extract | Llama may wrap JSON in markdown fences or add preamble. |
| Idempotent execution | Skip papers with existing output | Resume-safe if API calls fail mid-run. |
| Experiment-based structure | Each run gets own directory with full config snapshot | Supports multiple approaches; makes the methodology paper-ready. |
| Seed terms as separate config | seed_terms.yaml | Different experiments may use different seed lists; keeps them auditable. |
| Code in src/, notebooks as wrappers | GitHub repo + Colab | Developer gets version control and Claude Code; team gets accessible execution. |
| Raw text stored alongside outputs | raw_text/{paper_id}.txt | Enables quote verification and reviewer cross-referencing without opening PDFs. |

## 9. Open Questions (For Researcher to Resolve)

1. **Llama 3.1 size:** 8B (free, lower quality), 70B (needs Pro or endpoint), 405B (expensive, best). Recommend starting with 70B via HF Pro ($9/month) or a free Inference Endpoint if available.
2. **Seed term list:** Draft provided. Should be validated against the existing literature review spreadsheet's "How the paper defines Loss of Control" column before first run.
3. **HF Inference API vs Inference Endpoints:** Free API may have rate limits and model loading delays. If this is a problem, switch to a dedicated Inference Endpoint and document in APPROACH.md.
4. **The implicit-definition problem:** Swaptik's prompt says "extract ONLY what is explicitly stated, do not infer." Many papers define loss of control implicitly. Flag to Swaptik before running, or run as-is and let human review catch gaps. Recommend flagging it.
5. **`papers.yaml` population:** The 22 paper IDs and filenames need to be entered manually.

## 10. What Not To Build Yet

- Alternative extraction methods (chunking, RAG, multi-pass). Run baseline first, iterate with evidence.
- Scenario extraction pipeline (Swaptik explicitly said to wait).
- Any automation of the human review step.
- Inter-experiment comparison tooling (premature until we have >1 experiment).
