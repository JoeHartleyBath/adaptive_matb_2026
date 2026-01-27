# Repository Operating Principles (Solo PhD)

## Purpose

This repository supports a **solo PhD research project**.

Priorities (in order):

1. **Clarity**
2. **Reproducibility**
3. **Long-term maintainability**
4. **Safe data handling**
5. **Research velocity** (but never at the expense of understanding)

This is **not** a team/enterprise codebase. Avoid ceremony and bureaucracy.

Copilot should optimise for:

- minimal friction
- clear, inspectable history for future reference
- safe handling of data and experiments
- fast iteration **only when it does not reduce understanding**

---

## Canonical Documentation (Authoritative)

If instructions conflict, **these documents take precedence**:

- **Style & naming rules** → `docs/STYLEGUIDE.md`
- **High-level workflow notes** → `docs/WORKFLOW.md`

---

## Working Style: Slow, Stepwise, and Verified

**Default mode is deliberate and incremental.**

- Prefer **one small change at a time**, not large batches.
- After each small change:
  - explain what changed (briefly, concretely)
  - explain why it matters for the study
  - state what to check to confirm it worked
- **Do not move on** to the next step until:
  - the user understands the change, and
  - the user is happy with it

When proposing work, Copilot should:

- propose the *smallest* next step that creates value
- avoid “big refactors” unless explicitly requested
- avoid adding complexity “just in case”
- optimise for “do it right once” to avoid debugging unknown code later

If unsure, **ask before restructuring** or expanding scope.

---

## Core Constraints (Mandatory)

These rules are **non-negotiable**:

- Do **not** create new top-level directories unless explicitly instructed.
- Always respect `.gitignore` and data boundaries.
- **Never commit**:
  - raw data
  - identifiable participant data (PII)
  - large datasets
  - model checkpoints
- Prefer **editing existing files** over creating new ones.
- If a new file is necessary:
  - place it in an existing directory (`docs/`, `src/`, `analysis/`, `scripts/`, `results/`)
  - briefly justify its purpose in the response

---

## Version Control Rules (Solo-Optimised)

### Commits

- Prefer **small, focused commits** (one logical change per commit).
- Use **clear, descriptive commit messages**.
- Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`) are encouraged but **not required**.
- Do **not** batch unrelated changes into a single commit.

Clarity > strict formatting.

### Branching Strategy

This is a **trunk-based solo workflow**.

It is acceptable to commit directly to `main` for:

- small fixes
- exploratory analysis
- documentation edits
- incremental research work

Create a **short-lived feature branch** only when:

- making a substantial refactor
- adding a new pipeline/module
- doing work that may temporarily break things

**Examples:**
- `feat/subjective-esm-integration`
- `refactor/eeg-window-alignment`
- `docs/data-management-clarification`

Feature branches should be merged back into `main` once stable.

- Pull Requests are **optional**
- Suggest PRs only if useful as a self-review checkpoint
- Do **not** require PRs by default

---

## File Placement Rules

Use a **single, consistent structure**:

- Documentation → `docs/`
- Source code → `src/`
- Analysis scripts and notebooks → `analysis/`
- Utility or one-off scripts → `scripts/`
- Derived results (non-raw, non-large) → `results/`

Do **not** create parallel folder structures or duplicate concepts across directories.

---

## Data & Research Boundaries (Critical)

Code and data lifecycles are **strictly separate**.

Git **tracks**:

- code
- configs
- analysis logic
- documentation

Git **does not track**:

- raw participant data
- large intermediate artifacts
- trained models (unless explicitly designed and documented)

Assume:

- ethics matters
- reproducibility matters
- auditability matters

When in doubt, err on the side of **not committing**.

---

## Third-Party Code & Submodules

For third-party dependencies (e.g. OpenMATB):

- Keep them clearly separated (submodule or vendor directory).
- Document:
  - setup and update steps
  - licensing and usage constraints
  - how the code is used in this project
- Do **not** modify third-party code unless explicitly intended and documented.

---

## Copilot Behaviour Expectations

Copilot should:

- optimise for **understanding-first** progress
- recommend **one small step at a time**
- include a quick “what changed / why / how to verify” for each step
- avoid creating new branches unless substantial
- avoid suggesting PRs unless explicitly requested
- avoid unnecessary scope expansion
- prefer minimal, reversible changes over sweeping refactors
- ask before restructuring when unsure

---

## Guiding Principle

This repository exists for:

- thinking
- experimentation
- building defensible research

It is **not** for simulating a large engineering organisation.

Structure should **reduce cognitive load**, and changes should be:
- small
- explainable
- verifiable
- aligned with the study’s actual needs
