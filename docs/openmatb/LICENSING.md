# OpenMATB licensing + provenance

OpenMATB is treated as third-party/upstream code in this repository.

## License (detected)

OpenMATB includes the **CeCILL Free Software License Agreement, Version 2.1 (2013-06-21)**.

Source: [src/python/vendor/openmatb/LICENSE](../../src/python/vendor/openmatb/LICENSE)

## What this implies for this repo (practical summary)

This is a high-level, non-lawyer summary for engineering hygiene:

- We must **keep the upstream license text** with the upstream code when distributing.
- If we **redistribute** OpenMATB (e.g., as part of this repo), we should:
  - preserve attribution and license notices
  - clearly indicate any modifications (if we ever make them)
  - provide access to corresponding source (which we do by including the submodule)
- The license includes **warranty/liability disclaimers**; do not imply warranty.

If we add our own code that interfaces with OpenMATB:
- Keep our changes in our own folders (`src/`, `scripts/`, `docs/`) rather than patching upstream.
- If we must patch upstream later, record the rationale and the exact diff (see “provenance notes” below).

## Provenance (exact upstream reference)

Submodule path:
- `src/python/vendor/openmatb/`

Upstream repository URL (from git config):
- `https://github.com/juliencegarra/OpenMATB.git`

Current pinned commit in this repo:
- `77171c33820c0a0e2d4e2cafa1deb714eeb27bc6` (reported by `git submodule status`)

## Where we keep provenance notes

- This file: `docs/openmatb/LICENSING.md` (license + pointers)
- Submodule definition: [.gitmodules](../../.gitmodules)
- If we ever make repo-level decisions about third-party integration:
  - add an ADR in [docs/decisions/](../decisions/)

## Distribution checklist (for later)

- Confirm the submodule commit hash is pinned and documented.
- Confirm the upstream `LICENSE` file is included in distributions.
- If we ship binaries/installers that bundle OpenMATB, ensure the license accompanies them.
