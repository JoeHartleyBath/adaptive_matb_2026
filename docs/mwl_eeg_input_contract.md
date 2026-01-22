# MWL EEG model input contract (v0)

**Status:** draft

**Last updated:** 2026-01-22

## 1. Purpose

This input contract guarantees that *the exact same model-ready input unit* is produced across (a) public dataset pretraining, (b) participant calibration/baseline collection, and (c) real-time online inference. Concretely, it fixes the windowing scheme, channel semantics and ordering, reference scheme, preprocessing invariants, and normalisation behaviour so that a model trained in one context can be evaluated or deployed in another without silent distribution shifts caused by inconsistent signal handling.

## 2. Input Unit Definition

The **canonical model input unit** is a single, fixed-geometry window of EEG transformed into a fixed-length representation.

- **Window duration (seconds):** `TBD_FIXED_WINDOW_S` (must be constant everywhere; suggested default: `2.0`)
- **Window duration (seconds):** `TBD_FIXED_WINDOW_S` (must be constant everywhere; suggested default: `5.0`)
- **Step size (seconds):** `TBD_STEP_SIZE_S` (must be constant everywhere; suggested default: `0.25`)
- **Effective sampling rate (Hz):** `TBD_EFFECTIVE_FS_HZ` (must be constant everywhere; suggested default: `250`)
- **EEG representation class (name only):** `EEGLogBandpowerFeatureVectorV0`
- **Channel policy:** fixed ordered channel set `TBD_CHANNELS_ORDERED_V0` (the model input ordering is exactly this list)
- **Reference scheme:** common average reference (CAR) computed over `TBD_CHANNELS_ORDERED_V0`
- **Output per window:** one fixed-length feature vector `x ∈ R^D` (where `D` is fully determined by the representation class and channel policy)

Normative timing rule:

- Windows are defined on the *post-resampling* timebase at `TBD_EFFECTIVE_FS_HZ`. Window start times follow a regular grid with spacing `TBD_STEP_SIZE_S`.

## 3. Preprocessing Invariants

All steps below MUST be applied identically (same rules and outcomes) across public datasets, calibration sessions, and online inference.

- **Resampling rule:** all signals are resampled to `TBD_EFFECTIVE_FS_HZ`. If the native sampling rate differs, the output must be time-aligned to the resampled clock and use deterministic resampling settings.
- **Filtering:**
- **Filtering:**
  - **Causality requirement:** filtering MUST be causal (forward-only) in all contexts to avoid future-sample leakage. Offline preprocessing MUST use the same causal filtering behaviour (do not use zero-phase / forward-backward filtering).
  - **Bandpass:** apply a bandpass of `TBD_BANDPASS_HZ = [0.5, 40.0]` (inclusive endpoints) before feature extraction.
  - **Notch:** apply a notch at `TBD_MAINS_HZ ∈ {50, 60}` (and harmonics if configured as `TBD_NOTCH_HARMONICS = true/false`). The chosen mains frequency must be recorded per dataset/session.
- **Re-referencing:** apply CAR over the canonical channel set `TBD_CHANNELS_ORDERED_V0`.
- **Bad-channel handling policy:**
  - A channel is either **valid** or **bad** for a given recording.
  - If any required channel in `TBD_CHANNELS_ORDERED_V0` is marked **bad** (or missing) and `TBD_BAD_CHANNEL_POLICY_V0 = reject_window`, then any window overlapping that interval is **invalid** (must not be fed as a normal window).
  - No channel interpolation is permitted in v0 unless explicitly changed by a versioned contract update.
- **Artifact handling policy:**
  - Artifact handling is outcome-based: each window is tagged as **valid** or **artifact-contaminated** according to `TBD_ARTIFACT_POLICY_V0`.
  - For v0, set `TBD_ARTIFACT_POLICY_V0 = reject_window` (artifact-contaminated windows are excluded from training/inference outputs or must yield an explicit “invalid window” output).

## 4. Normalisation Contract

Normalisation is applied to the per-window feature vector `x` after feature extraction.

- **Statistics used for normalisation:** per-feature mean and standard deviation (`μ ∈ R^D`, `σ ∈ R^D`) for z-scoring: `x_norm = (x - μ) / (σ + ε)`, with `ε = 1e-8`.
- **When and from which data they are computed:**
- **When and from which data they are computed:**
  - **Pretraining:** compute **global** (`population`) statistics on the pretraining corpus *training split only*, restricted to windows that pass validity rules (bad-channel/artifact policies).
  - **Participant calibration:** compute **participant** statistics on a participant’s **baseline/reference segment(s)** collected during calibration (e.g., a fixed resting segment), using the same window definition and preprocessing invariants. The baseline/reference segment definition MUST be declared as `TBD_BASELINE_REFERENCE_SEGMENTS_V0` and applied consistently.
- **Fixed or adaptive during runtime:**
- **Fixed or adaptive during runtime:**
  - v0 is **fixed during runtime**: once baseline/reference segments are collected and `μ, σ` are computed, they do not update online.
- **Required behaviour if baseline data are missing or invalid:**
- **Required behaviour if baseline/reference data are missing or invalid:**
  - If participant baseline/reference segments are missing/invalid (insufficient valid windows, non-finite values), fall back to **global** statistics.
  - If global statistics are also missing, use identity normalisation (`μ = 0`, `σ = 1`) and mark the window output as **low-trust** with a required flag `normalization_source = none`.
  - Any window that cannot be normalised due to non-finite values in `x` or `σ` must be marked **invalid** (never silently imputed).

## 5. Dataset Compatibility Rules

A public EEG dataset is admissible under this contract only if all criteria below are satisfied.

- **Channel/montage requirements:**
  - The dataset must provide channel labels sufficient to map deterministically to `TBD_CHANNELS_ORDERED_V0`.
  - Every channel in `TBD_CHANNELS_ORDERED_V0` must be present as recorded data (not derived) or the dataset is **disqualified** in v0.
  - The dataset must provide enough metadata to determine (or reasonably assume) reference and units; if reference is unknown, the dataset is **disqualified**.
- **Sampling rate requirements:**
  - Native sampling rate must be `>= 2 * TBD_BANDPASS_HZ[1]` and must be stable across the recording.
  - The dataset must support deterministic resampling to `TBD_EFFECTIVE_FS_HZ`.
- **Label availability and alignment assumptions:**
  - Labels must be available in a way that can be aligned to the same timebase as EEG (timestamps, sample indices, or trial boundaries).
  - Label alignment assumes monotonic timestamps and a single consistent clock per recording.
  - If labels are provided at lower frequency (e.g., per-trial or sparse ratings), the dataset must include boundaries to assign labels to EEG windows deterministically.
- **Disqualifying conditions (non-exhaustive):**
  - Missing or ambiguous channel names that prevent mapping to `TBD_CHANNELS_ORDERED_V0`.
  - Unknown or mixed reference schemes without sufficient metadata to correct.
  - Non-monotonic or missing timestamps/sample indices that prevent window alignment.
  - Excessive missing/bad data such that valid-window coverage is below `TBD_MIN_VALID_WINDOW_FRACTION`.

## 6. Personalisation Class (v0)

Personalisation describes what is allowed to vary per participant while preserving contract compliance.

- **May vary per participant (allowed):**
  - Normalisation statistics (`μ_participant, σ_participant`) computed from that participant’s calibration data.
  - Optional participant-level affine calibration of model outputs (e.g., bias/scale) defined as `TBD_OUTPUT_CALIBRATION_V0 = none|affine`, provided it is applied *after* the frozen model and is fit only on calibration data.
  - Decision thresholds for mapping continuous MWL scores to discrete bins, if used (must be recorded as participant-specific parameters).
- **Frozen across all participants (must not change):**
  - Windowing (`TBD_FIXED_WINDOW_S`, `TBD_STEP_SIZE_S`) and `TBD_EFFECTIVE_FS_HZ`.
  - Preprocessing invariants (filter bands, notch rule, CAR definition, bad-channel and artifact outcomes).
  - Channel ordering `TBD_CHANNELS_ORDERED_V0` and representation class `EEGLogBandpowerFeatureVectorV0`.
  - Core model weights for feature-to-MWL mapping (except permitted output calibration described above).
- **Explicitly out of scope (not allowed in v0):**
  - Online gradient-based fine-tuning (backprop) during inference.
  - Adaptive filtering/re-referencing that changes over time.
  - Dynamic channel-set learning, channel-dropout substitution, or learned channel imputation.
  - Cross-participant meta-learning procedures that require changing the representation class or window definition.

## 7. Non-Goals

This contract intentionally does not support:

- Variable-length or event-triggered windows; only fixed-length, fixed-step windows are covered.
- Multiple concurrent sampling rates or asynchronous multi-device timebase reconciliation.
- Automatic montage inference when channel labels are missing/ambiguous.
- Guaranteeing robustness to severe artifacts; artifact policy is defined, not artifact immunity.
- Supporting datasets with different reference schemes without explicit, contract-versioned conversion rules.
- End-to-end raw waveform deep models that require different input geometry than `EEGLogBandpowerFeatureVectorV0`.
