# Implementation plan: simulate MWL-driven adaptation

**Date:** 2026-03-12  
**Status:** Not started  
**Depends on:** `personalisation_comparison_mixed_norm.json` (complete ✓)

---

## Goal

Test the existing `StaircaseController` with real EEG-derived MWL estimates rather than synthetic score patterns. For each of the 40 included participants, replay their 4 MATB condition blocks through a personalised FT-head classifier, feed `1 − P(high_MWL)` into the controller, and record the resulting `d(t)` trajectory.

This answers: *would the adaptation logic have increased difficulty in LOW-load blocks and decreased it in HIGH-load blocks?*

---

## Design decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Personalisation strategy | FT-head (B) | Uses group Nystroem mapping — no expensive kernel refit; median AUC 0.753 at 30 s |
| Calibration duration | 30 s per label | Shortest cal with FT-head median AUC > 0.70 (confirmed in personalisation_comparison results) |
| Threshold | Youden-J from cal data | Matches how classifier would be deployed |
| Score fed to staircase | `1 − P(high_MWL)` | High MWL → low score → controller eases difficulty |
| Smoothers compared | EMA(α=0.10), SMA(window=8), AdaptiveEMA | All 3 overlaid on same plot |
| d_init | 0.5 | Neutral start |
| Adaptation check interval | 5 s | Matches `AdaptationConfig` default |
| Window for evaluation | 45 s | Matches `AdaptationConfig` default |
| Normalisation | Mixed norm (pp z-score train + calibration test) | Consistent with all other mixed-norm results |
| Cal split seeding | Same `SEED + hash(pid)` pattern as personalisation_comparison | Reproducible |

**Score mapping note:** The staircase was designed around a *performance* score (proportion in-target). Here `score = 1 − P(high_MWL)` inverts the intended semantic: the system will call for a *difficulty decrease* when MWL is high. This is the intended adaptive behaviour and is noted in the script docstring.

---

## Caveats

- **Oracle calibration.** The 30 s cal sample is drawn randomly from all 4 condition blocks (same as `personalisation_comparison.py`). In real deployment the system would accumulate cal data sequentially. This makes personalisation quality slightly optimistic.
- **No timing jitter.** All 4 blocks are replayed end-to-end with no inter-block gap. Real sessions include Forest relaxation blocks (~180 s) between task blocks; this simulation ignores them.
- **Low-AUC participants.** P01, P17, P19, P30 have FT-head AUC ≤ 0.45 even with personalisation. Their `d(t)` trajectories will be noisy.

---

## Steps

Work through these one at a time. Each step is small and independently verifiable.

---

### Step 1 — Read and understand the smoother interface

Read `src/adaptation/mwl_smoother.py` in full. Confirm:
- `EmaSmoother`, `SmaSmoother`, `AdaptiveEmaSmoother` all have `.update(value) → float` and `.reset() → None`
- `MwlSmootherConfig` is a dataclass; instantiate with `method="ema"` etc.

No code changes. Just confirm the interface before building on it.

---

### Step 2 — Read the DifficultyState interface

Read `src/adaptation/difficulty_state.py`. Confirm:
- `DifficultyState(d_init, d_min, d_max, seed)` constructor
- `state.d` property returns current difficulty scalar
- `state.update(new_d)` clips to `[d_min, d_max]`

No code changes.

---

### Step 3 — Create the script skeleton

Create `scripts/simulate_mwl_adaptation.py` with:
- Module docstring describing the simulation (see Goal and caveats above)
- Imports: numpy, json, pathlib, sys, time, argparse, matplotlib
- Adaptation imports: `StaircaseController`, `DifficultyState`, `MwlSmootherConfig` + smoother classes
- ML imports: sklearn pipeline, SelectKBest, etc.
- Constants: `_REPO_ROOT`, `_FEATURE_CACHE`, `_NORM_CACHE`, `_EXCLUDE`, `_FIXED`, `_RBF_GAMMA/C/NYS/K`, `_CAL_DURATION_S = 30`, `_SEED`, `_CHECK_INTERVAL_S = 5.0`, `_STEP_S = 0.5`
- `main()` stub that prints "scaffold OK" and exits

Verify: `python scripts/simulate_mwl_adaptation.py` prints scaffold message with no import errors.

---

### Step 4 — Add feature loading

In `main()`, add:
- Feature cache load (`_load_feature_cache`) — copy the exact function from `personalised_logreg.py`
- Mixed norm (`load_baseline_from_cache` + `prepare_mixed_norm`)
- Print: `Loaded N participants, F features [key=...]`

Verify: running the script exits cleanly and prints the correct participant/feature counts (40, 54).

---

### Step 5 — Add group model training helper

Copy `_train_group_model()` from `personalisation_comparison.py` into the new script verbatim (same frozen config: k_candidates, gamma_C_grid, inner CV). Add `_make_logreg`, `_make_rbf` helpers.

No test run needed — will be exercised in Step 8.

---

### Step 6 — Add calibration split and FT-head helper

Copy `_random_cal_split()` and `_detect_blocks()` from `personalisation_comparison.py` verbatim.

Add a new function `_fit_fthead(selector, group_pipe, X_cal, y_cal) → (LogisticRegression, float)`:
- Projects `X_cal` through selector → `sc → nys`
- Fits `LogisticRegression(C=1.0)` on projected cal
- Computes Youden-J threshold from cal predictions
- Returns `(fitted_clf, threshold)`

No test run needed yet.

---

### Step 7 — Add per-smoother simulation function

Add `_simulate_one_smoother(pid, X_full, y_full, selector, group_pipe, clf_personal, threshold, smoother_cfg, rng) → dict`:

```
For each epoch i in chronological order:
    t = i * _STEP_S
    X_i = selector.transform(X_full[i:i+1])
    projected = nys.transform(sc.transform(X_i))
    p_high = clf_personal.predict_proba(projected)[0, 1]
    smoothed = smoother.update(p_high)
    score = 1.0 - smoothed
    staircase.push_performance(t, score)
    if (t - last_check_t) >= _CHECK_INTERVAL_S:
        delta = staircase.tick(t)
        if delta is not None:
            state.update(state.d + delta)
        last_check_t = t
    record: t, p_high, smoothed, score, state.d
Return: dict with arrays + step_events list
```

---

### Step 8 — Add per-participant outer loop

In `main()`, add the participant loop:

```
For each pid in pids:
    Train group model (27-fold)
    Cal split (30 s, seeded rng)
    Fit FT-head + Youden-J threshold
    Run _simulate_one_smoother for EMA, SMA, AdaptiveEMA
    Collect results[pid]
    Print one-liner: pid, FT-head AUC, n_steps (EMA), threshold
```

Verify: run `--only P05` (best-performing participant). Check the one-liner prints and no exceptions.

---

### Step 9 — Add per-participant plot

Add `_plot_participant(pid, y_full, results_by_smoother, out_dir)`:

- **Top panel:** raw P(high_MWL) from EMA smoother (grey line) + actual MWL label as background shading (LOW=blue α=0.1, HIGH=orange α=0.1) + horizontal Youden-J threshold line
- **Bottom panel:** d(t) for EMA (solid blue), SMA (dashed orange), AdaptiveEMA (dotted green) + vertical grey lines at block boundaries
- Save to `results/figures/mwl_adaptation/{pid}.png`

Verify with `--only P05`: check image is created and renders sensibly.

---

### Step 10 — Add JSON output and summary table

After the loop:
- Build output dict: `{config: {...}, participants: {pid: {ema: {...}, sma: {...}, adaptive_ema: {...}}}}`
- Each per-smoother dict: `d_trajectory`, `p_high_trajectory`, `step_events`, `n_steps_up`, `n_steps_down`, `pct_correct_direction` (fraction of steps where Δd sign is consistent with actual MWL label change)
- Write to `results/test_pretrain/simulate_mwl_adaptation.json`
- Print summary table: `pid | FT-head AUC | n_steps (EMA) | pct_correct (EMA)`

---

### Step 11 — Run full 40 participants

```
python scripts/simulate_mwl_adaptation.py
```

Expected: ~5–10 min (group model refit is the bottleneck, ~16 s/participant).

---

### Step 12 — Interpret results

Things to assess from the plots and summary JSON:

1. Do participants' d(t) trajectories go DOWN in HIGH-MWL blocks and UP in LOW-MWL blocks?
2. Which smoother (EMA/SMA/AdaptiveEMA) produces the most stable trajectory with least oscillation?
3. For how many participants does the controller fire ≥ 1 step in the "correct" direction?
4. Are P01/P17/P19/P30 (low FT-head AUC) visibly worse?
5. What `pct_correct_direction` would you expect by chance (50%) and does the mean exceed that?

This step is human review — no code changes.

---

## Output files

| File | Description |
|------|-------------|
| `results/test_pretrain/simulate_mwl_adaptation.json` | Full per-participant simulation results |
| `results/figures/mwl_adaptation/{pid}.png` | Per-participant timeline plot (40 files) |

---

## References

- `scripts/personalisation_comparison.py` — source of group model and cal split logic
- `src/adaptation/staircase_controller.py` — `StaircaseController`
- `src/adaptation/difficulty_state.py` — `DifficultyState`
- `src/adaptation/mwl_smoother.py` — smoother implementations
- `results/test_pretrain/personalisation_comparison_mixed_norm.json` — personalisation AUC values informing cal duration choice
