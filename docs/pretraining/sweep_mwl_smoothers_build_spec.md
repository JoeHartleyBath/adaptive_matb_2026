# Build spec: `scripts/sweep_mwl_smoothers.py`

Create a new script that sweeps smoother hyperparameters, calibration durations, and hysteresis across all 28 participants using LOSO. Reuse existing infrastructure from `simulate_mwl_adaptation.py`.

---

## Step 1 — Copy boilerplate from simulate_mwl_adaptation.py

Copy these verbatim (no changes needed):
- All imports
- All constants (`_DATASET`, `_FEATURE_CACHE`, `_NORM_CACHE`, `_EXCLUDE`, `_FIXED`, `_RBF_*`, etc.)
- `_make_rbf`, `_make_logreg`
- `_auc`, `_youden_threshold`
- `_train_group_model`, `_load_or_train_group_model`
- `_detect_blocks`, `_random_cal_split`
- `_fit_fthead`
- `_make_smoother`
- `_compute_smoother_stats`
- `_load_feature_cache`, `_cache_key`

Set new output paths:
```python
_DEFAULT_OUT   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep.json"
_CSV_PID       = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep.csv"
_CSV_SUMMARY   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep_summary.csv"
_FIG_DIR       = _REPO_ROOT / "results" / "figures" / "smoother_sweep"
_GROUP_CACHE   = _REPO_ROOT / "results" / "test_pretrain" / "group_model_cache"
```

---

## Step 2 — Modify `_simulate_one_smoother` to accept hysteresis

Add a `hysteresis_margin: float = 0.0` parameter. Replace the stateless threshold check with a stateful one:

```python
state_on = False
for i in range(n_epochs):
    ...
    if state_on:
        if smoothed < threshold - hysteresis_margin:
            state_on = False
    else:
        if smoothed >= threshold + hysteresis_margin:
            state_on = True
    assist_on[i] = state_on
```

---

## Step 3 — Define the parameter grid

```python
import itertools

_CAL_DURATIONS  = [30, 60, 120]   # seconds per label
_SEEDS          = [0, 1, 2, 3, 4]
_HYST_MARGINS   = [0.00, 0.02, 0.05, 0.08, 0.10]

def _build_smoother_grid() -> list[tuple[str, MwlSmootherConfig, dict]]:
    """Returns list of (config_id, smoother_cfg, param_dict) tuples."""
    configs = []

    for alpha in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        cfg = MwlSmootherConfig(method="ema", alpha=alpha)
        configs.append((f"ema_a{alpha}", cfg, {"alpha": alpha}))

    for w in [4, 6, 8, 10, 12, 16, 20]:
        cfg = MwlSmootherConfig(method="sma", window_n=w)
        configs.append((f"sma_w{w}", cfg, {"window_n": w}))

    for amin, amax in itertools.product(
        [0.03, 0.05, 0.10], [0.15, 0.20, 0.30, 0.50]
    ):
        if amin >= amax:
            continue
        cfg = MwlSmootherConfig(method="adaptive_ema", alpha_min=amin, alpha_max=amax)
        configs.append((f"aema_{amin}_{amax}", cfg, {"alpha_min": amin, "alpha_max": amax}))

    for lag, pn, mn in itertools.product(
        [2, 3, 4, 6, 8], [0.001, 0.005, 0.01], [0.05, 0.10, 0.20]
    ):
        cfg = MwlSmootherConfig(method="fixed_lag", lag_n=lag,
                                process_noise=pn, measurement_noise=mn)
        configs.append((f"fl_l{lag}_p{pn}_m{mn}", cfg,
                        {"lag_n": lag, "process_noise": pn, "measurement_noise": mn}))

    return configs
```

---

## Step 4 — Per-participant worker function

```python
def _process_participant(
    pid, X_by, y_by, X_by_test, dataset_key, smoother_grid
) -> list[dict]:
    """Returns list of flat result dicts — one per (seed, cal_dur, smoother, hysteresis)."""
    selector, group_pipe, _ = _load_or_train_group_model(X_by, y_by, pid, dataset_key)
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]

    # Pre-project all epochs (expensive — do once)
    X_full     = X_by_test[pid]
    y_full     = y_by[pid]
    X_sel      = selector.transform(X_full)
    X_proj_all = nys.transform(sc.transform(X_sel))  # (N, n_components)

    rows = []
    pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)

    for seed in _SEEDS:
        for cal_dur in _CAL_DURATIONS:
            rng = np.random.default_rng(SEED + pid_hash + seed * 1000)
            X_cal, y_cal, X_test, y_test, _ = _random_cal_split(
                X_full, y_full, cal_dur, rng)

            clf, threshold = _fit_fthead(selector, group_pipe, X_cal, y_cal)

            # FT-head AUC
            X_test_sel  = selector.transform(X_test)
            probs_test  = clf.predict_proba(nys.transform(sc.transform(X_test_sel)))[:, 1]
            ft_auc      = _auc(y_test, probs_test)

            # Pre-compute raw probabilities for the full session
            p_high_all = clf.predict_proba(X_proj_all)[:, 1]

            for cfg_id, smoother_cfg, param_dict in smoother_grid:
                for hyst in _HYST_MARGINS:
                    # Run smoother + hysteresis
                    smoother  = _make_smoother(smoother_cfg)
                    state_on  = False
                    assist_on = np.zeros(len(y_full), dtype=bool)
                    for i, p in enumerate(p_high_all):
                        sm = smoother.update(p)
                        if state_on:
                            if sm < threshold - hyst:
                                state_on = False
                        else:
                            if sm >= threshold + hyst:
                                state_on = True
                        assist_on[i] = state_on

                    stats = _compute_smoother_stats(assist_on, y_full)
                    mean_onset = float(np.nanmean(stats["onset_latencies"])) \
                        if stats["onset_latencies"] else float("nan")

                    rows.append({
                        "pid":           pid,
                        "seed":          seed,
                        "cal_dur_s":     cal_dur,
                        "config_id":     cfg_id,
                        "method":        smoother_cfg.method,
                        **param_dict,
                        "hysteresis":    hyst,
                        "threshold":     round(threshold, 4),
                        "ft_auc":        round(ft_auc, 4),
                        "bal_acc":       stats["bal_acc"],
                        "pct_on_hi":     stats["pct_on_hi"],
                        "pct_off_lo":    stats["pct_off_lo"],
                        "switch_rate_pm": stats["switch_rate_pm"],
                        "median_bout_s": stats["median_bout_s"],
                        "mean_onset_s":  round(mean_onset, 2),
                    })

    return rows
```

Key optimisation: **project all epochs through the Nystroem kernel once per (pid, cal_dur, seed)** rather than inside the smoother loop. Raw probabilities are then reused across all smoother/hysteresis combos.

---

## Step 5 — Main function

```python
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument("--out",     type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only",    type=str,  nargs="*", default=None)
    parser.add_argument("--jobs",    type=int,  default=4)
    parser.add_argument("--top-n",   type=int,  default=3, dest="top_n")
    args = parser.parse_args()

    # Load features (same pattern as simulate_mwl_adaptation.py)
    key    = _cache_key(args.dataset)
    cached = _load_feature_cache(_FEATURE_CACHE, key)
    if cached is None:
        print("ERROR: Feature cache missing — run personalised_logreg.py first")
        sys.exit(1)

    X_by, y_by, _ = cached
    X_by = {p: v for p, v in X_by.items() if p not in _EXCLUDE}
    y_by = {p: v for p, v in y_by.items() if p not in _EXCLUDE}
    pids = sorted(X_by.keys())
    if args.only:
        pids = [p for p in pids if p in args.only]

    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    if baseline_by is not None:
        X_by, X_by_test = prepare_mixed_norm(X_by, baseline_by)
    else:
        for pid in X_by:
            X_by[pid] = StandardScaler().fit_transform(X_by[pid])
        X_by_test = X_by

    smoother_grid = _build_smoother_grid()
    total = len(pids) * len(_SEEDS) * len(_CAL_DURATIONS) * len(smoother_grid) * len(_HYST_MARGINS)
    print(f"{len(pids)} pids × {len(_SEEDS)} seeds × {len(_CAL_DURATIONS)} cal × "
          f"{len(smoother_grid)} smoothers × {len(_HYST_MARGINS)} hyst = {total:,} runs")

    all_rows = []
    t0 = time.time()
    participant_results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_process_participant)(
            pid, X_by, y_by, X_by_test, key, smoother_grid
        ) for pid in pids
    )
    for rows in participant_results:
        all_rows.extend(rows)

    print(f"Done in {(time.time()-t0)/60:.1f} min  — {len(all_rows):,} rows")

    # --- CSV output ---
    _write_csv(all_rows, args)

    # --- Summary CSV (mean across seeds and pids) ---
    _write_summary(all_rows, args)

    # --- Heatmaps ---
    _plot_heatmaps(all_rows, args)

    # --- Top-N per-participant plots ---
    _plot_top_configs(all_rows, X_by_test, y_by, X_by, key, args)

    # --- Full JSON ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    print(f"JSON written to {args.out}")
```

---

## Step 6 — `_write_csv`

Write `all_rows` to `_CSV_PID`. Columns with missing smoother-specific params (e.g. `alpha` for a FixedLag row) should be written as empty string. Sort by `bal_acc` descending.

---

## Step 7 — `_write_summary`

Group `all_rows` by `(config_id, method, param_dict fields, hysteresis, cal_dur_s)`. For each group compute mean/std of `bal_acc`, `pct_on_hi`, `pct_off_lo`, `switch_rate_pm`, `median_bout_s`, `mean_onset_s` across pids and seeds. Also add a `wins` column = number of participants for which this config has the highest `bal_acc` (at seed=0, cal_dur=30 for comparability). Sort by `mean_bal_acc` descending.

---

## Step 8 — `_plot_heatmaps`

Use summary data. Save to `_FIG_DIR / "heatmaps"`. One figure per smoother method:

- **EMA**: bar chart — alpha (x) vs mean_bal_acc (y), colour = mean_switch_rate_pm
- **SMA**: bar chart — window_n (x) vs mean_bal_acc (y)
- **AdaptiveEMA**: 2D heatmap (alpha_min rows, alpha_max cols) → mean_bal_acc. One subplot per hysteresis level.
- **FixedLag**: for each lag_n, a 2D heatmap (process_noise rows, measurement_noise cols) → mean_bal_acc.

Also one figure: **hysteresis effect** — for each smoother method, line chart of mean_bal_acc vs hysteresis_margin.

---

## Step 9 — `_plot_top_configs`

1. From summary CSV, take the top `--top-n` configs by `mean_bal_acc` (averaged over cal_dur and seeds).
2. For each top config, re-simulate the full session for each participant (seed=0, cal_dur=30) and generate the 2-panel (LOW | HIGH MWL) time-series plot using the same `_plot_participant` logic from `simulate_mwl_adaptation.py`.
3. Save to `_FIG_DIR / "top_configs" / f"rank{rank}_{config_id}" / f"{pid}.png"`.

**Do not generate plots for any other configs.**

---

## Step 10 — CLI dry-run test

After writing, test with:
```
C:\vr_tsst_2025\.venv\Scripts\python.exe scripts/sweep_mwl_smoothers.py --only P05 --jobs 1
```
Verify:
- No crash
- CSV written with correct number of rows: `1 × 5 seeds × 3 cal × N_configs × 5 hyst`
- Summary CSV has N_configs rows
- At least one heatmap PNG saved
