# Methods decisions — EEG preprocessing, filters, and referencing (v0)

Status: draft (method decision record)

Last updated: 2026-01-28

Purpose: explain and make replicable the preprocessing choices used in the EEG pipeline (CAR, causal filtering, SOS/IIR implementation, Butterworth bandpass, notch).

Primary authority for invariants: [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md)

Primary implementation: `src/python/eeg/`

---

## Decision 1: Enforce causal (forward-only) filtering everywhere

**Decision statement:** All filtering MUST be causal (no zero-phase forward-backward filtering) to prevent future-sample leakage, so offline and online preprocessing match.

**Rationale (already specified):**
- The input contract explicitly requires causal filtering to avoid leakage and distribution shift between offline training and online inference.

**Evidence:**
- Contract: [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md)

**Replication record (must capture):**
- [ ] Confirm no offline path uses `filtfilt` or equivalent.
- [ ] Confirm streaming filter state is persisted across chunks.

---

## Decision 2: Use bandpass 0.5–40 Hz (Butterworth)

**Decision statement:** Apply a 0.5–40 Hz bandpass filter before feature extraction.

**Contract constraint:**
- Bandpass invariants are defined in [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md).

**Implementation evidence (current code defaults):**
- Default config values are defined in [src/python/eeg/eeg_preprocessing_config.py](../../src/python/eeg/eeg_preprocessing_config.py).
- Filter design uses Butterworth with SOS output in [src/python/eeg/eeg_filters.py](../../src/python/eeg/eeg_filters.py).

**Rationale (fill in / confirm):**
- [ ] Why 0.5 Hz high-pass (slow drift) and 40 Hz low-pass (muscle noise / line noise harmonics; feature focus).
- [ ] Why Butterworth (flat passband; standard baseline choice).

**Replication record (must capture):**
- [ ] Filter order (currently order=4).
- [ ] Effective sampling rate used at filtering time.

---

## Decision 3: Use notch filtering at mains frequency (50/60 Hz)

**Decision statement:** Apply notch filtering at the mains frequency (50 Hz or 60 Hz), with mains choice recorded per dataset/session.

**Contract constraint:**
- Notch invariants are defined in [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md).

**Implementation evidence (current code defaults):**
- Defaults (50 Hz, Q=30) in [src/python/eeg/eeg_preprocessing_config.py](../../src/python/eeg/eeg_preprocessing_config.py).
- Design uses `iirnotch` and converts to SOS in [src/python/eeg/eeg_filters.py](../../src/python/eeg/eeg_filters.py).

**Rationale (fill in / confirm):**
- [ ] Why notch is applied even with low-pass at 40 Hz (e.g., leakage/aliasing, robust cleanup).
- [ ] Why Q=30 was chosen (tradeoff: bandwidth vs ringing).

**Replication record (must capture):**
- [ ] Mains frequency (50 vs 60) and whether harmonics are filtered.

---

## Decision 4: Use CAR (Common Average Reference)

**Decision statement:** Apply Common Average Reference (CAR) across the canonical channel set.

**Contract constraint:**
- CAR is explicitly required by the input contract: [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md)

**Implementation evidence (current code defaults):**
- CAR enabled by default in [src/python/eeg/eeg_preprocessing_config.py](../../src/python/eeg/eeg_preprocessing_config.py).
- CAR implementation appears in the preprocessing pipeline: [src/python/eeg/eeg_preprocessor.py](../../src/python/eeg/eeg_preprocessor.py).

**Rationale (fill in / confirm):**
- [ ] Why CAR vs linked-mastoids vs average mastoids vs Laplacian.
- [ ] What channel set is included in CAR (must match `TBD_CHANNELS_ORDERED_V0`).

**Replication record (must capture):**
- [ ] Concrete `TBD_CHANNELS_ORDERED_V0` list and how bad channels affect CAR.

---

## Decision 5: Use SOS filter form for streaming stability and numerical robustness

**Decision statement:** Implement IIR filters in second-order sections (SOS) form and apply them as stateful streaming filters.

**Implementation evidence:**
- `RealTimeFilter` uses `scipy.signal.sosfilt` and maintains per-channel `zi` state in [src/python/eeg/eeg_filters.py](../../src/python/eeg/eeg_filters.py).

**Rationale (fill in / confirm):**
- [ ] Why SOS over direct-form IIR (numerical stability for higher-order filters).
- [ ] Why apply per-channel state (streaming chunks) rather than filtering each chunk independently.

**Replication record (must capture):**
- [ ] How initial conditions are set (e.g., `sosfilt_zi` broadcasting).
- [ ] Chunking behavior (chunk sizes; axis conventions).

---

## Related open decisions (tracked elsewhere)

- Effective resampling rate (`TBD_EFFECTIVE_FS_HZ`) and channel list are still open in [docs/decisions/PENDING_DECISIONS_INFERENCE.md](PENDING_DECISIONS_INFERENCE.md).
