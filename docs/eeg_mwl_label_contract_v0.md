## 1. Purpose

This label contract guarantees that the meaning, admissible sources, temporal support, and harmonisation rules for mental workload (MWL) labels are identical across public dataset pretraining, participant calibration, and real-time online inference, such that any MWL label used by the system is interpretable under one fixed semantic definition and can be aligned to EEG windows deterministically.

## 2. MWL Semantic Definition

Mental workload (MWL) is defined as a latent, task-related, continuous demand/effort construct reflecting the level of cognitive processing load required to meet task requirements under current conditions.

Higher MWL means greater required mental effort to maintain task performance given task demands, time pressure, and concurrent tasking; lower MWL means lower required effort under the same operational definition.

The following constructs are explicitly out of scope and must not be treated as equivalent labels for MWL under this contract: stress, affect, arousal, fatigue, sleepiness, boredom, engagement/motivation, emotion, pain, and any clinical symptom scales.

## 3. Admissible Label Types

All labels must be explicitly defined, externally provided by the dataset/protocol, and alignable to EEG time.

Permitted label source categories:

- Task-demand or task-difficulty labels
  - Allowed for: pretraining, calibration, validation, analysis.
  - Requirements: labels must represent task-defined demand states, difficulty levels, or condition identifiers that are declared to operationalise MWL under this contract.

- Subjective self-report labels
  - Allowed for: pretraining, calibration, validation, analysis.
  - Requirements: the instrument must be declared to measure MWL as defined in Section 2, and the directionality (higher = higher MWL) must be determinable.

- Performance-derived proxy labels
  - Allowed for: pretraining, validation, analysis.
  - Allowed for calibration: not permitted.
  - Requirements: the proxy must be derived from task performance measures that are declared to index MWL under this contract and must be timestamp-alignable to EEG.

Not allowed label sources (disallowed in all contexts):

- Physiology-derived pseudo-labels (including but not limited to labels computed from EEG, ECG/HRV, EDA, respiration, pupilometry, eye metrics, or any other biosignal).
- Any labels inferred from the same EEG stream used as input (including teacher-student pseudo-labeling, self-supervised targets, or circular definitions).
- Free-text annotations without a deterministic mapping to the MWL target space.

## 4. Temporal Support and Window Alignment

The contract supports three label temporal supports:

- Block-level labels: labels apply over a contiguous time interval with a defined start and end time.
- Trial-level labels: labels apply over a contiguous time interval corresponding to a trial with a defined start and end time.
- Continuous labels: labels are defined as a time series on a declared sampling grid with explicit timestamps.

Deterministic label assignment rule to EEG windows:

- Each EEG window is assigned a label based on the window’s centre timestamp.
- For block-level and trial-level labels, a window is assigned the label of the unique interval that contains the window centre timestamp.
- For continuous labels, a window is assigned the label value at the window centre timestamp using the dataset-provided timestamped value at that timebase.

Boundary handling:

- If the window centre timestamp is exactly equal to an interval boundary, the window is assigned to the earlier interval.
- If no label interval/value covers the window centre timestamp, the window is unlabeled and is not eligible for supervised use.
- If label timing is ambiguous (overlapping intervals without a deterministic precedence rule), the recording is incompatible.

## 5. Cross-Dataset Harmonisation Rules

All admissible datasets must be mapped to a common MWL target space under the following rules:

Harmonisation is within-dataset only: each dataset’s labels may be transformed using rules in this section without using information from other datasets.

- The contract assumes that MWL labels provide at least a valid monotonic ordering with respect to the MWL semantic definition in Section 2.
- Absolute numeric comparability across datasets is not assumed unless the dataset explicitly declares a common, shared scale definition that is consistent across datasets.

Permitted harmonisation operations:

- Monotonic direction correction (inversion) only when directionality is determinable and must yield higher value = higher MWL.
- Monotonic re-scaling of numeric labels to a common bounded target range declared as `TBD_MWL_TARGET_RANGE_V0`.
- Discretisation into a fixed number of ordered levels declared as `TBD_MWL_NUM_LEVELS_V0`, preserving ordering.

Incompatibility conditions:

- Label directionality cannot be determined.
- Labels are defined for a construct outside Section 2.
- Labels cannot be aligned to EEG windows deterministically under Section 4.
- Label semantics vary within a dataset in a way that prevents a single mapping to the common target space.

## 6. Calibration Label Usage

Calibration labels may be used only if they are admissible under Section 3 and alignable under Section 4.

Calibration labels are allowed to influence:

- Participant-specific output scaling parameters.
- Participant-specific decision thresholds or bin boundaries applied to the MWL target space.
- Selection or validation of baseline/reference segments used to compute participant-specific normalisation statistics, provided the baseline/reference segment definition is fixed and declared.

Calibration labels must not influence:

- The semantic definition of MWL in Section 2.
- Any cross-dataset harmonisation rule in Section 5.
- Any preprocessing, windowing, or alignment rules defined by the system’s input contract.
- Any online-time updating of label mappings during inference.

## 7. Online Inference Output Semantics

For each EEG window, the system outputs a single MWL estimate `y` in the declared MWL target space `TBD_MWL_TARGET_SPACE_V0`.

Semantic guarantees:

- `y` is ordered such that larger values correspond to higher MWL as defined in Section 2.
- If the system emits a validity flag, an invalid window output indicates that `y` is not semantically interpretable under this contract for that window.

Guarantees not provided:

- No guarantee that `y` is absolutely comparable across participants.
- No guarantee that `y` is absolutely comparable across datasets.
- No guarantee that differences in `y` correspond to equal-magnitude changes in MWL.
- No guarantee that `y` is a calibrated probability or a clinical score.

## 8. Non-Goals

This contract intentionally does not support:

- Labeling for constructs outside the MWL semantic definition in Section 2.
- Physiology-derived pseudo-labeling or any circular label construction from EEG.
- Ambiguous or non-deterministic label-to-window alignment.
- Cross-participant absolute MWL comparability guarantees.
- Automated discovery of label semantics when a dataset does not declare them.
