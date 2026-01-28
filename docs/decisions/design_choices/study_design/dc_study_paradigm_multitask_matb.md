# Design choice: Multitask MATB(-II)-style paradigm

**Decision**  
Use an MATB(-II)-style multitask paradigm (sysmon + tracking + communications + resource management, with scheduling display) as the core experimental task ecology.

**Rationale**  
- Is well-established in mental workload research, enabling comparison with prior MATB(-II)-style studies while remaining implementable within the current instrumentation constraints.
- Provides a structured multitask environment in which workload can be manipulated via difficulty and event structure without introducing major task-identity confounds, supporting stable interpretation of behavioural and physiological effects.
- Has existing public datasets from MATB(-II)-style paradigms, enabling optional pretraining or benchmarking of workload models if required.
- Involves minimal whole-body movement and limited speech, reducing motion- and muscle-related EEG artefacts and improving the interpretability of neural measures during adaptation.
- Is open-source and inspectable, reducing platform lock-in and development overhead while improving reproducibility and transparency relative to proprietary or high-fidelity simulators.



**Alternatives considered**  
- Single-task paradigms (e.g., n-back, Stroop, SART).
- Higher-fidelity simulators (e.g., driving/flight) with heavier infrastructure demands.
- A custom-built multitask UI (higher maintenance and replication burden).

**Implications**  
This constrains session design, logging, and analysis to a multitask workload construct; all workload manipulations and adaptive logic are defined relative to concurrent task demands rather than isolated tasks.

**Status**  
Final (foundational)

**References**  
- Internal: `docs/decisions/2026-01-28_methods_paradigm_and_platform.md`
- Internal: `docs/pilot/PILOT_STUDY_SPEC_V0.md`
- Pontiggia et al., 2024 — review of MATB(-II)-style paradigms for mental workload assessment
