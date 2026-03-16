"""adaptation_logger.py

Structured CSV audit logger for MWL-driven adaptation decisions.

This logger is SEPARATE from the OpenMATB session CSV.  It records
every adaptation decision tick with full context so that post-hoc
analysis can reconstruct the controller's behaviour without parsing
OpenMATB's internal log format.

No vendor / LSL / torch imports — safe to import and test anywhere.

Schema
------
timestamp_lsl       – LSL local_clock() at decision time
scenario_time_s     – elapsed scenario time (seconds)
mwl_raw             – most recent raw MWL from the estimator
mwl_smoothed        – EMA-smoothed MWL value
signal_quality      – signal quality channel from estimator [0, 1]
threshold           – current decision threshold
action              – one of: assist_on | assist_off | hold
assistance_on       – whether tracking assistance is currently active (True/False)
cooldown_remaining_s – seconds left in cooldown at tick time
hold_counter_s      – continuous seconds spent in current side of threshold
reason              – human-readable reason string
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TextIO


# Column order — used for header and row dictionaries.
COLUMNS = [
    "timestamp_lsl",
    "scenario_time_s",
    "mwl_raw",
    "mwl_smoothed",
    "signal_quality",
    "threshold",
    "action",
    "assistance_on",
    "cooldown_remaining_s",
    "hold_counter_s",
    "reason",
]


class AdaptationLogger:
    """Append-only CSV logger for MWL adaptation audit trail.

    Parameters
    ----------
    path : str | Path
        File path for the CSV.  Parent directory is created if needed.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        """Open the CSV file and write the header row."""
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "AdaptationLogger":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- logging -------------------------------------------------------------

    def log(
        self,
        *,
        timestamp_lsl: float,
        scenario_time_s: float,
        mwl_raw: float,
        mwl_smoothed: float,
        signal_quality: float,
        threshold: float,
        action: str,
        assistance_on: bool,
        cooldown_remaining_s: float,
        hold_counter_s: float,
        reason: str,
    ) -> None:
        """Write one decision-tick row to the audit CSV.

        All parameters are keyword-only to prevent column-order bugs.
        """
        if self._writer is None:
            raise RuntimeError("AdaptationLogger is not open; call .open() first")

        self._writer.writerow(
            {
                "timestamp_lsl": f"{timestamp_lsl:.6f}",
                "scenario_time_s": f"{scenario_time_s:.3f}",
                "mwl_raw": f"{mwl_raw:.6f}",
                "mwl_smoothed": f"{mwl_smoothed:.6f}",
                "signal_quality": f"{signal_quality:.3f}",
                "threshold": f"{threshold:.4f}",
                "action": action,
                "assistance_on": assistance_on,
                "cooldown_remaining_s": f"{cooldown_remaining_s:.1f}",
                "hold_counter_s": f"{hold_counter_s:.2f}",
                "reason": reason,
            }
        )
        self._file.flush()
