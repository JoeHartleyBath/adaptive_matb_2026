"""adaptation package

Online staircase calibration for MATB sessions.

Public API
----------
DifficultyState        — continuous difficulty scalar + derived task parameters
StaircaseController    — sliding-window 1-up/1-down staircase
PoissonEventGenerator  — Poisson-process event scheduler driven by DifficultyState
AdaptationScheduler    — OpenMATB Scheduler subclass wiring the above together
"""

from .difficulty_state import DifficultyState, TaskParams
from .staircase_controller import StaircaseController
from .event_generators import PoissonEventGenerator

__all__ = [
    "DifficultyState",
    "TaskParams",
    "StaircaseController",
    "PoissonEventGenerator",
]
