from abc import ABC, abstractmethod
from dataclasses import dataclass

from flybrain_robot.telemetry import Telemetry
from flybrain_robot.vision import SensoryInput


@dataclass(frozen=True)
class MotorActivity:
    left: float = 0.0
    right: float = 0.0
    escape: float = 0.0


class BrainBackend(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, sensory_input: SensoryInput, telemetry: Telemetry, dt: float): ...

    @abstractmethod
    def get_motor_activity(self) -> MotorActivity: ...
