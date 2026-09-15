"""Whole-CNS MaleCNS v1.0 backend: camera and body features in, descending spikes out.

All 166,700 neurons and their signed synapse counts are simulated with the LIF
model in ``lif.py``. Only the interfaces are chosen by hand, and all are declared
below: which sensory neurons receive camera and tilt features, and which
descending neurons are read out as motor commands. Everything between them is the
connectome. See docs/MALECNS_INTEGRATION.md for the evidence behind each choice and
docs/VALIDATION.md for measured responses.
"""

import math
from dataclasses import dataclass

import numpy as np

from flybrain_robot.telemetry import tilt_features

from .base import BrainBackend, MotorActivity

SOURCE = "https://male-cns.janelia.org/download/"
# Metabotropic transmitters act over hundreds of ms to seconds, not as fast synapses.
MODULATORS = ("dopamine", "octopamine", "serotonin")


@dataclass(frozen=True)
class Drive:
    feature: str
    populations: tuple
    max_rate_hz: float = 150.0  # stimulation rate used by Shiu et al. 2024
    # Feature values up to this floor are encoder noise (a camera pan reaches ~0.09).
    floor: float = 0.1

    def rate(self, value):
        if not math.isfinite(value):
            return 0.0
        return self.max_rate_hz * min(max((value - self.floor) / (1 - self.floor), 0.0), 1.0)


SENSORY_MAP = (
    # Looming-sensitive visual projection neurons (LPLC2: Klapoetke et al. 2017;
    # LC4: von Reyn et al. 2017). Soma side L = dendrites in the left optic lobe.
    Drive("looming_left", ("LPLC2:L", "LC4:L")),
    Drive("looming_right", ("LPLC2:R", "LC4:R")),
    # LC10a responds to small moving objects and supports visual pursuit
    # (Ribeiro et al. 2018; Hindmarsh Sten et al. 2021).
    Drive("object_left", ("LC10a:L",)),
    Drive("object_right", ("LC10a:R",)),
)

BODY_MAP = (
    # Johnston's organ: JO-C neurons respond when the antenna is pushed toward the head,
    # JO-E neurons when it is pulled away (Kamikouchi et al. 2009; Yorozu et al. 2009;
    # Patella & Wilson 2018). HYPOTHESIS, not measured: nose-up tilt swings both aristae
    # back (push), nose-down forward (pull), and a roll pushes the lower-side antenna and
    # pulls the upper one. Tilt features already contain their 3 degree floor.
    Drive("tilt_back", ("JO-C*:L", "JO-C*:R"), floor=0.0),
    Drive("tilt_forward", ("JO-E*:L", "JO-E*:R"), floor=0.0),
    Drive("tilt_left", ("JO-C*:L", "JO-E*:R"), floor=0.0),
    Drive("tilt_right", ("JO-C*:R", "JO-E*:L"), floor=0.0),
)

READOUT = {
    # Giant Fiber, the command neuron for the escape takeoff (von Reyn et al. 2014).
    "escape": ("DNp01:L", "DNp01:R"),
    # DNa01/DNa02 activity predicts ipsilateral turning (Rayshubskiy et al. 2020).
    "turn_left": ("DNa01:L", "DNa02:L"),
    "turn_right": ("DNa01:R", "DNa02:R"),
}
# Logged and visualised only; they do not drive motors. DNp18 and DNp20 (DNOVS1) have
# the strongest mirrored responses to Johnston's organ input (VALIDATION.md, body).
MONITOR = ("DNp02", "DNp04", "DNp11", "DNp09", "MDN", "DNp18", "DNp20")

RATE_TAU_MS = 100.0  # smoothing of spike counts into firing rates
TURN_FULL_HZ = 60.0  # left/right rate difference that commands a full turn
ESCAPE_FULL_HZ = 60.0  # Giant Fiber rate that maps to escape activity 1.0
MAX_STEP_MS = 100.0  # upper bound on simulated time per control tick


class MaleCNSBackend(BrainBackend):
    def __init__(
        self,
        dataset_path=None,
        seed=0,
        cruise=0.4,
        fast_modulators=False,
        wiring="connectome",
        params=None,
        body_feedback=False,
    ):
        from flybrain_robot.connectome.graph import load_connectome

        from .lif import LIFSimulator

        if not dataset_path:
            raise ValueError(f"MaleCNS dataset path is missing. Obtain data from {SOURCE}")
        graph = load_connectome(dataset_path)
        if not fast_modulators:
            graph = graph.without_fast_synapses_from(MODULATORS)
        if wiring == "shuffled":  # negative control: same neurons, random targets
            graph = graph.shuffled(seed)
        elif wiring != "connectome":
            raise ValueError("wiring must be 'connectome' or 'shuffled'")
        if not 0 <= cruise <= 1:
            raise ValueError("cruise must be between 0 and 1")
        self.graph = graph
        self.cruise = cruise
        # Off by default: in VALIDATION.md a 20 degree roll reaches the Giant Fiber and
        # DNa02 through the connectome and changes motor commands.
        self.body_feedback = body_feedback
        self.sim = LIFSimulator(graph, params=params, seed=seed)
        # Visual populations first, in map order: the Poisson generator draws in this order.
        self.populations = {
            p: graph.select(p) for d in SENSORY_MAP + BODY_MAP for p in d.populations
        }
        self.readout = {
            name: [graph.select(spec) for spec in specs] for name, specs in READOUT.items()
        }
        self.groups = dict(self.populations)
        self.groups.update({s: graph.select(s) for specs in READOUT.values() for s in specs})
        self.groups.update(
            {f"{name}:{side}": graph.select(f"{name}:{side}") for name in MONITOR for side in "LR"}
        )
        self.sim.run(0.1)  # compile the simulation kernel before the control loop starts
        self.reset()

    def reset(self):
        self.sim.reset()
        self.rates = np.zeros(self.graph.size)
        self.activity = MotorActivity()

    @property
    def neuron_count(self):
        return self.graph.size

    def drive_rates(self, sensory_input, telemetry):
        """Poisson rate (Hz) per driven population; overlapping drives add, capped at 150 Hz."""
        features = tilt_features(telemetry)
        rates = dict.fromkeys(self.populations, 0.0)
        for d in SENSORY_MAP + (BODY_MAP if self.body_feedback else ()):
            value = features[d.feature] if d in BODY_MAP else getattr(sensory_input, d.feature)
            for population in d.populations:
                rates[population] = min(rates[population] + d.rate(value), d.max_rate_hz)
        return rates

    def step(self, sensory_input, telemetry, dt):
        """Advance the connectome by ``dt`` seconds (at most MAX_STEP_MS)."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        duration = min(dt * 1000, MAX_STEP_MS)
        rates = self.drive_rates(sensory_input, telemetry)
        self.sim.set_stimulus([(self.populations[p], hz) for p, hz in rates.items()])
        self.sim.run(duration)
        counts = self.sim.take_counts()
        alpha = 1 - math.exp(-duration / RATE_TAU_MS)
        self.rates += alpha * (counts / (duration / 1000) - self.rates)
        escape = max(self.rate(idx) for idx in self.readout["escape"]) / ESCAPE_FULL_HZ
        left = sum(self.rate(idx) for idx in self.readout["turn_left"])
        right = sum(self.rate(idx) for idx in self.readout["turn_right"])
        turn = float(np.clip((left - right) / TURN_FULL_HZ, -1, 1))  # positive = turn left
        self.activity = MotorActivity(
            left=float(np.clip(self.cruise - turn, -1, 1)),
            right=float(np.clip(self.cruise + turn, -1, 1)),
            escape=float(min(escape, 1.0)),
        )
        return self.activity

    def rate(self, indices):
        return float(self.rates[indices].mean())

    def named_rates(self):
        """Smoothed firing rates (Hz) of every stimulated, read-out and monitored group."""
        return {name: self.rate(idx) for name, idx in self.groups.items()}

    def get_motor_activity(self):
        return self.activity
