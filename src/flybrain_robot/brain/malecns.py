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

# Vision "retina" (experimental): instead of flow features, darkening at each lamina
# column's pixel drives that column's L2 cell. L2 is cholinergic and feeds the OFF
# motion pathway (Tm1, Tm2, Tm4 -> T5). The ON pathway starts with glutamatergic L1,
# which signals light by releasing less glutamate; a model that only adds spikes to
# neurons at rest cannot express that, so L1 is not driven. Column positions come from
# the connectome (connectome/retinotopy.py). Rates: 0 Hz up to 5 % darkening, 150 Hz
# at 50 % or more.
RETINA_TYPE = "L2"
RETINA_FLOOR = 0.05
RETINA_FULL = 0.5
RETINA_MONITOR = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "HSE", "HSN", "HSS")

# Vision "eye" (experimental): the camera drives the photoreceptors themselves, at a rate
# proportional to the brightness of their pixel, and the optic lobe carries the signal.
# Two changes make that possible and are declared here:
#   1. Columnar neurons of the optic lobe fire tonically (a resting potential above
#      threshold, EYE_TONIC_HZ without input). Most of these cells do not spike in the
#      fly: they release transmitter continuously and signal in both directions. R1-R6
#      are histaminergic, so light inhibits the lamina and darkness releases it; without
#      a baseline to release, a spiking model cannot carry the signal at all.
#   2. CT1, one cell per eye, is split into its columns (connectome/columns.py), because
#      its compartments are electrically isolated in the fly (Meier & Borst 2019). As a
#      single point neuron it inhibited every column of the eye at once.
# Columns missing from the release (T4, T5, Tm3, L4, the photoreceptors) are recovered
# from the wiring by connectome/columns.py.
# The inputs of T4 and T5 that recordings report as slow and sustained, against the fast
# transient ones (Mi1, Tm3, Tm1, Tm2): Mi9 and Tm9 (Arenz et al. 2017; Strother et al.
# 2017; Serbe et al. 2016) and the delayed inhibition of Mi4 and CT1 (Arenz et al. 2017;
# Meier & Borst 2019). With one time constant for every synapse, a moving pattern
# delivers the same input in both directions and no motion detector can be selective.
# EYE_SLOW_TAU_MS is a hand-chosen value in the reported range, not a measurement.
# Off by default: VALIDATION.md 5c measures no direction selectivity either way, and the
# slow inhibitory arms suppress T4 and T5 about tenfold (T4a of the left eye 8.9 -> 0.7 Hz).
EYE_SLOW_TYPES = ("Mi9", "Tm9", "Mi4", "CT1")
EYE_SLOW_TAU_MS = 50.0
EYE_PHOTORECEPTORS = ("R1-R6", "R7*", "R8*")
EYE_TONIC_HZ = 20.0
EYE_FULL_HZ = 150.0  # photoreceptor rate at full brightness
EYE_MONITOR = ("L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Tm3", "Tm9", "CT1",
               "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "HSE", "HSN", "HSS")  # fmt: skip
VISION_MODES = ("flow", "retina", "eye")

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
        device="auto",
        vision="flow",
        slow_inputs=False,
    ):
        from flybrain_robot.connectome.graph import load_connectome

        from .lif import make_simulator

        if not dataset_path:
            raise ValueError(f"MaleCNS dataset path is missing. Obtain data from {SOURCE}")
        graph = load_connectome(dataset_path)
        if not fast_modulators:
            graph = graph.without_fast_synapses_from(MODULATORS)
        if vision not in VISION_MODES:
            raise ValueError(f"vision must be one of {VISION_MODES}")
        self.vision = vision
        self.v_rest = self.slow = None
        if vision == "eye":
            from flybrain_robot.connectome.columns import compartmentalize, impute_columns

            graph = compartmentalize(impute_columns(graph))
        if vision in ("retina", "eye"):
            # From the real wiring even for the shuffled control: the interface stays fixed.
            from flybrain_robot.connectome.retinotopy import column_image_positions, eye_axes

            driven = RETINA_TYPE if vision == "retina" else EYE_PHOTORECEPTORS
            self.eye_axes = {side: eye_axes(graph, side) for side in "LR"}
            columns = [column_image_positions(graph, driven, self.eye_axes[side])
                       for side in "LR"]  # fmt: skip
            self.retina_cells = np.concatenate([c[0] for c in columns])
            self.retina_x = np.concatenate([c[1] for c in columns])
            self.retina_y = np.concatenate([c[2] for c in columns])
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
        if vision == "eye":
            from dataclasses import replace

            from .lif import LIFParams, resting_for_rate

            settings = params or LIFParams.matched_to(graph.manifest)
            if slow_inputs:
                settings = replace(settings, tau_slow_ms=EYE_SLOW_TAU_MS)
                self.slow = np.isin(np.asarray(graph.type), list(EYE_SLOW_TYPES))
            params = settings
            tonic = (np.asarray(graph.superclass) == "ol_intrinsic") & (np.asarray(graph.hex1) >= 0)
            self.v_rest = np.full(graph.size, settings.v_0)
            self.v_rest[tonic] = resting_for_rate(settings, EYE_TONIC_HZ)
            self.tonic_cells = int(tonic.sum())
        self.sim = make_simulator(graph, params=params, seed=seed, device=device,
                                  v_rest=self.v_rest, slow=self.slow)  # fmt: skip
        visual = SENSORY_MAP if vision == "flow" else ()
        self.populations = {p: graph.select(p) for d in visual + BODY_MAP for p in d.populations}
        self.readout = {
            name: [graph.select(spec) for spec in specs] for name, specs in READOUT.items()
        }
        self.groups = {p: graph.select(p) for d in SENSORY_MAP + BODY_MAP for p in d.populations}
        self.groups.update({s: graph.select(s) for specs in READOUT.values() for s in specs})
        self.groups.update(
            {f"{name}:{side}": graph.select(f"{name}:{side}") for name in MONITOR for side in "LR"}
        )
        if vision in ("retina", "eye"):
            monitor = RETINA_MONITOR if vision == "retina" else EYE_MONITOR
            self.groups.update({f"{name}:{side}": graph.select(f"{name}:{side}")
                                for name in monitor for side in "LR"})  # fmt: skip
        self.device = self.sim.device
        # One full block compiles every kernel before the control loop starts.
        self.sim.run(self.sim.params.t_dly_ms)
        self.reset()

    def reset(self):
        self.sim.reset()
        self.rates = np.zeros(self.graph.size)
        self.last_counts = np.zeros(self.graph.size, np.int32)
        self.activity = MotorActivity()

    @property
    def neuron_count(self):
        return self.graph.size

    def drive_rates(self, sensory_input, telemetry):
        """Poisson rate (Hz) per driven population; overlapping drives add, capped at 150 Hz."""
        features = tilt_features(telemetry)
        rates = dict.fromkeys(self.populations, 0.0)
        visual = SENSORY_MAP if self.vision == "flow" else ()
        for d in visual + (BODY_MAP if self.body_feedback else ()):
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
        stimulus = [(self.populations[p], hz) for p, hz in rates.items()]
        if self.vision in ("retina", "eye"):
            stimulus.append((self.retina_cells, self.retina_rates(sensory_input.retina)))
        self.sim.set_stimulus(stimulus)
        self.sim.run(duration)
        counts = self.last_counts = self.sim.take_counts()
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

    def retina_rates(self, values):
        """Poisson rate per driven cell: darkening in retina mode, brightness in eye mode."""
        if values is None:
            return 0.0
        value = np.nan_to_num(np.asarray(values, np.float64))
        if self.vision == "eye":
            return EYE_FULL_HZ * np.clip(value, 0.0, 1.0)
        scaled = (value - RETINA_FLOOR) / (RETINA_FULL - RETINA_FLOOR)
        return 150.0 * np.clip(scaled, 0.0, 1.0)

    def make_encoder(self):
        """The camera encoder this backend expects."""
        from flybrain_robot.vision import LuminanceEncoder, RetinaEncoder, VisionEncoder

        if self.vision == "retina":
            return RetinaEncoder(self.retina_x, self.retina_y)
        if self.vision == "eye":
            return LuminanceEncoder(self.retina_x, self.retina_y)
        return VisionEncoder()

    def rate(self, indices):
        return float(self.rates[indices].mean())

    def named_rates(self):
        """Smoothed firing rates (Hz) of every stimulated, read-out and monitored group."""
        return {name: self.rate(idx) for name, idx in self.groups.items()}

    def get_motor_activity(self):
        return self.activity
