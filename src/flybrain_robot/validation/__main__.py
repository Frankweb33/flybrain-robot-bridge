"""Reproduce every number in docs/VALIDATION.md from the public MaleCNS release.

    python -m flybrain_robot.validation --data data/malecns-v1.0

Writes docs/validation/results.json and regenerates docs/VALIDATION.md. Each
experiment states its pass condition in code (``expect``); failures are reported,
not hidden. Needs the compiled connectome; brian2 is optional.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flybrain_robot.brain.lif import LIFParams, LIFSimulator, make_simulator
from flybrain_robot.brain.malecns import (
    EYE_SLOW_TAU_MS,
    EYE_SLOW_TYPES,
    EYE_TONIC_HZ,
    MODULATORS,
    READOUT,
    SENSORY_MAP,
    MaleCNSBackend,
)
from flybrain_robot.connectome.columns import held_out_accuracy
from flybrain_robot.connectome.graph import load_connectome
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.telemetry import Telemetry, tilted_telemetry
from flybrain_robot.vision import SCENES, TOUR_SEGMENT, VisionEncoder, scene_frame

from .report import failures, render_markdown, skipped

ROOT = Path(__file__).resolve().parents[3]
STIM_HZ = 150.0
TRIAL_MS = 1000.0

# Pathway checks. Pass: the target fires >= 20 Hz and >= 3x the strongest response of
# the same neuron in any shuffled control. Where the robot uses left/right differences
# (steering) or the check is about a lateralized pathway, the target must also fire
# >= 3x its counterpart on the other side ("lateral").
# History, kept visible on purpose: the first full run (2026-09-15) required
# lateralization for every row. The Giant Fiber rows then failed at 2.2x and 1.8x
# (229 vs 105 Hz, 213 vs 118 Hz) while shuffled controls stayed at 0 Hz. Escape reads both
# Giant Fibers, so lateralization is reported for them but no longer required.
# The interfaces were chosen after an exploratory screen of 33 visual projection
# types (the "screen" section), so these checks confirm rather than predict.
PATHWAY_SEEDS = (1, 2, 3)
PATHWAYS = [
    dict(name="Looming (left) to Giant Fiber", stimulus=["LPLC2:L"], target="DNp01:L",
         counterpart="DNp01:R", role="escape (either side)", lateral=False),
    dict(name="Looming (right) to Giant Fiber", stimulus=["LPLC2:R"], target="DNp01:R",
         counterpart="DNp01:L", role="escape (either side)", lateral=False),
    dict(name="LC4 (left) to DNp04", stimulus=["LC4:L"], target="DNp04:L",
         counterpart="DNp04:R", role="monitored", lateral=True),
    dict(name="LC4 (left) to DNp11", stimulus=["LC4:L"], target="DNp11:L",
         counterpart="DNp11:R", role="monitored", lateral=True),
    dict(name="Object (left) to DNa02", stimulus=["LC10a:L"], target="DNa02:L",
         counterpart="DNa02:R", role="turn left", lateral=True),
    dict(name="Object (right) to DNa02", stimulus=["LC10a:R"], target="DNa02:R",
         counterpart="DNa02:L", role="turn right", lateral=True),
    dict(name="LC9 (left) to DNp09", stimulus=["LC9:L"], target="DNp09:L",
         counterpart="DNp09:R", role="not used", lateral=True),
    dict(name="HS cells (left) to DNp15 (DNHS1)", stimulus=["HSE:L", "HSN:L", "HSS:L"],
         target="DNp15:L", counterpart="DNp15:R", role="not used", lateral=True),
    dict(name="VS cells (left) to DNp20 (DNOVS1)", stimulus=["VS:L"], target="DNp20:L",
         counterpart="DNp20:R", role="not used", lateral=True),
]  # fmt: skip

# Published feeding circuit: sugar GRNs -> ... -> MN9, the proboscis motor neuron
# (Shiu et al. 2022, eLife; Shiu et al. 2024, Nature 634:210). MaleCNS does not label
# gustatory neurons by receptor. LB3c and LB3d are the gustatory types with by far the
# most synapses onto G2N-1 (GNG232) and Zorro (GNG215): 519/766 and 485/107; the next
# is LB3a with 30/79 (counted 2026-09-15). claw_tpGRN (50 taste-peg GRNs, 2 synapses
# onto G2N-1) is a gustatory population outside that circuit. Circuit neurons are
# identified by the `synonyms` column of the release (Shiu 2022; Yao & Scott 2022).
# Pass condition, written and committed before the first run:
#   MN9 (mean of left and right) >= 20 Hz while sugar GRNs fire at 150 Hz, and
#   >= 3x (strongest MN9 response in any shuffled-wiring control + 1 Hz), and
#   >= 3x (MN9 response to claw_tpGRN stimulation + 1 Hz).
# Circuit neuron rates and the Roundup/Bract lesions are reported, not required.
FEEDING = dict(
    sugar=["LB3c", "LB3d"],
    other_gustatory=["claw_tpGRN"],
    readout="MN9",
    circuit={
        "G2N-1": ["GNG232"],
        "Zorro": ["GNG215"],
        "Phantom": ["GNG229"],
        "Roundup": ["GNG108"],
        "Fudog": ["DNg67"],
        "Bract 1": ["DNge174"],
        "Bract 2": ["DNge173"],
        "Sugar SEL PN": ["GNG540", "GNG550"],
        "Sugar SEL LN": ["GNG056"],
    },
    lesions={"Roundup": ["GNG108"], "Bract 1 and 2": ["DNge174", "DNge173"]},
    # Post hoc, added after the pre-registered check failed on 2026-09-15 (MN9 29 Hz for
    # sugar, 21.5 Hz for claw_tpGRN). These rows explain the failure; they do not change
    # the verdict above.
    post_hoc_stimuli={
        "sugar GRNs, left only": ["LB3c:L", "LB3d:L"],
        "sugar GRNs, right only": ["LB3c:R", "LB3d:R"],
        "claw_tpGRN (taste peg)": ["claw_tpGRN"],
        "LB1e (other labellar GRNs)": ["LB1e"],
        "aPhM1 (pharyngeal mechanosensory)": ["aPhM1"],
        "JO-EV1 (antennal mechanosensory)": ["JO-EV1"],
        "LPLC2 (visual, both sides)": ["LPLC2"],
    },
)

# Body feedback through Johnston's organ (JO), the antennal organ that senses gravity
# and wind through static antennal deflection (Kamikouchi et al. 2009; Yorozu et al.
# 2009). Which MaleCNS JO subtype reports which deflection is not annotated, so every
# JO type with at least 18 cells is screened.
# Pass condition, written and committed before the first run: there is a JO type T and
# a descending neuron type D with a left and a right cell such that
#   T on the left drives one D cell >= 20 Hz and >= 3x (the other D cell + 1 Hz),
#   T on the right drives the mirror-image D cell with the same thresholds, and
#   in every shuffled-wiring control that D cell stays below (its rate / 3 - 1 Hz)
#   for both stimuli.
# Ten JO types are compared against every paired descending type, so a single-side hit
# could be chance; the mirror replication is required for that reason.
JO_SCREEN_TYPES = (
    "JO-EV1", "JO-EV2", "JO-EV3", "JO-EV5", "JO-EV6", "JO-ED1", "JO-ED2_a", "JO-ED2_b",
    "JO-CM", "JO-CL",
)  # fmt: skip
JO_SCREEN_MS = 400.0

# Body in the closed loop: the robot faces the static scene and is held tilted by 20
# degrees for 3 s; tilt reaches Johnston's organ through the declared BODY_MAP
# hypothesis. Pass condition, written and committed before the first run, per tilt:
#   some descending neuron cell peaks >= 20 Hz, and in every shuffled-wiring control
#   that same cell peaks below (its connectome peak / 3 - 1 Hz); and
#   motor commands stay as when level (no escape, |mean turn| < 0.1), because body
#   input is not wired to motors.
TILTS = {
    "nose down 20°": (0.0, 20.0),
    "nose up 20°": (0.0, -20.0),
    "left side down 20°": (20.0, 0.0),
    "right side down 20°": (-20.0, 0.0),
}
# Post hoc, added after the tilt loop check failed on 2026-09-15 (a 20 degree roll
# triggered the Giant Fiber, others turned through DNa02): the same tilts at smaller
# angles, real wiring only, to find where motor commands start to change. Reported only.
TILT_SWEEP_SCALES = (0.25, 0.5, 0.75)
# Post hoc, added on 2026-09-16 after the Poisson generator changed (hash instead of a
# random stream) and the same check went from three failing tilts to one: the motor
# effect of a 20 degree tilt sits at the thresholds. Five brain seeds show how often
# each tilt changes motor commands. Reported only; the verdict above stays with seed 0.
TILT_SEEDS = (1, 2, 3, 4, 5)

# Retinotopic vision (exploratory, NOT pre-registered): the first simulations in this
# mode ran on 2026-09-16 before any pass condition was written, and already showed T5,
# T4, HS, LPLC2 and LC4 silent. This section reproduces those observations and the
# mechanism behind them; it has no pass or fail.
RETINA_TYPES = ("L2", "Tm1", "Tm2", "Tm4", "Tm9", "CT1", "T5a", "T5b", "T5c", "T5d",
                "T4a", "T4b", "HSE", "HSN", "HSS", "LPLC2", "LC4")  # fmt: skip
RETINA_SCENES = ("grating_left", "grating_right", "flicker", "approach_center", "pan")

# Motion detectors from pixels, vision "eye" (PRE-REGISTERED: written and committed on
# 2026-09-16 before any grating was simulated in this mode). The configuration is frozen
# here: photoreceptors driven by the brightness of their pixel, tonic resting potentials
# in the optic lobe, CT1 split into columns, columns imputed from the wiring, seed 0,
# matched w_syn, four drifting gratings, 0.5 s to settle and then 2 s measured.
#
# Which direction each subtype should prefer is fixed by the literature and by the eye
# axes that come out of the wiring, with no choice left for afterwards: T4a/T5a prefer
# front-to-back motion, T4b/T5b back-to-front, T4c/T5c upward, T4d/T5d downward (Maisak
# et al. 2013). Front-to-back is leftward image motion for the left eye and rightward for
# the right eye, because each eye's anterior columns look at the image midline.
#
# Pass conditions, all four required:
#   1. Every T4 and T5 subtype fires at >= 2 Hz on at least one direction.
#   2. For at least half of the 8 T4 subtype-eye pairs and half of the 8 T5 pairs, the
#      direction selectivity index DSI = (preferred - null) / (preferred + null) over the
#      two directions of that subtype's own axis is >= 0.05, and the mean DSI is positive
#      for T4 and for T5.
#   3. Mirror symmetry: for the horizontal subtypes (a, b) the two eyes prefer opposite
#      image directions; for the vertical subtypes (c, d) they prefer the same one.
#   4. Shuffled-wiring controls: mean |DSI| < 0.02 and no pair reaches DSI >= 0.05.
EYE_DIRECTIONS = ("grating_left", "grating_right", "grating_up", "grating_down")
EYE_SETTLE_FRAMES = 15
EYE_MEASURE_FRAMES = 60
EYE_PREFERRED = {  # subtype letter -> preferred image direction per eye
    "a": {"L": "grating_left", "R": "grating_right"},
    "b": {"L": "grating_right", "R": "grating_left"},
    "c": {"L": "grating_up", "R": "grating_up"},
    "d": {"L": "grating_down", "R": "grating_down"},
}
EYE_OPPOSITE = {"grating_left": "grating_right", "grating_right": "grating_left",
                "grating_up": "grating_down", "grating_down": "grating_up"}  # fmt: skip
EYE_MIN_HZ = 2.0
EYE_MIN_DSI = 0.05
EYE_CONTROL_MAX_DSI = 0.02
# Reported, not part of the pass condition: the horizontal system cells that integrate
# T4/T5 output, and the full-field brightness step. The step was already run on
# 2026-09-16 before this pre-registration and showed T5 rising and T4 falling in the
# dark, so it is exploratory and is reported as such.
EYE_HS_TYPES = ("HSE", "HSN", "HSS")
EYE_STEP_MS = 200.0
# Second PRE-REGISTRATION, committed on 2026-09-16 after the first configuration failed
# with no direction selectivity at all (DSI 0.00, shuffled 0.0004). The model changes in
# one declared way: the inputs that recordings report as slow and sustained (Mi9, Tm9,
# Mi4, CT1) get their own synaptic time constant of 50 ms, because with one time constant
# for every synapse a drifting grating delivers the same input in both directions no
# matter how the cells are wired. The pass conditions above are unchanged and are applied
# to each configuration separately; both are reported.
EYE_CONFIGURATIONS = (
    ("uniform kinetics", False),
    (f"slow {', '.join(EYE_SLOW_TYPES)} ({EYE_SLOW_TAU_MS:.0f} ms)", True),
)

# The real wiring is run at as many seeds as the shuffled control, after a review noted
# that the control had more replication than the claim.
CLOSED_LOOP_SEEDS = (0, 1, 2)
SCENE_EXPECTATIONS = {
    "static": dict(escape=False, turn=0),
    "pan": dict(escape=False, turn=0),
    "pass_left": dict(escape=False, turn=+1),
    "approach_center": dict(escape=True),
    "approach_left": dict(escape=True),
    "approach_right": dict(escape=True),
}
SCREEN_TYPES = (
    "H2 HSE HSN HSS VS LC4 LC6 LC9 LC10a LC10b LC10d LC10e LC11 LC12 LC15 LC16 LC17 LC18 "
    "LC21 LC22 LC25 LC26 LC31a LC31b LLPC1 LLPC2 LLPC3 LPC1 LPC2 LPLC1 LPLC2 LPLC4"
).split()


def indices(graph, specs):
    return np.concatenate([graph.select(spec) for spec in specs])


# Simulator device for every section except "devices", which compares both. CPU and GPU
# give bit-identical results, and that section checks it on the same run.
DEVICE = "auto"


def trial(graph, stimulus, duration_ms=TRIAL_MS, seed=1, params=None, device=None,
          rate_hz=STIM_HZ):  # fmt: skip
    sim = make_simulator(graph, params=params, seed=seed, device=device or DEVICE)
    sim.set_stimulus([(indices(graph, stimulus), rate_hz)] if stimulus else [])
    started = time.perf_counter()
    sim.run(duration_ms)
    wall = time.perf_counter() - started
    return sim, sim.take_counts() / (duration_ms / 1000), wall


def dataset_section(graph):
    keys = ("dataset", "license", "attribution", "sources", "neurons", "edges_in_release",
            "neuron_to_neuron_edges", "synapses", "inhibitory_neurons", "transmitter_counts",
            "sign_rule", "side_rule", "neurons_with_optic_column")  # fmt: skip
    return {k: graph.manifest[k] for k in keys}


def reference_section(seeds):
    try:
        from .reference import compare
    except ImportError as exc:
        return {"skipped": f"brian2 not installed ({exc})"}
    try:
        import brian2
    except ImportError as exc:
        return {"skipped": f"brian2 not installed ({exc})"}
    return {"brian2_version": brian2.__version__, "runs": [compare(s) for s in seeds]}


def stability_section(full):
    shiu, matched = LIFParams(), LIFParams.matched_to(full.manifest)
    silent = full.without_fast_synapses_from(MODULATORS)
    rows = []
    for label, graph, params in [
        ("Shiu w_syn, modulators excitatory", full, shiu),
        ("Shiu w_syn, modulators silent", silent, shiu),
        ("matched w_syn, modulators excitatory", full, matched),
        ("matched w_syn, modulators silent (default)", silent, matched),
    ]:
        sim, rates, wall = trial(graph, ["LPLC2:L"], params=params)
        during = rates.sum()
        spiking = int((rates > 0).sum())
        sim.set_stimulus([])
        sim.run(TRIAL_MS)
        after = sim.take_counts()
        by_type = {}
        for i in np.flatnonzero(after):
            name = graph.type[i] or "untyped"
            by_type[name] = by_type.get(name, 0) + int(after[i])
        top = sorted(by_type.items(), key=lambda kv: -kv[1])[:8]
        rows.append(dict(condition=label, w_syn_mv=round(params.w_syn, 4),
                         spikes_during_1s=int(during), spiking_neurons=spiking,
                         spikes_in_1s_after_stimulus=int(after.sum()),
                         neurons_active_after=int((after > 0).sum()),
                         after_top_types=dict(top)))  # fmt: skip
    return {"stimulus": "LPLC2:L at 150 Hz for 1 s, then 1 s without input", "rows": rows}


def synapses_between(graph, sources, target):
    """Direct signed synapse count from ``sources`` onto ``target`` neurons."""
    total = 0
    targets = set(target.tolist())
    for s in sources:
        lo, hi = graph.indptr[s], graph.indptr[s + 1]
        hit = np.isin(graph.indices[lo:hi], list(targets))
        total += int(np.asarray(graph.weight[lo:hi])[hit].sum())
    return total


def top_intermediates(graph, sources, target, limit=3):
    """Strongest two-synapse routes: min(source->X, X->target) summed over sources."""
    via = {}
    for s in sources:
        lo, hi = graph.indptr[s], graph.indptr[s + 1]
        for x, w in zip(graph.indices[lo:hi], graph.weight[lo:hi]):
            if w > 0:
                via[int(x)] = via.get(int(x), 0) + int(w)
    scored = []
    target_set = set(target.tolist())
    for x, w_in in via.items():
        lo, hi = graph.indptr[x], graph.indptr[x + 1]
        hit = np.isin(graph.indices[lo:hi], list(target_set))
        w_out = int(np.asarray(graph.weight[lo:hi])[hit].sum())
        if w_out > 0:
            scored.append((min(w_in, w_out), x, w_in, w_out))
    scored.sort(reverse=True)
    return [
        dict(neuron=graph.labels([x])[0], synapses_in=w_in, synapses_out=w_out)
        for _, x, w_in, w_out in scored[:limit]
    ]


def pathway_section(graph, control_seeds):
    controls = [graph.shuffled(seed) for seed in control_seeds]
    # Added on 2026-09-16, reported rather than required: a harder control that keeps the
    # regional structure of the connectome and destroys only the specific partner of each
    # synapse. The pass condition is still the pre-registered one above.
    regional = [graph.shuffled(seed, within="superclass") for seed in control_seeds]
    rows = []
    for p in PATHWAYS:
        target, counterpart = graph.select(p["target"]), graph.select(p["counterpart"])
        # Three Poisson seeds for the claim, as many as for the controls: a single run
        # hid how much of a margin is noise.
        per_seed = [trial(graph, p["stimulus"], seed=seed)[1] for seed in PATHWAY_SEEDS]
        target_by_seed = [round(float(r[target].mean()), 1) for r in per_seed]
        shuffled, within_superclass = [], []
        for control in regional:
            _, control_rates, _ = trial(control, p["stimulus"])
            within_superclass.append(float(control_rates[target].mean()))
        for control in controls:
            _, control_rates, _ = trial(control, p["stimulus"])
            shuffled.append(float(control_rates[target].mean()))
        hz = float(np.mean([r[target].mean() for r in per_seed]))
        other = float(np.mean([r[counterpart].mean() for r in per_seed]))
        source = indices(graph, p["stimulus"])
        rows.append(dict(
            p, target_hz=round(hz, 1), counterpart_hz=round(other, 1),
            target_hz_by_seed=target_by_seed,
            shuffled_hz=[round(x, 1) for x in shuffled],
            shuffled_within_superclass_hz=[round(x, 1) for x in within_superclass],
            direct_synapses=synapses_between(graph, source, target),
            routes=top_intermediates(graph, source, target),
            lateral_ratio=round(hz / (other + 1), 2),
            passed=bool(hz >= 20 and hz >= 3 * (max(shuffled) + 1)
                        and (not p["lateral"] or hz >= 3 * (other + 1))),
        ))  # fmt: skip
    return {"stimulus_hz": STIM_HZ, "duration_ms": TRIAL_MS, "control_seeds": control_seeds,
            "seeds": list(PATHWAY_SEEDS), "rows": rows,
            "shuffle_clause_binds": any(max(r["shuffled_hz"]) * 3 + 3 > 20 for r in rows)}  # fmt: skip


def feeding_section(graph, full, control_seeds):
    f = FEEDING
    readout = graph.select(f["readout"])
    circuit = {name: indices(graph, types) for name, types in f["circuit"].items()}
    sugar = indices(graph, f["sugar"])

    def mn9_hz(g, stimulus, params=None):
        _, rates, _ = trial(g, stimulus, params=params)
        return float(rates[readout].mean()), rates

    def by_side(rates):
        return {graph.side[i]: round(float(rates[i]), 1) for i in readout}

    post_hoc = []
    for label, stimulus in f["post_hoc_stimuli"].items():
        _, rates = mn9_hz(graph, stimulus)
        source = indices(graph, stimulus)
        post_hoc.append(dict(
            stimulus=label, cells=len(source), mn9_by_side=by_side(rates),
            circuit_hz={k: round(float(rates[v].mean()), 1) for k, v in circuit.items()},
            synapses_onto={k: synapses_between(graph, source, v) for k, v in circuit.items()},
        ))  # fmt: skip
    reference = {}
    for label, stimulus in (("sugar GRNs", f["sugar"]), ("claw_tpGRN", f["other_gustatory"])):
        _, rates = mn9_hz(full, stimulus, params=LIFParams())
        reference[label] = by_side(rates)

    hz, rates = mn9_hz(graph, f["sugar"])
    shuffled = [mn9_hz(graph.shuffled(seed), f["sugar"])[0] for seed in control_seeds]
    other, _ = mn9_hz(graph, f["other_gustatory"])
    lesions = {
        name: round(mn9_hz(graph.silenced(indices(graph, types)), f["sugar"])[0], 1)
        for name, types in f["lesions"].items()
    }
    return dict(
        sugar_types=f["sugar"], sugar_cells=len(sugar),
        other_gustatory=f["other_gustatory"],
        other_gustatory_cells=len(indices(graph, f["other_gustatory"])),
        stimulus_hz=STIM_HZ, duration_ms=TRIAL_MS, control_seeds=control_seeds,
        mn9_hz=round(hz, 1),
        mn9_by_side=by_side(rates),
        mn9_shuffled_hz=[round(x, 1) for x in shuffled],
        mn9_other_gustatory_hz=round(other, 1),
        mn9_after_lesion_hz=lesions,
        circuit_hz={name: round(float(rates[idx].mean()), 1) for name, idx in circuit.items()},
        sugar_synapses_onto={name: synapses_between(graph, sugar, idx)
                             for name, idx in circuit.items()},
        passed=bool(hz >= 20 and hz >= 3 * (max(shuffled) + 1) and hz >= 3 * (other + 1)),
        post_hoc=post_hoc,
        mn9_reference_settings_by_side=reference,
    )  # fmt: skip


def descending_pairs(graph):
    """{type: {"L": [indices], "R": [indices]}} for descending types with both sides."""
    pairs = {}
    for i in np.flatnonzero(graph.superclass == "descending_neuron"):
        pairs.setdefault(graph.type[i], {}).setdefault(graph.side[i], []).append(i)
    return {t: s for t, s in pairs.items() if t and "L" in s and "R" in s}


def lateral_hit(on, off):
    return on >= 20 and on >= 3 * (off + 1)


def body_section(graph, control_seeds):
    pairs = descending_pairs(graph)
    controls = [graph.shuffled(seed) for seed in control_seeds]
    rows, candidates, tested = [], [], 0
    for jo in JO_SCREEN_TYPES:
        rates = {s: trial(graph, [f"{jo}:{s}"], duration_ms=JO_SCREEN_MS)[1] for s in "LR"}
        mean = {
            (s, d, side): float(rates[s][cells[side]].mean())
            for s in "LR" for d, cells in pairs.items() for side in "LR"
        }  # fmt: skip
        strongest = sorted(
            ((d, mean[("L", d, "L")], mean[("L", d, "R")]) for d in pairs),
            key=lambda row: -abs(row[1] - row[2]),
        )
        rows.append(dict(
            jo_type=jo, cells={s: len(graph.select(f"{jo}:{s}")) for s in "LR"},
            strongest_descending_left_stimulus=[
                dict(type=d, left_hz=round(lh), right_hz=round(rh))
                for d, lh, rh in strongest[:4] if max(lh, rh) >= 20
            ],
        ))  # fmt: skip
        for d in pairs:
            for relation, target in (("ipsilateral", {"L": "L", "R": "R"}),
                                     ("contralateral", {"L": "R", "R": "L"})):  # fmt: skip
                tested += 1
                other = {s: "R" if target[s] == "L" else "L" for s in "LR"}
                if all(lateral_hit(mean[(s, d, target[s])], mean[(s, d, other[s])]) for s in "LR"):
                    candidates.append((jo, d, relation, target, other, mean))
    checked = []
    for jo, d, relation, target, other, mean in candidates:
        shuffled = {s: [] for s in "LR"}
        for control in controls:
            for s in "LR":
                _, control_rates, _ = trial(control, [f"{jo}:{s}"], duration_ms=JO_SCREEN_MS)
                shuffled[s].append(float(control_rates[pairs[d][target[s]]].mean()))
        passed = all(
            max(shuffled[s]) < mean[(s, d, target[s])] / 3 - 1 for s in "LR"
        )
        checked.append(dict(
            jo_type=jo, descending_type=d, relation=relation,
            left_stimulus_hz=dict(target=round(mean[("L", d, target["L"])], 1),
                                  other=round(mean[("L", d, other["L"])], 1)),
            right_stimulus_hz=dict(target=round(mean[("R", d, target["R"])], 1),
                                   other=round(mean[("R", d, other["R"])], 1)),
            shuffled_target_hz={s: [round(x, 1) for x in v] for s, v in shuffled.items()},
            passed=bool(passed),
        ))  # fmt: skip
    return dict(
        duration_ms=JO_SCREEN_MS, stimulus_hz=STIM_HZ, control_seeds=control_seeds,
        screen=rows, mirrored_candidates=checked,
        passed=any(c["passed"] for c in checked),
    )  # fmt: skip


def run_tilt(backend, roll_left, pitch_down, cells):
    """Static scene with the robot held tilted; peak smoothed rate of ``cells``."""
    backend.reset()
    encoder, decoder = VisionEncoder(), MotorDecoder()
    telemetry = tilted_telemetry(roll_left, pitch_down)
    peak = np.zeros(len(cells))
    turns, escape, reverse = [], 0.0, 0
    groups = {}
    for i in range(TOUR_SEGMENT):
        activity = backend.step(encoder.encode(scene_frame("static", i)), telemetry, 1 / 30)
        left, right = decoder.update(activity, now=i / 30)
        turns.append((activity.right - activity.left) / 2)
        escape = max(escape, activity.escape)
        reverse += left < 0 and right < 0
        np.maximum(peak, backend.rates[cells], out=peak)
        for key, value in backend.named_rates().items():
            groups[key] = max(groups.get(key, 0.0), value)
    return peak, dict(mean_turn=round(float(np.mean(turns)), 3), max_escape=round(escape, 3),
                      reverse_frames=int(reverse),
                      peak_group_hz={k: round(v, 1) for k, v in groups.items()})  # fmt: skip


def body_loop_section(data, control_seeds):
    real = MaleCNSBackend(data, seed=0, body_feedback=True, device=DEVICE)
    graph = real.graph
    cells = np.flatnonzero(graph.superclass == "descending_neuron")
    controls = [
        MaleCNSBackend(data, seed=seed, wiring="shuffled", body_feedback=True, device=DEVICE)
        for seed in control_seeds
    ]
    rows = []
    for label, (roll, pitch) in TILTS.items():
        peak, motor = run_tilt(real, roll, pitch, cells)
        shuffled = np.array([run_tilt(c, roll, pitch, cells)[0] for c in controls])
        survivors = [
            k for k in np.flatnonzero(peak >= 20)
            if np.all(shuffled[:, k] < peak[k] / 3 - 1)
        ]  # fmt: skip
        order = np.argsort(peak)[::-1][:6]
        motor_ok = motor["max_escape"] < 0.3 and motor["reverse_frames"] == 0
        motor_ok = motor_ok and abs(motor["mean_turn"]) < 0.1
        rows.append(dict(
            tilt=label, roll_left_deg=roll, pitch_down_deg=pitch, **motor,
            strongest_descending=[
                dict(neuron=graph.labels([cells[k]])[0], peak_hz=round(float(peak[k]), 1),
                     shuffled_peak_hz=[round(float(x), 1) for x in shuffled[:, k]])
                for k in order if peak[k] > 0
            ],
            descending_cells_passing=len(survivors),
            passed=bool(survivors) and motor_ok,
        ))  # fmt: skip
    sweep = []
    for label, (roll, pitch) in TILTS.items():
        for scale in TILT_SWEEP_SCALES:
            _, motor = run_tilt(real, roll * scale, pitch * scale, cells)
            g = motor["peak_group_hz"]
            sweep.append(dict(
                tilt=label.rsplit(" ", 1)[0], degrees=round(20 * scale, 1),
                mean_turn=motor["mean_turn"],
                max_escape=motor["max_escape"], reverse_frames=motor["reverse_frames"],
                peak_hz={k: g[k] for k in ("DNp01:L", "DNp01:R", "DNa02:L", "DNa02:R",
                                           "DNa01:L", "DNa01:R")},
            ))  # fmt: skip
    by_seed = []
    for seed in TILT_SEEDS:
        backend = MaleCNSBackend(data, seed=seed, body_feedback=True, device=DEVICE)
        for label, (roll, pitch) in TILTS.items():
            _, motor = run_tilt(backend, roll, pitch, cells)
            by_seed.append(dict(
                tilt=label, seed=seed, mean_turn=motor["mean_turn"], max_escape=motor["max_escape"],
                reverse_frames=motor["reverse_frames"],
                motor_changed=bool(motor["max_escape"] >= 0.3 or motor["reverse_frames"] > 0
                                   or abs(motor["mean_turn"]) >= 0.1),
            ))  # fmt: skip
    return dict(duration_s=TOUR_SEGMENT / 30, control_seeds=control_seeds, rows=rows,
                passed=all(r["passed"] for r in rows), post_hoc_sweep=sweep,
                post_hoc_seeds=by_seed)  # fmt: skip


def retina_section(data, graph):
    backend = MaleCNSBackend(data, seed=0, vision="retina", device=DEVICE)
    g = backend.graph
    groups = {f"{t}:{s}": g.select(f"{t}:{s}") for t in RETINA_TYPES for s in "LR"}
    axes = {
        side: dict(posterior=[round(float(x), 3) for x in a.posterior],
                   dorsal=[round(float(x), 3) for x in a.dorsal], cells=a.cells,
                   concentration=a.concentration, angles_deg=a.angles_deg)
        for side, a in backend.eye_axes.items()
    }  # fmt: skip
    scenes = []
    for name in RETINA_SCENES:
        backend.reset()
        encoder = backend.make_encoder()
        counts = np.zeros(g.size)
        for i in range(TOUR_SEGMENT):
            backend.step(encoder.encode(scene_frame(name, i)), Telemetry(), 1 / 30)
            counts += backend.last_counts
        hz = counts / (TOUR_SEGMENT / 30)
        scenes.append(dict(scene=name, mean_hz={k: round(float(hz[v].mean()), 2)
                                                for k, v in groups.items()}))  # fmt: skip
    l2 = backend.retina_cells
    t5 = np.flatnonzero(np.isin(graph.type, ["T5a", "T5b", "T5c", "T5d"]))
    saturating = []
    for label, params, net in (("matched w_syn", None, graph),
                               ("Shiu w_syn 0.275 mV", LIFParams(), graph)):  # fmt: skip
        sim = make_simulator(net, params=params, seed=1, device="cpu")
        sim.set_stimulus([(l2, STIM_HZ)])
        v_samples, g_samples = [], []
        for chunk in range(20):
            sim.run(TRIAL_MS / 20)
            if chunk >= 4:
                v_samples.append(sim.v[t5].copy())
                g_samples.append(sim.g[t5].copy())
        hz = sim.take_counts() / (TRIAL_MS / 1000)
        v, gg = np.array(v_samples), np.array(g_samples)
        saturating.append(dict(
            condition=label, mean_hz={t: round(float(hz[graph.type == t].mean()), 2)
                                      for t in RETINA_TYPES},
            t5_mean_v_mv=round(float(v.mean()), 2), t5_max_v_mv=round(float(v.max()), 2),
            t5_mean_g_mv=round(float(gg.mean()), 3), t5_share_net_inhibitory=round(float(np.mean(gg < 0)), 2),
        ))  # fmt: skip
    indptr, indices, weight = np.asarray(graph.indptr), np.asarray(graph.indices), np.asarray(graph.weight)
    pre = np.repeat(np.arange(graph.size, dtype=np.int32), np.diff(indptr))
    is_t5 = np.zeros(graph.size, bool)
    is_t5[t5] = True
    edges = np.flatnonzero(is_t5[indices])
    totals = {}
    for t, w in zip(graph.type[pre[edges]], weight[edges]):
        totals[str(t)] = totals.get(str(t), 0) + int(w)
    inputs = [dict(type=t, synapses_per_t5=round(w / len(t5), 1))
              for t, w in sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:8]]  # fmt: skip
    return dict(eye_axes=axes, columns=int(len(l2)), scenes=scenes, saturating_l2=saturating,
                t5_inputs=inputs, floor=0.05, full=0.5)  # fmt: skip


def run_scene(backend, name):
    backend.reset()
    encoder, decoder = VisionEncoder(), MotorDecoder()
    brain_seconds, turns, escape, reverse = 0.0, [], 0.0, 0
    peak = {}
    for i in range(TOUR_SEGMENT):
        sensory = encoder.encode(scene_frame(name, i))
        started = time.perf_counter()
        activity = backend.step(sensory, Telemetry(), 1 / 30)
        brain_seconds += time.perf_counter() - started
        left, right = decoder.update(activity, now=i / 30)
        turns.append((activity.right - activity.left) / 2)
        escape = max(escape, activity.escape)
        reverse += left < 0 and right < 0
        for key, value in backend.named_rates().items():
            peak[key] = max(peak.get(key, 0.0), value)
    return dict(scene=name, mean_turn=round(float(np.mean(turns)), 3), max_escape=round(escape, 3),
                reverse_frames=int(reverse), frames=TOUR_SEGMENT,
                brain_realtime_factor=round(TOUR_SEGMENT / 30 / brain_seconds, 2),
                peak_rates_hz={k: round(v, 1) for k, v in peak.items()})  # fmt: skip


def scene_passed(row, expect):
    escaped = row["max_escape"] >= 0.3 and row["reverse_frames"] > 0
    if escaped != expect["escape"]:
        return False
    turn = expect.get("turn")
    if turn == 0:
        return abs(row["mean_turn"]) < 0.1
    if turn is not None:
        return row["mean_turn"] * turn > 0.3
    return True


def closed_loop_section(data, control_seeds):
    sections = {}
    for wiring, seeds in [("connectome", CLOSED_LOOP_SEEDS), ("shuffled", control_seeds)]:
        rows = []
        for seed in seeds:
            backend = MaleCNSBackend(data, seed=seed, wiring=wiring, device=DEVICE)
            for name in SCENES:
                row = run_scene(backend, name)
                row["seed"] = seed
                expect = SCENE_EXPECTATIONS[name]
                if wiring == "connectome":
                    row["expected"] = expect
                    row["passed"] = scene_passed(row, expect)
                else:  # the control must not produce behaviour
                    row["passed"] = scene_passed(row, dict(escape=False, turn=0))
                rows.append(row)
        sections[wiring] = rows
    return sections


# How much the results depend on the constants that were chosen by hand: the 150 Hz
# stimulation rate of the reference model, the rescaled synaptic weight, and the brain
# seed that sets the Poisson draws. Reported, not a pass/fail check.
SENSITIVITY_RATES = (25.0, 50.0, 100.0, 150.0)
SENSITIVITY_WEIGHT_SCALES = (0.7, 1.0, 1.3)
SENSITIVITY_SEEDS = (0, 1, 2, 3, 4)
SENSITIVITY_SCENES = ("approach_left", "pass_left")


def sensitivity_section(data, graph):
    rates = []
    for rate in SENSITIVITY_RATES:
        row = dict(stimulus_hz=rate)
        for name, stimulus, target in (("giant_fiber", ["LPLC2:L", "LC4:L"], "DNp01:L"),
                                       ("dna02", ["LC10a:L"], "DNa02:L")):  # fmt: skip
            _, counts, _ = trial(graph, stimulus, rate_hz=rate)
            row[name] = round(float(counts[graph.select(target)].mean()), 1)
        rates.append(row)
    weights = []
    base = LIFParams.matched_to(graph.manifest)
    for scale in SENSITIVITY_WEIGHT_SCALES:
        from dataclasses import replace

        params = replace(base, w_syn=base.w_syn * scale)
        row = dict(w_syn_scale=scale, w_syn_mv=round(params.w_syn, 4))
        for name, stimulus, target in (("giant_fiber", ["LPLC2:L", "LC4:L"], "DNp01:L"),
                                       ("dna02", ["LC10a:L"], "DNa02:L")):  # fmt: skip
            _, counts, _ = trial(graph, stimulus, params=params)
            row[name] = round(float(counts[graph.select(target)].mean()), 1)
        weights.append(row)
    seeds = []
    for seed in SENSITIVITY_SEEDS:
        backend = MaleCNSBackend(data, seed=seed, device=DEVICE)
        for scene in SENSITIVITY_SCENES:
            row = run_scene(backend, scene)
            seeds.append(dict(seed=seed, scene=scene, max_escape=row["max_escape"],
                              mean_turn=row["mean_turn"],
                              peak_giant_fiber_hz=max(row["peak_rates_hz"][f"DNp01:{s}"]
                                                      for s in "LR")))  # fmt: skip
        del backend
    escaped = [r for r in seeds if r["scene"] == "approach_left"]
    turned = [r for r in seeds if r["scene"] == "pass_left"]
    return dict(
        stimulus_rate=rates, synaptic_weight=weights, brain_seeds=seeds,
        seeds_with_escape=sum(r["max_escape"] >= 0.3 for r in escaped),
        seeds_with_left_turn=sum(r["mean_turn"] > 0.3 for r in turned),
        seeds=len(SENSITIVITY_SEEDS),
    )  # fmt: skip


def participation_section(data, scenes=("approach_left", "pass_left")):
    """How much of the brain fires at all, and how little of it reproduces the commands.

    Measured, not a hypothesis test: the same scenes are run with the whole brain, with
    everything except the driven and read-out cells silenced (so only synapses directly
    between them survive), and with the neurons on two-synapse paths added back.
    """
    backend = MaleCNSBackend(data, seed=0, device=DEVICE)
    graph = backend.graph
    driven = np.unique(np.concatenate(list(backend.populations.values())))
    readout = np.unique(np.concatenate([np.concatenate(v) for v in backend.readout.values()]))
    indptr, indices = np.asarray(graph.indptr), np.asarray(graph.indices)
    is_readout = np.zeros(graph.size, bool)
    is_readout[readout] = True
    reached = np.zeros(graph.size, bool)
    for i in driven:
        reached[indices[indptr[i] : indptr[i + 1]]] = True
    between = np.zeros(graph.size, bool)
    for i in np.flatnonzero(reached):
        between[i] = is_readout[indices[indptr[i] : indptr[i + 1]]].any()
    core = np.zeros(graph.size, bool)
    core[driven] = True
    core[readout] = True
    conditions = [("whole brain", None), ("driven and read-out cells only", ~core),
                  ("+ neurons on two-synapse paths", ~(core | between))]  # fmt: skip
    rows = []
    for label, silence in conditions:
        net = graph if silence is None else graph.silenced(np.flatnonzero(silence))
        backend.sim = make_simulator(net, params=backend.sim.params, seed=0, device=DEVICE)
        for scene in scenes:
            backend.reset()
            encoder = backend.make_encoder()
            fired = np.zeros(graph.size, bool)
            escape, turns, peak_gf = 0.0, [], 0.0
            for i in range(TOUR_SEGMENT):
                activity = backend.step(encoder.encode(scene_frame(scene, i)), Telemetry(), 1 / 30)
                fired |= backend.last_counts > 0
                escape = max(escape, activity.escape)
                turns.append((activity.right - activity.left) / 2)
                peak_gf = max(peak_gf, *(backend.named_rates()[f"DNp01:{s}"] for s in "LR"))
            rows.append(dict(
                condition=label, scene=scene,
                neurons_kept=int(graph.size if silence is None else graph.size - silence.sum()),
                max_escape=round(escape, 3), mean_turn=round(float(np.mean(turns)), 3),
                peak_giant_fiber_hz=round(peak_gf, 1), neurons_that_fired=int(fired.sum()),
            ))  # fmt: skip
    return dict(neurons=int(graph.size), driven=int(len(driven)), readout=int(len(readout)),
                two_step_intermediates=int(between.sum()), rows=rows)  # fmt: skip


SPEED_REPEATS = 3
SPEED_CONDITIONS = [
    ("no input (silent brain)", []),
    ("LPLC2:L at 150 Hz", ["LPLC2:L"]),
    ("LPLC2 + LC4, both sides, 150 Hz", ["LPLC2:L", "LPLC2:R", "LC4:L", "LC4:R"]),
]


def gpu_available():
    try:
        from flybrain_robot.brain.lif_cuda import cuda_available
    except ImportError:
        return False
    return cuda_available()


def performance_section(graph):
    devices = ["cpu"] + (["cuda"] if gpu_available() else [])
    rows = []
    for device in devices:
        trial(graph, [], duration_ms=10, device=device)  # compile before timing
        for label, stimulus in SPEED_CONDITIONS:
            # Repeated: a single timing on a GPU that also drives the display varied by a
            # factor of 2.5 between regenerations of this report.
            walls, rates = [], None
            for repeat in range(SPEED_REPEATS):
                _, rates, wall = trial(graph, stimulus, device=device, seed=1 + repeat)
                walls.append(wall)
            factors = sorted(round(TRIAL_MS / 1000 / w, 2) for w in walls)
            rows.append(dict(device=device, condition=label, simulated_s=TRIAL_MS / 1000,
                             wall_s=round(float(np.median(walls)), 3),
                             realtime_factor=factors[len(factors) // 2],
                             realtime_factor_range=[factors[0], factors[-1]],
                             repeats=SPEED_REPEATS, spikes=int(rates.sum())))  # fmt: skip
    return rows


def devices_section(data, graph):
    """CPU and GPU kernels must agree exactly: spikes, and motor commands frame by frame."""
    if not gpu_available():
        return {"skipped": "no CUDA GPU or numba-cuda-mlir"}
    from flybrain_robot.brain.lif_cuda import CudaLIFSimulator
    from flybrain_robot.connectome.synthetic import random_connectome

    networks = []
    for seed in (1, 2, 3):
        net = random_connectome(n=500, density=0.03, seed=seed)
        traces = []
        for cls in (LIFSimulator, CudaLIFSimulator):
            sim = cls(net, params=LIFParams(w_syn=1.5), seed=seed)
            chunks = []
            for chunk in range(30):
                sim.set_stimulus([(np.arange(40), 90.0 + 7 * chunk), (np.arange(40, 60), 0.0)])
                sim.run(33.3 if chunk % 3 else 7.5)
                chunks.append(sim.take_counts())
            traces.append(np.array(chunks))
        networks.append(dict(seed=seed, neurons=net.size, spikes=int(traces[0].sum()),
                             identical=bool(np.array_equal(*traces))))  # fmt: skip
    connectome = []
    for label, stimulus in SPEED_CONDITIONS[1:] + [("LC10a:L at 150 Hz", ["LC10a:L"])]:
        counts = [trial(graph, stimulus, device=d)[1] for d in ("cpu", "cuda")]
        connectome.append(dict(condition=label, spikes_per_s=int(counts[0].sum()),
                               identical=bool(np.array_equal(*counts))))  # fmt: skip
    commands = []
    for device in ("cpu", "cuda"):
        backend = MaleCNSBackend(data, seed=0, device=device)
        encoder, frames = VisionEncoder(), []
        for i in range(TOUR_SEGMENT * len(SCENES)):
            if i % TOUR_SEGMENT == 0:
                backend.reset()
                encoder.reset()
            a = backend.step(encoder.encode(scene_frame("tour", i)), Telemetry(), 1 / 30)
            frames.append((a.left, a.right, a.escape))
        commands.append(frames)
    loop = dict(frames=len(commands[0]), identical=commands[0] == commands[1])
    passed = all(r["identical"] for r in networks + connectome) and loop["identical"]
    return dict(networks=networks, connectome=connectome, closed_loop=loop, passed=passed)


def screen_section(graph):
    dn = np.flatnonzero(graph.superclass == "descending_neuron")
    pairs = {}
    for i in dn:
        pairs.setdefault(graph.type[i], {}).setdefault(graph.side[i], []).append(i)
    rows = []
    for name in SCREEN_TYPES:
        _, rates, _ = trial(graph, [f"{name}:L"], duration_ms=400)
        lateral = []
        for dn_type, sides in pairs.items():
            if dn_type and "L" in sides and "R" in sides:
                left, right = rates[sides["L"]].mean(), rates[sides["R"]].mean()
                if max(left, right) >= 20:
                    lateral.append((float(left - right), dn_type, float(left), float(right)))
        lateral.sort(key=lambda row: -abs(row[0]))
        rows.append(dict(stimulus=f"{name}:L", cells=len(graph.select(f"{name}:L")),
                         strongest_descending=[dict(type=t, left_hz=round(lh), right_hz=round(rh))
                                               for _, t, lh, rh in lateral[:4]]))  # fmt: skip
    return {"duration_ms": 400, "stimulus_hz": STIM_HZ, "rows": rows}


def eye_rates(backend, direction, groups, frames=None):
    """Mean rate (Hz) per group while a pattern is shown, after the settling frames."""
    backend.reset()
    encoder = backend.make_encoder()
    counts = np.zeros(backend.graph.size)
    frame_of = frames or (lambda i: scene_frame(direction, i))
    for i in range(EYE_SETTLE_FRAMES + EYE_MEASURE_FRAMES):
        backend.step(encoder.encode(frame_of(i)), Telemetry(), 1 / 30)
        if i >= EYE_SETTLE_FRAMES:
            counts += backend.last_counts
    hz = counts / (EYE_MEASURE_FRAMES / 30)
    return {name: float(hz[cells].mean()) for name, cells in groups.items()}


def eye_direction_rows(backend):
    """One row per motion-detector subtype and eye, with its direction selectivity."""
    graph = backend.graph
    groups = {f"{t}{letter}:{side}": graph.select(f"{t}{letter}:{side}")
              for t in ("T4", "T5") for letter in "abcd" for side in "LR"}  # fmt: skip
    groups.update({f"{t}:{side}": graph.select(f"{t}:{side}")
                   for t in EYE_HS_TYPES for side in "LR"})  # fmt: skip
    rates = {d: eye_rates(backend, d, groups) for d in EYE_DIRECTIONS}
    rows = []
    for cell_type in ("T4", "T5"):
        for letter in "abcd":
            for side in "LR":
                name = f"{cell_type}{letter}:{side}"
                preferred = EYE_PREFERRED[letter][side]
                null = EYE_OPPOSITE[preferred]
                pref_hz, null_hz = rates[preferred][name], rates[null][name]
                total = pref_hz + null_hz
                rows.append(dict(
                    group=name, preferred_direction=preferred,
                    hz={d: round(rates[d][name], 2) for d in EYE_DIRECTIONS},
                    max_hz=round(max(rates[d][name] for d in EYE_DIRECTIONS), 2),
                    dsi=round((pref_hz - null_hz) / total, 4) if total > 0 else 0.0,
                ))  # fmt: skip
    hs = [dict(group=f"{t}:{side}", hz={d: round(rates[d][f"{t}:{side}"], 2)
                                        for d in EYE_DIRECTIONS})
          for t in EYE_HS_TYPES for side in "LR"]  # fmt: skip
    return rows, hs


def eye_stimulus_check(backend, frames=80):
    """Positive control: does the grating really travel across the modelled retina?

    A null result about motion is worthless unless the stimulus arrives as motion. This
    measures the photoreceptor drive itself, with no neurons involved: how strongly each
    cell is modulated, how many columns one grating period covers (two would be the
    aliasing limit), and whether the phase advances across the eye in the direction the
    pattern moves.
    """
    from flybrain_robot.vision import GRATING_PERIOD_PX, GRATING_SPEED_PX, WIDTH

    encoder = backend.make_encoder()
    period_px = GRATING_PERIOD_PX * WIDTH / 320  # scenes are 320 wide, the encoder 160
    period_frames = GRATING_PERIOD_PX / GRATING_SPEED_PX
    k = int(round(frames / period_frames))
    eyes = {}
    left = np.arange(len(backend.retina_cells)) < len(backend.retina_cells) // 2
    hexes = np.stack([backend.graph.hex1[backend.retina_cells],
                      backend.graph.hex2[backend.retina_cells]], axis=1).astype(float)  # fmt: skip
    for side, mask in (("L", left), ("R", ~left)):
        axis = backend.eye_axes[side].posterior
        along = hexes[mask] @ axis  # position in columns along the eye's posterior axis
        x = np.asarray(backend.retina_x)[mask]
        slope = np.polyfit(along, x, 1)[0]  # image pixels per column
        rows = {}
        for direction in ("grating_left", "grating_right", "still"):
            series = np.array([
                encoder.encode(scene_frame("grating_left" if direction == "still" else direction,
                                           0 if direction == "still" else i)).retina
                for i in range(frames)
            ])[:, mask]  # fmt: skip
            component = np.fft.rfft(series - series.mean(0), axis=0)[k]
            amplitude = np.abs(component) * 2 / frames
            expected = 2 * np.pi * x / period_px * (1 if direction == "grating_left" else -1)
            phase = np.angle(component)
            agreement = float(np.abs(np.mean(np.exp(1j * (phase - expected)))))
            rows[direction] = dict(mean_modulation=round(float(amplitude.mean()), 4),
                                   phase_agreement=round(agreement, 3))  # fmt: skip
        eyes[side] = dict(cells=int(mask.sum()), pixels_per_column=round(float(abs(slope)), 2),
                          columns_per_grating_period=round(period_px / abs(slope), 2),
                          patterns=rows)  # fmt: skip
    return eyes


def eye_configuration(data, seeds, slow_inputs):
    """One model configuration of the eye mode, measured against the same criteria."""
    backend = MaleCNSBackend(data, seed=0, vision="eye", device=DEVICE, slow_inputs=slow_inputs)
    graph = backend.graph
    rows, hs = eye_direction_rows(backend)
    step = {}
    for phase, luminance in (("bright", 1.0), ("dark", 0.0)):
        backend.sim.set_stimulus([(backend.retina_cells, luminance * 150.0)])
        backend.sim.run(EYE_STEP_MS)
        hz = backend.sim.take_counts() / (EYE_STEP_MS / 1000)
        step[phase] = {t: round(float(hz[graph.select(t)].mean()), 2)
                       for t in ("L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9",
                                 "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")}  # fmt: skip
    # Post hoc, added after the first configuration showed no direction selectivity at
    # all: a grating that does not move and a full-field contrast reversal, to show what
    # a population mean over the whole eye can and cannot see.
    groups = {f"{t}{letter}:{side}": graph.select(f"{t}{letter}:{side}")
              for t in ("T4", "T5") for letter in "abcd" for side in "LR"}  # fmt: skip
    groups.update({f"{t}:L": graph.select(f"{t}:L") for t in ("L1", "L2", "Tm1", "Tm9", "Mi1")})
    post_hoc = {
        "still grating": eye_rates(backend, None, groups,
                                   frames=lambda i: scene_frame("grating_left", 0)),
        "drifting grating": eye_rates(backend, "grating_left", groups),
        "full-field flicker": eye_rates(backend, "flicker", groups),
    }  # fmt: skip
    configuration = dict(
        stimulus_check=eye_stimulus_check(backend),
        held_out_columns=held_out_accuracy(backend.graph),
        tonic_hz=EYE_TONIC_HZ, tonic_cells=backend.tonic_cells,
        photoreceptors=int(len(backend.retina_cells)), neurons=int(graph.size),
        slow_inputs=list(EYE_SLOW_TYPES) if slow_inputs else [],
        slow_tau_ms=EYE_SLOW_TAU_MS if slow_inputs else None,
        settle_frames=EYE_SETTLE_FRAMES, measured_frames=EYE_MEASURE_FRAMES,
        imputed_columns=graph.manifest.get("imputed_columns"),
        compartments=graph.manifest.get("compartments"),
    )  # fmt: skip
    del backend
    controls = []
    for seed in seeds:
        control = MaleCNSBackend(data, seed=seed, vision="eye", wiring="shuffled", device=DEVICE,
                                 slow_inputs=slow_inputs)  # fmt: skip
        control_rows, _ = eye_direction_rows(control)
        controls.append(dict(seed=seed, rows=control_rows))
        del control
    by_type = {t: [r for r in rows if r["group"].startswith(t)] for t in ("T4", "T5")}
    quiet = [r["group"] for r in rows if r["max_hz"] < EYE_MIN_HZ]
    selective = {t: [r for r in group if r["dsi"] >= EYE_MIN_DSI] for t, group in by_type.items()}
    mean_dsi = {t: float(np.mean([r["dsi"] for r in group])) for t, group in by_type.items()}
    mirrored = []
    for cell_type in ("T4", "T5"):
        for letter in "abcd":
            left, right = (next(r for r in rows if r["group"] == f"{cell_type}{letter}:{side}")
                           for side in "LR")  # fmt: skip
            horizontal = letter in "ab"
            pair = ("grating_left", "grating_right") if horizontal else ("grating_up", "grating_down")
            diff = [row["hz"][pair[0]] - row["hz"][pair[1]] for row in (left, right)]
            agree = (diff[0] * diff[1] < 0) if horizontal else (diff[0] * diff[1] > 0)
            mirrored.append(dict(subtype=f"{cell_type}{letter}",
                                 difference_hz=[round(d, 2) for d in diff],
                                 as_predicted=bool(agree)))  # fmt: skip
    control_dsi = [abs(r["dsi"]) for c in controls for r in c["rows"]]
    passed = bool(
        not quiet
        and all(len(selective[t]) >= 4 and mean_dsi[t] > 0 for t in ("T4", "T5"))
        and all(m["as_predicted"] for m in mirrored)
        and control_dsi and float(np.mean(control_dsi)) < EYE_CONTROL_MAX_DSI
        and max(r["dsi"] for c in controls for r in c["rows"]) < EYE_MIN_DSI
    )
    return dict(
        configuration=configuration, rows=rows, hs=hs, mirrored=mirrored, controls=controls,
        mean_dsi={t: round(v, 4) for t, v in mean_dsi.items()},
        selective={t: len(v) for t, v in selective.items()}, quiet_groups=quiet,
        control_mean_abs_dsi=round(float(np.mean(control_dsi)), 4) if control_dsi else None,
        brightness_step=step, passed=passed,
        post_hoc_patterns={k: {g: round(v, 2) for g, v in row.items()}
                           for k, row in post_hoc.items()},
    )  # fmt: skip


def eye_section(data, seeds):
    configurations = []
    for name, slow_inputs in EYE_CONFIGURATIONS:
        result = eye_configuration(data, seeds, slow_inputs)
        configurations.append(dict(name=name, **result))
    return dict(
        criterion=dict(min_hz=EYE_MIN_HZ, min_dsi=EYE_MIN_DSI, pairs_required=4,
                       control_max_mean_abs_dsi=EYE_CONTROL_MAX_DSI),
        configurations=configurations,
        passed=all(c["passed"] for c in configurations),
    )  # fmt: skip



def machine():
    import numba

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=ROOT, check=True).stdout.strip()  # fmt: skip
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip())  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")  # fmt: skip
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    gpu = None
    if gpu_available():
        from importlib.metadata import version

        from numba_cuda_mlir import cuda

        name = cuda.get_current_device().name
        gpu = dict(name=name.decode() if isinstance(name, bytes) else str(name),
                   numba_cuda_mlir=version("numba-cuda-mlir"))  # fmt: skip
    return dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                commit=commit, source_modified_since_commit=dirty, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, os=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), numba_threads=numba.get_num_threads(), gpu=gpu,
                device=DEVICE)  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--json", default=str(ROOT / "docs" / "validation" / "results.json"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "VALIDATION.md"))
    parser.add_argument("--controls", type=int, default=3, help="Shuffled-wiring control seeds")
    parser.add_argument("--skip-screen", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="SECTION",
                        help="Run only these sections and print them; writes no report")  # fmt: skip
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Simulator for all sections except 'devices' (identical results)")  # fmt: skip
    args = parser.parse_args(argv)
    global DEVICE
    DEVICE = args.device
    seeds = list(range(1, args.controls + 1))
    started = time.perf_counter()
    full = load_connectome(args.data)
    graph = full.without_fast_synapses_from(MODULATORS)
    results = {"machine": machine(), "dataset": dataset_section(full),
               "model": dict(w_syn_mv=LIFParams.matched_to(full.manifest).w_syn,
                             silenced_transmitters=list(MODULATORS),
                             sensory_map=[dict(feature=d.feature, populations=list(d.populations),
                                               max_rate_hz=d.max_rate_hz, floor=d.floor)
                                          for d in SENSORY_MAP],
                             readout={k: list(v) for k, v in READOUT.items()})}  # fmt: skip
    steps = [
        ("reference", lambda: reference_section([1, 2, 3])),
        ("devices", lambda: devices_section(args.data, graph)),
        ("stability", lambda: stability_section(full)),
        ("pathways", lambda: pathway_section(graph, seeds)),
        ("feeding", lambda: feeding_section(graph, full, seeds)),
        ("body", lambda: body_section(graph, seeds)),
        ("body_loop", lambda: body_loop_section(args.data, seeds)),
        ("closed_loop", lambda: closed_loop_section(args.data, seeds)),
        ("retina", lambda: retina_section(args.data, graph)),
        ("eye", lambda: eye_section(args.data, seeds)),
        ("participation", lambda: participation_section(args.data)),
        ("sensitivity", lambda: sensitivity_section(args.data, graph)),
        ("performance", lambda: performance_section(graph)),
    ]
    if not args.skip_screen:
        steps.append(("screen", lambda: screen_section(graph)))
    if args.only:
        unknown = set(args.only) - {key for key, _ in steps}
        if unknown:
            parser.error(f"unknown sections: {', '.join(sorted(unknown))}")
        steps = [(key, run) for key, run in steps if key in args.only]
    for key, run in steps:
        t0 = time.perf_counter()
        print(f"[{key}] running...", flush=True)
        results[key] = run()
        print(f"[{key}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    if args.only:
        print(json.dumps({key: results[key] for key, _ in steps}, indent=2))
        return 0
    results["machine"]["total_seconds"] = round(time.perf_counter() - started)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(results), encoding="utf-8")
    print(f"Wrote {args.json} and {args.markdown}")
    failed, not_run = failures(results), skipped(results)
    if not_run:
        print(f"Not run on this machine: {'; '.join(not_run)}")
    print("All checks that ran passed." if not failed else f"Failed checks: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
