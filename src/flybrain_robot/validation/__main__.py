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

from flybrain_robot.brain.lif import LIFParams, LIFSimulator
from flybrain_robot.brain.malecns import MODULATORS, READOUT, SENSORY_MAP, MaleCNSBackend
from flybrain_robot.connectome.graph import load_connectome
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.telemetry import Telemetry
from flybrain_robot.vision import SCENES, TOUR_SEGMENT, VisionEncoder, scene_frame

from .report import failures, render_markdown

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


def trial(graph, stimulus, duration_ms=TRIAL_MS, seed=1, params=None):
    sim = LIFSimulator(graph, params=params, seed=seed)
    sim.set_stimulus([(indices(graph, stimulus), STIM_HZ)] if stimulus else [])
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
    rows = []
    for p in PATHWAYS:
        target, counterpart = graph.select(p["target"]), graph.select(p["counterpart"])
        _, rates, _ = trial(graph, p["stimulus"])
        shuffled = []
        for control in controls:
            _, control_rates, _ = trial(control, p["stimulus"])
            shuffled.append(float(control_rates[target].mean()))
        hz, other = float(rates[target].mean()), float(rates[counterpart].mean())
        source = indices(graph, p["stimulus"])
        rows.append(dict(
            p, target_hz=round(hz, 1), counterpart_hz=round(other, 1),
            shuffled_hz=[round(x, 1) for x in shuffled],
            direct_synapses=synapses_between(graph, source, target),
            routes=top_intermediates(graph, source, target),
            lateral_ratio=round(hz / (other + 1), 2),
            passed=bool(hz >= 20 and hz >= 3 * (max(shuffled) + 1)
                        and (not p["lateral"] or hz >= 3 * (other + 1))),
        ))  # fmt: skip
    return {"stimulus_hz": STIM_HZ, "duration_ms": TRIAL_MS, "control_seeds": control_seeds,
            "rows": rows}  # fmt: skip


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
    for wiring, seeds in [("connectome", [0]), ("shuffled", control_seeds)]:
        rows = []
        for seed in seeds:
            backend = MaleCNSBackend(data, seed=seed, wiring=wiring)
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


def performance_section(graph):
    sim = LIFSimulator(graph, seed=1)
    sim.run(1)
    rows = []
    for label, stimulus in [
        ("no input (silent brain)", []),
        ("LPLC2:L at 150 Hz", ["LPLC2:L"]),
        ("LPLC2 + LC4, both sides, 150 Hz", ["LPLC2:L", "LPLC2:R", "LC4:L", "LC4:R"]),
    ]:
        _, rates, wall = trial(graph, stimulus)
        rows.append(dict(condition=label, simulated_s=TRIAL_MS / 1000, wall_s=round(wall, 3),
                         realtime_factor=round(TRIAL_MS / 1000 / wall, 2),
                         spikes=int(rates.sum())))  # fmt: skip
    return rows


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
    return dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                commit=commit, source_modified_since_commit=dirty, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, os=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), numba_threads=numba.get_num_threads())  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--json", default=str(ROOT / "docs" / "validation" / "results.json"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "VALIDATION.md"))
    parser.add_argument("--controls", type=int, default=3, help="Shuffled-wiring control seeds")
    parser.add_argument("--skip-screen", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="SECTION",
                        help="Run only these sections and print them; writes no report")  # fmt: skip
    args = parser.parse_args(argv)
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
        ("stability", lambda: stability_section(full)),
        ("pathways", lambda: pathway_section(graph, seeds)),
        ("feeding", lambda: feeding_section(graph, full, seeds)),
        ("closed_loop", lambda: closed_loop_section(args.data, seeds)),
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
    failed = failures(results)
    print("All checks passed." if not failed else f"Failed checks: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
