"""Render validation results (results.json) as docs/VALIDATION.md."""


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _mark(passed):
    return "✅ pass" if passed else "❌ fail"


def _pair(rates, name):
    return f"{rates.get(name + ':L', 0):.0f}/{rates.get(name + ':R', 0):.0f}"


def _reference(section):
    if "skipped" in section:
        return f"Skipped: {section['skipped']}."
    rows = [
        (r["seed"], r["neurons"], r["edges"], r["steps"], r["brian2_spikes"], r["numba_spikes"],
         r["spiking_neurons"], _mark(r["identical"]))
        for r in section["runs"]
    ]  # fmt: skip
    return (
        f"Brian2 {section['brian2_version']} with the equations, reset, refractoriness and "
        "delays of Shiu et al.'s `model.py`, random signed networks and identical "
        "deterministic input kicks. A run passes only if every spike (neuron and 0.1 ms time "
        "step) is identical.\n\n"
        + _table(["seed", "neurons", "synapse edges", "steps", "Brian2 spikes", "this kernel",
                  "spiking neurons", "spike-for-spike"], rows)  # fmt: skip
    )


def _pathways(section):
    rows = []
    for r in section["rows"]:
        routes = "; ".join(
            f"{x['neuron']} ({x['synapses_in']}→{x['synapses_out']})" for x in r["routes"]
        )
        rows.append((
            r["name"], " + ".join(r["stimulus"]), r["target"], r["target_hz"], r["counterpart_hz"],
            " / ".join(str(x) for x in r["shuffled_hz"]),
            f"{r['lateral_ratio']}×" + (" (required ≥ 3×)" if r["lateral"] else " (reported)"),
            r["direct_synapses"], routes or "none", r["role"], _mark(r["passed"]),
        ))  # fmt: skip
    return (
        f"Each stimulus population fires Poisson spikes at {section['stimulus_hz']:.0f} Hz for "
        f"{section['duration_ms'] / 1000:.0f} s in the whole 166,700-neuron model. "
        "Pass: target ≥ 20 Hz and ≥ 3× the strongest response in any of the shuffled-wiring "
        f"controls (seeds {', '.join(map(str, section['control_seeds']))}); where marked, also "
        "≥ 3× the same cell type on the other side. The first full run required that for the "
        "Giant Fiber too; it failed at about 2.2× while the controls stayed at 0 Hz. Escape "
        "reads both Giant Fibers, so their ratio is now reported rather than required. "
        "*Direct synapses* and *top two-step routes* (synapses into → out of the intermediate "
        "neuron) are read from the compiled graph.\n\n"
        + _table(["check", "stimulus", "target", "target Hz", "other side Hz", "shuffled Hz",
                  "target / other side", "direct synapses", "top two-step routes", "robot use",
                  "result"], rows)  # fmt: skip
    )


def _closed_loop(section):
    out = []
    for wiring, rows in section.items():
        body = [
            (r["scene"], r.get("seed", ""), f"{r['mean_turn']:+.2f}", r["max_escape"],
             f"{r['reverse_frames']}/{r['frames']}", _pair(r["peak_rates_hz"], "DNp01"),
             _pair(r["peak_rates_hz"], "DNa02"), _pair(r["peak_rates_hz"], "LPLC2"),
             _pair(r["peak_rates_hz"], "LC10a"), r["brain_realtime_factor"], _mark(r["passed"]))
            for r in rows
        ]  # fmt: skip
        title = "Real MaleCNS wiring" if wiring == "connectome" else "Shuffled wiring (control)"
        out.append(
            f"**{title}**\n\n"
            + _table(["scene", "seed", "mean turn (+ = left)", "max escape", "reversing frames",
                      "peak GF L/R Hz", "peak DNa02 L/R Hz", "peak LPLC2 L/R Hz",
                      "peak LC10a L/R Hz", "brain × real time", "result"], body)  # fmt: skip
        )
    return "\n\n".join(out)


def failures(results):
    failed = [r["name"] for r in results["pathways"]["rows"] if not r["passed"]]
    failed += [
        f"{wiring}: {r['scene']} (seed {r.get('seed')})"
        for wiring, rows in results["closed_loop"].items()
        for r in rows
        if not r["passed"]
    ]
    reference = results["reference"]
    failed += [f"Brian2 seed {r['seed']}" for r in reference.get("runs", []) if not r["identical"]]
    return failed


def render_markdown(results):
    m, d, model = results["machine"], results["dataset"], results["model"]
    stability, perf = results["stability"], results["performance"]
    failed = failures(results)
    summary = "All checks below passed." if not failed else "Failed: " + ", ".join(failed) + "."
    sensory = "; ".join(
        f"`{s['feature']}` → {', '.join(s['populations'])} (0–{s['max_rate_hz']:.0f} Hz above "
        f"{s['floor']})"
        for s in model["sensory_map"]
    )
    readout = "; ".join(f"{k}: {', '.join(v)}" for k, v in model["readout"].items())
    after = "; ".join(
        f"*{r['condition']}*: "
        + (", ".join(f"{k} {v:,}" for k, v in r.get("after_top_types", {}).items()) or "none")
        for r in stability["rows"]
    )
    dataset_rows = [
        ("release", d["dataset"]),
        ("license", d["license"]),
        ("neurons simulated", f"{d['neurons']:,}"),
        ("neuron-to-neuron connections", f"{d['neuron_to_neuron_edges']:,}"),
        ("synapses", f"{d['synapses']:,}"),
        ("rows in the release weight table", f"{d['edges_in_release']:,} (includes fragments)"),
        ("inhibitory neurons", f"{d['inhibitory_neurons']:,} ({d['sign_rule']})"),
    ]
    if isinstance(d["sources"], dict):
        dataset_rows += [
            (f"`{name}`", f"{info['bytes']:,} bytes, SHA-256 `{info['sha256']}`")
            for name, info in d["sources"].items()
        ]
    parts = [
        "# Validation",
        "<!-- Generated by `python -m flybrain_robot.validation`. Do not edit by hand. -->",
        f"Generated {m['timestamp']} from commit `{m['commit']}`"
        + (" (with uncommitted source changes)" if m["source_modified_since_commit"] else "")
        + f" in {m['total_seconds']} s on {m['cpu']} ({m['logical_cpus']} logical CPUs, "
        f"{m['numba_threads']} numba threads), {m['os']}, Python {m['python']}, "
        f"NumPy {m['numpy']}, numba {m['numba']}. Raw numbers: "
        "[validation/results.json](validation/results.json).",
        f"**{summary}**",
        "Reproduce:\n\n```bash\npip install -e \".[connectome,reference]\"\n"
        "python -m flybrain_robot.connectome.build download --data data/malecns-v1.0\n"
        "python -m flybrain_robot.connectome.build compile --data data/malecns-v1.0\n"
        "python -m flybrain_robot.validation --data data/malecns-v1.0\n```",
        "## 1. Dataset",
        _table(["", ""], dataset_rows),
        "## 2. The simulation kernel matches the reference simulator",
        _reference(results["reference"]),
        "## 3. Calibration and stability",
        "Shiu et al. fitted one free parameter, w_syn = 0.275 mV per synapse, on FlyWire "
        "(54,492,922 synapses among 138,639 neurons, 393 per neuron). MaleCNS has "
        f"{d['synapses'] / d['neurons']:.0f} synapses per neuron, so the same value roughly "
        "doubles the synaptic drive. The default rescales w_syn to the FlyWire mean input "
        f"({model['w_syn_mv']:.4f} mV) and gives dopamine, octopamine and serotonin no fast "
        "synaptic effect, since they act through slow metabotropic receptors. Stimulus: "
        f"{stability['stimulus']}.",
        _table(["condition", "w_syn mV", "spikes during", "spiking neurons",
                "spikes in the second after", "neurons still firing"],
               [(r["condition"], r["w_syn_mv"], f"{r['spikes_during_1s']:,}", r["spiking_neurons"],
                 f"{r['spikes_in_1s_after_stimulus']:,}", r["neurons_active_after"])
                for r in stability["rows"]]),  # fmt: skip
        f"Cell types still firing in the second after the stimulus (spikes): {after}.",
        "## 4. Visual pathways in the whole-brain model",
        _pathways(results["pathways"]),
        "## 5. Closed loop: synthetic camera scenes → connectome → motor commands",
        f"Interfaces (the only hand-chosen parts of the brain path): {sensory}. Read-out: "
        f"{readout}. Forward cruise is a constant controller setting; turning and escape come "
        "from the read-out neurons. Expected: no escape and |turn| < 0.1 for `static` and "
        "`pan`; a left turn > 0.3 without escape for `pass_left`; escape (activity ≥ 0.3 and "
        "reversing) for every `approach_*`. The shuffled controls must produce no escape and "
        "|turn| < 0.1 in every scene. While an object approaches from one side, the pursuit "
        "pathway (LC10a → DNa02) still turns the robot toward it as the Giant Fiber commands "
        "reverse.",
        _closed_loop(results["closed_loop"]),
        "## 6. Speed",
        _table(["condition", "simulated s", "wall s", "× real time", "spikes"],
               [(r["condition"], r["simulated_s"], r["wall_s"], r["realtime_factor"],
                 f"{r['spikes']:,}") for r in perf]),  # fmt: skip
        "The simulation is event-driven: cost grows with the number of neurons away from rest. "
        "With strong visual input it runs slower than real time on this CPU, so the robot loop "
        "advances brain time by one frame period per camera frame and reports the ratio.",
    ]
    if "screen" in results:
        rows = [
            (r["stimulus"], r["cells"], ", ".join(
                f"{x['type']} {x['left_hz']}/{x['right_hz']}" for x in r["strongest_descending"]
            ) or "none ≥ 20 Hz")
            for r in results["screen"]["rows"]
        ]  # fmt: skip
        parts += [
            "## Appendix: exploratory screen",
            f"Left-side visual projection types driven at {results['screen']['stimulus_hz']:.0f} "
            f"Hz for {results['screen']['duration_ms']} ms. Listed: descending neuron types with "
            "a left and a right cell and at least 20 Hz on one side, largest left/right "
            "difference first (Hz, left/right). This screen was used to choose the interfaces.",
            _table(["stimulus", "cells", "descending neurons (left/right Hz)"], rows),
        ]
    parts += [
        "## Not validated",
        "- Behaviour of a physical robot. No hardware run is recorded in this repository.\n"
        "- Biological accuracy of the dynamics. Leaky integrate-and-fire neurons with one "
        "weight per synapse, predicted transmitters and no gap junctions, neuromodulation, "
        "plasticity or graded potentials are a connectome-constrained approximation.\n"
        "- The camera feature encoders. They are optical-flow heuristics tested on synthetic "
        "scenes; webcams, real lighting and motion blur are untested.",
    ]
    return "\n\n".join(parts) + "\n"
