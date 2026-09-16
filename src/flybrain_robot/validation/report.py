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


def _devices(section):
    if "skipped" in section:
        return f"Skipped: {section['skipped']}."
    networks = [(r["seed"], r["neurons"], f"{r['spikes']:,}", _mark(r["identical"]))
                for r in section["networks"]]  # fmt: skip
    connectome = [(r["condition"], f"{r['spikes_per_s']:,}", _mark(r["identical"]))
                  for r in section["connectome"]]  # fmt: skip
    loop = section["closed_loop"]
    return (
        "The CPU kernel (numba) and the GPU kernel (numba-cuda-mlir) compile the same Python "
        "source for the per-neuron update, sum synaptic input in fixed point, draw Poisson "
        "input from a counter-based hash and compute the membrane update with an explicit "
        "fused multiply-add, so their results can be compared exactly. A row passes only if "
        "every spike count, per neuron and per interval, is identical.\n\n"
        + _table(["random network seed", "neurons", "spikes", "CPU = GPU"], networks)
        + "\n\n"
        + _table(["whole MaleCNS, 1 s", "spikes", "CPU = GPU"], connectome)
        + f"\n\nClosed loop over the synthetic tour: left, right and escape commands of all "
        f"{loop['frames']} frames {'are identical' if loop['identical'] else 'differ'} "
        f"between devices ({_mark(loop['identical'])})."
    )  # fmt: skip


def _pathways(section):
    rows = []
    for r in section["rows"]:
        routes = "; ".join(
            f"{x['neuron']} ({x['synapses_in']}→{x['synapses_out']})" for x in r["routes"]
        )
        rows.append((
            r["name"], " + ".join(r["stimulus"]), r["target"], r["target_hz"], r["counterpart_hz"],
            " / ".join(str(x) for x in r["shuffled_hz"]),
            " / ".join(str(x) for x in r.get("shuffled_within_superclass_hz", [])) or "-",
            f"{r['lateral_ratio']}×" + (" (required ≥ 3×)" if r["lateral"] else " (reported)"),
            r["direct_synapses"], routes or "none", r["role"], _mark(r["passed"]),
        ))  # fmt: skip
    return (
        f"Each stimulus population fires Poisson spikes at {section['stimulus_hz']:.0f} Hz for "
        f"{section['duration_ms'] / 1000:.0f} s in the whole 166,700-neuron model. "
        "Pass: target ≥ 20 Hz and ≥ 3× the strongest response in any of the shuffled-wiring "
        f"controls (seeds {', '.join(map(str, section['control_seeds']))}); where marked, also "
        "≥ 3× the same cell type on the other side. The first full run required that for the "
        "Giant Fiber too; it failed at 1.8× and 2.2× while the controls stayed at 0 Hz. Escape "
        "reads both Giant Fibers, so their ratio is now reported rather than required. "
        "The *within superclass* column is a harder control added later and reported only: "
        "targets are permuted among neurons of the same superclass, so the regional structure "
        "of the connectome survives and only the specific partner of each synapse is destroyed. "
        "*Direct synapses* and *top two-step routes* (synapses into → out of the intermediate "
        "neuron) are read from the compiled graph.\n\n"
        + _table(["check", "stimulus", "target", "target Hz", "other side Hz", "shuffled Hz",
                  "shuffled within superclass Hz", "target / other side", "direct synapses",
                  "top two-step routes", "robot use", "result"], rows)  # fmt: skip
    )


def _sides(values):
    return f"{values.get('L', 0):.0f}/{values.get('R', 0):.0f}"


def _feeding(f):
    circuit = list(f["circuit_hz"])
    rows = [
        ("MN9 mean of left and right, sugar GRNs", f"{f['mn9_hz']} Hz", "≥ 20 Hz"),
        ("MN9, shuffled wiring (seeds " + ", ".join(map(str, f["control_seeds"])) + ")",
         " / ".join(f"{x} Hz" for x in f["mn9_shuffled_hz"]),
         f"≤ {f['mn9_hz'] / 3 - 1:.1f} Hz for a pass"),
        (f"MN9, {', '.join(f['other_gustatory'])} ({f['other_gustatory_cells']} cells)",
         f"{f['mn9_other_gustatory_hz']} Hz", f"≤ {f['mn9_hz'] / 3 - 1:.1f} Hz for a pass"),
    ]  # fmt: skip
    lesions = [(name, f"{hz} Hz") for name, hz in f["mn9_after_lesion_hz"].items()]
    post_hoc = [
        (r["stimulus"], r["cells"], _sides(r["mn9_by_side"]),
         " · ".join(f"{k} {r['circuit_hz'][k]:.0f}" for k in circuit if r["circuit_hz"][k])
         or "none",
         " · ".join(f"{k} {r['synapses_onto'][k]}" for k in circuit if r["synapses_onto"][k])
         or "none")
        for r in f["post_hoc"]
    ]  # fmt: skip
    reference = "; ".join(
        f"{label}: MN9 L/R {_sides(v)} Hz" for label, v in f["mn9_reference_settings_by_side"].items()
    )
    return (
        "Shiu et al. (2022, 2024) found that sugar gustatory neurons drive MN9, the motor neuron "
        "of proboscis extension, through a named set of interneurons. MaleCNS has no receptor "
        f"labels, so the sugar neurons are the types with the most synapses onto G2N-1 and "
        f"Zorro: {', '.join(f['sugar_types'])} ({f['sugar_cells']} cells), stimulated at "
        f"{f['stimulus_hz']:.0f} Hz for {f['duration_ms'] / 1000:.0f} s. The pass condition "
        "was committed before the first run. "
        f"**Result: {_mark(f['passed'])}.**\n\n"
        + _table(["pre-registered condition", "measured", "required"], rows)
        + f"\n\nMN9 left/right during sugar stimulation: {_sides(f['mn9_by_side'])} Hz.\n\n"
        + "Circuit neurons during sugar stimulation (Hz) and synapses from the sugar neurons:\n\n"
        + _table(["neuron", "Hz", "synapses from sugar GRNs"],
                 [(k, f["circuit_hz"][k], f["sugar_synapses_onto"][k]) for k in circuit])
        + "\n\nMN9 after silencing a circuit neuron during sugar stimulation (lesions; "
        "reported, not required):\n\n"
        + _table(["silenced", "MN9 mean"], lesions)
        + "\n\n**Post hoc, added after the check failed.** These rows explain the result and do "
        "not change it. Stimulus 150 Hz for 1 s; MN9 left/right; circuit neurons that fired "
        "(Hz); synapses from the stimulated cells onto circuit neurons.\n\n"
        + _table(["stimulus", "cells", "MN9 L/R Hz", "circuit neurons firing", "synapses onto circuit"],
                 post_hoc)
        + f"\n\nWith the reference settings of Shiu et al. (w_syn 0.275 mV, modulators "
        f"excitatory): {reference}."
    )  # fmt: skip


def _body(b):
    screen = [
        (r["jo_type"], _sides(r["cells"]), ", ".join(
            f"{x['type']} {x['left_hz']}/{x['right_hz']}" for x in r["strongest_descending_left_stimulus"]
        ) or "none ≥ 20 Hz")
        for r in b["screen"]
    ]  # fmt: skip
    pairs = [
        (c["jo_type"], c["descending_type"], c["relation"],
         f"{c['left_stimulus_hz']['target']:.0f} / {c['left_stimulus_hz']['other']:.0f}",
         f"{c['right_stimulus_hz']['target']:.0f} / {c['right_stimulus_hz']['other']:.0f}",
         " / ".join(f"{max(c['shuffled_target_hz'][s]):.0f}" for s in "LR"), _mark(c["passed"]))
        for c in b["mirrored_candidates"]
    ]  # fmt: skip
    return (
        "Johnston's organ (JO) neurons sense static antennal deflection, which is how the fly "
        "senses gravity and wind; JO-C neurons respond when the antenna is pushed toward the "
        "head, JO-E neurons when it is pulled away (Kamikouchi et al. 2009; Yorozu et al. 2009; "
        f"Patella & Wilson 2018). Each JO type fires at {b['stimulus_hz']:.0f} Hz on one side "
        f"for {b['duration_ms']:.0f} ms. The pass condition, committed before the first run, "
        "requires a lateralized descending response that replicates mirror-image from the other "
        "side and stays below a third on every shuffled control (seeds "
        f"{', '.join(map(str, b['control_seeds']))}). **Result: {_mark(b['passed'])}** — "
        f"{sum(c['passed'] for c in b['mirrored_candidates'])} of "
        f"{len(b['mirrored_candidates'])} mirrored pairs pass.\n\n"
        + _table(["JO type", "cells L/R", "strongest descending responses to left stimulus (L/R Hz)"],
                 screen)
        + "\n\nMirrored pairs (target / other cell of the same type, Hz):\n\n"
        + _table(["JO type", "descending type", "relation", "left stimulus", "right stimulus",
                  "shuffled max, L / R stimulus", "result"], pairs)
        + "\n\nNone of the responding descending types is in the motor read-out (DNp01, DNa01, "
        "DNa02), so body input is not wired to motors."
    )  # fmt: skip


def _body_loop(b):
    rows = [
        (r["tilt"], f"{r['peak_group_hz']['JO-C*:L']:.0f}/{r['peak_group_hz']['JO-C*:R']:.0f}",
         f"{r['peak_group_hz']['JO-E*:L']:.0f}/{r['peak_group_hz']['JO-E*:R']:.0f}",
         r["descending_cells_passing"],
         ", ".join(f"{x['neuron']} {x['peak_hz']:.0f}" for x in r["strongest_descending"][:3]),
         f"{r['mean_turn']:+.3f}", r["max_escape"], r["reverse_frames"],
         _pair(r["peak_group_hz"], "DNp01"), _pair(r["peak_group_hz"], "DNa02"),
         _mark(r["passed"]))
        for r in b["rows"]
    ]  # fmt: skip
    sweep = [
        (r["tilt"], r["degrees"], f"{r['mean_turn']:+.3f}", r["max_escape"], r["reverse_frames"],
         _pair(r["peak_hz"], "DNp01"), _pair(r["peak_hz"], "DNa02"))
        for r in b["post_hoc_sweep"]
    ]  # fmt: skip
    return (
        "The robot faces the static scene for "
        f"{b['duration_s']:.0f} s while held tilted; `body_feedback` is on and tilt drives JO "
        "neurons through the declared hypothesis (nose-up and the lower side push the antenna → "
        "JO-C; nose-down and the upper side pull it → JO-E). Pass condition, committed before "
        "the first run, per tilt: some descending cell peaks ≥ 20 Hz and below a third of that "
        "in every shuffled control, **and** motor commands stay as when level (no escape, "
        f"|mean turn| < 0.1). **Result: {_mark(b['passed'])}.**\n\n"
        + _table(["tilt", "JO-C L/R Hz", "JO-E L/R Hz", "descending cells passing",
                  "strongest descending (Hz)", "mean turn", "max escape", "reversing frames",
                  "peak GF L/R Hz", "peak DNa02 L/R Hz", "result"], rows)
        + "\n\nTilt reaches hundreds of descending cells that stay silent on shuffled wiring, but "
        "it also reaches the Giant Fiber and DNa02 through the connectome, so motor commands "
        "change although body input is not wired to them. `body_feedback` is therefore off by "
        "default.\n\n"
        "**Post hoc, added after the check failed:** the same tilts at smaller angles, real "
        "wiring only.\n\n"
        + _table(["tilt", "degrees", "mean turn", "max escape", "reversing frames",
                  "peak GF L/R Hz", "peak DNa02 L/R Hz"], sweep)
        + _tilt_seeds(b.get("post_hoc_seeds"))
    )  # fmt: skip


def _tilt_seeds(rows):
    if not rows:
        return ""
    tilts = list(dict.fromkeys(r["tilt"] for r in rows))
    table = []
    for tilt in tilts:
        mine = [r for r in rows if r["tilt"] == tilt]
        table.append((
            tilt, f"{sum(r['motor_changed'] for r in mine)} of {len(mine)}",
            " / ".join(f"{r['mean_turn']:+.3f}" for r in mine),
            " / ".join(f"{r['max_escape']:.2f}" for r in mine),
        ))  # fmt: skip
    seeds = ", ".join(str(s) for s in dict.fromkeys(r["seed"] for r in rows))
    return (
        "\n\n**Post hoc, added when a change of the Poisson generator moved the verdict** "
        "(the previous kernel failed three of four tilts, this one fails fewer): the 20° tilts "
        f"with brain seeds {seeds}. A seed counts as a motor change if escape ≥ 0.3, the "
        "robot reverses, or |mean turn| ≥ 0.1. The effect of a 20° tilt on motor commands sits "
        "at these thresholds.\n\n"
        + _table(["tilt", "seeds with a motor change", "mean turn per seed", "max escape per seed"],
                 table)
    )  # fmt: skip


def _retina(r):
    axes = [
        (side, a["cells"]["T4a"], a["angles_deg"]["T4a-T4b"], a["angles_deg"]["T4c-T4d"],
         a["angles_deg"]["T4a-T4c"], " / ".join(f"{a['concentration'][k]:.2f}" for k in
                                                ("T4a", "T4b", "T4c", "T4d")))
        for side, a in r["eye_axes"].items()
    ]  # fmt: skip
    shown = ("L2", "Tm1", "Tm2", "Tm4", "Tm9", "CT1", "T5a", "T5b", "T4a", "HSE", "LPLC2")
    scenes = [
        (s["scene"],) + tuple(_sides({side: s["mean_hz"][f"{t}:{side}"] for side in "LR"})
                               .replace("/", " / ") for t in shown)
        for s in r["scenes"]
    ]  # fmt: skip
    saturating = [
        (x["condition"], ", ".join(f"{t} {hz:g}" for t, hz in x["mean_hz"].items() if hz > 0),
         x["t5_mean_v_mv"], x["t5_max_v_mv"], x["t5_mean_g_mv"],
         f"{x['t5_share_net_inhibitory']:.0%}")
        for x in r["saturating_l2"]
    ]  # fmt: skip
    inputs = ", ".join(f"{x['type']} {x['synapses_per_t5']:+g}" for x in r["t5_inputs"])
    return (
        "**Exploratory, not pre-registered.** In `vision: retina` the camera is not reduced to "
        "optical flow: each of the "
        f"{r['columns']:,} lamina L2 cells with column coordinates receives Poisson spikes in "
        f"proportion to how much darker its pixel became (0 Hz up to {r['floor']:.0%}, 150 Hz "
        f"from {r['full']:.0%}), and motion has to be computed by the connectome. The first "
        "simulations in this mode ran before any pass condition was written and already showed "
        "the motion detectors silent, so this section reports observations and a mechanism, "
        "with no pass or fail.\n\n"
        "*Where the columns look.* The orientation of each eye's column lattice comes from the "
        "wiring: every T4 cell gets Mi9 input from the side a preferred-direction stimulus "
        "approaches from and Mi4 input from the other side (Takemura et al. 2017), so the "
        "Mi4-minus-Mi9 input centroid points along its preferred direction. Opposite subtypes "
        "come out opposite and orthogonal subtypes orthogonal:\n\n"
        + _table(["eye", "T4a cells", "T4a vs T4b", "T4c vs T4d", "T4a vs T4c",
                  "agreement across cells (a/b/c/d)"], axes)
        + "\n\nMean firing rate over 3 s per scene (Hz, left / right eye):\n\n"
        + _table(["scene"] + list(shown), scenes)
        + "\n\nEvery L2 cell of both eyes driven at 150 Hz for 1 s, T5 membrane sampled from 0.2 s:\n\n"
        + _table(["weight", "mean rates (Hz)", "T5 mean v mV", "T5 max v mV", "T5 mean g mV",
                  "samples with net inhibition"], saturating)
        + f"\n\nSigned synapses onto an average T5 cell: {inputs}.\n\n"
        "The OFF pathway carries the signal from L2 to Tm1, Tm2 and Tm4, but T5 stays below "
        "threshold even under saturating input: CT1, two GABAergic cells that contact every "
        "column, fire at over 100 Hz and keep T5 inhibited, and Tm9, T5's largest excitatory "
        "input, is silent because its input L3 has column coordinates only in the right eye and "
        "is not driven. In the fly CT1 computes locally in each column; as a single point neuron "
        "it inhibits the whole eye at once. The ON pathway is not driven at all, because "
        "glutamatergic L1 signals light by releasing less transmitter. Vision therefore stays "
        "on optical flow by default. (The approaching disk of the synthetic scenes is mostly "
        "brighter than the background, so in `approach_center` almost no column darkens.)"
    )  # fmt: skip


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


def _eye_configuration(c, crit):
    gap = chr(10) * 2
    directions = ['grating_left', 'grating_right', 'grating_up', 'grating_down']
    cfg = c['configuration']
    rows = [
        (row['group'], row['preferred_direction'].replace('grating_', ''),
         *(f"{row['hz'][d]:.1f}" for d in directions), f"{row['dsi']:+.3f}")
        for row in c['rows']
    ]  # fmt: skip
    control_rows = [
        (f"shuffled seed {x['seed']}",
         f"{sum(abs(r['dsi']) for r in x['rows']) / len(x['rows']):.3f}",
         f"{max(r['dsi'] for r in x['rows']):+.3f}",
         f"{max(r['max_hz'] for r in x['rows']):.1f}")
        for x in c['controls']
    ]  # fmt: skip
    step = c['brightness_step']
    kinetics = (
        f"slow synapses ({', '.join(cfg['slow_inputs'])}) with a {cfg['slow_tau_ms']:.0f} ms "
        'time constant' if cfg['slow_inputs'] else
        'one synaptic time constant for every synapse, as in the reference model'
    )  # fmt: skip
    mirrored = ', '.join(f"{m['subtype']} {'yes' if m['as_predicted'] else 'NO'}"
                         for m in c['mirrored'])  # fmt: skip
    quiet = (f"; below {crit['min_hz']:.0f} Hz: " + ', '.join(c['quiet_groups'])
             if c['quiet_groups'] else '')  # fmt: skip
    return (
        f"### {c['name']}{gap}{_mark(c['passed'])}. Model: {kinetics}. Mean DSI: T4 "
        f"{c['mean_dsi']['T4']:+.3f}, T5 {c['mean_dsi']['T5']:+.3f}; pairs at or above "
        f"{crit['min_dsi']}: T4 {c['selective']['T4']}/8, T5 {c['selective']['T5']}/8; "
        f"shuffled wiring mean |DSI| {c['control_mean_abs_dsi']}; mirrored as predicted: "
        f"{mirrored}{quiet}.{gap}"
        'Mean rate over 2 s of drifting grating (Hz), after 0.5 s of settling:' + gap
        + _table(['group', 'preferred', 'left', 'right', 'up', 'down', 'DSI'], rows)
        + gap + 'Shuffled-wiring controls:' + gap
        + _table(['control', 'mean |DSI|', 'largest DSI', 'largest rate'], control_rows)
        + gap + 'Horizontal system cells, which integrate T4/T5 output (reported only):' + gap
        + _table(['group', 'left', 'right', 'up', 'down'],
                 [(x['group'], *(f"{x['hz'][d]:.1f}" for d in directions)) for x in c['hs']])
        + gap + 'Full-field brightness step, 200 ms per phase, mean rate per type (Hz). The '
        'ON/OFF split of the fly follows from the transmitter signs alone, and was first seen '
        'before the pre-registration:' + gap
        + _table(['type', 'bright', 'dark', 'change'],
                 [(t, f"{step['bright'][t]:.1f}", f"{step['dark'][t]:.1f}",
                   f"{step['dark'][t] - step['bright'][t]:+.1f}") for t in step['bright']])
        + gap + 'Post hoc: a population mean over the whole eye is blind to which column is '
        'bright, so a still grating, a drifting one and a full-field flicker give the same '
        'rates; the verdict rests on the direction selectivity index, not on these:' + gap
        + _table(['pattern', 'T4a:L', 'T4c:L', 'T5a:L', 'T5c:L', 'L1:L', 'L2:L', 'Tm9:L'],
                 [(name, *(f"{row[g]:.2f}" for g in ('T4a:L', 'T4c:L', 'T5a:L', 'T5c:L',
                                                     'L1:L', 'L2:L', 'Tm9:L')))
                  for name, row in c['post_hoc_patterns'].items()])  # fmt: skip
    )


def _eye(r):
    gap = chr(10) * 2
    crit = r['criterion']
    first = r['configurations'][0]['configuration']
    columns = first['imputed_columns']
    header = (
        '**Pre-registered** (criteria committed before the first grating was simulated in '
        f"this mode). In `vision: eye` the camera drives the {first['photoreceptors']:,} "
        'photoreceptors themselves, at a rate proportional to the brightness of their pixel, '
        'and the connectome has to do the rest. Two declared changes make that possible: '
        f"{first['tonic_cells']:,} columnar optic-lobe neurons fire tonically at "
        f"{first['tonic_hz']:.0f} Hz without input (most of these cells do not spike in the "
        'fly; they release transmitter continuously, and since the photoreceptors are '
        'histaminergic the signal is a *release* from inhibition, which a model without a '
        'baseline cannot carry at all), and CT1 is split into one neuron per column, its '
        'compartments being electrically isolated in the fly (Meier & Borst 2019). Columns '
        'the release does not assign — T4, T5, Tm3, L4, the photoreceptors — come from the '
        'wiring: on the neurons that do have one the rule is exact for '
        f"{columns['accuracy']['exact']:.1%} and within one column for "
        f"{columns['accuracy']['within_one']:.1%}, and {columns['neurons_with_column']:,} "
        f"neurons end up with a column.{gap}"
        'Pass condition, the same for every configuration: each T4 and T5 subtype reaches '
        f"{crit['min_hz']:.0f} Hz on some direction; at least {crit['pairs_required']} of the "
        '8 subtype-eye pairs of T4 and of T5 have DSI ≥ '
        f"{crit['min_dsi']} in the direction the wiring predicts, with positive mean DSI; the "
        'eyes prefer mirrored directions; and shuffled wiring stays below '
        f"{crit['control_max_mean_abs_dsi']} mean |DSI|. The second configuration was "
        'pre-registered after the first failed and changes the model in one declared way.'
        + gap
    )  # fmt: skip
    return header + gap.join(_eye_configuration(c, crit) for c in r['configurations'])

def _participation(p):
    gap = chr(10) * 2
    rows = [
        (x['condition'], x['scene'], f"{x['neurons_kept']:,}", x['neurons_that_fired'],
         x['max_escape'], f"{x['mean_turn']:+.3f}", x['peak_giant_fiber_hz'])
        for x in p['rows']
    ]  # fmt: skip
    whole = [x for x in p['rows'] if x['condition'] == 'whole brain']
    return (
        'How much of the connectome carries the behaviour, measured by silencing the rest. '
        f"The camera drives {p['driven']} cells and the motor read-out reads {p['readout']}; "
        f"{p['two_step_intermediates']} neurons lie on a two-synapse path between them, out of "
        f"{p['neurons']:,}. A silenced neuron still fires but reaches no target.{gap}"
        + _table(['kept', 'scene', 'neurons kept', 'neurons that fired', 'max escape',
                  'mean turn', 'peak Giant Fiber Hz'], rows)
        + gap + 'Read this as a limit on the headline: in the approach scene '
        f"{whole[0]['neurons_that_fired']:,} neurons of {p['neurons']:,} fire at all, the "
        'escape command is reproduced by the driven and read-out cells alone (their direct '
        'synapses are the Giant Fiber pathway of the fly), and the turn needs the '
        'two-synapse neurons. The rest of the brain shifts the numbers by a few percent '
        'rather than producing the behaviour.'
    )  # fmt: skip

def _sensitivity(x):
    gap = chr(10) * 2
    rate_rows = [(f"{r['stimulus_hz']:.0f} Hz", r['giant_fiber'], r['dna02'])
                 for r in x['stimulus_rate']]  # fmt: skip
    weight_rows = [(f"{r['w_syn_scale']:.1f}x ({r['w_syn_mv']} mV)", r['giant_fiber'],
                    r['dna02']) for r in x['synaptic_weight']]  # fmt: skip
    seed_rows = [(r['seed'], r['scene'], r['max_escape'], f"{r['mean_turn']:+.3f}",
                  r['peak_giant_fiber_hz']) for r in x['brain_seeds']]  # fmt: skip
    return (
        'Three constants of the model were chosen rather than measured: the 150 Hz '
        'stimulation rate of the reference model, the rescaled synaptic weight, and the seed '
        'that sets every Poisson draw. These tables show how much the results depend on them. '
        'Reported, with no pass condition.'
        + gap + 'Stimulation rate of the driven population (mean rate of the read-out, Hz):'
        + gap
        + _table(['drive', 'Giant Fiber (LPLC2+LC4 left)', 'DNa02 left (LC10a left)'], rate_rows)
        + gap + 'Synaptic weight, scaled around the matched value:' + gap
        + _table(['w_syn', 'Giant Fiber', 'DNa02 left'], weight_rows)
        + gap
        + f"Closed loop over {x['seeds']} brain seeds: escape in "
        f"{x['seeds_with_escape']}/{x['seeds']} seeds for the approach scene, a left turn in "
        f"{x['seeds_with_left_turn']}/{x['seeds']} for the passing object.{gap}"
        + _table(['seed', 'scene', 'max escape', 'mean turn', 'peak Giant Fiber Hz'], seed_rows)
    )  # fmt: skip

def skipped(results):
    """Headline checks that did not run here, so that a partial run cannot read as a pass."""
    missing = []
    if "skipped" in results.get("reference", {}):
        missing.append("Brian2 spike-for-spike comparison (" + results["reference"]["skipped"] + ")")
    if "skipped" in results.get("devices", {}):
        missing.append("CPU and GPU agreement (" + results["devices"]["skipped"] + ")")
    return missing


def failures(results):
    failed = [r["name"] for r in results["pathways"]["rows"] if not r["passed"]]
    if "feeding" in results and not results["feeding"]["passed"]:
        failed.append("Published feeding circuit (sugar GRNs to MN9)")
    if "body" in results and not results["body"]["passed"]:
        failed.append("Johnston's organ screen")
    if "body_loop" in results and not results["body_loop"]["passed"]:
        failed += [f"Tilt loop: {r['tilt']}" for r in results["body_loop"]["rows"] if not r["passed"]]
    failed += [
        f"{wiring}: {r['scene']} (seed {r.get('seed')})"
        for wiring, rows in results["closed_loop"].items()
        for r in rows
        if not r["passed"]
    ]
    reference = results["reference"]
    failed += [f"Brian2 seed {r['seed']}" for r in reference.get("runs", []) if not r["identical"]]
    if results.get("devices", {}).get("passed") is False:
        failed.append("CPU and GPU kernels agree")
    if "eye" in results:
        failed += [f"Motion detectors from pixels ({c['name']})"
                   for c in results["eye"]["configurations"] if not c["passed"]]
    return failed


def render_markdown(results):
    m, d, model = results["machine"], results["dataset"], results["model"]
    stability, perf = results["stability"], results["performance"]
    failed, not_run = failures(results), skipped(results)
    summary = "All checks below passed." if not failed else "Failed: " + ", ".join(failed) + "."
    if not_run:
        summary += " **Not run on this machine: " + "; ".join(not_run) + ".**"
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
    if "side_rule" in d:
        dataset_rows += [
            ("side of a neuron", d["side_rule"]),
            ("neurons with an optic-lobe column", f"{d['neurons_with_optic_column']:,}"),
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
        f"NumPy {m['numpy']}, numba {m['numba']}"
        + (f", GPU {m['gpu']['name']} with numba-cuda-mlir {m['gpu']['numba_cuda_mlir']}"
           if m.get("gpu") else "")
        + (f"; simulations on device `{m['device']}`" if m.get("device") else "")
        + ". Raw numbers: [validation/results.json](validation/results.json).",
        f"**{summary}**",
        "Reproduce:\n\n```bash\npip install -e \".[connectome,reference]\"\n"
        "python -m flybrain_robot.connectome.build download --data data/malecns-v1.0\n"
        "python -m flybrain_robot.connectome.build compile --data data/malecns-v1.0\n"
        "python -m flybrain_robot.validation --data data/malecns-v1.0\n```",
        "## 1. Dataset",
        _table(["", ""], dataset_rows),
        "## 2. The simulation kernel matches the reference simulator",
        _reference(results["reference"]),
    ]
    if "devices" in results:
        parts += ["## 2b. CPU and GPU kernels agree exactly", _devices(results["devices"])]
    parts += [
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
    ]
    if "feeding" in results:
        parts += ["## 4b. Published feeding circuit (pre-registered)", _feeding(results["feeding"])]
    if "body" in results:
        parts += ["## 4c. Body sense: Johnston's organ (pre-registered)", _body(results["body"])]
    if "body_loop" in results:
        parts += ["## 4d. Tilted robot in the closed loop (pre-registered)",
                  _body_loop(results["body_loop"])]  # fmt: skip
    parts += [
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
    ]
    if "retina" in results:
        parts += ["## 5b. Retinotopic vision through the lamina (exploratory)",
                  _retina(results["retina"])]  # fmt: skip
    if "eye" in results:
        parts += ["## 5c. Motion detectors from pixels (pre-registered)",
                  _eye(results["eye"])]  # fmt: skip
    if "participation" in results:
        parts += ["## 5d. How much of the brain participates",
                  _participation(results["participation"])]  # fmt: skip
    if "sensitivity" in results:
        parts += ["## 5e. Dependence on the hand-chosen constants",
                  _sensitivity(results["sensitivity"])]  # fmt: skip
    parts += [
        "## 6. Speed",
        _table(["device", "condition", "simulated s", "wall s", "× real time", "spikes"],
               [(r.get("device", "cpu"), r["condition"], r["simulated_s"], r["wall_s"],
                 r["realtime_factor"], f"{r['spikes']:,}") for r in perf]),  # fmt: skip
        "Both kernels advance in blocks of 1.8 ms, the synaptic delay, inside which neurons do "
        "not interact. The CPU kernel processes only neurons away from rest; the GPU kernel "
        "runs two launches per block over all neurons. Timings on a desktop GPU vary by tens of "
        "percent between runs because it also drives the display. When a device is slower than "
        "real time, the robot loop still advances brain time by one frame period per camera "
        "frame and reports the ratio.",
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
