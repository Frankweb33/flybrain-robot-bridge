"""Connectome compilation, LIF kernel and MaleCNS backend on small synthetic graphs.

These run without the 1 GB MaleCNS download. Results on the real connectome are
produced by ``python -m flybrain_robot.validation`` and recorded in docs/VALIDATION.md.
"""

import math

import numpy as np
import pytest

pytest.importorskip("numba")

from flybrain_robot.brain.base import MotorActivity  # noqa: E402
from flybrain_robot.brain.lif import (  # noqa: E402
    FLYWIRE_783_SYNAPSES_PER_NEURON,
    SHIU_W_SYN_MV,
    LIFParams,
    LIFSimulator,
)
from flybrain_robot.connectome.graph import load_connectome, save_connectome  # noqa: E402
from flybrain_robot.connectome.synthetic import (  # noqa: E402
    connectome_from_edges,
    random_connectome,
)
from flybrain_robot.telemetry import Telemetry  # noqa: E402
from flybrain_robot.vision import SensoryInput  # noqa: E402


def test_compile_from_janelia_tables(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from flybrain_robot.connectome import build

    pd.DataFrame(
        dict(
            bodyId=[30, 10, 20, 99, 40, 50, 60],
            superclass=["descending_neuron", "visual_projection", "cb_intrinsic", None,
                        "cb_sensory", "cb_sensory", "ol_intrinsic"],
            type=["DNp01", "LPLC2", "X", None, "JO-EV1", "JO-CM", "L1"],
            instance=["DNp01(GF)_L", "LPLC2_L", "X_L", None, "JO-EV1_R", "JO-CM_L", "L1_R"],
            somaSide=["L", "L", "R", None, None, None, "R"],
            rootSide=[None, None, None, None, "R", "unknown", None],
            assignedOlHex1=[None, None, None, None, None, None, 5.0],
            assignedOlHex2=[None, None, None, None, None, None, 7.0],
        )
    ).to_feather(tmp_path / build.ANNOTATIONS)  # fmt: skip
    pd.DataFrame(
        dict(
            body=[10, 20, 30, 99],
            consensus_nt=["acetylcholine", "unclear", "unclear", "gaba"],
            celltype_predicted_nt=["acetylcholine", "gaba", "unclear", "gaba"],
        )
    ).to_feather(tmp_path / build.TRANSMITTERS)
    pd.DataFrame(
        dict(body_pre=[10, 20, 20, 99, 10], body_post=[30, 30, 10, 30, 99], weight=[7, 3, 2, 50, 4])
    ).to_feather(tmp_path / build.WEIGHTS)

    manifest = build.compile_graph(tmp_path, verify=False)
    graph = load_connectome(tmp_path)

    assert manifest["neurons"] == 6  # body 99 has no superclass: not a neuron
    assert manifest["neuron_to_neuron_edges"] == 3 and manifest["synapses"] == 12
    assert list(graph.body_id) == [10, 20, 30, 40, 50, 60]
    assert list(graph.transmitter[:3]) == ["acetylcholine", "gaba", "unclear"]  # type fallback
    # Sensory neurons have no soma side: root side first, "unknown" falls to the suffix.
    assert list(graph.side) == ["L", "R", "L", "R", "L", "R"]
    assert graph.select("JO-EV1:R").tolist() == [3] and graph.select("JO-CM:L").tolist() == [4]
    assert graph.select("JO-*").tolist() == [3, 4] and graph.select("JO-C*:L").tolist() == [4]
    assert graph.hex1.tolist() == [-1, -1, -1, -1, -1, 5] and graph.hex2[5] == 7
    assert manifest["neurons_with_optic_column"] == 1
    lplc2, x, gf = graph.select("LPLC2:L")[0], graph.select("X")[0], graph.select("DNp01:L")[0]
    edges = {
        (pre, int(post)): int(w)
        for pre in range(graph.size)
        for post, w in zip(
            graph.indices[graph.indptr[pre] : graph.indptr[pre + 1]],
            graph.weight[graph.indptr[pre] : graph.indptr[pre + 1]],
        )
    }
    assert edges == {(lplc2, gf): 7, (x, lplc2): -2, (x, gf): -3}  # GABA is inhibitory
    with pytest.raises(ValueError, match="No neurons match"):
        graph.select("LPLC2:R")


def test_matched_weight_rescales_to_flywire_input():
    params = LIFParams.matched_to({"synapses": 2_000_000, "neurons": 1_000})
    assert params.w_syn == pytest.approx(SHIU_W_SYN_MV * FLYWIRE_783_SYNAPSES_PER_NEURON / 2000)
    assert params.poisson_kick_mv == pytest.approx(SHIU_W_SYN_MV * 250)


def test_single_synapse_psp_is_exact():
    graph = connectome_from_edges(2, [0], [1], [10])
    params = LIFParams(w_syn=0.3)
    sim = LIFSimulator(graph, params=params)
    sim.set_stimulus([([0], 1000 / params.dt_ms)])
    sim.run(params.dt_ms)  # kick applied after this step's threshold test
    sim.set_stimulus([([0], 0.0)])
    sim.run(params.dt_ms)  # neuron 0 crosses threshold and spikes
    assert sim.take_counts().tolist() == [1, 0]
    sim.run(params.t_dly_ms)  # spike lands on neuron 1 after the synaptic delay
    g0 = 10 * params.w_syn
    assert sim.g[1] == pytest.approx(g0)
    sim.run(5.0)
    k = 50
    a = math.exp(-k * params.dt_ms / params.t_mbr_ms)
    b = math.exp(-k * params.dt_ms / params.tau_ms)
    expected = params.v_0 + g0 * params.tau_ms / (params.tau_ms - params.t_mbr_ms) * (b - a)
    assert sim.v[1] == pytest.approx(expected, rel=1e-9)
    assert sim.g[1] == pytest.approx(g0 * b, rel=1e-9)


def test_silent_network_returns_to_rest_and_sleeps():
    graph = random_connectome(n=200, seed=3)
    sim = LIFSimulator(graph, params=LIFParams(w_syn=0.5), seed=4)
    sim.set_stimulus([(np.arange(10), 200.0)])
    sim.run(200)
    assert sim.take_counts().sum() > 0
    sim.set_stimulus([])
    sim.run(1000)
    assert sim.n_active == 0
    assert np.all(sim.v == sim.params.v_0) and np.all(sim.g == 0)


def test_simulation_is_deterministic_per_seed():
    graph = random_connectome(n=300, seed=5)
    runs = []
    for seed in (7, 7, 8):
        sim = LIFSimulator(graph, params=LIFParams(w_syn=1.0), seed=seed)
        sim.set_stimulus([(np.arange(20), 100.0)])
        sim.run(300)
        runs.append(sim.take_counts())
    assert np.array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_shuffled_control_keeps_degrees_and_weights():
    graph = random_connectome(n=300, seed=6)
    control = graph.shuffled(seed=1)
    assert np.array_equal(control.indptr, graph.indptr)
    assert np.array_equal(control.weight, graph.weight)
    assert np.array_equal(np.bincount(control.indices, minlength=300),
                          np.bincount(graph.indices, minlength=300))  # fmt: skip
    assert not np.array_equal(control.indices, graph.indices)


def test_silencing_transmitters_zeroes_outgoing_weights():
    graph = random_connectome(n=50, seed=2)
    graph.transmitter[:5] = "octopamine"
    silent = graph.without_fast_synapses_from(["octopamine"])
    assert not silent.weight[: graph.indptr[5]].any()
    assert np.array_equal(silent.weight[graph.indptr[5] :], graph.weight[graph.indptr[5] :])


def test_silencing_neurons_removes_only_their_output():
    graph = connectome_from_edges(3, [0, 1, 2], [1, 2, 0], [5, 6, 7])
    lesioned = graph.silenced([1])
    assert lesioned.weight.tolist() == [5, 0, 7]
    assert graph.weight.tolist() == [5, 6, 7]  # the original is untouched


def robot_circuit():
    """Mirror-symmetric toy brain using the cell types the MaleCNS backend reads."""
    cells = []
    for side in "LR":
        cells += [("LPLC2", side)] * 6 + [("LC4", side)] * 4 + [("LC10a", side)] * 6
        cells += [(t, side) for t in ("DNp01", "DNa01", "DNa02", "DNp02", "DNp04", "DNp11")]
        cells += [("DNp09", side), ("MDN", side), ("DNp18", side), ("DNp20", side)]
        cells += [("JO-CM", side)] * 4 + [("JO-EV1", side)] * 4
    types, sides = zip(*cells)
    index = {}
    for i, key in enumerate(cells):
        index.setdefault(key, []).append(i)
    pre, post, weight = [], [], []
    for side in "LR":
        for src, dst in (("LPLC2", "DNp01"), ("LC4", "DNp01"), ("LC10a", "DNa02"),
                         ("JO-EV1", "DNp20")):  # fmt: skip
            for s in index[(src, side)]:
                pre.append(s), post.append(index[(dst, side)][0]), weight.append(40)
    return connectome_from_edges(len(cells), pre, post, weight, types=types, sides=sides)


def test_malecns_backend_escapes_and_steers(tmp_path):
    from flybrain_robot.brain.malecns import MaleCNSBackend

    save_connectome(robot_circuit(), tmp_path)
    brain = MaleCNSBackend(tmp_path, params=LIFParams(w_syn=0.4))
    for _ in range(30):
        activity = brain.step(SensoryInput(object_left=0.9), Telemetry(), 1 / 30)
    assert activity.escape == 0 and activity.right > activity.left  # turn left toward object
    assert brain.named_rates()["DNa02:L"] > 20

    brain.reset()
    for _ in range(30):
        activity = brain.step(SensoryInput(looming_right=0.9), Telemetry(), 1 / 30)
    assert activity.escape == 1.0
    assert brain.named_rates()["DNp01:R"] > 20 and brain.named_rates()["DNp01:L"] == 0

    brain.reset()
    for _ in range(30):
        noise = SensoryInput(looming_left=0.05, object_right=0.08)
        activity = brain.step(noise, Telemetry(), 1 / 30)
    assert activity == MotorActivity(0.4, 0.4, 0.0)  # below the encoder noise floor


def test_tilt_reaches_johnstons_organ_without_steering(tmp_path):
    from flybrain_robot.brain.malecns import MaleCNSBackend
    from flybrain_robot.telemetry import tilted_telemetry

    save_connectome(robot_circuit(), tmp_path)
    brain = MaleCNSBackend(tmp_path, params=LIFParams(w_syn=0.4))
    rates = brain.drive_rates(SensoryInput(), tilted_telemetry(roll_left_deg=20))
    assert rates["JO-C*:L"] == 150 and rates["JO-E*:R"] == 150  # lower side pushed
    assert rates["JO-C*:R"] == 0 and rates["JO-E*:L"] == 0 and rates["LPLC2:L"] == 0
    both = brain.drive_rates(SensoryInput(), tilted_telemetry(roll_left_deg=15, pitch_down_deg=15))
    assert both["JO-E*:R"] == 150  # roll and pitch drives add, capped at 150 Hz
    assert 0 < both["JO-E*:L"] < 150
    for _ in range(30):
        activity = brain.step(SensoryInput(), tilted_telemetry(pitch_down_deg=20), 1 / 30)
    named = brain.named_rates()
    assert named["JO-E*:L"] > 20 and named["DNp20:L"] > 20 and named["JO-C*:L"] == 0
    assert activity == MotorActivity(0.4, 0.4, 0.0)  # body input is not wired to motors


def test_malecns_backend_requires_compiled_data(tmp_path):
    from flybrain_robot.brain.malecns import MaleCNSBackend

    with pytest.raises(ValueError, match="Obtain data"):
        MaleCNSBackend()
    with pytest.raises(ValueError, match="No compiled connectome"):
        MaleCNSBackend(tmp_path)


def test_cli_runs_connectome_backend(tmp_path, capsys):
    from flybrain_robot.main import main

    save_connectome(robot_circuit(), tmp_path)
    main(["--backend", "malecns", "--data", str(tmp_path), "--synthetic", "--scene",
          "approach_left", "--dry-run", "--steps", "12"])  # fmt: skip
    out = capsys.readouterr().out
    assert f"neurons={robot_circuit().size}" in out and "GF=" in out
    main(["--backend", "malecns", "--data", str(tmp_path), "--synthetic", "--scene", "static",
          "--tilt", "0,20", "--dry-run", "--steps", "11"])  # fmt: skip
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert "tilt(roll/pitch)=+0/+20deg" in last and "JO-C=0/0Hz" in last
    assert "JO-E=0/0Hz" not in last  # nose down pulls both antennae: JO-E fires
