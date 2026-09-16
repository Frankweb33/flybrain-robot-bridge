"""Retinotopy from T4 input offsets, the retina encoder and the retina backend mode."""

import numpy as np
import pytest

from flybrain_robot.connectome.synthetic import connectome_from_edges
from flybrain_robot.vision import RetinaEncoder, grating, scene_frame


def eye_graph(extra_types=()):
    """One left eye: T4a-d with Mi9/Mi4 inputs offset along known directions, 25 L2 columns."""
    cells, hex1, hex2, sides = [], [], [], []

    def add(cell_type, h1=-1, h2=-1, side="L"):
        cells.append(cell_type), hex1.append(h1), hex2.append(h2), sides.append(side)
        return len(cells) - 1

    pre, post, weight = [], [], []
    # Preferred direction = Mi4 centroid minus Mi9 centroid.
    offsets = {"T4a": (1, 0), "T4b": (-1, 0), "T4c": (0, 1), "T4d": (0, -1)}
    for subtype, (d1, d2) in offsets.items():
        t4 = add(subtype)
        mi9, mi4 = add("Mi9", 5, 5), add("Mi4", 5 + d1, 5 + d2)
        pre += [mi9, mi4]
        post += [t4, t4]
        weight += [-4, -6]
    for h1 in range(5):
        for h2 in range(5):
            add("L2", h1, h2)
    for cell_type in extra_types:
        for side in "LR":
            add(cell_type, side=side)
    return connectome_from_edges(len(cells), pre, post, weight, types=cells, sides=sides,
                                 hex1=hex1, hex2=hex2)  # fmt: skip


def test_eye_axes_follow_mi9_to_mi4_offsets():
    from flybrain_robot.connectome.retinotopy import column_image_positions, eye_axes

    axes = eye_axes(eye_graph(), "L")
    assert np.allclose(axes.posterior, [1, 0]) and np.allclose(axes.dorsal, [0, 1])
    assert axes.angles_deg == {"T4a-T4b": 180.0, "T4c-T4d": 180.0, "T4a-T4c": 90.0}
    cells, x, y = column_image_positions(eye_graph(), "L2", axes)
    hex1 = eye_graph().hex1[cells]
    # Left eye: more posterior columns sit further left; anterior columns at the midline.
    assert x[hex1 == 4].max() < x[hex1 == 0].min() and x.max() <= 80
    with pytest.raises(ValueError, match="column coordinates"):
        column_image_positions(eye_graph(), "L3", axes)


def columnar_graph(columns=12, spread=1):
    """Two cells per column plus one wide-field cell that touches every column."""
    cells, hex1, hex2 = [], [], []

    def add(cell_type, h1=-1, h2=-1):
        cells.append(cell_type), hex1.append(h1), hex2.append(h2)
        return len(cells) - 1

    known = [add("L2", h, 3) for h in range(columns)]
    unknown = [add("T4a") for _ in range(columns)]  # same columns, no coordinates
    wide = add("CT1")
    pre, post, weight = [], [], []
    for column, (source, target) in enumerate(zip(known, unknown)):
        for neighbour in range(max(0, column - spread), min(columns, column + spread + 1)):
            pre.append(known[neighbour]), post.append(target), weight.append(10)
            if neighbour != column:  # lamina cells also contact their neighbours
                pre.append(source), post.append(known[neighbour]), weight.append(10)
        pre.append(target), post.append(wide), weight.append(5)
        pre.append(wide), post.append(source), weight.append(-3)
    return connectome_from_edges(len(cells), pre, post, weight, types=cells,
                                 sides=["L"] * len(cells), hex1=hex1, hex2=hex2)  # fmt: skip


def test_columns_are_recovered_from_partner_columns():
    from flybrain_robot.connectome.columns import accuracy, compartmentalize, impute_columns

    graph = columnar_graph()
    filled = impute_columns(graph, min_synapses=10)
    manifest = filled.manifest["imputed_columns"]
    assert manifest["neurons_with_column"] == 24  # the 12 T4a cells gained one, CT1 did not
    t4 = filled.hex1[np.asarray(filled.type) == "T4a"]
    # Each gets the column of its inputs; at the edge of the eye the centroid is half a
    # column short, because a border column has neighbours on one side only.
    assert np.abs(t4 - np.arange(12)).max() <= 1 and (t4 == np.arange(12)).sum() >= 10
    assert filled.hex1[np.asarray(filled.type) == "CT1"][0] == -1
    measured = accuracy(graph, min_synapses=10)  # the cells that do have a column
    assert measured["exact"] >= 0.8 and measured["within_one"] == 1.0

    split = compartmentalize(filled)
    # One compartment per column and partner group: the T4 partners of a column are kept
    # apart from the rest, as the medulla and lobula compartments of CT1 are in the fly.
    assert split.manifest["compartments"]["compartments"] == 23 and split.size == graph.size + 22
    assert len(np.asarray(split.indices)) == len(np.asarray(graph.indices))
    assert np.abs(np.asarray(split.weight)).sum() == np.abs(np.asarray(graph.weight)).sum()
    compartments = np.flatnonzero(np.asarray(split.type) == "CT1")
    degree = np.diff(np.asarray(split.indptr))[compartments]
    assert degree.max() == 1  # each compartment reaches only its own column
    instances = {str(x) for x in np.asarray(split.instance)[compartments]}
    assert any(name.endswith("_T4") for name in instances)  # the T4 side is its own unit
    assert any(not name.endswith("_T4") and "_col" in name for name in instances)


def test_luminance_encoder_reports_brightness():
    from flybrain_robot.vision import LuminanceEncoder

    encoder = LuminanceEncoder([10, 100], [20, 60])
    frame = np.full((240, 320, 3), 255, np.uint8)
    frame[30:50, 10:30] = 0  # around pixel (10, 20) at 160x120
    values = encoder.encode(frame).retina
    assert values[0] < 0.1 < 0.9 < values[1]
    assert np.array_equal(values, encoder.encode(frame).retina)  # no temporal difference


def test_retina_encoder_reports_darkening_only():
    encoder = RetinaEncoder([10, 100], [20, 60])
    bright = np.full((240, 320, 3), 200, np.uint8)
    first = encoder.encode(bright).retina
    darker = bright.copy()
    darker[30:50, 10:30] = 50  # around pixel (10, 20) at 160x120
    second = encoder.encode(darker).retina
    third = encoder.encode(bright).retina  # brightening back is not darkening
    assert not first.any()
    assert second[0] > 0.5 and second[1] == 0
    assert not third.any()


def test_flicker_matches_grating_darkening():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(1, 159, 300), rng.uniform(1, 119, 300)

    def mean_darkening(name):
        encoder = RetinaEncoder(x, y)
        return np.mean([encoder.encode(scene_frame(name, i)).retina.mean() for i in range(90)])

    moving, flicker = mean_darkening("grating_left"), mean_darkening("flicker")
    assert abs(flicker - moving) / moving < 0.1
    assert np.array_equal(grating(3)[0, :-3], grating(0)[0, 3:])  # bars move toward the left


def test_backend_retina_mode_drives_darkened_columns(tmp_path):
    pytest.importorskip("numba")
    from flybrain_robot.brain.lif import LIFParams
    from flybrain_robot.brain.malecns import MaleCNSBackend
    from flybrain_robot.connectome.graph import save_connectome
    from flybrain_robot.telemetry import Telemetry

    needed = ("LPLC2", "LC4", "LC10a", "DNp01", "DNa01", "DNa02", "DNp02", "DNp04", "DNp11",
              "DNp09", "MDN", "DNp18", "DNp20", "JO-CM", "JO-EV1", "T5a", "T5b", "T5c", "T5d",
              "HSE", "HSN", "HSS", "L3")  # fmt: skip
    save_connectome(mirror_right_eye(eye_graph(needed)), tmp_path)  # both eyes need a lattice
    brain = MaleCNSBackend(tmp_path, params=LIFParams(w_syn=0.4), vision="retina", device="cpu")
    encoder = brain.make_encoder()
    assert isinstance(encoder, RetinaEncoder) and len(brain.retina_cells) == 50
    bright = np.full((240, 320, 3), 200, np.uint8)
    brain.step(encoder.encode(bright), Telemetry(), 1 / 30)
    assert not brain.last_counts[brain.retina_cells].any()
    brain.step(encoder.encode(np.full_like(bright, 20)), Telemetry(), 1 / 30)
    assert brain.last_counts[brain.retina_cells].sum() > 50  # every column darkened by 90 %
    lplc2 = brain.graph.select("LPLC2")
    assert not brain.last_counts[lplc2].any()  # flow populations are not driven in this mode


def mirror_right_eye(graph):
    """Copy the left-eye lattice to a right eye so both eyes have columns."""
    n = graph.size
    lattice = ["T4a", "T4b", "T4c", "T4d", "Mi9", "Mi4", "L2"]
    eye = np.flatnonzero((graph.side == "L") & np.isin(graph.type, lattice))
    types = np.concatenate([graph.type, graph.type[eye]])
    sides = np.concatenate([graph.side, np.full(len(eye), "R")])
    hex1 = np.concatenate([graph.hex1, graph.hex1[eye]])
    hex2 = np.concatenate([graph.hex2, graph.hex2[eye]])
    new_index = {int(old): n + k for k, old in enumerate(eye)}
    pre = list(np.repeat(np.arange(n), np.diff(graph.indptr)))
    post, weight = list(graph.indices), list(graph.weight)
    for p, q, w in list(zip(pre, post, weight)):
        if int(p) in new_index and int(q) in new_index:
            pre.append(new_index[int(p)]), post.append(new_index[int(q)]), weight.append(w)
    return connectome_from_edges(len(types), pre, post, weight, types=types, sides=sides,
                                 hex1=hex1, hex2=hex2)  # fmt: skip
