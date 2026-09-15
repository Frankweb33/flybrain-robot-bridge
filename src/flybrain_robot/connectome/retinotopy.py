"""Where each optic-lobe column looks, derived from the connectome alone.

MaleCNS gives lamina and medulla neurons hexagonal column coordinates
(``assignedOlHex1/2``) but not the orientation of that lattice in the visual field.
T4 cells supply it: each receives Mi9 input from columns on the side a stimulus moving
in its preferred direction comes from, and Mi4 input from the opposite side (Takemura
et al. 2017; Haag et al. 2016). The synapse-weighted centroid of its Mi4 inputs minus
that of its Mi9 inputs points along its preferred direction. T4a cells prefer
front-to-back and T4c cells upward motion (Maisak et al. 2013), so their mean vectors
are the posterior and dorsal axes of each eye in column coordinates.

The mapping from the eye onto a forward-facing camera is a hand-made approximation:
the anterior-most columns of each eye look at the image centre line, the posterior-
most at that eye's image edge, dorsal at the top. A fly eye covers far more than half
of a camera image, and the frontal binocular overlap is ignored.
"""

from dataclasses import dataclass

import numpy as np

WIDTH, HEIGHT = 160, 120  # VisionEncoder resolution


@dataclass
class EyeAxes:
    side: str
    posterior: np.ndarray  # unit vector in (hex1, hex2) coordinates, from T4a
    dorsal: np.ndarray  # from T4c
    cells: dict  # cells with both Mi9 and Mi4 input, per T4 subtype
    concentration: dict  # length of the mean unit vector per subtype (1 = all agree)
    angles_deg: dict  # a-b, c-d and a-c angles between subtype mean vectors


def preferred_directions(graph, side):
    """Mean Mi4-minus-Mi9 input centroid vector per T4 subtype of one eye."""
    indptr, indices = np.asarray(graph.indptr), np.asarray(graph.indices)
    weight = np.abs(np.asarray(graph.weight)).astype(np.float64)
    pre = np.repeat(np.arange(graph.size, dtype=np.int32), np.diff(indptr))
    columns = np.stack([graph.hex1, graph.hex2], axis=1).astype(np.float64)
    result = {}
    for subtype in ("T4a", "T4b", "T4c", "T4d"):
        cells = np.flatnonzero((graph.type == subtype) & (graph.side == side))
        row = np.full(graph.size, -1, np.int64)
        row[cells] = np.arange(len(cells))
        sums = np.zeros((2, len(cells), 3))  # [Mi9, Mi4] x cell x (hex1, hex2, weight)
        for which, source_type in enumerate(("Mi9", "Mi4")):
            is_source = (graph.type == source_type) & (graph.hex1 >= 0)
            edges = np.flatnonzero(is_source[pre] & (row[indices] >= 0))
            k, s, w = row[indices[edges]], pre[edges], weight[edges]
            for axis in range(2):
                np.add.at(sums[which, :, axis], k, w * columns[s, axis])
            np.add.at(sums[which, :, 2], k, w)
        ok = (sums[0, :, 2] > 0) & (sums[1, :, 2] > 0)
        vectors = sums[1, ok, :2] / sums[1, ok, 2:] - sums[0, ok, :2] / sums[0, ok, 2:]
        units = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        result[subtype] = (vectors.mean(0), int(ok.sum()), float(np.linalg.norm(units.mean(0))))
    return result


def _angle(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def eye_axes(graph, side):
    pd = preferred_directions(graph, side)
    posterior, dorsal = pd["T4a"][0], pd["T4c"][0]
    return EyeAxes(
        side=side,
        posterior=posterior / np.linalg.norm(posterior),
        dorsal=dorsal / np.linalg.norm(dorsal),
        cells={k: v[1] for k, v in pd.items()},
        concentration={k: round(v[2], 3) for k, v in pd.items()},
        angles_deg={
            "T4a-T4b": round(_angle(pd["T4a"][0], pd["T4b"][0]), 1),
            "T4c-T4d": round(_angle(pd["T4c"][0], pd["T4d"][0]), 1),
            "T4a-T4c": round(_angle(pd["T4a"][0], pd["T4c"][0]), 1),
        },
    )


def column_image_positions(graph, cell_type, axes):
    """(neuron indices, x, y) in VisionEncoder pixels for the cells of one eye.

    ``cell_type`` is a type name, a prefix ending in ``*``, or a tuple of either.
    """
    names = (cell_type,) if isinstance(cell_type, str) else tuple(cell_type)
    match = np.zeros(graph.size, bool)
    for name in names:
        match |= (np.char.startswith(graph.type.astype(str), name[:-1]) if name.endswith("*")
                  else graph.type == name)  # fmt: skip
    cells = np.flatnonzero(match & (graph.side == axes.side) & (graph.hex1 >= 0))
    if len(cells) < 10:
        raise ValueError(f"{cell_type} has no column coordinates in the {axes.side} eye")
    hexes = np.stack([graph.hex1[cells], graph.hex2[cells]], axis=1).astype(np.float64)
    centred = hexes - hexes.mean(0)
    posterior, dorsal = centred @ axes.posterior, centred @ axes.dorsal

    def unit(values):
        lo, hi = np.percentile(values, [1, 99])
        return np.clip((values - lo) / (hi - lo), 0, 1)

    p, d = unit(posterior), unit(dorsal)
    half = WIDTH / 2
    x = half - p * (half - 1) if axes.side == "L" else half + p * (half - 1)
    y = (1 - d) * (HEIGHT - 1)
    return cells.astype(np.int32), x, y
