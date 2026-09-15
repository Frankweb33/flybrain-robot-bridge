"""Compiled connectome: neuron table plus signed synapse counts in CSR form (by presynapse)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2
COMPILED_DIR = "compiled"
TEXT_COLUMNS = ("type", "instance", "side", "superclass", "transmitter")


@dataclass
class Connectome:
    body_id: np.ndarray
    type: np.ndarray
    instance: np.ndarray
    side: np.ndarray  # "L", "R", "M" or ""
    hex1: np.ndarray  # optic-lobe column coordinates (int16), -1 if none
    hex2: np.ndarray
    superclass: np.ndarray
    transmitter: np.ndarray
    sign: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray
    manifest: dict

    @property
    def size(self):
        return len(self.body_id)

    def select(self, spec):
        """Indices for ``TYPE`` or ``TYPE:L`` / ``TYPE:R``; errors if empty.

        The side is the soma side, or for sensory neurons the side their root enters.
        """
        name, _, side = spec.partition(":")
        mask = self.type == name
        if side:
            mask &= self.side == side
        found = np.flatnonzero(mask).astype(np.int32)
        if not len(found):
            raise ValueError(f"No neurons match {spec!r}")
        return found

    def labels(self, indices):
        return [f"{self.instance[i] or self.type[i]} ({self.body_id[i]})" for i in indices]

    def without_fast_synapses_from(self, transmitters):
        """Copy whose neurons releasing ``transmitters`` have no fast synaptic effect."""
        silent = np.isin(self.transmitter, list(transmitters))
        manifest = dict(self.manifest, silenced_transmitters=sorted(transmitters))
        return self._without_output(silent, manifest)

    def silenced(self, indices):
        """Copy in which the given neurons still fire but reach no target (a lesion)."""
        silent = np.zeros(self.size, bool)
        silent[np.asarray(indices)] = True
        manifest = dict(self.manifest, silenced_neurons=int(silent.sum()))
        return self._without_output(silent, manifest)

    def _without_output(self, silent, manifest):
        per_edge = np.repeat(silent, np.diff(self.indptr))
        weight = np.where(per_edge, 0, self.weight).astype(self.weight.dtype)
        return Connectome(**{**self.__dict__, "weight": weight, "manifest": manifest})

    def shuffled(self, seed=0):
        """Control graph: postsynaptic targets are permuted across all edges.

        Every neuron keeps its number of outgoing and incoming connections, its
        transmitter sign and its outgoing synapse counts; only who connects to whom
        is destroyed.
        """
        indices = np.random.default_rng(seed).permutation(np.asarray(self.indices))
        manifest = dict(self.manifest, control=f"permuted targets, seed {seed}")
        return Connectome(**{**self.__dict__, "indices": indices, "manifest": manifest})


def save_connectome(graph, data_dir):
    root = Path(data_dir) / COMPILED_DIR
    root.mkdir(parents=True, exist_ok=True)
    for name in Connectome.__dataclass_fields__:
        if name != "manifest":
            np.save(root / f"{name}.npy", np.asarray(getattr(graph, name)), allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps(graph.manifest, indent=2), encoding="utf-8")
    return root


def load_connectome(data_dir):
    root = Path(data_dir) / COMPILED_DIR
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"No compiled connectome in {root}. Run: python -m flybrain_robot.connectome.build "
            f"download --data {data_dir} && python -m flybrain_robot.connectome.build compile "
            f"--data {data_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Compiled connectome format is outdated; run the compile step again")
    arrays = {}
    for name in Connectome.__dataclass_fields__:
        if name != "manifest":
            mmap = None if name in TEXT_COLUMNS else "r"
            arrays[name] = np.load(root / f"{name}.npy", mmap_mode=mmap, allow_pickle=False)
    return Connectome(**arrays, manifest=manifest)
