"""Extension seam only: no graph loader or biological simulation is provided."""

from pathlib import Path

from .base import BrainBackend

SOURCE = "https://www.janelia.org/project-team/flyem/male-cns-connectome"


class MaleCNSBackend(BrainBackend):
    def __init__(self, dataset_path=None):
        if not dataset_path or not Path(dataset_path).exists():
            raise ValueError(f"MaleCNS dataset path is missing. Obtain data from {SOURCE}")
        self.dataset_path = Path(dataset_path)
        self.load_graph()

    def load_graph(self):
        """Future: load licensed connectivity, neuron IDs and population annotations."""
        raise NotImplementedError("Not implemented yet: MaleCNS graph loading and simulation")

    def reset(self):
        raise NotImplementedError("Not implemented yet")

    def feed_sensory(self, sensory_input, telemetry):
        """Future: map camera and IMU features to annotated input populations."""
        raise NotImplementedError("Not implemented yet")

    def step(self, sensory_input, telemetry, dt):
        """Future: feed sensory input and advance the selected neuron dynamics."""
        raise NotImplementedError("Not implemented yet")

    def get_motor_activity(self):
        """Future: read explicitly mapped descending/motor populations."""
        raise NotImplementedError("Not implemented yet")
