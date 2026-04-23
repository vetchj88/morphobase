from __future__ import annotations

import numpy as np

from morphobase.ports.base import BasePort, BoundaryWindow


class ToyPatternPort(BasePort):
    def __init__(
        self,
        num_cells: int = 49,
        *,
        width: int | None = None,
        support_margin: int = 3,
    ) -> None:
        super().__init__()
        width = int(width or max(4, num_cells // 10))
        self._base_input_window = BoundaryWindow("left_pattern_input", 0, width, "left")
        self._base_output_window = BoundaryWindow("right_pattern_output", num_cells - width, num_cells, "right")
        self.scale = 1.0
        self.phase_offset = 0.0
        self.input_shift = 0
        self.output_shift = 0
        self.configure_boundary(
            num_cells=num_cells,
            input_window=self._base_input_window,
            output_window=self._base_output_window,
            support_margin=support_margin,
            name="toy_pattern_port",
        )

    def _shifted_input_window(self, shift: int) -> BoundaryWindow:
        base = self._base_input_window
        shift = int(np.clip(shift, 0, max(self.num_cells - base.width, 0)))
        start = min(shift, self.num_cells - base.width)
        stop = start + base.width
        return BoundaryWindow(base.label, start, stop, base.side)

    def _shifted_output_window(self, shift: int) -> BoundaryWindow:
        base = self._base_output_window
        shift = int(np.clip(shift, 0, max(self.num_cells - base.width, 0)))
        stop = max(base.width, self.num_cells - shift)
        start = max(0, stop - base.width)
        return BoundaryWindow(base.label, start, stop, base.side)

    def encode(self, external_input):
        signal = np.asarray(external_input, dtype=float).reshape(-1)
        width = self.boundary_window("input").width
        if signal.size == 0:
            signal = np.zeros(width, dtype=float)
        elif signal.size == 1:
            amplitude = float(signal[0])
            phase = np.linspace(0.0, np.pi, width)
            signal = amplitude + 0.18 * np.sin(phase + self.phase_offset)
        elif signal.size != width:
            source = np.linspace(0.0, 1.0, signal.size)
            target = np.linspace(0.0, 1.0, width)
            signal = np.interp(target, source, signal)
        return np.clip(signal * self.scale, 0.0, 1.0)

    def decode(self, boundary_state):
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        width = hidden.shape[0]
        phase = np.linspace(0.0, np.pi, width)
        weights = 0.55 + 0.45 * np.sin(phase + self.phase_offset)
        weights = weights / max(np.sum(weights), 1e-8)
        hidden_score = np.clip(hidden[:, 0] / 1.25, 0.0, 1.0)
        membrane_score = np.clip(0.5 * (membrane + 1.0), 0.0, 1.0)
        z_score = np.clip(0.5 * (z_alignment + 1.0), 0.0, 1.0)
        return float(
            np.clip(
                np.sum(
                    weights
                    * (
                        0.42 * hidden_score
                        + 0.24 * membrane_score
                        + 0.22 * field
                        + 0.12 * z_score
                    )
                ),
                0.0,
                1.0,
            )
        )

    def loss_fn(self, external_output, target) -> float:
        return float(np.mean((np.asarray(external_output, dtype=float) - np.asarray(target, dtype=float)) ** 2))

    def remap(self, mapping_spec: dict) -> None:
        if "scale" in mapping_spec:
            self.scale = float(mapping_spec["scale"])
        if "phase_offset" in mapping_spec:
            self.phase_offset = float(mapping_spec["phase_offset"])
        if "input_shift" in mapping_spec:
            self.input_shift = int(max(mapping_spec["input_shift"], 0))
            self.input_window = self._shifted_input_window(self.input_shift)
        if "output_shift" in mapping_spec:
            self.output_shift = int(max(mapping_spec["output_shift"], 0))
            self.output_window = self._shifted_output_window(self.output_shift)

    def damage(self, mask_spec: dict) -> None:
        if "input_attenuation" in mask_spec:
            self.input_attenuation = float(np.clip(mask_spec["input_attenuation"], 0.0, 1.0))
        if "readout_attenuation" in mask_spec:
            self.readout_attenuation = float(np.clip(mask_spec["readout_attenuation"], 0.0, 1.0))
