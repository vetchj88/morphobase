from __future__ import annotations

import numpy as np

from morphobase.ports.base import BasePort, BoundaryWindow


class ControlPort(BasePort):
    def __init__(self, num_cells: int = 49, *, width: int | None = None, support_margin: int = 3) -> None:
        super().__init__()
        width = int(width or max(4, num_cells // 10))
        self._base_input_window = BoundaryWindow("left_control_input", 0, width, "left")
        self._base_output_window = BoundaryWindow("right_control_output", num_cells - width, num_cells, "right")
        self.scale = 1.0
        self.flip = False
        self.input_shift = 0
        self.output_shift = 0
        self.configure_boundary(
            num_cells=num_cells,
            input_window=self._base_input_window,
            output_window=self._base_output_window,
            support_margin=support_margin,
            name="control_port",
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
        values = np.asarray(external_input, dtype=float).reshape(-1)
        if values.size == 0:
            values = np.zeros(1, dtype=float)
        if self.flip:
            values = values[::-1]
        width = self.boundary_window("input").width
        if values.size == 1:
            signal = np.full(width, float(values[0]), dtype=float)
        elif values.size != width:
            source = np.linspace(0.0, 1.0, values.size)
            target = np.linspace(0.0, 1.0, width)
            signal = np.interp(target, source, values)
        else:
            signal = values
        return np.clip(signal * self.scale, 0.0, 1.0)

    def decode(self, boundary_state):
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        z_memory = np.asarray(boundary_state["z_memory"], dtype=float)
        hidden_primary = np.clip(hidden[:, 0] / 1.25, 0.0, 1.0)
        hidden_secondary = np.clip(hidden[:, 1] / 1.25, 0.0, 1.0) if hidden.shape[1] > 1 else hidden_primary
        membrane_score = np.clip(0.5 * (membrane + 1.0), 0.0, 1.0)
        z_score = np.clip(0.25 * (z_alignment + z_memory + 2.0), 0.0, 1.0)
        score = float(
            np.clip(
                0.32 * hidden_primary.mean()
                + 0.20 * hidden_secondary.mean()
                + 0.18 * membrane_score.mean()
                + 0.18 * field.mean()
                + 0.12 * z_score.mean(),
                0.0,
                1.0,
            )
        )
        return float(np.clip(self.readout_attenuation * score + (1.0 - self.readout_attenuation) * 0.5, 0.0, 1.0))

    def loss_fn(self, external_output, target) -> float:
        return float(np.mean(np.abs(np.asarray(external_output, dtype=float) - np.asarray(target, dtype=float))))

    def remap(self, mapping_spec: dict) -> None:
        if "scale" in mapping_spec:
            self.scale = float(mapping_spec["scale"])
        if "flip" in mapping_spec:
            self.flip = bool(mapping_spec["flip"])
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
