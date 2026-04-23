from __future__ import annotations

import numpy as np

from morphobase.ports.base import BasePort, BoundaryWindow


class ToyRulePort(BasePort):
    def __init__(
        self,
        num_cells: int = 49,
        *,
        width: int | None = None,
        support_margin: int = 3,
        flip: bool = False,
    ) -> None:
        super().__init__()
        width = int(width or max(4, num_cells // 10))
        self._base_input_window = BoundaryWindow("left_input", 0, width, "left")
        self._base_output_window = BoundaryWindow("right_output", num_cells - width, num_cells, "right")
        self.flip = bool(flip)
        self.scale = 1.0
        self.input_shift = 0
        self.output_shift = 0
        self.configure_boundary(
            num_cells=num_cells,
            input_window=self._base_input_window,
            output_window=self._base_output_window,
            support_margin=support_margin,
            name="toy_rule_port",
        )

    def _shifted_output_window(self, shift: int) -> BoundaryWindow:
        base = self._base_output_window
        shift = int(np.clip(shift, 0, max(self.num_cells - base.width, 0)))
        stop = max(base.width, self.num_cells - shift)
        start = max(0, stop - base.width)
        return BoundaryWindow("right_output", start, stop, "right")

    def _shifted_input_window(self, shift: int) -> BoundaryWindow:
        base = self._base_input_window
        shift = int(np.clip(shift, 0, max(self.num_cells - base.width, 0)))
        start = min(shift, self.num_cells - base.width)
        stop = start + base.width
        return BoundaryWindow("left_input", start, stop, "left")

    def encode(self, external_input):
        value = float(np.clip(np.asarray(external_input, dtype=float).mean(), 0.0, 1.0))
        window = self.boundary_window("input")
        ramp = np.linspace(-0.06, 0.06, window.width)
        signal = np.clip(value + ramp, 0.0, 1.0)
        return np.clip(signal * self.scale, 0.0, 1.0)

    def decode(self, boundary_state):
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        z_memory = np.asarray(boundary_state["z_memory"], dtype=float)
        hidden_score = np.clip(hidden[:, 0] / 1.25, 0.0, 1.0)
        membrane_score = np.clip(0.5 * (membrane + 1.0), 0.0, 1.0)
        z_score = np.clip(0.5 * (z_alignment + z_memory + 2.0) / 2.0, 0.0, 1.0)
        score = float(
            np.clip(
                0.38 * hidden_score.mean()
                + 0.22 * membrane_score.mean()
                + 0.24 * field.mean()
                + 0.16 * z_score.mean(),
                0.0,
                1.0,
            )
        )
        score = self.readout_attenuation * score + (1.0 - self.readout_attenuation) * 0.5
        if self.flip:
            score = 1.0 - score
        return float(np.clip(score, 0.0, 1.0))

    def loss_fn(self, external_output, target) -> float:
        return float(np.mean(np.abs(np.asarray(external_output, dtype=float) - np.asarray(target, dtype=float))))

    def remap(self, mapping_spec: dict) -> None:
        if "flip" in mapping_spec:
            self.flip = bool(mapping_spec["flip"])
        if "scale" in mapping_spec:
            self.scale = float(mapping_spec["scale"])
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
