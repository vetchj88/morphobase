from __future__ import annotations

import numpy as np

from morphobase.ports.base import BasePort, BoundaryWindow


class MNISTPort(BasePort):
    def __init__(self, num_cells: int = 49, *, width: int | None = None, support_margin: int = 3) -> None:
        super().__init__()
        width = int(width or max(6, num_cells // 7))
        self._base_input_window = BoundaryWindow("left_mnist_input", 0, width, "left")
        self._base_output_window = BoundaryWindow("right_mnist_output", num_cells - width, num_cells, "right")
        self.scale = 1.0
        self.input_shift = 0
        self.output_shift = 0
        self.reset_episode()
        self.configure_boundary(
            num_cells=num_cells,
            input_window=self._base_input_window,
            output_window=self._base_output_window,
            support_margin=support_margin,
            name="mnist_port",
        )

    def reset_episode(self) -> None:
        self.row_count = 0
        self.energy_accumulator = 0.0
        self.gradient_accumulator = 0.0
        self.profile_accumulator: np.ndarray | None = None
        self.column_energy_accumulator: np.ndarray | None = None

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

    def _interpolate_vector(self, values: np.ndarray, target_width: int) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == target_width:
            return values
        source = np.linspace(0.0, 1.0, values.size)
        target = np.linspace(0.0, 1.0, target_width)
        return np.interp(target, source, values)

    def encode(self, external_input):
        values = np.asarray(external_input, dtype=float)
        if values.ndim == 2:
            values = values.mean(axis=0)
        row = self._interpolate_vector(values.reshape(-1), self.boundary_window("input").width)
        row = np.clip(row * self.scale, 0.0, 1.0)

        if self.profile_accumulator is None:
            self.profile_accumulator = np.zeros_like(row)
            self.column_energy_accumulator = np.zeros_like(row)
        self.row_count += 1
        self.energy_accumulator += float(np.mean(row))
        self.gradient_accumulator += float(np.mean(np.abs(np.diff(row)))) if row.size > 1 else 0.0
        self.profile_accumulator += row
        self.column_energy_accumulator += row ** 2
        return row

    def decode(self, boundary_state):
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        z_memory = np.asarray(boundary_state["z_memory"], dtype=float)

        output_width = hidden.shape[0]
        hidden_primary = np.clip(hidden[:, 0] / 1.5, 0.0, 1.0)
        hidden_secondary = np.clip(hidden[:, 1] / 1.5, 0.0, 1.0) if hidden.shape[1] > 1 else hidden_primary
        membrane_score = np.clip(0.5 * (membrane + 1.0), 0.0, 1.0)
        z_score = np.clip(0.25 * (z_alignment + z_memory + 2.0), 0.0, 1.0)

        if self.row_count == 0 or self.profile_accumulator is None or self.column_energy_accumulator is None:
            image_profile = np.zeros(output_width, dtype=float)
            image_energy = np.zeros(output_width, dtype=float)
            mean_ink = 0.0
            mean_gradient = 0.0
        else:
            image_profile = self._interpolate_vector(self.profile_accumulator / self.row_count, output_width)
            image_energy = self._interpolate_vector(
                np.sqrt(self.column_energy_accumulator / self.row_count),
                output_width,
            )
            mean_ink = self.energy_accumulator / self.row_count
            mean_gradient = self.gradient_accumulator / max(self.row_count, 1)

        embedding = np.concatenate(
            [
                image_profile,
                image_energy,
                hidden_primary,
                hidden_secondary,
                field,
                membrane_score,
                z_score,
                np.array(
                    [
                        mean_ink,
                        mean_gradient,
                        float(field.mean()),
                        float(membrane_score.mean()),
                        float(z_score.mean()),
                        float(np.linalg.norm(hidden.mean(axis=0))),
                    ],
                    dtype=float,
                ),
            ]
        )
        return embedding.astype(float, copy=False)

    def loss_fn(self, external_output, target) -> float:
        pred = np.asarray(external_output, dtype=float)
        target = np.asarray(target, dtype=float)
        return float(np.mean((pred - target) ** 2))

    def remap(self, mapping_spec: dict) -> None:
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
