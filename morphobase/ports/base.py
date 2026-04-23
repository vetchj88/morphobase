from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class BoundaryWindow:
    label: str
    start: int
    stop: int
    side: str

    @property
    def width(self) -> int:
        return max(self.stop - self.start, 0)

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    def mask(self, num_cells: int, *, margin: int = 0) -> np.ndarray:
        left = max(0, self.start - margin)
        right = min(num_cells, self.stop + margin)
        mask = np.zeros(num_cells, dtype=bool)
        mask[left:right] = True
        return mask


class BasePort(ABC):
    def __init__(self) -> None:
        self.name = type(self).__name__.lower()
        self.num_cells: int | None = None
        self.support_margin: int = 2
        self.input_window: BoundaryWindow | None = None
        self.output_window: BoundaryWindow | None = None
        self.input_attenuation: float = 1.0
        self.readout_attenuation: float = 1.0

    def configure_boundary(
        self,
        *,
        num_cells: int,
        input_window: BoundaryWindow,
        output_window: BoundaryWindow,
        support_margin: int = 2,
        name: str | None = None,
    ) -> None:
        self.num_cells = int(num_cells)
        self.input_window = input_window
        self.output_window = output_window
        self.support_margin = int(max(support_margin, 0))
        if name is not None:
            self.name = name

    def _window(self, kind: str) -> BoundaryWindow:
        window = self.input_window if kind == "input" else self.output_window
        if window is None or self.num_cells is None:
            raise ValueError(f"{self.__class__.__name__} boundary interface is not configured.")
        return window

    def boundary_window(self, kind: str = "output") -> BoundaryWindow:
        return self._window(kind)

    def boundary_slice(self, kind: str = "output") -> slice:
        return self._window(kind).as_slice()

    def boundary_mask(self, kind: str = "output", *, margin: int = 0) -> np.ndarray:
        if self.num_cells is None:
            raise ValueError(f"{self.__class__.__name__} boundary interface is not configured.")
        return self._window(kind).mask(self.num_cells, margin=margin)

    def support_mask(self, kind: str = "output") -> np.ndarray:
        return self.boundary_mask(kind, margin=self.support_margin)

    def distal_mask(self) -> np.ndarray:
        if self.num_cells is None:
            raise ValueError(f"{self.__class__.__name__} boundary interface is not configured.")
        excluded = self.support_mask("input") | self.support_mask("output")
        if np.all(excluded):
            return ~self.boundary_mask("output")
        return ~excluded

    def capture_boundary_state(self, body, *, kind: str = "output") -> dict[str, np.ndarray | BoundaryWindow]:
        boundary_slice = self.boundary_slice(kind)
        return {
            "window": self.boundary_window(kind),
            "hidden": body.state.hidden[boundary_slice].copy(),
            "membrane": body.state.membrane[boundary_slice].copy(),
            "energy": body.state.energy[boundary_slice].copy(),
            "stress": body.state.stress[boundary_slice].copy(),
            "field_alignment": body.state.field_alignment[boundary_slice].copy(),
            "z_alignment": body.state.z_alignment[boundary_slice].copy(),
            "z_memory": body.state.z_memory[boundary_slice].copy(),
        }

    def _signal_vector(self, external_input) -> np.ndarray:
        width = self.boundary_window("input").width
        signal = np.asarray(self.encode(external_input), dtype=float).reshape(-1)
        if signal.size == 0:
            return np.zeros(width, dtype=float)
        if signal.size == 1:
            return np.full(width, float(signal[0]), dtype=float)
        if signal.size != width:
            source = np.linspace(0.0, 1.0, signal.size)
            target = np.linspace(0.0, 1.0, width)
            signal = np.interp(target, source, signal)
        return signal.astype(float, copy=False)

    def apply_input(self, body, external_input, *, gain: float = 0.42) -> None:
        signal = np.clip(self._signal_vector(external_input) * self.input_attenuation, 0.0, 1.0)
        input_slice = self.boundary_slice("input")
        input_hidden = body.state.hidden[input_slice]
        hidden_dims = input_hidden.shape[1]
        if hidden_dims:
            input_hidden[:, 0] = np.clip(
                (1.0 - gain) * input_hidden[:, 0] + gain * signal,
                -2.0,
                2.0,
            )
        if hidden_dims > 1:
            input_hidden[:, 1] = np.clip(
                (1.0 - gain) * input_hidden[:, 1] + gain * (1.0 - signal),
                -2.0,
                2.0,
            )
        if hidden_dims > 2:
            ramp = np.linspace(-0.18, 0.18, signal.shape[0])
            input_hidden[:, 2] = np.clip(
                (1.0 - 0.55 * gain) * input_hidden[:, 2] + 0.55 * gain * (signal + ramp),
                -2.0,
                2.0,
            )
        body.state.membrane[input_slice] = np.clip(
            (1.0 - gain) * body.state.membrane[input_slice] + gain * (2.0 * signal - 1.0),
            -1.0,
            1.0,
        )
        body.state.field_alignment[input_slice] = np.clip(
            body.state.field_alignment[input_slice] + 0.14 * gain + 0.18 * signal,
            0.0,
            1.0,
        )
        body.state.z_alignment[input_slice] = np.clip(
            0.82 * body.state.z_alignment[input_slice] + 0.18 * (2.0 * signal - 1.0),
            -1.0,
            1.0,
        )
        body.state.z_memory[input_slice] = np.clip(
            0.90 * body.state.z_memory[input_slice] + 0.10 * (2.0 * signal - 1.0),
            -1.0,
            1.0,
        )
        body.state.energy[input_slice] = np.clip(body.state.energy[input_slice] + 0.02, 0.0, 1.0)
        body.state.stress[input_slice] = np.clip(body.state.stress[input_slice] * 0.96, 0.0, 5.0)

    def read_output(self, body) -> float:
        decoded = self.decode(self.capture_boundary_state(body, kind="output"))
        return float(np.asarray(decoded, dtype=float).mean())

    @staticmethod
    def union_mask(*masks: np.ndarray) -> np.ndarray:
        if not masks:
            raise ValueError("At least one mask is required.")
        merged = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            merged |= mask.astype(bool)
        return merged

    @abstractmethod
    def encode(self, external_input): ...

    @abstractmethod
    def decode(self, boundary_state): ...

    @abstractmethod
    def loss_fn(self, external_output, target) -> float: ...

    @abstractmethod
    def remap(self, mapping_spec: dict) -> None: ...

    @abstractmethod
    def damage(self, mask_spec: dict) -> None: ...
