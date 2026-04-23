from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from morphobase.assays.common import AssayResult, AssayRunner, build_synthetic_body
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.scheduler import Scheduler
from morphobase.ports.mnist_port import MNISTPort
from morphobase.training.trainer import Trainer


class MNISTSanityAssay(AssayRunner):
    CLASS_IDS = (0, 1, 2, 3, 4)
    SUPPORT_PER_CLASS = 6
    EVAL_PER_CLASS = 6
    SETTLE_STEPS = 6
    CONDITION_SPECS = {
        "baseline": {
            "allow_growth": True,
            "stress_sharing": True,
            "z_field": True,
        },
        "no_growth": {
            "allow_growth": False,
            "stress_sharing": True,
            "z_field": True,
        },
        "no_stress": {
            "allow_growth": True,
            "stress_sharing": False,
            "z_field": True,
        },
        "no_z_field": {
            "allow_growth": True,
            "stress_sharing": True,
            "z_field": False,
        },
    }

    @staticmethod
    def _balanced_select(images: np.ndarray, labels: np.ndarray, *, classes: tuple[int, ...], per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        chosen_images = []
        chosen_labels = []
        for label in classes:
            indices = np.flatnonzero(labels == label)
            if indices.size < per_class:
                raise ValueError(f"Not enough samples for class {label}.")
            selected = rng.choice(indices, size=per_class, replace=False)
            chosen_images.append(images[selected])
            chosen_labels.append(labels[selected])
        return np.concatenate(chosen_images, axis=0), np.concatenate(chosen_labels, axis=0)

    def _load_dataset(self, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        try:
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor

            train_set = MNIST(root=str(root), train=True, download=True, transform=ToTensor())
            test_set = MNIST(root=str(root), train=False, download=True, transform=ToTensor())
            train_images = train_set.data.numpy().astype(np.float32) / 255.0
            train_labels = train_set.targets.numpy()
            test_images = test_set.data.numpy().astype(np.float32) / 255.0
            test_labels = test_set.targets.numpy()
            return train_images, train_labels, test_images, test_labels, "mnist"
        except Exception:
            from sklearn.datasets import load_digits

            digits = load_digits(n_class=max(self.CLASS_IDS) + 1)
            images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
            resized = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False).squeeze(1).numpy()
            labels = digits.target.astype(int)
            split_index = int(0.6 * resized.shape[0])
            return resized[:split_index], labels[:split_index], resized[split_index:], labels[split_index:], "sklearn_digits"

    def _condition_spec(self, condition_name: str) -> dict[str, bool]:
        if condition_name not in self.CONDITION_SPECS:
            raise ValueError(f"Unknown MNIST rollout condition '{condition_name}'.")
        return dict(self.CONDITION_SPECS[condition_name])

    def _apply_row_load(self, body, port: MNISTPort, row: np.ndarray, row_index: int, condition_name: str) -> None:
        row_signal = np.asarray(row, dtype=float).reshape(-1)
        row_gradient = float(np.mean(np.abs(np.diff(row_signal)))) if row_signal.size > 1 else 0.0
        row_intensity = float(np.mean(row_signal))
        metabolic_load = 0.08 * row_gradient + 0.05 * row_intensity
        input_slice = port.boundary_slice("input")
        body.state.stress[input_slice] = np.clip(
            body.state.stress[input_slice] + metabolic_load,
            0.0,
            5.0,
        )
        body.state.energy[input_slice] = np.clip(
            body.state.energy[input_slice] - 0.10 * metabolic_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[input_slice] = np.clip(
            body.state.field_alignment[input_slice] - 0.12 * metabolic_load,
            0.0,
            1.0,
        )

    def _apply_settle_load(self, body, port: MNISTPort, image: np.ndarray, settle_index: int, condition_name: str) -> None:
        return None

    @staticmethod
    def _apply_condition_post_step(body, port: MNISTPort, condition_name: str) -> None:
        if condition_name == "baseline":
            return

        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        distal_mask = port.distal_mask()

        if condition_name == "no_stress":
            body.state.stress[distal_mask] *= 0.08
            body.state.stress[support_mask] = np.clip(
                body.state.stress[support_mask] + 0.035,
                0.0,
                5.0,
            )
            body.state.hidden[distal_mask] *= 0.982
            body.state.membrane[distal_mask] *= 0.95
            body.state.field_alignment[distal_mask] *= 0.82
            body.state.plasticity[distal_mask] *= 0.985
            support_indices = np.flatnonzero(support_mask)
            distal_indices = np.flatnonzero(distal_mask)
            if support_indices.size and distal_indices.size:
                body.state.conductance[np.ix_(distal_indices, support_indices)] *= 0.90
                body.state.conductance[np.ix_(support_indices, distal_indices)] *= 0.90
                diagonal = np.diag_indices_from(body.state.conductance)
                body.state.conductance = np.clip(body.state.conductance, 0.0, 2.0)
                body.state.conductance[diagonal] = 1.0

        if condition_name == "no_z_field":
            body.state.z_alignment.fill(0.0)
            body.state.z_memory.fill(0.0)

    def _rollout_image(self, cfg, image: np.ndarray, *, condition_name: str = "baseline") -> dict:
        condition = self._condition_spec(condition_name)
        body = build_synthetic_body(cfg)
        port = MNISTPort(cfg.body.num_cells)
        port.reset_episode()
        scheduler = Scheduler()
        history = []
        state_history = [body.state.copy()]
        z_history = [body.state.z_alignment.copy()]

        for row in image:
            due = scheduler.due(body.state.step_count)
            target_value = float(np.mean(row))
            port.apply_input(body, row, gain=0.48)
            task_context_hook = getattr(self, "_apply_task_context", None)
            if callable(task_context_hook):
                task_context_hook(body, port, body.state.step_count)
            self._apply_row_load(body, port, row, body.state.step_count, condition_name)
            if not condition["z_field"]:
                body.state.z_alignment.fill(0.0)
                body.state.z_memory.fill(0.0)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                target_value,
                allow_growth=condition["allow_growth"],
            )
            self._apply_condition_post_step(body, port, condition_name)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if body.state.step_count % cfg.runtime.log_every == 0:
                history.append(summarize_state(body.state, z_history=z_history))

        for _ in range(self.SETTLE_STEPS):
            due = scheduler.due(body.state.step_count)
            self._apply_settle_load(body, port, image, body.state.step_count, condition_name)
            settle_context_hook = getattr(self, "_apply_settle_context", None)
            if callable(settle_context_hook):
                settle_context_hook(body, port, body.state.step_count)
            if not condition["z_field"]:
                body.state.z_alignment.fill(0.0)
                body.state.z_memory.fill(0.0)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                float(np.mean(image)),
                allow_growth=condition["allow_growth"],
            )
            self._apply_condition_post_step(body, port, condition_name)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if body.state.step_count % cfg.runtime.log_every == 0:
                history.append(summarize_state(body.state, z_history=z_history))

        final_metrics = summarize_state(body.state, z_history=z_history)
        final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
        final_metrics["mnist_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        final_metrics["mnist_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        final_metrics["mnist_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        final_metrics["mnist_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0
        final_metrics["mnist_peak_growth_pressure"] = float(
            max(item.get("mean_growth_pressure", 0.0) for item in history) if history else final_metrics["mean_growth_pressure"]
        )
        boundary_embedding = np.asarray(port.decode(port.capture_boundary_state(body, kind="output")), dtype=float)
        physiology_signature = np.array(
            [
                final_metrics["mean_energy"],
                final_metrics["energy_variance"],
                final_metrics["mean_stress"],
                final_metrics["stress_variance"],
                final_metrics["mean_plasticity"],
                final_metrics["mean_commitment"],
                final_metrics["mean_field_alignment"],
                final_metrics["mean_z_alignment"],
                final_metrics["mean_z_memory"],
                final_metrics["z_memory_alignment_gap"],
                final_metrics["mean_growth_pressure"],
                final_metrics["lightcone_proxy"],
            ],
            dtype=float,
        )
        embedding = np.concatenate([boundary_embedding, physiology_signature], axis=0)
        return {
            "embedding": embedding,
            "history": history,
            "final_metrics": final_metrics,
        }

    def run(self, cfg):
        dataset_root = Path(cfg.run.output_dir) / "datasets"
        train_images, train_labels, test_images, test_labels, source = self._load_dataset(dataset_root)
        support_images, support_labels = self._balanced_select(
            train_images,
            train_labels,
            classes=self.CLASS_IDS,
            per_class=self.SUPPORT_PER_CLASS,
            seed=cfg.run.seed,
        )
        eval_images, eval_labels = self._balanced_select(
            test_images,
            test_labels,
            classes=self.CLASS_IDS,
            per_class=self.EVAL_PER_CLASS,
            seed=cfg.run.seed + 17,
        )

        support_embeddings = []
        representative_history = None
        representative_metrics = None
        for image in support_images:
            rollout = self._rollout_image(cfg, image)
            support_embeddings.append(rollout["embedding"])
            if representative_history is None:
                representative_history = rollout["history"]
                representative_metrics = rollout["final_metrics"].copy()

        eval_embeddings = []
        for image in eval_images:
            rollout = self._rollout_image(cfg, image)
            eval_embeddings.append(rollout["embedding"])

        support_embeddings = np.stack(support_embeddings, axis=0)
        eval_embeddings = np.stack(eval_embeddings, axis=0)
        trainer = Trainer()
        model = trainer.train_step(support_embeddings, support_labels)
        eval_accuracy = model.score(eval_embeddings, eval_labels)
        chance_accuracy = 1.0 / len(self.CLASS_IDS)
        mean_margin = model.mean_margin(eval_embeddings)

        final_metrics = representative_metrics or {}
        final_metrics["mnist_eval_accuracy"] = float(eval_accuracy)
        final_metrics["mnist_chance_accuracy"] = float(chance_accuracy)
        final_metrics["mnist_accuracy_advantage"] = float(eval_accuracy - chance_accuracy)
        final_metrics["mnist_mean_margin"] = float(mean_margin)
        final_metrics["mnist_support_count"] = float(len(support_labels))
        final_metrics["mnist_eval_count"] = float(len(eval_labels))
        final_metrics["mnist_class_count"] = float(len(self.CLASS_IDS))
        final_metrics["mnist_dataset_source_mnist"] = 1.0 if source == "mnist" else 0.0
        final_metrics["mnist_embedding_dim"] = float(support_embeddings.shape[1])

        notes = (
            "MNIST sanity assay ran a tiny balanced subset through the explicit MNIST port with prototype readout. "
            f"source={source}; accuracy={eval_accuracy:.4f}; chance={chance_accuracy:.4f}; margin={mean_margin:.4f}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )
