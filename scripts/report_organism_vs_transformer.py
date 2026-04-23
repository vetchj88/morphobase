from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphobase.assays.lesion_gridworld_remap import LesionGridworldRemapAssay
from morphobase.assays.lesion_sequential_rules import LesionSequentialRulesAssay
from morphobase.assays.lesion_split_mnist import LesionSplitMNISTAssay
from morphobase.assays.split_mnist import SplitMNISTAssay
from morphobase.config.validate import load_config
from morphobase.seeds import set_seed
from morphobase.training.transformer_baselines import (
    TinyMLPClassifier,
    TinyTransformerClassifier,
    fit_mlp_classifier,
    fit_static_prototype,
    fit_transformer_classifier,
    score_mlp_classifier,
    score_static_prototype,
)
from morphobase.training.trainer import SequentialLinearTrainer, Trainer


@dataclass(slots=True)
class RunStats:
    primary_metric_name: str
    primary_metric: float
    support_count: float
    eval_count: float
    wall_time_sec: float
    samples_per_sec: float
    score_per_sec: float
    mean_margin: float
    mean_energy: float | None = None
    mean_stress: float | None = None
    energy_spent_estimate: float | None = None
    score_per_energy_spent: float | None = None
    parameter_count: int | None = None
    train_wall_time_sec: float | None = None
    eval_wall_time_sec: float | None = None
    train_samples_per_sec: float | None = None
    eval_samples_per_sec: float | None = None


def _safe_rate(sample_count: float, wall_time_sec: float) -> float:
    return float(sample_count / max(wall_time_sec, 1e-8))


def _energy_efficiency(primary_metric: float, mean_energy: float | None) -> tuple[float | None, float | None]:
    if mean_energy is None:
        return None, None
    energy_spent = float(max(1.0 - np.clip(mean_energy, 0.0, 1.0), 1e-6))
    return energy_spent, float(primary_metric / energy_spent)


def _predict_logits_transformer(model: TinyTransformerClassifier, tokens: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(tokens, dtype=np.float32))).cpu().numpy()
    return logits, float(perf_counter() - start)


def _predict_logits_mlp(model: TinyMLPClassifier, features: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(features, dtype=np.float32))).cpu().numpy()
    return logits, float(perf_counter() - start)


def _classification_rollup(task_peak: dict[int, float], task_initial: dict[int, float], task_final: dict[int, float]) -> tuple[float, float, float]:
    forgetting_values = [
        task_peak[index] - task_final.get(index, 0.0)
        for index in range(max(len(task_final) - 1, 0))
    ]
    bwt_values = [
        task_final.get(index, 0.0) - task_initial.get(index, 0.0)
        for index in range(max(len(task_final) - 1, 0))
    ]
    final_mean = float(np.mean(list(task_final.values()))) if task_final else 0.0
    forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
    bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
    return final_mean, forgetting, bwt


def _run_organism(
    *,
    cfg_path: str,
    assay_cls,
    condition_name: str,
    primary_key: str,
    support_key: str,
    eval_key: str,
) -> RunStats:
    cfg = load_config(cfg_path)
    set_seed(cfg.run.seed)
    assay = assay_cls()
    start = perf_counter()
    result = assay.run_condition(cfg, condition_name=condition_name) if hasattr(assay, "run_condition") else assay.run(cfg)
    elapsed = float(perf_counter() - start)
    metrics = result.final_metrics
    support_count = float(metrics[support_key])
    eval_count = float(metrics[eval_key])
    margin_key = primary_key.replace("final_accuracy_mean", "mean_margin").replace("final_success_mean", "mean_margin")
    mean_energy = float(metrics.get("mean_energy")) if "mean_energy" in metrics else None
    energy_spent, score_per_energy = _energy_efficiency(float(metrics[primary_key]), mean_energy)
    return RunStats(
        primary_metric_name=primary_key,
        primary_metric=float(metrics[primary_key]),
        support_count=support_count,
        eval_count=eval_count,
        wall_time_sec=elapsed,
        samples_per_sec=_safe_rate(support_count + eval_count, elapsed),
        score_per_sec=float(metrics[primary_key] / max(elapsed, 1e-8)),
        mean_margin=float(metrics.get(margin_key, 0.0)),
        mean_energy=mean_energy,
        mean_stress=float(metrics.get("mean_stress")) if "mean_stress" in metrics else None,
        energy_spent_estimate=energy_spent,
        score_per_energy_spent=score_per_energy,
    )


def _fit_static(_model_unused, features: np.ndarray, labels: np.ndarray, _seed: int) -> dict[str, float]:
    model, summary = fit_static_prototype(features, labels)
    return {
        "model": model,
        "parameter_count": summary.parameter_count,
        "train_wall_time_sec": summary.train_wall_time_sec,
        "train_presentations": float(len(features)),
    }


def _score_static(model, features: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict[str, float]]:
    accuracy, margin, summary = score_static_prototype(model, features, labels)
    return accuracy, margin, {"eval_wall_time_sec": summary.eval_wall_time_sec}


def _fit_mlp(model: TinyMLPClassifier, features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    summary = fit_mlp_classifier(
        model,
        features,
        labels,
        epochs=70,
        learning_rate=3e-3,
        weight_decay=2e-4,
        batch_size=32,
        seed=seed,
    )
    return {
        "model": model,
        "parameter_count": summary.parameter_count,
        "train_wall_time_sec": summary.train_wall_time_sec,
        "train_presentations": float(len(features) * 70),
    }


def _score_mlp(model: TinyMLPClassifier, features: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict[str, float]]:
    accuracy, margin, summary = score_mlp_classifier(model, features, labels)
    return accuracy, margin, {"eval_wall_time_sec": summary.eval_wall_time_sec}


def _fit_transformer(model: TinyTransformerClassifier, tokens: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    summary = fit_transformer_classifier(
        model,
        tokens,
        labels,
        epochs=60,
        learning_rate=3e-3,
        weight_decay=5e-4,
        batch_size=32,
        seed=seed,
    )
    return {
        "model": model,
        "parameter_count": summary.parameter_count,
        "train_wall_time_sec": summary.train_wall_time_sec,
        "train_presentations": float(len(tokens) * 60),
    }


def _score_transformer(model: TinyTransformerClassifier, tokens: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict[str, float]]:
    logits, elapsed = _predict_logits_transformer(model, tokens)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == labels))
    if logits.shape[1] < 2:
        margin = 0.0
    else:
        sorted_logits = np.sort(logits, axis=1)
        margin = float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2]))
    return accuracy, margin, {"eval_wall_time_sec": elapsed}


def _run_incremental_classifier(
    *,
    task_support_inputs: list[np.ndarray],
    task_support_labels: list[np.ndarray],
    task_eval_inputs: list[np.ndarray],
    task_eval_labels: list[np.ndarray],
    primary_metric_name: str,
    build_model,
    fit_model,
    score_model,
) -> RunStats:
    total_train_time = 0.0
    total_eval_time = 0.0
    total_train_presentations = 0.0
    total_eval_samples = 0.0
    task_peak: dict[int, float] = {}
    task_initial: dict[int, float] = {}
    task_final: dict[int, float] = {}
    final_margin = 0.0
    model_parameter_count = 0
    seen_inputs: list[np.ndarray] = []
    seen_labels: list[np.ndarray] = []

    for task_index, support_inputs in enumerate(task_support_inputs):
        seen_inputs.append(np.asarray(support_inputs, dtype=np.float32))
        seen_labels.append(np.asarray(task_support_labels[task_index], dtype=np.int64))
        train_inputs = np.concatenate(seen_inputs, axis=0)
        train_labels = np.concatenate(seen_labels, axis=0)
        model = build_model(task_index)
        fit_summary = fit_model(model, train_inputs, train_labels, 1000 + 37 * task_index)
        total_train_time += fit_summary["train_wall_time_sec"]
        total_train_presentations += fit_summary["train_presentations"]
        model_parameter_count = int(fit_summary["parameter_count"])
        model = fit_summary["model"]

        for seen_task_index, eval_inputs in enumerate(task_eval_inputs[: task_index + 1]):
            accuracy, margin, eval_summary = score_model(
                model,
                np.asarray(eval_inputs, dtype=np.float32),
                np.asarray(task_eval_labels[seen_task_index], dtype=np.int64),
            )
            total_eval_time += eval_summary["eval_wall_time_sec"]
            total_eval_samples += float(len(task_eval_labels[seen_task_index]))
            task_peak[seen_task_index] = max(task_peak.get(seen_task_index, 0.0), accuracy)
            if seen_task_index == task_index:
                task_initial[seen_task_index] = accuracy
            task_final[seen_task_index] = accuracy
            if seen_task_index == task_index:
                final_margin = margin

    final_mean, _, _ = _classification_rollup(task_peak, task_initial, task_final)
    total_wall = total_train_time + total_eval_time
    return RunStats(
        primary_metric_name=primary_metric_name,
        primary_metric=final_mean,
        support_count=float(sum(len(labels) for labels in task_support_labels)),
        eval_count=float(sum(len(labels) for labels in task_eval_labels)),
        wall_time_sec=total_wall,
        samples_per_sec=_safe_rate(total_train_presentations + total_eval_samples, total_wall),
        score_per_sec=float(final_mean / max(total_wall, 1e-8)),
        mean_margin=float(final_margin),
        parameter_count=model_parameter_count,
        train_wall_time_sec=total_train_time,
        eval_wall_time_sec=total_eval_time,
        train_samples_per_sec=_safe_rate(total_train_presentations, total_train_time),
        eval_samples_per_sec=_safe_rate(total_eval_samples, total_eval_time),
    )


def _gridworld_tokens(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    tokens = np.zeros((5, 9), dtype=np.float32)
    tokens[0] = obs[0:9]
    tokens[1] = obs[9:18]
    tokens[2] = obs[18:27]
    tokens[3, :7] = obs[27:34]
    tokens[4, :5] = obs[34:39]
    return tokens


def _predict_static(model, features: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter()
    predictions = model.predict(np.asarray(features, dtype=np.float32))
    return np.asarray(predictions, dtype=np.int64), float(perf_counter() - start)


def _predict_labels_mlp(model: TinyMLPClassifier, features: np.ndarray) -> tuple[np.ndarray, float]:
    logits, elapsed = _predict_logits_mlp(model, features)
    return np.argmax(logits, axis=1).astype(np.int64), elapsed


def _predict_labels_transformer(model: TinyTransformerClassifier, tokens: np.ndarray) -> tuple[np.ndarray, float]:
    logits, elapsed = _predict_logits_transformer(model, tokens)
    return np.argmax(logits, axis=1).astype(np.int64), elapsed


def _sequential_rules_task_data() -> dict:
    cfg = load_config("configs/assay/lesion_sequential_rules.yaml")
    set_seed(cfg.run.seed)
    assay = LesionSequentialRulesAssay()
    support_sequences, support_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_sequences, eval_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )

    support_features: list[np.ndarray] = []
    support_tokens: list[np.ndarray] = []
    support_targets: list[np.ndarray] = []
    eval_features: list[np.ndarray] = []
    eval_tokens: list[np.ndarray] = []
    eval_targets: list[np.ndarray] = []

    for task_classes in assay.TASK_SPLITS:
        support_mask = np.isin(support_labels, task_classes)
        eval_mask = np.isin(eval_labels, task_classes)
        task_support = np.asarray(support_sequences[support_mask], dtype=np.float32)
        task_eval = np.asarray(eval_sequences[eval_mask], dtype=np.float32)
        support_features.append(task_support.reshape(task_support.shape[0], -1))
        support_tokens.append(task_support[:, :, None])
        support_targets.append(np.asarray(support_labels[support_mask], dtype=np.int64))
        eval_features.append(task_eval.reshape(task_eval.shape[0], -1))
        eval_tokens.append(task_eval[:, :, None])
        eval_targets.append(np.asarray(eval_labels[eval_mask], dtype=np.int64))

    return {
        "cfg_path": "configs/assay/lesion_sequential_rules.yaml",
        "assay_cls": LesionSequentialRulesAssay,
        "primary_key": "lesion_sequential_rules_final_accuracy_mean",
        "support_key": "lesion_sequential_rules_support_count",
        "eval_key": "lesion_sequential_rules_eval_count",
        "primary_label": "final_accuracy_mean",
        "task_support_features": support_features,
        "task_support_tokens": support_tokens,
        "task_support_labels": support_targets,
        "task_eval_features": eval_features,
        "task_eval_tokens": eval_tokens,
        "task_eval_labels": eval_targets,
    }


def _split_mnist_task_data() -> dict:
    cfg = load_config("configs/assay/split_mnist.yaml")
    set_seed(cfg.run.seed)
    assay = SplitMNISTAssay()
    dataset_root = Path(cfg.run.output_dir) / "datasets"
    train_images, train_labels, test_images, test_labels, _ = assay._load_dataset(dataset_root)
    support_images, support_labels = assay._balanced_select(
        train_images,
        train_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_images, eval_labels = assay._balanced_select(
        test_images,
        test_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )

    support_features: list[np.ndarray] = []
    support_tokens: list[np.ndarray] = []
    support_targets: list[np.ndarray] = []
    eval_features: list[np.ndarray] = []
    eval_tokens: list[np.ndarray] = []
    eval_targets: list[np.ndarray] = []

    for task_classes in assay.TASK_SPLITS:
        support_mask = np.isin(support_labels, task_classes)
        eval_mask = np.isin(eval_labels, task_classes)
        task_support = np.asarray(support_images[support_mask], dtype=np.float32)
        task_eval = np.asarray(eval_images[eval_mask], dtype=np.float32)
        support_features.append(task_support.reshape(task_support.shape[0], -1))
        support_tokens.append(task_support.reshape(task_support.shape[0], task_support.shape[1], task_support.shape[2]))
        support_targets.append(np.asarray(support_labels[support_mask], dtype=np.int64))
        eval_features.append(task_eval.reshape(task_eval.shape[0], -1))
        eval_tokens.append(task_eval.reshape(task_eval.shape[0], task_eval.shape[1], task_eval.shape[2]))
        eval_targets.append(np.asarray(eval_labels[eval_mask], dtype=np.int64))

    return {
        "cfg_path": "configs/assay/split_mnist.yaml",
        "assay_cls": SplitMNISTAssay,
        "primary_key": "split_mnist_final_accuracy_mean",
        "support_key": "split_mnist_support_count",
        "eval_key": "split_mnist_eval_count",
        "primary_label": "final_accuracy_mean",
        "task_support_features": support_features,
        "task_support_tokens": support_tokens,
        "task_support_labels": support_targets,
        "task_eval_features": eval_features,
        "task_eval_tokens": eval_tokens,
        "task_eval_labels": eval_targets,
    }


def _lesion_image_rows(image: np.ndarray, *, start: int, stop: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lesion_image = np.asarray(image, dtype=np.float32).copy()
    lesion_band = lesion_image[start:stop]
    if lesion_band.size:
        noise = rng.normal(0.0, 0.06, size=lesion_band.shape)
        lesion_image[start:stop] = np.clip(0.34 * lesion_band + noise, 0.0, 1.0)
    return lesion_image


def _lesion_split_mnist_task_data() -> dict:
    cfg = load_config("configs/assay/lesion_split_mnist.yaml")
    set_seed(cfg.run.seed)
    assay = LesionSplitMNISTAssay()
    dataset_root = Path(cfg.run.output_dir) / "datasets"
    train_images, train_labels, test_images, test_labels, _ = assay._load_dataset(dataset_root)
    support_images, support_labels = assay._balanced_select(
        train_images,
        train_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_images, eval_labels = assay._balanced_select(
        test_images,
        test_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )

    support_features: list[np.ndarray] = []
    support_tokens: list[np.ndarray] = []
    support_targets: list[np.ndarray] = []
    eval_features: list[np.ndarray] = []
    eval_tokens: list[np.ndarray] = []
    eval_targets: list[np.ndarray] = []

    for task_index, task_classes in enumerate(assay.TASK_SPLITS):
        support_mask = np.isin(support_labels, task_classes)
        eval_mask = np.isin(eval_labels, task_classes)
        task_support = np.asarray(support_images[support_mask], dtype=np.float32)
        task_eval = np.asarray(eval_images[eval_mask], dtype=np.float32)
        lesioned_support = np.stack(
            [
                _lesion_image_rows(
                    image,
                    start=assay.ROW_LESION_START,
                    stop=assay.ROW_LESION_STOP,
                    seed=cfg.run.seed + 701 * task_index + sample_index,
                )
                for sample_index, image in enumerate(task_support)
            ],
            axis=0,
        )
        lesioned_eval = np.stack(
            [
                _lesion_image_rows(
                    image,
                    start=assay.ROW_LESION_START,
                    stop=assay.ROW_LESION_STOP,
                    seed=cfg.run.seed + 1701 * task_index + sample_index,
                )
                for sample_index, image in enumerate(task_eval)
            ],
            axis=0,
        )
        support_features.append(lesioned_support.reshape(lesioned_support.shape[0], -1))
        support_tokens.append(lesioned_support.reshape(lesioned_support.shape[0], lesioned_support.shape[1], lesioned_support.shape[2]))
        support_targets.append(np.asarray(support_labels[support_mask], dtype=np.int64))
        eval_features.append(lesioned_eval.reshape(lesioned_eval.shape[0], -1))
        eval_tokens.append(lesioned_eval.reshape(lesioned_eval.shape[0], lesioned_eval.shape[1], lesioned_eval.shape[2]))
        eval_targets.append(np.asarray(eval_labels[eval_mask], dtype=np.int64))

    return {
        "cfg_path": "configs/assay/lesion_split_mnist.yaml",
        "assay_cls": LesionSplitMNISTAssay,
        "primary_key": "lesion_split_mnist_final_accuracy_mean",
        "support_key": "lesion_split_mnist_support_count",
        "eval_key": "lesion_split_mnist_eval_count",
        "primary_label": "final_accuracy_mean",
        "task_support_features": support_features,
        "task_support_tokens": support_tokens,
        "task_support_labels": support_targets,
        "task_eval_features": eval_features,
        "task_eval_tokens": eval_tokens,
        "task_eval_labels": eval_targets,
    }


def _gridworld_task_data() -> dict:
    cfg = load_config("configs/assay/lesion_gridworld_remap.yaml")
    set_seed(cfg.run.seed)
    assay = LesionGridworldRemapAssay()

    task_support_features: list[np.ndarray] = []
    task_support_tokens: list[np.ndarray] = []
    task_support_labels: list[np.ndarray] = []

    for task_index, task_spec in enumerate(assay.TASK_SPECS):
        support_obs, support_labels = assay._support_samples(task_spec, seed=cfg.run.seed + 43 * task_index)
        task_support_features.append(np.asarray(support_obs, dtype=np.float32))
        task_support_tokens.append(np.stack([_gridworld_tokens(obs) for obs in support_obs], axis=0))
        task_support_labels.append(np.asarray(support_labels, dtype=np.int64))

    return {
        "cfg_path": "configs/assay/lesion_gridworld_remap.yaml",
        "assay_cls": LesionGridworldRemapAssay,
        "primary_key": "lesion_gridworld_remap_final_success_mean",
        "support_key": "lesion_gridworld_remap_support_count",
        "eval_key": "lesion_gridworld_remap_eval_count",
        "primary_label": "final_success_mean",
        "cfg": cfg,
        "assay": assay,
        "task_support_features": task_support_features,
        "task_support_tokens": task_support_tokens,
        "task_support_labels": task_support_labels,
    }


def _run_gridworld_baseline(
    *,
    task_support_inputs: list[np.ndarray],
    task_support_labels: list[np.ndarray],
    task_specs: tuple[dict, ...],
    assay: LesionGridworldRemapAssay,
    cfg,
    primary_metric_name: str,
    build_model,
    fit_model,
    predict_fn,
    rollout_input_builder,
) -> RunStats:
    total_train_time = 0.0
    total_eval_time = 0.0
    total_train_presentations = 0.0
    total_eval_samples = 0.0
    task_peak: dict[int, float] = {}
    task_initial: dict[int, float] = {}
    task_final: dict[int, float] = {}
    task_efficiency: dict[int, float] = {}
    model_parameter_count = 0
    seen_inputs: list[np.ndarray] = []
    seen_labels: list[np.ndarray] = []
    final_margin = 0.0

    for task_index, support_inputs in enumerate(task_support_inputs):
        seen_inputs.append(np.asarray(support_inputs, dtype=np.float32))
        seen_labels.append(np.asarray(task_support_labels[task_index], dtype=np.int64))
        train_inputs = np.concatenate(seen_inputs, axis=0)
        train_labels = np.concatenate(seen_labels, axis=0)
        model = build_model(task_index)
        fit_summary = fit_model(model, train_inputs, train_labels, 2000 + 17 * task_index)
        total_train_time += float(fit_summary["train_wall_time_sec"])
        total_train_presentations += float(fit_summary["train_presentations"])
        model_parameter_count = int(fit_summary["parameter_count"])
        model = fit_summary["model"]

        for seen_task_index, task_spec in enumerate(task_specs[: task_index + 1]):
            successes: list[float] = []
            efficiencies: list[float] = []
            for episode_idx in range(assay.EVAL_EPISODES):
                position = assay._initial_position(task_spec, episode_seed=cfg.run.seed + 1701 + 89 * seen_task_index + episode_idx)
                prev_action = 2
                reward_zone = assay._positions(task_spec, "reward_zone")
                episode_eval_calls = 0
                for step in range(assay.HORIZON):
                    observation = assay._observation(task_spec, position, prev_action)
                    inputs = rollout_input_builder(observation)
                    predictions, elapsed = predict_fn(model, np.asarray(inputs, dtype=np.float32))
                    total_eval_time += float(elapsed)
                    total_eval_samples += 1.0
                    episode_eval_calls += 1
                    action = int(np.asarray(predictions).reshape(-1)[0])
                    position = assay._transition(task_spec, position, action)
                    prev_action = action
                    if position in reward_zone:
                        successes.append(1.0)
                        efficiencies.append(float(np.clip(1.0 - (step / max(assay.HORIZON - 1, 1)), 0.0, 1.0)))
                        break
                else:
                    successes.append(0.0)
                    efficiencies.append(0.0)

            success_rate = float(np.mean(successes)) if successes else 0.0
            efficiency_rate = float(np.mean(efficiencies)) if efficiencies else 0.0
            task_peak[seen_task_index] = max(task_peak.get(seen_task_index, 0.0), success_rate)
            if seen_task_index == task_index:
                task_initial[seen_task_index] = success_rate
            task_final[seen_task_index] = success_rate
            task_efficiency[seen_task_index] = efficiency_rate

        # Classification margin on the current seen support set as a readout confidence proxy.
        current_inputs = np.concatenate(task_support_inputs[: task_index + 1], axis=0)
        current_labels = np.concatenate(task_support_labels[: task_index + 1], axis=0)
        if isinstance(model, TinyTransformerClassifier):
            _, final_margin, eval_summary = _score_transformer(model, current_inputs, current_labels)
        elif isinstance(model, TinyMLPClassifier):
            _, final_margin, eval_summary = _score_mlp(model, current_inputs, current_labels)
        else:
            _, final_margin, eval_summary = _score_static(model, current_inputs, current_labels)
        total_eval_time += float(eval_summary["eval_wall_time_sec"])
        total_eval_samples += float(len(current_labels))

    final_mean, forgetting, bwt = _classification_rollup(task_peak, task_initial, task_final)
    total_wall = total_train_time + total_eval_time
    eval_count = float(len(task_specs) * assay.EVAL_EPISODES)
    return RunStats(
        primary_metric_name=primary_metric_name,
        primary_metric=final_mean,
        support_count=float(sum(len(labels) for labels in task_support_labels)),
        eval_count=eval_count,
        wall_time_sec=total_wall,
        samples_per_sec=_safe_rate(total_train_presentations + total_eval_samples, total_wall),
        score_per_sec=float(final_mean / max(total_wall, 1e-8)),
        mean_margin=float(final_margin),
        parameter_count=model_parameter_count,
        train_wall_time_sec=total_train_time,
        eval_wall_time_sec=total_eval_time,
        train_samples_per_sec=_safe_rate(total_train_presentations, total_train_time),
        eval_samples_per_sec=_safe_rate(total_eval_samples, total_eval_time),
    )


def _run_lesion_sequential_organism_fixed(condition_name: str) -> RunStats:
    cfg = load_config("configs/assay/lesion_sequential_rules.yaml")
    set_seed(cfg.run.seed)
    assay = LesionSequentialRulesAssay()
    support_sequences, support_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_sequences, eval_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )

    support_baseline_start = perf_counter()
    support_baseline = np.stack(
        [assay._rollout_sequence(cfg, sequence, condition_name="baseline")["embedding"] for sequence in support_sequences],
        axis=0,
    )
    support_baseline_time = float(perf_counter() - support_baseline_start)

    eval_start = perf_counter()
    eval_condition = []
    metric_energy = []
    metric_stress = []
    for sequence in eval_sequences:
        rollout = assay._rollout_sequence(cfg, sequence, condition_name=condition_name)
        eval_condition.append(rollout["embedding"])
        metric_energy.append(float(rollout["final_metrics"].get("mean_energy", 0.0)))
        metric_stress.append(float(rollout["final_metrics"].get("mean_stress", 0.0)))
    eval_condition = np.stack(eval_condition, axis=0)
    eval_rollout_time = float(perf_counter() - eval_start)

    trainer = SequentialLinearTrainer(np.array(assay.CLASS_IDS, dtype=int), support_baseline.shape[1], seed=cfg.run.seed + 503)
    seen_classes: list[int] = []
    seen_train_embeddings: list[np.ndarray] = []
    seen_train_labels: list[np.ndarray] = []
    total_train_time = 0.0
    total_eval_time = eval_rollout_time
    total_train_presentations = 0.0
    total_eval_samples = float(len(eval_labels))
    final_margin = 0.0

    for task_index, task_classes in enumerate(assay.TASK_SPLITS):
        support_mask = np.isin(support_labels, task_classes)
        seen_train_embeddings.append(support_baseline[support_mask])
        seen_train_labels.append(support_labels[support_mask])
        train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
        train_labels = np.concatenate(seen_train_labels, axis=0)
        train_start = perf_counter()
        final_model = trainer.train_task(train_embeddings, train_labels, epochs=88, learning_rate=0.16, l2=2e-4)
        train_elapsed = float(perf_counter() - train_start)
        total_train_time += train_elapsed
        total_train_presentations += float(len(train_labels) * 88)
        seen_classes.extend(int(label) for label in task_classes)

    allowed_classes = np.array(sorted(set(seen_classes)), dtype=int)
    task_scores = []
    for task_classes in assay.TASK_SPLITS:
        eval_mask = np.isin(eval_labels, task_classes)
        score_start = perf_counter()
        accuracy = final_model.score(eval_condition[eval_mask], eval_labels[eval_mask], allowed_classes=allowed_classes)
        total_eval_time += float(perf_counter() - score_start)
        total_eval_samples += float(np.sum(eval_mask))
        task_scores.append(float(accuracy))

    margin_start = perf_counter()
    final_margin = final_model.mean_margin(eval_condition, allowed_classes=allowed_classes)
    total_eval_time += float(perf_counter() - margin_start)
    final_mean = float(np.mean(task_scores)) if task_scores else 0.0
    total_wall = support_baseline_time + total_train_time + total_eval_time
    parameter_count = int(support_baseline.shape[1] * len(assay.CLASS_IDS) + len(assay.CLASS_IDS))
    mean_energy = float(np.mean(metric_energy)) if metric_energy else None
    energy_spent, score_per_energy = _energy_efficiency(final_mean, mean_energy)
    return RunStats(
        primary_metric_name="lesion_sequential_rules_final_accuracy_mean",
        primary_metric=final_mean,
        support_count=float(len(support_labels)),
        eval_count=float(len(eval_labels)),
        wall_time_sec=total_wall,
        samples_per_sec=_safe_rate(total_train_presentations + total_eval_samples, total_wall),
        score_per_sec=float(final_mean / max(total_wall, 1e-8)),
        mean_margin=float(final_margin),
        mean_energy=mean_energy,
        mean_stress=float(np.mean(metric_stress)) if metric_stress else None,
        energy_spent_estimate=energy_spent,
        score_per_energy_spent=score_per_energy,
        parameter_count=parameter_count,
        train_wall_time_sec=total_train_time + support_baseline_time,
        eval_wall_time_sec=total_eval_time,
        train_samples_per_sec=_safe_rate(total_train_presentations + len(support_labels), total_train_time + support_baseline_time),
        eval_samples_per_sec=_safe_rate(total_eval_samples, total_eval_time),
    )


def _run_lesion_gridworld_organism_fixed(condition_name: str) -> RunStats:
    cfg = load_config("configs/assay/lesion_gridworld_remap.yaml")
    set_seed(cfg.run.seed)
    assay = LesionGridworldRemapAssay()

    support_embeddings_baseline: list[np.ndarray] = []
    support_labels_by_task: list[np.ndarray] = []
    support_rollout_time = 0.0
    for task_index, task_spec in enumerate(assay.TASK_SPECS):
        support_obs, support_labels = assay._support_samples(task_spec, seed=cfg.run.seed + 43 * task_index)
        task_rows = []
        for obs, label in zip(support_obs, support_labels, strict=True):
            start = perf_counter()
            rollout = assay._rollout_observation(cfg, task_spec, obs, int(label), condition_name="baseline")
            support_rollout_time += float(perf_counter() - start)
            task_rows.append(rollout["embedding"])
        support_embeddings_baseline.append(np.stack(task_rows, axis=0))
        support_labels_by_task.append(np.asarray(support_labels, dtype=np.int64))

    trainer = SequentialLinearTrainer(
        assay.ACTION_LABELS.copy(),
        support_embeddings_baseline[0].shape[1],
        seed=cfg.run.seed + 719,
    )
    seen_train_embeddings: list[np.ndarray] = []
    seen_train_labels: list[np.ndarray] = []
    total_train_time = 0.0
    total_train_presentations = 0.0
    for task_index, task_embeddings in enumerate(support_embeddings_baseline):
        seen_train_embeddings.append(task_embeddings)
        seen_train_labels.append(support_labels_by_task[task_index])
        train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
        train_labels = np.concatenate(seen_train_labels, axis=0)
        train_start = perf_counter()
        final_model = trainer.train_task(train_embeddings, train_labels, epochs=80, learning_rate=0.12, l2=2e-4)
        total_train_time += float(perf_counter() - train_start)
        total_train_presentations += float(len(train_labels) * 80)

    task_peak: dict[int, float] = {}
    task_initial: dict[int, float] = {}
    task_final: dict[int, float] = {}
    total_eval_time = 0.0
    total_eval_samples = 0.0
    metric_energy = []
    metric_stress = []
    final_margin = 0.0

    for task_index, task_spec in enumerate(assay.TASK_SPECS):
        successes = []
        condition_embeddings = []
        condition_labels = []
        for episode_idx in range(assay.EVAL_EPISODES):
            position = assay._initial_position(task_spec, episode_seed=cfg.run.seed + 1701 + 89 * task_index + episode_idx)
            prev_action = 2
            reward_zone = assay._positions(task_spec, "reward_zone")
            for step in range(assay.HORIZON):
                observation = assay._observation(task_spec, position, prev_action)
                target_action = assay._expert_action(task_spec, position, prev_action)
                rollout_start = perf_counter()
                rollout = assay._rollout_observation(
                    cfg,
                    task_spec,
                    observation,
                    target_action,
                    condition_name=condition_name,
                )
                total_eval_time += float(perf_counter() - rollout_start)
                metric_energy.append(float(rollout["final_metrics"].get("mean_energy", 0.0)))
                metric_stress.append(float(rollout["final_metrics"].get("mean_stress", 0.0)))
                condition_embeddings.append(rollout["embedding"])
                condition_labels.append(target_action)
                score_start = perf_counter()
                action = int(final_model.predict(rollout["embedding"][None, :])[0])
                total_eval_time += float(perf_counter() - score_start)
                total_eval_samples += 1.0
                position = assay._transition(task_spec, position, action)
                prev_action = action
                if position in reward_zone:
                    successes.append(1.0)
                    break
            else:
                successes.append(0.0)

        success_rate = float(np.mean(successes)) if successes else 0.0
        task_peak[task_index] = success_rate
        task_initial[task_index] = success_rate
        task_final[task_index] = success_rate
        if condition_embeddings:
            embeddings_np = np.stack(condition_embeddings, axis=0)
            labels_np = np.asarray(condition_labels, dtype=np.int64)
            margin_start = perf_counter()
            final_margin = final_model.mean_margin(embeddings_np)
            total_eval_time += float(perf_counter() - margin_start)

    final_mean, _, _ = _classification_rollup(task_peak, task_initial, task_final)
    total_wall = support_rollout_time + total_train_time + total_eval_time
    parameter_count = int(support_embeddings_baseline[0].shape[1] * len(assay.ACTION_LABELS) + len(assay.ACTION_LABELS))
    mean_energy = float(np.mean(metric_energy)) if metric_energy else None
    energy_spent, score_per_energy = _energy_efficiency(final_mean, mean_energy)
    return RunStats(
        primary_metric_name="lesion_gridworld_remap_final_success_mean",
        primary_metric=final_mean,
        support_count=float(sum(len(labels) for labels in support_labels_by_task)),
        eval_count=float(len(assay.TASK_SPECS) * assay.EVAL_EPISODES),
        wall_time_sec=total_wall,
        samples_per_sec=_safe_rate(total_train_presentations + total_eval_samples, total_wall),
        score_per_sec=float(final_mean / max(total_wall, 1e-8)),
        mean_margin=float(final_margin),
        mean_energy=mean_energy,
        mean_stress=float(np.mean(metric_stress)) if metric_stress else None,
        energy_spent_estimate=energy_spent,
        score_per_energy_spent=score_per_energy,
        parameter_count=parameter_count,
        train_wall_time_sec=total_train_time + support_rollout_time,
        eval_wall_time_sec=total_eval_time,
        train_samples_per_sec=_safe_rate(total_train_presentations + sum(len(labels) for labels in support_labels_by_task), total_train_time + support_rollout_time),
        eval_samples_per_sec=_safe_rate(total_eval_samples, total_eval_time),
    )


def _run_lesion_split_mnist_organism_fixed(condition_name: str) -> RunStats:
    cfg = load_config("configs/assay/lesion_split_mnist.yaml")
    set_seed(cfg.run.seed)
    assay = LesionSplitMNISTAssay()
    dataset_root = Path(cfg.run.output_dir) / "datasets"
    train_images, train_labels, test_images, test_labels, _ = assay._load_dataset(dataset_root)
    support_images, support_labels = assay._balanced_select(
        train_images,
        train_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_images, eval_labels = assay._balanced_select(
        test_images,
        test_labels,
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )

    support_baseline_start = perf_counter()
    support_baseline = np.stack(
        [assay._rollout_image(cfg, image, condition_name="baseline")["embedding"] for image in support_images],
        axis=0,
    )
    support_baseline_time = float(perf_counter() - support_baseline_start)

    eval_start = perf_counter()
    eval_condition = []
    metric_energy = []
    metric_stress = []
    for image in eval_images:
        rollout = assay._rollout_image(cfg, image, condition_name=condition_name)
        eval_condition.append(rollout["embedding"])
        metric_energy.append(float(rollout["final_metrics"].get("mean_energy", 0.0)))
        metric_stress.append(float(rollout["final_metrics"].get("mean_stress", 0.0)))
    eval_condition = np.stack(eval_condition, axis=0)
    eval_rollout_time = float(perf_counter() - eval_start)

    seen_classes: list[int] = []
    task_final_accuracies: dict[int, float] = {}
    total_train_time = 0.0
    total_eval_time = eval_rollout_time
    total_train_presentations = 0.0
    total_eval_samples = float(len(eval_labels))
    final_margin = 0.0
    seen_train_embeddings: list[np.ndarray] = []
    seen_train_labels: list[np.ndarray] = []

    for task_index, task_classes in enumerate(assay.TASK_SPLITS):
        task_classes = tuple(int(label) for label in task_classes)
        train_mask = np.isin(support_labels, task_classes)
        seen_train_embeddings.append(support_baseline[train_mask])
        seen_train_labels.append(support_labels[train_mask])
        train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
        train_labels = np.concatenate(seen_train_labels, axis=0)

        train_start = perf_counter()
        final_model = Trainer().train_step(train_embeddings, train_labels)
        total_train_time += float(perf_counter() - train_start)
        total_train_presentations += float(len(train_labels))
        seen_classes.extend(task_classes)

    allowed_classes = np.array(sorted(set(seen_classes)), dtype=int)
    for task_index, task_classes in enumerate(assay.TASK_SPLITS):
        eval_mask = np.isin(eval_labels, task_classes)
        score_start = perf_counter()
        accuracy = final_model.score(eval_condition[eval_mask], eval_labels[eval_mask], allowed_classes=allowed_classes)
        total_eval_time += float(perf_counter() - score_start)
        total_eval_samples += float(np.sum(eval_mask))
        task_final_accuracies[task_index] = float(accuracy)

    margin_start = perf_counter()
    final_margin = final_model.mean_margin(eval_condition, allowed_classes=allowed_classes)
    total_eval_time += float(perf_counter() - margin_start)
    final_mean = float(np.mean(list(task_final_accuracies.values()))) if task_final_accuracies else 0.0
    total_wall = support_baseline_time + total_train_time + total_eval_time
    parameter_count = int(support_baseline.shape[1] * len(assay.CLASS_IDS))
    mean_energy = float(np.mean(metric_energy)) if metric_energy else None
    energy_spent, score_per_energy = _energy_efficiency(final_mean, mean_energy)
    return RunStats(
        primary_metric_name="lesion_split_mnist_final_accuracy_mean",
        primary_metric=final_mean,
        support_count=float(len(support_labels)),
        eval_count=float(len(eval_labels)),
        wall_time_sec=total_wall,
        samples_per_sec=_safe_rate(total_train_presentations + total_eval_samples, total_wall),
        score_per_sec=float(final_mean / max(total_wall, 1e-8)),
        mean_margin=float(final_margin),
        mean_energy=mean_energy,
        mean_stress=float(np.mean(metric_stress)) if metric_stress else None,
        energy_spent_estimate=energy_spent,
        score_per_energy_spent=score_per_energy,
        parameter_count=parameter_count,
        train_wall_time_sec=total_train_time + support_baseline_time,
        eval_wall_time_sec=total_eval_time,
        train_samples_per_sec=_safe_rate(total_train_presentations + len(support_labels), total_train_time + support_baseline_time),
        eval_samples_per_sec=_safe_rate(total_eval_samples, total_eval_time),
    )


def _task_report_entry(
    *,
    task_name: str,
    label: str,
    primary_metric_label: str,
    organism: RunStats,
    organism_no_growth: RunStats,
    organism_no_z_field: RunStats,
    static_routing: RunStats,
    mlp: RunStats,
    transformer: RunStats,
) -> dict:
    lesion_retention = task_name.startswith("lesion_")
    competitors = {
        "organism": asdict(organism),
        "organism_no_growth": asdict(organism_no_growth),
        "organism_no_z_field": asdict(organism_no_z_field),
        "static_routing": asdict(static_routing),
        "mlp": asdict(mlp),
        "transformer": asdict(transformer),
    }
    comparisons = {
        "organism_vs_transformer_primary_advantage": organism.primary_metric - transformer.primary_metric,
        "organism_vs_transformer_score_per_sec_advantage": organism.score_per_sec - transformer.score_per_sec,
        "organism_vs_static_primary_advantage": organism.primary_metric - static_routing.primary_metric,
        "organism_vs_mlp_primary_advantage": organism.primary_metric - mlp.primary_metric,
        "organism_no_growth_drop": organism.primary_metric - organism_no_growth.primary_metric,
        "organism_no_z_field_drop": organism.primary_metric - organism_no_z_field.primary_metric,
        "organism_no_growth_score_per_energy_drop": (
            organism.score_per_energy_spent - organism_no_growth.score_per_energy_spent
            if organism.score_per_energy_spent is not None and organism_no_growth.score_per_energy_spent is not None
            else None
        ),
        "organism_no_z_field_score_per_energy_drop": (
            organism.score_per_energy_spent - organism_no_z_field.score_per_energy_spent
            if organism.score_per_energy_spent is not None and organism_no_z_field.score_per_energy_spent is not None
            else None
        ),
    }
    return {
        "task": task_name,
        "label": label,
        "primary_metric_label": primary_metric_label,
        "evaluation_contract": {
            "mode": "baseline_trained_retained_competence" if lesion_retention else "standard_sequential_bridge",
            "fixed_readout": lesion_retention,
            "lesion_aware": lesion_retention,
            "description": (
                "Baseline-trained competence retained after lesion, scored with a readout trained on baseline organism embeddings."
                if lesion_retention
                else "Standard sequential bridge score with the organism and baseline models evaluated directly on the task family."
            ),
        },
        "competitors": competitors,
        "comparisons": comparisons,
    }


def _format_optional(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Single Organism vs Baselines",
        "",
        f"- task_count: {int(summary['task_count'])}",
        f"- baseline_trained_retention_task_count: {int(summary['baseline_trained_retention_task_count'])}",
        f"- organism_raw_metric_win_count_vs_transformer: {int(summary['organism_raw_metric_win_count_vs_transformer'])}",
        f"- organism_score_per_sec_win_count_vs_transformer: {int(summary['organism_score_per_sec_win_count_vs_transformer'])}",
        f"- organism_mechanism_drop_supported_task_count: {int(summary['organism_mechanism_drop_supported_task_count'])}",
        f"- organism_energy_efficiency_reported_task_count: {int(summary['organism_energy_efficiency_reported_task_count'])}",
        f"- organism_score_per_energy_mean: {_format_optional(summary['organism_score_per_energy_mean'])}",
        "",
    ]
    for task in report["tasks"]:
        contract = task["evaluation_contract"]
        lines.extend(
            [
                f"## {task['label']}",
                "",
                f"- evaluation_mode: {contract['mode']}",
                f"- fixed_readout: {contract['fixed_readout']}",
                f"- contract: {contract['description']}",
                "",
                "| Competitor | Primary | Score/Sec | Score/Energy | Margin | Params |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, stats in task["competitors"].items():
            lines.append(
                "| {name} | {primary:.4f} | {score_per_sec:.4f} | {score_per_energy} | {margin:.4f} | {params} |".format(
                    name=name,
                    primary=stats["primary_metric"],
                    score_per_sec=stats["score_per_sec"],
                    score_per_energy=_format_optional(stats.get("score_per_energy_spent")),
                    margin=stats["mean_margin"],
                    params=int(stats["parameter_count"] or 0),
                )
            )
        lines.extend(
            [
                "",
                f"- organism_vs_transformer_primary_advantage: {task['comparisons']['organism_vs_transformer_primary_advantage']:.4f}",
                f"- organism_vs_transformer_score_per_sec_advantage: {task['comparisons']['organism_vs_transformer_score_per_sec_advantage']:.4f}",
                f"- organism_no_growth_drop: {task['comparisons']['organism_no_growth_drop']:.4f}",
                f"- organism_no_z_field_drop: {task['comparisons']['organism_no_z_field_drop']:.4f}",
                f"- organism_no_growth_score_per_energy_drop: {_format_optional(task['comparisons']['organism_no_growth_score_per_energy_drop'])}",
                f"- organism_no_z_field_score_per_energy_drop: {_format_optional(task['comparisons']['organism_no_z_field_score_per_energy_drop'])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="artifacts/single_organism_vs_transformer.json")
    parser.add_argument("--markdown-out", default="artifacts/single_organism_vs_transformer.md")
    args = parser.parse_args()

    sequential_task = _sequential_rules_task_data()
    lesion_split_task = _lesion_split_mnist_task_data()
    split_task = _split_mnist_task_data()
    gridworld_task = _gridworld_task_data()

    tasks: list[dict] = []

    for task_name, label, task_data in (
        ("lesion_sequential_rules", "Lesion Sequential Rules", sequential_task),
        ("lesion_gridworld_remap", "Lesion Gridworld Remap", gridworld_task),
        ("lesion_split_mnist", "Lesion Split-MNIST", lesion_split_task),
        ("split_mnist", "Split-MNIST", split_task),
    ):
        if task_name == "lesion_sequential_rules":
            organism = _run_lesion_sequential_organism_fixed("baseline")
            organism_no_growth = _run_lesion_sequential_organism_fixed("no_growth")
            organism_no_z_field = _run_lesion_sequential_organism_fixed("no_z_field")
        elif task_name == "lesion_gridworld_remap":
            organism = _run_lesion_gridworld_organism_fixed("baseline")
            organism_no_growth = _run_lesion_gridworld_organism_fixed("no_growth")
            organism_no_z_field = _run_lesion_gridworld_organism_fixed("no_z_field")
        elif task_name == "lesion_split_mnist":
            organism = _run_lesion_split_mnist_organism_fixed("baseline")
            organism_no_growth = _run_lesion_split_mnist_organism_fixed("no_growth")
            organism_no_z_field = _run_lesion_split_mnist_organism_fixed("no_z_field")
        else:
            organism = _run_organism(
                cfg_path=task_data["cfg_path"],
                assay_cls=task_data["assay_cls"],
                condition_name="baseline",
                primary_key=task_data["primary_key"],
                support_key=task_data["support_key"],
                eval_key=task_data["eval_key"],
            )
            organism_no_growth = _run_organism(
                cfg_path=task_data["cfg_path"],
                assay_cls=task_data["assay_cls"],
                condition_name="no_growth",
                primary_key=task_data["primary_key"],
                support_key=task_data["support_key"],
                eval_key=task_data["eval_key"],
            )
            organism_no_z_field = _run_organism(
                cfg_path=task_data["cfg_path"],
                assay_cls=task_data["assay_cls"],
                condition_name="no_z_field",
                primary_key=task_data["primary_key"],
                support_key=task_data["support_key"],
                eval_key=task_data["eval_key"],
            )

        if task_name == "lesion_gridworld_remap":
            static_routing = _run_gridworld_baseline(
                task_support_inputs=task_data["task_support_features"],
                task_support_labels=task_data["task_support_labels"],
                task_specs=task_data["assay"].TASK_SPECS,
                assay=task_data["assay"],
                cfg=task_data["cfg"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: None,
                fit_model=_fit_static,
                predict_fn=_predict_static,
                rollout_input_builder=lambda observation: np.asarray(observation, dtype=np.float32)[None, :],
            )
            mlp = _run_gridworld_baseline(
                task_support_inputs=task_data["task_support_features"],
                task_support_labels=task_data["task_support_labels"],
                task_specs=task_data["assay"].TASK_SPECS,
                assay=task_data["assay"],
                cfg=task_data["cfg"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: TinyMLPClassifier(input_dim=39, num_classes=5, hidden_dim=64),
                fit_model=_fit_mlp,
                predict_fn=_predict_labels_mlp,
                rollout_input_builder=lambda observation: np.asarray(observation, dtype=np.float32)[None, :],
            )
            transformer = _run_gridworld_baseline(
                task_support_inputs=task_data["task_support_tokens"],
                task_support_labels=task_data["task_support_labels"],
                task_specs=task_data["assay"].TASK_SPECS,
                assay=task_data["assay"],
                cfg=task_data["cfg"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: TinyTransformerClassifier(
                    input_dim=9,
                    max_seq_len=5,
                    num_classes=5,
                    d_model=32,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=64,
                ),
                fit_model=_fit_transformer,
                predict_fn=_predict_labels_transformer,
                rollout_input_builder=lambda observation: _gridworld_tokens(observation)[None, :, :],
            )
        else:
            static_routing = _run_incremental_classifier(
                task_support_inputs=task_data["task_support_features"],
                task_support_labels=task_data["task_support_labels"],
                task_eval_inputs=task_data["task_eval_features"],
                task_eval_labels=task_data["task_eval_labels"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: None,
                fit_model=_fit_static,
                score_model=_score_static,
            )
            mlp = _run_incremental_classifier(
                task_support_inputs=task_data["task_support_features"],
                task_support_labels=task_data["task_support_labels"],
                task_eval_inputs=task_data["task_eval_features"],
                task_eval_labels=task_data["task_eval_labels"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: TinyMLPClassifier(
                    input_dim=task_data["task_support_features"][0].shape[1],
                    num_classes=10,
                    hidden_dim=96 if task_name == "split_mnist" else 64,
                ),
                fit_model=_fit_mlp,
                score_model=_score_mlp,
            )
            transformer = _run_incremental_classifier(
                task_support_inputs=task_data["task_support_tokens"],
                task_support_labels=task_data["task_support_labels"],
                task_eval_inputs=task_data["task_eval_tokens"],
                task_eval_labels=task_data["task_eval_labels"],
                primary_metric_name=task_data["primary_key"],
                build_model=lambda _idx: TinyTransformerClassifier(
                    input_dim=task_data["task_support_tokens"][0].shape[2],
                    max_seq_len=task_data["task_support_tokens"][0].shape[1],
                    num_classes=10,
                    d_model=48,
                    nhead=4,
                    num_layers=2 if task_name == "split_mnist" else 1,
                    dim_feedforward=96,
                ),
                fit_model=_fit_transformer,
                score_model=_score_transformer,
            )

        tasks.append(
            _task_report_entry(
                task_name=task_name,
                label=label,
                primary_metric_label=task_data["primary_label"],
                organism=organism,
                organism_no_growth=organism_no_growth,
                organism_no_z_field=organism_no_z_field,
                static_routing=static_routing,
                mlp=mlp,
                transformer=transformer,
            )
        )

    summary = {
        "task_count": float(len(tasks)),
        "baseline_trained_retention_task_count": float(
            sum(1 for task in tasks if task["evaluation_contract"]["fixed_readout"])
        ),
        "organism_raw_metric_win_count_vs_transformer": float(
            sum(
                1
                for task in tasks
                if task["competitors"]["organism"]["primary_metric"] > task["competitors"]["transformer"]["primary_metric"]
            )
        ),
        "organism_score_per_sec_win_count_vs_transformer": float(
            sum(
                1
                for task in tasks
                if task["competitors"]["organism"]["score_per_sec"] > task["competitors"]["transformer"]["score_per_sec"]
            )
        ),
        "organism_mechanism_drop_supported_task_count": float(
            sum(
                1
                for task in tasks
                if task["comparisons"]["organism_no_growth_drop"] > 0.0 and task["comparisons"]["organism_no_z_field_drop"] > 0.0
            )
        ),
        "organism_energy_efficiency_reported_task_count": float(
            sum(1 for task in tasks if task["competitors"]["organism"]["score_per_energy_spent"] is not None)
        ),
        "organism_score_per_energy_mean": float(
            np.mean(
                [
                    task["competitors"]["organism"]["score_per_energy_spent"]
                    for task in tasks
                    if task["competitors"]["organism"]["score_per_energy_spent"] is not None
                ]
            )
        )
        if any(task["competitors"]["organism"]["score_per_energy_spent"] is not None for task in tasks)
        else None,
    }

    payload = {"summary": summary, "tasks": tasks}
    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
