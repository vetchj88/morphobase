import numpy as np

from morphobase.training.transformer_baselines import (
    TinyMLPClassifier,
    TinyTransformerClassifier,
    fit_mlp_classifier,
    fit_static_prototype,
    fit_transformer_classifier,
    parameter_count,
    score_mlp_classifier,
    score_static_prototype,
    score_transformer_classifier,
)


def test_tiny_transformer_classifier_forward_shape():
    model = TinyTransformerClassifier(
        input_dim=3,
        max_seq_len=5,
        num_classes=4,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
    )
    tokens = np.random.default_rng(0).normal(size=(7, 5, 3)).astype(np.float32)
    accuracy, margin, _ = score_transformer_classifier(model, tokens, np.zeros(7, dtype=np.int64))
    assert isinstance(accuracy, float)
    assert isinstance(margin, float)
    assert parameter_count(model) > 0


def test_tiny_transformer_classifier_fits_simple_sequence_problem():
    rng = np.random.default_rng(1)
    class0 = rng.normal(loc=-0.75, scale=0.08, size=(16, 6, 1)).astype(np.float32)
    class1 = rng.normal(loc=0.75, scale=0.08, size=(16, 6, 1)).astype(np.float32)
    tokens = np.concatenate([class0, class1], axis=0)
    labels = np.concatenate([np.zeros(16, dtype=np.int64), np.ones(16, dtype=np.int64)], axis=0)

    model = TinyTransformerClassifier(
        input_dim=1,
        max_seq_len=6,
        num_classes=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
    )
    fit_transformer_classifier(
        model,
        tokens,
        labels,
        epochs=40,
        learning_rate=5e-3,
        weight_decay=1e-4,
        batch_size=8,
        seed=7,
    )
    accuracy, margin, _ = score_transformer_classifier(model, tokens, labels)
    assert accuracy >= 0.95
    assert margin > 0.5


def test_static_prototype_fits_simple_feature_problem():
    rng = np.random.default_rng(5)
    class0 = rng.normal(loc=-0.8, scale=0.05, size=(16, 6)).astype(np.float32)
    class1 = rng.normal(loc=0.8, scale=0.05, size=(16, 6)).astype(np.float32)
    features = np.concatenate([class0, class1], axis=0)
    labels = np.concatenate([np.zeros(16, dtype=np.int64), np.ones(16, dtype=np.int64)], axis=0)

    model, summary = fit_static_prototype(features, labels)
    accuracy, margin, _ = score_static_prototype(model, features, labels)
    assert summary.parameter_count > 0
    assert accuracy >= 0.95
    assert margin > 0.1


def test_tiny_mlp_classifier_fits_simple_feature_problem():
    rng = np.random.default_rng(9)
    class0 = rng.normal(loc=-0.7, scale=0.07, size=(20, 8)).astype(np.float32)
    class1 = rng.normal(loc=0.7, scale=0.07, size=(20, 8)).astype(np.float32)
    features = np.concatenate([class0, class1], axis=0)
    labels = np.concatenate([np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)], axis=0)

    model = TinyMLPClassifier(input_dim=8, num_classes=2, hidden_dim=24)
    fit_mlp_classifier(
        model,
        features,
        labels,
        epochs=50,
        learning_rate=4e-3,
        weight_decay=1e-4,
        batch_size=8,
        seed=11,
    )
    accuracy, margin, _ = score_mlp_classifier(model, features, labels)
    assert accuracy >= 0.95
    assert margin > 0.5
