import torch

from turbulens.models.architecture import build_model_from_config, parameter_counts


def _config(targets):
    return {
        "model": {
            "architecture": "resnet18",
            "in_channels": 1,
            "pretrained": False,
            "dropout": 0.1,
        },
        "task": {"targets": targets},
    }


def test_single_target_forward_shape():
    model = build_model_from_config(_config(["k_min"]))
    model.eval()

    output = model(torch.zeros(2, 1, 64, 64))

    assert output.shape == (2, 1)


def test_multi_target_forward_shape():
    model = build_model_from_config(_config(["k_min", "sigma"]))
    model.eval()

    output = model(torch.zeros(3, 1, 64, 64))

    assert output.shape == (3, 2)
    assert model.targets == ("k_min", "sigma")


def test_parameter_counts_are_positive():
    model = build_model_from_config(_config(["k_min"]))

    counts = parameter_counts(model)

    assert counts["total"] > 0
    assert counts["trainable"] > 0
