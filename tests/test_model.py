import copy

import torch

from adp.model import ForecastModel, constant_velocity, forecast_loss, model_inputs


def batch_fixture():
    history = torch.zeros(2, 3, 50, 8)
    history[..., 5:7] = 1
    history[:, :, :, 0] = torch.arange(50)*.1
    history[..., 2] = 1
    history[:, 1, :, 1] = 5
    return {"history": history, "history_mask": torch.ones(2, 3, 50, dtype=torch.bool),
            "actor_mask": torch.ones(2, 3, dtype=torch.bool), "dynamic_mask": torch.ones(2, 3, dtype=torch.bool),
            "actor_types": torch.zeros(2, 3, dtype=torch.long), "future": torch.zeros(2, 3, 60, 2),
            "future_mask": torch.ones(2, 3, 60, dtype=torch.bool), "map_points": torch.zeros(2, 2, 20, 2),
            "map_features": torch.zeros(2, 2, 4), "map_mask": torch.ones(2, 2, dtype=torch.bool),
            "sdf": torch.ones(2, 1, 256, 256), "has_drivable": torch.ones(2, dtype=torch.bool)}


def test_future_labels_cannot_enter_model_and_scene_graphs_are_isolated():
    torch.manual_seed(4)
    batch = batch_fixture()
    model = ForecastModel({"width": 32, "heads": 4, "temporal_layers": 1, "graph_layers": 1}).eval()
    with torch.no_grad():
        first = model(**model_inputs(batch))
        changed = copy.deepcopy(batch)
        changed["future"] += 10000
        second = model(**model_inputs(changed))
        torch.testing.assert_close(first["means"], second["means"])
        changed["history"][1, :, :, :4] += 100
        third = model(**model_inputs(changed))
        torch.testing.assert_close(first["means"][0], third["means"][0])
    assert first["means"].shape == (2, 6, 3, 60, 2)
    torch.testing.assert_close(first["logits"].softmax(-1).sum(-1), torch.ones(2))
    assert first["scales"].min() > 0


def test_missing_actors_map_and_future_keep_loss_finite():
    batch = batch_fixture()
    batch["history_mask"][:, 2] = False
    batch["actor_mask"][:, 2] = False
    batch["dynamic_mask"][:, 2] = False
    batch["future_mask"][:, 2] = False
    batch["future"][:, 2] = float("nan")
    batch["map_mask"][:] = False
    model = ForecastModel({"width": 32, "heads": 4, "temporal_layers": 1, "graph_layers": 1})
    outputs = model(**model_inputs(batch))
    losses = forecast_loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_constant_velocity_uses_observed_motion_only():
    batch = batch_fixture()
    outputs = constant_velocity(model_inputs(batch))
    torch.testing.assert_close(outputs["means"][0, 0, 0, -1], torch.tensor([10.9, 0.]))
    torch.testing.assert_close(outputs["means"][:, 0], outputs["means"][:, 5])


def test_map_penalty_requires_vehicle_type_and_known_map_coverage():
    batch = batch_fixture()
    batch['sdf'].fill_(-5)
    outputs = constant_velocity(model_inputs(batch))
    assert forecast_loss(outputs, batch)['offroad'].item() == 5
    outputs['means'].fill_(200)
    assert forecast_loss(outputs, batch)['offroad'].item() == 0
    outputs['means'].fill_(0)
    batch['actor_types'].fill_(1)
    assert forecast_loss(outputs, batch)['offroad'].item() == 0
    batch['actor_types'].fill_(0)
    outputs['map_enabled'] = False
    assert forecast_loss(outputs, batch)['offroad'].item() == 0
