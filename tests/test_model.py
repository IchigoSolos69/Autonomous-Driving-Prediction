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


def test_shared_world_ce_uses_detached_complete_scored_joint_fde():
    batch = batch_fixture()
    batch["scored_mask"] = torch.tensor([[True, False, True], [False, False, False]])
    batch["future_mask"][0, 2, -1] = False
    batch["future"][0, 2] = float("nan")
    means = torch.zeros(2, 2, 3, 60, 2)
    means[0, 0, 0, -1, 0] = 4
    means[0, 1, 0, :-1, 0] = 100
    means[0, 1, 1] = 1000
    means.requires_grad_()
    logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    outputs = {"means": means, "scales": torch.ones_like(means), "logits": logits,
               "risk_logits": torch.zeros(2, 3), "map_enabled": False}
    baseline = forecast_loss(outputs, batch)
    candidate = forecast_loss({**outputs, "shared_world_ce_weight": .1}, batch)
    expected = torch.nn.functional.cross_entropy(logits[:1], torch.tensor([1]))
    torch.testing.assert_close(candidate["shared_world_ce"], expected)
    torch.testing.assert_close(candidate["loss"], baseline["loss"] + .1*expected)
    assert int(candidate["world_supervision_count"]) == 1
    candidate["shared_world_ce"].backward()
    assert means.grad is None
    torch.testing.assert_close(logits.grad[1], torch.zeros(2))
    assert logits.grad[0, 0] > 0


def test_shared_world_ce_skips_scenes_without_complete_scored_actors():
    batch = batch_fixture()
    batch["future_mask"][:] = False
    batch["future"][:] = float("nan")
    outputs = constant_velocity(model_inputs(batch))
    outputs["logits"].requires_grad_()
    outputs["shared_world_ce_weight"] = .1
    losses = forecast_loss(outputs, batch)
    assert losses["shared_world_ce"].item() == 0
    assert losses["world_supervision_count"].item() == 0
    assert torch.isfinite(losses["loss"])
    losses["shared_world_ce"].backward()
    torch.testing.assert_close(outputs["logits"].grad, torch.zeros_like(outputs["logits"]))
