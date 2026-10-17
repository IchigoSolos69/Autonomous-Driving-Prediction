"""Held-out temperature and Platt calibration for forecast probabilities."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mode_targets(outputs: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the mode-ECE oracle: joint FDE on complete scored dynamic actors."""
    dynamic = batch["dynamic_mask"].bool() & batch["actor_mask"].bool()
    complete = (batch["future_mask"].bool() & torch.isfinite(batch["future"]).all(dim=-1)).all(dim=-1)
    scored = batch.get("scored_mask", dynamic).bool() & dynamic & complete
    distances = torch.linalg.vector_norm(
        outputs["means"][..., -1, :] - batch["future"][..., -1, :].unsqueeze(1), dim=-1
    )
    errors = torch.where(scored.unsqueeze(1), distances, 0).sum(dim=-1)
    errors = errors / scored.sum(dim=-1).clamp_min(1).unsqueeze(-1)
    return errors.argmin(dim=-1), scored.any(dim=-1)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Fit a positive scalar on the calibration split, with identity fallback."""
    logits = logits.detach().cpu().double().clone()
    targets = targets.detach().cpu().long().clone()
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError("Mode logits must be [N,K] and targets [N].")
    count = int(targets.numel())
    if not count:
        return {"mode_temperature": 1.0, "fitted": False, "count": 0, "reason": "no eligible scenes"}
    if not torch.isfinite(logits).all():
        raise ValueError("Mode calibration logits contain non-finite values.")
    before = float(F.cross_entropy(logits, targets))
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-3.0, 3.0).exp()
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-3.0, 3.0).exp())
    after = float(F.cross_entropy(logits / temperature, targets))
    if not torch.isfinite(torch.tensor(after)) or after > before:
        temperature, after = 1.0, before
    return {
        "mode_temperature": temperature,
        "fitted": True,
        "count": count,
        "nll_before": before,
        "nll_after": after,
    }


def fit_platt(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Fit monotone Platt scaling: calibrated logit = raw / T + bias."""
    logits = logits.detach().cpu().double().reshape(-1).clone()
    targets = targets.detach().cpu().double().reshape(-1).clone()
    if logits.shape != targets.shape:
        raise ValueError("Risk logits and labels must have matching shapes.")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Risk calibration inputs contain non-finite values.")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Risk calibration labels must be binary.")
    count = int(targets.numel())
    positives = int(targets.sum())
    if not count or targets.unique().numel() < 2:
        return {
            "risk_temperature": 1.0,
            "risk_bias": 0.0,
            "fitted": False,
            "count": count,
            "positives": positives,
            "reason": "both risk classes are required",
        }
    before = float(F.binary_cross_entropy_with_logits(logits, targets))
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        calibrated = logits / log_temperature.clamp(-3.0, 3.0).exp() + bias.clamp(-20.0, 20.0)
        loss = F.binary_cross_entropy_with_logits(calibrated, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-3.0, 3.0).exp())
    fitted_bias = float(bias.detach().clamp(-20.0, 20.0))
    after = float(F.binary_cross_entropy_with_logits(logits / temperature + fitted_bias, targets))
    if not torch.isfinite(torch.tensor(after)) or after > before:
        temperature, fitted_bias, after = 1.0, 0.0, before
    return {
        "risk_temperature": temperature,
        "risk_bias": fitted_bias,
        "fitted": True,
        "count": count,
        "positives": positives,
        "nll_before": before,
        "nll_after": after,
    }


def apply_calibration(outputs: dict, calibration: dict | None = None) -> dict:
    """Return calibrated logits while preserving trajectory tensors."""
    calibration = calibration or {}
    mode_temperature = float(calibration.get("mode_temperature", 1.0))
    risk_temperature = float(calibration.get("risk_temperature", 1.0))
    risk_bias = float(calibration.get("risk_bias", 0.0))
    if not all(math.isfinite(value) for value in (mode_temperature, risk_temperature, risk_bias)):
        raise ValueError("Calibration parameters must be finite.")
    if mode_temperature <= 0 or risk_temperature <= 0:
        raise ValueError("Calibration temperatures must be positive.")
    return {
        **outputs,
        "logits": outputs["logits"] / mode_temperature,
        "risk_logits": outputs["risk_logits"] / risk_temperature + risk_bias,
    }
