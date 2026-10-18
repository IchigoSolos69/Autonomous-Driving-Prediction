"""Check a real model step and portable checkpoint inference without changing weights."""
import argparse
import io
import json
from pathlib import Path
import platform
import time

import torch

from adp.model import ForecastModel, model_inputs
from adp.training import _smoke_batch, load_checkpoint


def qualify(expected_torch=None, device="cpu", checkpoint=None):
    if expected_torch and torch.__version__ != expected_torch:
        raise RuntimeError(f"Expected torch {expected_torch}; found {torch.__version__}")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The NVIDIA edition needs a compatible NVIDIA driver for GPU qualification")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = ForecastModel().to(selected)
    batch = _smoke_batch(selected)
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(network, optim):
        optim.zero_grad(set_to_none=True)
        outputs = network(**model_inputs(batch))
        loss = sum(outputs[key].square().mean() for key in ("means", "scales", "logits", "risk_logits"))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
        optim.step()

    step(model, optimizer)
    snapshot = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if selected.type == "cuda" else []}, snapshot)
    step(model, optimizer)
    snapshot.seek(0)
    saved_step = torch.load(snapshot, map_location="cpu", weights_only=True)
    continued = ForecastModel().to(selected)
    continued.load_state_dict(saved_step["model"])
    continued_optimizer = torch.optim.AdamW(continued.parameters(), lr=1e-4)
    continued_optimizer.load_state_dict(saved_step["optimizer"])
    torch.set_rng_state(saved_step["cpu_rng"])
    if selected.type == "cuda":
        torch.cuda.set_rng_state_all(saved_step["cuda_rng"])
    step(continued, continued_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, continued.state_dict()[name], rtol=1e-5, atol=1e-5)
    if selected.type == "cuda":
        torch.cuda.synchronize()
    result = {"python": platform.python_version(), "torch": torch.__version__, "device": str(selected),
              "cuda_runtime": torch.version.cuda, "forward_backward_seconds": time.perf_counter() - start,
              "actual_forward_backward": True, "optimizer_rng_continuation": "passed",
              "checkpoint_portability": "not_requested"}
    if checkpoint:
        saved = load_checkpoint(checkpoint)
        cpu = ForecastModel(saved["config"].get("model", saved["config"])).eval()
        cpu.load_state_dict(saved["model_state_dict"])
        inputs = model_inputs(_smoke_batch(torch.device("cpu")))
        with torch.inference_mode():
            expected = cpu(**inputs)
            actual = cpu.to(selected)(**{key: value.to(selected) for key, value in inputs.items()})
        for key in ("means", "scales", "logits", "risk_logits"):
            torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), rtol=1e-4, atol=1e-4)
        result["checkpoint_portability"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-torch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify(args.expected_torch, args.device, args.checkpoint)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
