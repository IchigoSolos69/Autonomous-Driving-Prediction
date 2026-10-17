"""Reproducible JSON reports and standalone research figures."""
from pathlib import Path
import os

from adp.data import write_json


def write_report(directory, report, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", report)
    write_json(directory / "records.json", records)
    os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve()/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    risk_calibration = report.get("risk", {}).get("calibration", {})
    for name in (("raw", "calibrated") if risk_calibration.get("fitted") else ("raw",)):
        bins = report.get("risk", {}).get(name, {}).get("reliability", {}).get("bins", [])
        usable = [b for b in bins if b.get("count", 0)]
        if usable:
            axes[0].plot([b["mean_probability"] for b in usable],
                         [b["event_frequency"] for b in usable], marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0].set(xlabel="Predicted near-conflict probability", ylabel="Observed event frequency", title="Risk reliability")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    if report.get("risk", {}).get("status") == "insufficient_positive_events":
        axes[0].set_title("Risk reliability: insufficient positive events")
        if not report.get("risk", {}).get("eligible_count", 0):
            axes[0].text(.5, .3, "No eligible event labels\nCalibration not established", ha="center", transform=axes[0].transAxes)
    elif report.get("risk", {}).get("status") in ("uncalibrated", "not_fitted"):
        axes[0].set_title("Risk reliability: calibration not established")
    axes[1].hist([v for r in records for v in r.get("minFDE", [])], bins=20, color="#176b87")
    axes[1].set(xlabel="Final displacement error (m)", ylabel="Scored actor count", title="Best-of-six forecast error")
    mode_series = [("raw", "mode_reliability_raw")]
    if risk_calibration.get("applied"):
        mode_series.append(("adjusted", "mode_reliability"))
    for label, key in mode_series:
        bins = [b for b in report.get(key, {}).get("bins", []) if b["count"]]
        if bins:
            axes[2].plot([b["mean_probability"] for b in bins], [b["event_frequency"] for b in bins], marker="o", label=label)
    axes[2].plot([0,1], [0,1], ":", color="gray")
    axes[2].set(xlabel="Maximum world probability", ylabel="Nearest joint-FDE world accuracy", title="World confidence reliability")
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend()
    figure.savefig(directory / "evaluation.png", dpi=160)
    plt.close(figure)


def plot_training(directory):
    from adp.data import read_json
    os.environ.setdefault("MPLCONFIGDIR", str(Path(directory).resolve()/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    epochs = read_json(directory / "history.json")["epochs"]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key in ("train_loss", "validation_loss"):
        axis.plot([r["epoch"] for r in epochs], [r[key] for r in epochs], label=key.replace("_", " "))
    axis.set(xlabel="Epoch", ylabel="Composite training objective", title="Real-data training")
    axis.legend()
    figure.savefig(directory / "training.png", dpi=160)
    plt.close(figure)
