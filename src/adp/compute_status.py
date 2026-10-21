"""Bounded host evidence published by controllers; HTTP polling never contacts hosts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

DEFAULT_HOSTS_PATH = Path("D:/InteractionLab/remote-compute/hosts.json")
MAX_HOSTS_BYTES = 1024**2
STALE_AFTER_SECONDS = 4 * 3600
LABELS = {"local": "Local NVIDIA GPU", "private": "Private Ubuntu server",
          "colab": "Google Colab", "kaggle": "Kaggle"}
STATUSES = {"unknown", "not_started", "waiting", "qualifying", "ready", "running",
            "paused", "completed", "failed", "unavailable", "offline", "budget_exhausted", "blocked"}
TRAINING_FIELDS = ("completed_epochs", "planned_epochs", "committed_examples", "total_examples", "examples_per_second")


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _text(value, limit=600):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid host text")
    return value


def _timestamp(value, now):
    if value is None:
        return None, None
    value = _text(value, 80)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Host timestamps require a timezone")
    timestamp = parsed.timestamp()
    if not math.isfinite(timestamp) or timestamp > now + 300:
        raise ValueError("Host timestamp exceeds clock-skew tolerance")
    return _iso(timestamp), timestamp


def _number(value, integer=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid host progress number")
    if not math.isfinite(value) or value < 0 or value > 10**12 or (integer and int(value) != value):
        raise ValueError("Host progress number is outside supported bounds")
    return int(value) if integer else float(value)


def _base(identifier, message=None, status="unknown"):
    return {"id": identifier, "label": LABELS[identifier], "status": status, "phase": None,
            "message": message, "gpu": None, "updated_at": None, "observed_at": None,
            "training": dict.fromkeys(TRAINING_FIELDS),
            "transfer": {"status": "unknown", "verified_at": None, "local_path": None},
            "cleanup": {"status": "unknown", "message": None},
            "connection": {"status": "unconfigured", "phase": None, "scope": None,
                           "message": None, "checked_at": None, "saved_run_status": None,
                           "source_observed_at": None, "published_at": None, "completed_jobs": None,
                           "completed_epochs": None, "available_archives": None, "expected_archives": None}}


def _freshness(row, now, threshold=STALE_AFTER_SECONDS):
    _, observed = _timestamp(row.get("observed_at") or row.get("updated_at"), now)
    age = max(0., now - observed) if observed is not None else None
    stale = age is None or age > threshold
    row.update(stale=stale, last_observed_status=row["status"], freshness={
        "age_seconds": age, "stale": stale, "checked_at": _iso(now),
        "observed_at": row.get("observed_at") or row.get("updated_at")})
    if stale and row["status"] in {"running", "qualifying"}:
        row["status"] = "unknown"
        row["message"] = "Host evidence is stale; current execution is unconfirmed. " + (row["message"] or "")
    return row


def _host(value, now):
    if not isinstance(value, dict) or value.get("id") not in LABELS:
        raise ValueError("Unknown host identifier")
    identifier = value["id"]
    row = _base(identifier)
    status = value.get("status", "unknown")
    if status not in STATUSES:
        raise ValueError("Unknown host status")
    row.update(status=status, label=_text(value.get("label"), 120) or LABELS[identifier],
               phase=_text(value.get("phase"), 120), message=_text(value.get("message")),
               gpu=_text(value.get("gpu"), 160))
    row["updated_at"], _ = _timestamp(value.get("updated_at"), now)
    row["observed_at"], _ = _timestamp(value.get("observed_at"), now)
    training = value.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("Training evidence must contain an object")
    row["training"] = {key: _number(training.get(key), integer=key != "examples_per_second") for key in TRAINING_FIELDS}
    for numerator, denominator in (("completed_epochs", "planned_epochs"), ("committed_examples", "total_examples")):
        if row["training"][numerator] is not None and row["training"][denominator] is not None and row["training"][numerator] > row["training"][denominator]:
            raise ValueError("Host progress exceeds its declared total")
    transfer, cleanup = value.get("transfer", {}), value.get("cleanup", {})
    if not isinstance(transfer, dict) or not isinstance(cleanup, dict):
        raise ValueError("Transfer and cleanup evidence must contain objects")
    verified_at, _ = _timestamp(transfer.get("verified_at"), now)
    row["transfer"] = {"status": _text(transfer.get("status"), 80) or "unknown",
                       "verified_at": verified_at, "local_path": _text(transfer.get("local_path"), 1024)}
    if row["transfer"]["status"] == "verified" and not (verified_at and row["transfer"]["local_path"]):
        raise ValueError("Verified transfer requires its timestamp and local receipt path")
    row["cleanup"] = {"status": _text(cleanup.get("status"), 80) or "unknown", "message": _text(cleanup.get("message"))}
    connection = value.get("connection", {})
    if not isinstance(connection, dict):
        raise ValueError("Connection evidence must contain an object")
    checked_at, _ = _timestamp(connection.get("checked_at"), now)
    source_observed_at, _ = _timestamp(connection.get("source_observed_at"), now)
    published_at, _ = _timestamp(connection.get("published_at"), now)
    row["connection"] = {"status": _text(connection.get("status"), 80) or "unconfigured",
        "phase": _text(connection.get("phase"), 120), "scope": _text(connection.get("scope"), 80),
        "message": _text(connection.get("message")), "checked_at": checked_at,
        "saved_run_status": _text(connection.get("saved_run_status"), 80),
        "source_observed_at": source_observed_at, "published_at": published_at,
        **{name:_number(connection.get(name), integer=True) for name in
            ("completed_jobs", "completed_epochs", "available_archives", "expected_archives")}}
    return _freshness(row, now, threshold=600 if row['phase'] == 'distributed development v2' else STALE_AFTER_SECONDS)


def _local(studies, now, published=None):
    row = _base("local", "No local study progress is available.")
    if published:
        row["gpu"] = published.get("gpu")
        row["cleanup"] = dict(published["cleanup"])
    candidates = [study for study in studies if isinstance(study, dict)]
    study = next((study for study in candidates if study.get("status") == "running"), candidates[0] if candidates else None)
    if not study:
        return _freshness(row, now)
    status = study.get("status")
    row["status"] = "paused" if status == "stopped" else status if status in STATUSES else "unknown"
    row["phase"] = study.get("phase")
    row["message"] = study.get("message") or "Read from local study records; checkpoint progress does not prove a worker is live."
    rows = study.get("runs", [])
    active = next((run for run in rows if run.get("id") == study.get("current_run_id")), None)
    progress = study.get("progress", {})
    row["training"].update(completed_epochs=progress.get("completed_epochs"), planned_epochs=progress.get("planned_epochs"))
    if active:
        row["training"].update(committed_examples=active.get("committed_examples"), total_examples=active.get("training_examples"),
                               examples_per_second=active.get("examples_per_second"))
        row["updated_at"] = active.get("checkpoint", {}).get("updated_at")
    else:
        dates = [run.get("checkpoint", {}).get("updated_at") for run in rows if run.get("checkpoint", {}).get("updated_at")]
        row["updated_at"] = max(dates, default=None)
    row["observed_at"] = row["updated_at"]
    return _freshness(row, now, threshold=600 if active else STALE_AFTER_SECONDS)


def read_compute_hosts(path=DEFAULT_HOSTS_PATH, *, studies=(), now=None):
    """Return four bounded records, including truthful unknown/malformed/stale states."""
    now = time.time() if now is None else now
    records, issue = {}, None
    try:
        with Path(path).open("rb") as stream:
            content = stream.read(MAX_HOSTS_BYTES + 1)
        if not content or len(content) > MAX_HOSTS_BYTES:
            raise ValueError("Host file exceeds its supported size")
        value = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1 or not isinstance(value.get("hosts"), list) or len(value["hosts"]) > 4:
            raise ValueError("Unsupported host metadata format")
        for item in value["hosts"]:
            if not isinstance(item, dict) or item.get("id") not in LABELS or item["id"] in records:
                raise ValueError("Host records require unique supported identifiers")
            try:
                records[item["id"]] = _host(item, now)
            except (ValueError, TypeError, OverflowError):
                records[item["id"]] = _freshness(_base(item["id"], "Host evidence is invalid; refresh this host in the compute controller.", "unavailable"), now)
    except FileNotFoundError:
        issue = "No host evidence has been published yet; run the compute qualification/transfer controller."
    except (OSError, ValueError, TypeError, OverflowError):
        issue = "Compute-host records are unreadable; refresh hosts.json using the compute controller."
        records = {}
    local = _local(studies, now, records.get("local"))
    return [local] + [records.get(identifier) or _freshness(_base(identifier, issue or "No evidence has been published for this host.",
                             "unavailable" if issue and "unreadable" in issue else "unknown"), now)
                      for identifier in ("private", "colab", "kaggle")]
