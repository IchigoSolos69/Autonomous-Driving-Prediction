"""A finite follow-up allocation; no benchmark jobs or previous results are reused."""
from __future__ import annotations

from adp.compute_queue import PAYLOAD_SHA256, spec_hash

HOST_SEEDS = {"private": (45, 46, 47), "colab": (48, 49, 50), "kaggle": (51, 52, 53)}
CONFIGURATIONS = ("full", "no_graph", "no_map", "world_ce")
HOST_SECONDS = 21600
# Kaggle's lost paused job is charged its entire 1,800 second allowance.
PRIOR_SECONDS = {"private": 13157.440873906, "colab": 0., "kaggle": 1800.}


def plan(host):
    if host not in HOST_SEEDS:
        raise ValueError("Unsupported follow-up host")
    jobs = [{"version": 1, "purpose": "development_only", "budget_unit": "allocated_gpu_seconds",
        "host": host, "recipe": "distributed-pilot-v2", "configuration": configuration,
        "seed": seed, "epochs": 10, "maximum_seconds": 1800,
        "payload_sha256": PAYLOAD_SHA256, "train_count": 1000, "selection_count": 64,
        "dependencies": []} for configuration in CONFIGURATIONS for seed in HOST_SEEDS[host]]
    return {"version": 2, "host": host, "host_seconds": HOST_SECONDS,
        "prior_charged_seconds": PRIOR_SECONDS[host], "jobs": jobs,
        "authorization": "Colab additional six GPU-hours; other hosts retain original total cap"}


def validate_spec(specification):
    if not isinstance(specification, dict) or specification not in plan(specification.get("host"))["jobs"]:
        raise ValueError("Job differs from the finite distributed follow-up")
    return specification


def protocol_hash(host):
    return spec_hash(plan(host))


def initial_state(host):
    return {"version": 2, "protocol_sha256": protocol_hash(host), "host": host,
        "charged_seconds": PRIOR_SECONDS[host], "setup_charged_seconds": 0., "attempts": [], "completed": [], "active": None,
        "exports": [], "state": "ready"}


def validate_state(state, host):
    import math
    expected = plan(host)
    ids = {spec_hash(job) for job in expected["jobs"]}
    if (state.get("version") != 2 or state.get("protocol_sha256") != protocol_hash(host)
            or state.get("host") != host or not isinstance(state.get("attempts"), list)
            or len(state["attempts"]) > 12 or not isinstance(state.get("completed"), list)
            or len(set(state["completed"])) != len(state["completed"])
            or not set(state["completed"]) <= ids):
        raise ValueError("Invalid follow-up ledger or protocol")
    setup = state.get("setup_charged_seconds")
    if type(setup) not in (int, float) or not math.isfinite(setup) or not 0 <= setup <= HOST_SECONDS:
        raise ValueError("Invalid setup charge")
    charged = PRIOR_SECONDS[host] + setup
    attempted = []
    for attempt in state["attempts"]:
        if (attempt.get("job_id") not in ids or attempt["job_id"] in attempted
                or type(attempt.get("charged_seconds")) not in (int, float)
                or not math.isfinite(attempt["charged_seconds"])
                or not 0 <= attempt["charged_seconds"] <= 3600
                or attempt.get("state") not in ("reserved", "completed", "paused", "failed")):
            raise ValueError("Invalid follow-up attempt")
        attempted.append(attempt["job_id"])
        charged += attempt["charged_seconds"]
    if (type(state.get("charged_seconds")) not in (int, float)
            or not math.isfinite(state["charged_seconds"])
            or not math.isclose(state["charged_seconds"], charged, abs_tol=1e-7)
            or not set(state["completed"]) <= {a["job_id"] for a in state["attempts"] if a["state"] == "completed"}
            or state.get("active") not in (None, *attempted)):
        raise ValueError("Follow-up ledger charges or completion disagree")
    return state
