"""Compact temporal/graph encoder with a joint six-world mixture decoder."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


INPUT_KEYS = ("history", "history_mask", "actor_mask", "actor_types", "dynamic_mask", "map_points", "map_features", "map_mask")


def model_inputs(batch):
    """An explicit trust boundary: reference futures can never enter forward()."""
    return {key: batch[key] for key in INPUT_KEYS}


def lane_tangents(points, features, mask):
    """Signed unit travel directions; preprocessing preserves lane point order."""
    differences = torch.zeros_like(points)
    if points.shape[-2] > 1:
        differences[..., 0, :] = points[..., 1, :] - points[..., 0, :]
        differences[..., -1, :] = points[..., -1, :] - points[..., -2, :]
        differences[..., 1:-1, :] = points[..., 2:, :] - points[..., :-2, :]
    norm = torch.linalg.vector_norm(differences, dim=-1, keepdim=True)
    valid = mask & (features[..., 0] > .5) & (features[..., 1] < .5)
    return torch.where(valid[..., None, None] & (norm > 1e-6), differences / norm.clamp_min(1e-6), 0)


class InteractionLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.depth = heads, width//heads
        self.qkv = nn.Linear(width, width*3)
        self.edge = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, heads))
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.final_norm = nn.LayerNorm(width)

    def forward(self, nodes, state, valid):
        b, a, width = nodes.shape
        q, k, v = self.qkv(nodes).reshape(b, a, 3, self.heads, self.depth).unbind(2)
        rel_xy = state[:, :, None, :2]-state[:, None, :, :2]
        rel_v = state[:, :, None, 2:4]-state[:, None, :, 2:4]
        rel_h = state[:, :, None, 4:6]-state[:, None, :, 4:6]
        distance = torch.linalg.vector_norm(rel_xy, dim=-1, keepdim=True)
        edge = torch.cat([rel_xy/50, rel_v/20, rel_h, distance/50], dim=-1)
        scores = torch.einsum("bihd,bjhd->bhij", q, k)/math.sqrt(self.depth)
        scores = scores+self.edge(edge).permute(0, 3, 1, 2)
        allowed = valid[:, None, :] & (distance.squeeze(-1) <= 50)
        # Padded queries still need a finite softmax; they are zeroed below.
        allowed = allowed | torch.eye(a, device=nodes.device, dtype=torch.bool)[None]
        scores = scores.masked_fill(~allowed[:, None], -1e4)
        messages = torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v).reshape(b, a, width)
        nodes = self.norm(nodes+self.output(messages))
        return self.final_norm(nodes+self.ff(nodes))*valid[..., None]


class ForecastModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or {})
        c = self.config
        self.map_point_encoding = c.get("map_point_encoding", "xy")
        if self.map_point_encoding not in ("xy", "xy_zero_tangent", "xy_tangent"):
            raise ValueError("map_point_encoding must be xy, xy_zero_tangent or xy_tangent")
        self.shared_world_ce_weight = float(c.get("shared_world_ce_weight", 0))
        if not math.isfinite(self.shared_world_ce_weight) or self.shared_world_ce_weight < 0:
            raise ValueError("shared_world_ce_weight must be finite and nonnegative")
        width, heads = c.get("width", 128), c.get("heads", 4)
        self.modes = c.get("modes", 6)
        self.history_embedding = nn.Linear(8, width)
        self.type_embedding = nn.Embedding(10, width)
        self.time_embedding = nn.Parameter(torch.randn(50, width)*0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width*2, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, c.get("temporal_layers", 2), enable_nested_tensor=False)
        self.graph = nn.ModuleList([InteractionLayer(width, heads) for _ in range(c.get("graph_layers", 2))])
        self.map_encoder = nn.Sequential(nn.Linear(6 if self.map_point_encoding == "xy" else 8, width), nn.GELU(), nn.Linear(width, width))
        self.map_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.map_norm = nn.LayerNorm(width)
        self.world_queries = nn.Parameter(torch.randn(self.modes, width)*0.1)
        self.scene_logits = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.modes))
        self.decoder = nn.Sequential(nn.Linear(width*2, width*2), nn.GELU(), nn.Linear(width*2, 60*4))
        self.risk = nn.Sequential(nn.Linear(width+3, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, history, history_mask, actor_mask, actor_types, dynamic_mask, map_points, map_features, map_mask):
        b, a, length, _ = history.shape
        if length != 50:
            raise ValueError("Exactly 50 observed steps are required")
        safe_mask = history_mask.clone()
        safe_mask[..., 0] |= ~safe_mask.any(-1)
        scaled = history*history.new_tensor([.01, .01, .05, .05, 1, 1, 1, 1])
        encoded = self.history_embedding(scaled)+self.time_embedding
        encoded = encoded+self.type_embedding(actor_types)[:, :, None]
        temporal = self.temporal(encoded.reshape(b*a, 50, -1), src_key_padding_mask=~safe_mask.reshape(b*a, 50))
        indices = (history_mask*torch.arange(50, device=history.device)).amax(-1)
        nodes = temporal.reshape(b, a, 50, -1).gather(2, indices[..., None, None].expand(b, a, 1, temporal.shape[-1])).squeeze(2)
        state = history.gather(2, indices[..., None, None].expand(b, a, 1, 8)).squeeze(2)
        nodes = nodes*actor_mask[..., None]
        if self.config.get("use_map", True):
            map_input = torch.cat([map_points/100, map_features[:, :, None].expand(-1, -1, 20, -1)], dim=-1)
            if self.map_point_encoding != "xy":
                tangents = lane_tangents(map_points, map_features, map_mask) if self.map_point_encoding == "xy_tangent" else torch.zeros_like(map_points)
                map_input = torch.cat([map_input, tangents], dim=-1)
            map_tokens = self.map_encoder(map_input).amax(2)
            safe_map_mask = map_mask.clone()
            safe_map_mask[:, 0] |= ~safe_map_mask.any(-1)
            map_tokens = map_tokens*map_mask[..., None]
            context, _ = self.map_attention(nodes, map_tokens, map_tokens, key_padding_mask=~safe_map_mask, need_weights=False)
            nodes = self.map_norm(nodes+context)*actor_mask[..., None]
        if self.config.get("use_graph", True):
            for layer in self.graph:
                nodes = layer(nodes, state, actor_mask)
        pooled = (nodes*actor_mask[..., None]).sum(1)/actor_mask.sum(1, keepdim=True).clamp_min(1)
        worlds = pooled[:, None]+self.world_queries[None]
        expanded_nodes = nodes[:, None].expand(-1, self.modes, -1, -1)
        decoded = self.decoder(torch.cat([expanded_nodes, worlds[:, :, None].expand(-1, -1, a, -1)], dim=-1))
        decoded = decoded.reshape(b, self.modes, a, 60, 4)
        times = torch.arange(1, 61, device=history.device, dtype=history.dtype)[None, None, None, :, None]*.1
        cv = state[:, None, :, None, :2]+state[:, None, :, None, 2:4]*times
        means = cv+decoded[..., :2]*times
        scales = (.15+F.softplus(decoded[..., 2:])).clamp_max(20)
        logits = self.scene_logits(pooled)
        # Clearance features are invariant to arbitrary permutations of the shared worlds.
        xy = means.detach()
        separation = torch.linalg.vector_norm(xy[:, :, :, None]-xy[:, :, None, :], dim=-1)
        dynamic = dynamic_mask & actor_mask
        pair_valid = dynamic[:, :, None] & dynamic[:, None, :]
        pair_valid &= ~torch.eye(a, device=history.device, dtype=torch.bool)[None]
        separation = separation.masked_fill(~pair_valid[:, None, :, :, None], 1000)
        clearance = separation.amin(dim=(-1, -2))
        probabilities = logits.softmax(-1)
        summaries = torch.stack([(clearance*probabilities[..., None]).sum(1).clamp_max(100)/100,
                                 clearance.amin(1).clamp_max(100)/100,
                                 ((3-clearance).sigmoid()*probabilities[..., None]).sum(1)], dim=-1)
        risk_logits = self.risk(torch.cat([nodes, summaries], dim=-1)).squeeze(-1)
        return {"means": means, "scales": scales, "logits": logits, "risk_logits": risk_logits,
                "map_enabled": self.config.get("use_map", True),
                "shared_world_ce_weight": self.shared_world_ce_weight}


def forecast_loss(outputs, batch):
    from adp.calibration import mode_targets
    from adp.metrics import risk_labels
    means, scales = outputs["means"].float(), outputs["scales"].float()
    raw_target = batch["future"].float()
    valid = batch["future_mask"] & torch.isfinite(raw_target).all(-1) & batch["dynamic_mask"][..., None] & batch["actor_mask"][..., None]
    target = torch.where(valid[..., None], raw_target, 0)
    count = valid.sum((1, 2)).clamp_min(1)
    log_density = -.5*((target[:, None]-means)/scales).square()-scales.log()-.5*math.log(2*math.pi)
    world_ll = (log_density.sum(-1)*valid[:, None]).sum((2, 3))
    nll = (-torch.logsumexp(outputs["logits"].float().log_softmax(-1)+world_ll, dim=1)/(2*count)).mean()
    distance = F.smooth_l1_loss(means, target[:, None].expand_as(means), reduction="none").sum(-1)
    trajectory = ((distance*valid[:, None]).sum((2, 3))/count[:, None]).amin(1).mean()
    labels, eligible = risk_labels(target, valid, batch["dynamic_mask"] & batch["actor_mask"])
    risk_terms = F.binary_cross_entropy_with_logits(outputs["risk_logits"].float(), labels.float(), reduction="none")
    risk = (risk_terms*eligible).sum()/eligible.sum().clamp_min(1)
    offroad = means.new_zeros(())
    if "sdf" in batch and outputs.get("map_enabled", True):
        b, k, a, t, _ = means.shape
        grid = (means/128).reshape(b, k*a, t, 2)
        sdf = F.grid_sample(batch["sdf"].float(), grid, mode="bilinear", padding_mode="border", align_corners=False).reshape(b, k, a, t)
        vehicle = ((batch["actor_types"] == 0) | (batch["actor_types"] == 2) | (batch["actor_types"] == 4)) & batch["actor_mask"]
        mask = valid & vehicle[..., None] & batch.get("has_drivable", torch.ones(b, dtype=torch.bool, device=means.device))[:, None, None]
        # The cached tile says nothing about roads outside its coverage.
        known = mask[:, None] & (means.abs() <= 127.5).all(-1)
        penalties = F.relu(-sdf)*known
        offroad = (penalties.sum((1, 2, 3))/known.sum((1, 2, 3)).clamp_min(1)).mean()
    world_targets, world_eligible = mode_targets({"means": means.detach()}, batch)
    world_ce = (F.cross_entropy(outputs["logits"][world_eligible].float(), world_targets[world_eligible])
                if world_eligible.any() else outputs["logits"].float().sum()*0)
    loss = nll+.1*trajectory+.2*risk+.01*offroad+outputs.get("shared_world_ce_weight", 0)*world_ce
    return {"loss": loss, "nll": nll, "trajectory": trajectory, "risk": risk, "offroad": offroad,
            "shared_world_ce": world_ce, "world_supervision_count": world_eligible.sum(),
            "risk_supervision_count": eligible.sum()}


def constant_velocity(batch):
    history = batch["history"]
    b, a, _, _ = history.shape
    indices = (batch["history_mask"]*torch.arange(50, device=history.device)).amax(-1)
    last = history.gather(2, indices[..., None, None].expand(b, a, 1, 8)).squeeze(2)
    times = torch.arange(1, 61, device=history.device, dtype=history.dtype)*.1
    means = (last[:, :, None, :2]+last[:, :, None, 2:4]*times[None, None, :, None])[:, None].repeat(1, 6, 1, 1, 1)
    separation = torch.linalg.vector_norm(means[:, 0, :, None]-means[:, 0, None, :], dim=-1)
    pairs = batch["dynamic_mask"][:, :, None] & batch["dynamic_mask"][:, None, :]
    pairs &= ~torch.eye(a, device=history.device, dtype=torch.bool)[None]
    distance = separation.masked_fill(~pairs[..., None], 1000).amin((-1, -2))
    logits = history.new_full((b, 6), -30)
    logits[:, 0] = 0
    return {"means": means, "scales": torch.ones_like(means), "logits": logits,
            "risk_logits": (3-distance).clamp(-15, 15)}
