"""Opt-in fast rollout path (`--rollout_fast`, default off; docs/user/rollout_and_analysis.md).

Same math as `MeshSimulator.predict_velocity` + `gns.graph_network.EncodeProcessDecode` on the
static rollout graph, restructured for inference:
  - normalizer statistics, the node-type one-hot and node property folded into constants once;
  - the edge encoder run once (edge features never change during a rollout);
  - each edge MLP's first layer factorized, W [x_i, x_j, e] = (x W_i)[dst] + (x W_j)[src] + e W_e,
    so the two 128-wide node blocks are multiplied per node instead of per edge;
  - edges stored as (node, slot) with each node's incoming edges in its own row (slots padded to
    the maximum in-degree, 6 on the 2D meshes): the dst term is a broadcast, and the 'add'
    aggregation a masked sum over slots, so there is no scatter, no atomics, and the order of
    summation is fixed (deterministic, and capturable in a CUDA graph under deterministic mode);
  - optional TF32 matmuls or fp16 / bf16 autocast, torch.compile, and one CUDA graph per step.
Rounding differs from the default path, and the autoregressive rollout amplifies it, so this path
is for evaluation and screening, never for the paper-parity gate.
"""
import torch
import torch_geometric.transforms as T

from meshnet.utils import datas_to_graph, NodeType

PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
_AUTOCAST = {"fp16": torch.float16, "bf16": torch.bfloat16}
_transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


class FastStep(torch.nn.Module):
    """One rollout step on a fixed graph: (velocities, ground truth at boundary nodes) -> next velocities."""

    def __init__(self, simulator, edge_index, edge_attr, node_type, node_property, dims):
        super().__init__()
        self.epd = simulator._encode_process_decode
        src, dst = edge_index
        nnodes, nedges = node_type.shape[0], dst.shape[0]
        degree = torch.bincount(dst, minlength=nnodes)
        slots = int(degree.max())
        order = torch.argsort(dst, stable=True)
        first = torch.cumsum(degree, 0) - degree
        rank = torch.arange(nedges, device=dst.device) - first[dst[order]]
        flat = dst[order] * slots + rank
        self.src = torch.zeros(nnodes * slots, dtype=src.dtype, device=src.device)
        self.src[flat] = src[order]
        self.src = self.src.view(nnodes, slots)
        valid = torch.zeros(nnodes * slots, dtype=torch.bool, device=dst.device)
        valid[flat] = True
        self.valid = valid.view(nnodes, slots, 1)
        attr = torch.zeros(nnodes * slots, edge_attr.shape[1], dtype=edge_attr.dtype, device=edge_attr.device)
        attr[flat] = edge_attr[order]
        edge_attr = attr.view(nnodes, slots, -1)
        mean = simulator._node_normalizer._mean()
        std = simulator._node_normalizer._std_with_epsilon()
        onehot = torch.nn.functional.one_hot(node_type.long().squeeze(), simulator._node_type_embedding_size)
        static = torch.cat([onehot.float(), node_property.unsqueeze(-1)], 1)
        self.v_mean, self.v_std = mean[:, :dims], std[:, :dims]
        self.static = ((static - mean[:, dims:]) / std[:, dims:]).contiguous()
        self.out_mean = simulator._output_normalizer._mean()
        self.out_std = simulator._output_normalizer._std_with_epsilon()
        with torch.no_grad():
            self.edge_latent = self.epd._encoder.edge_fn(edge_attr).contiguous()
        kinematic = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.HIGH_STRESS)
        self.boundary = (~kinematic).reshape(-1, 1)

    def interaction(self, layer, x, e):
        mlp, norm = layer.edge_fn[0], layer.edge_fn[1]
        w, b = mlp[0].weight, mlp[0].bias
        width, hidden = x.shape[1], w.shape[0]
        xw = x @ torch.cat([w[:, :width], w[:, width:2 * width]], 0).t()  # (nnodes, 2 * hidden)
        h = xw[:, None, :hidden] + xw[self.src, hidden:] + torch.nn.functional.linear(e, w[:, 2 * width:], b)
        for module in list(mlp)[1:]:
            h = module(h)
        h = norm(h)  # (nnodes, slots, hidden); padded slots carry junk and are masked out here
        agg = torch.where(self.valid, h, torch.zeros((), dtype=h.dtype, device=h.device)).sum(1)
        return layer.node_fn(torch.cat([agg, x], -1)) + x, h + e

    def forward(self, velocities, truth):
        x = self.epd._encoder.node_fn(torch.cat([(velocities - self.v_mean) / self.v_std, self.static], 1))
        e = self.edge_latent
        for layer in self.epd._processor.gnn_stacks:
            x, e = self.interaction(layer, x, e)
        acc = self.epd._decoder(x).float()
        return torch.where(self.boundary, truth, velocities + acc * self.out_std + self.out_mean)


def _graphed(fn, velocities, truth):
    """Capture fn once as a CUDA graph; replay it with copied-in inputs."""
    v_in, t_in = velocities.clone(), truth.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):  # warm-up (and compilation) outside the capture
            fn(v_in, t_in)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(v_in, t_in)

    def step(v, t):
        v_in.copy_(v)
        t_in.copy_(t)
        graph.replay()
        return out.clone()
    return step


def rollout_fast(simulator, features_list, nsteps, device, precision="tf32",
                 compile=True, cuda_graph=True, dt=1.0):
    """Roll out same-length trajectories (one disjoint graph); same outputs as train.rollout_batched."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    parts = []
    for features in features_list:
        node_coords, node_types, node_property, velocities, pressures, cells = (
            f.to(device) for f in features[:6])
        current = velocities[0]
        nnodes = current.shape[0]
        example = ((node_coords[0], node_types[0], node_property[0], current, pressures[0], cells[0],
                    torch.zeros(nnodes, device=device)), velocities[1])
        graph = _transformer(datas_to_graph(example, dt=dt, device=device))
        parts.append(dict(nnodes=nnodes, graph=graph, initial=velocities[:1], truth=velocities[1:],
                          current=current, node_coords=node_coords, node_types=node_types,
                          node_property=node_property))
    offsets = [0]
    for p in parts[:-1]:
        offsets.append(offsets[-1] + p['nnodes'])
    edge_index = torch.cat([p['graph'].edge_index + o for p, o in zip(parts, offsets)], dim=1)
    edge_attr = torch.cat([p['graph'].edge_attr for p in parts], dim=0)
    node_type = torch.cat([p['graph'].x[:, 0] for p in parts], dim=0)
    node_prop = torch.cat([p['graph'].x[:, 1] for p in parts], dim=0)
    truth = torch.cat([p['truth'] for p in parts], dim=1)
    velocities = torch.cat([p['current'] for p in parts], dim=0)

    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = precision == "tf32"
    # Deterministic mode NaN-fills every torch.empty, which a CUDA graph capture cannot take. Every
    # buffer here is written before it is read, so the fill changes nothing; results stay
    # deterministic (fixed-order slot sums, no atomics).
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        with torch.no_grad():
            model = FastStep(simulator, edge_index, edge_attr, node_type, node_prop, velocities.shape[1])
            fn = model.forward
            if precision in _AUTOCAST:
                plain = fn

                def fn(v, t):
                    with torch.autocast(device.type, dtype=_AUTOCAST[precision]):
                        return plain(v, t)
            if compile:
                fn = torch.compile(fn, dynamic=False)
            if cuda_graph and device.type == "cuda":
                with torch.cuda.device(device):  # capture streams live on the current device
                    fn = _graphed(fn, velocities, truth[0])
            predictions = torch.empty((nsteps,) + tuple(velocities.shape), dtype=velocities.dtype, device=device)
            for step in range(nsteps):
                velocities = fn(velocities, truth[step])
                predictions[step] = velocities
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.utils.deterministic.fill_uninitialized_memory = fill

    outputs = []
    for p, o in zip(parts, offsets):
        pred = predictions[:, o:o + p['nnodes']]
        loss = (pred - p['truth']) ** 2
        outputs.append({
            'initial_velocities': p['initial'].cpu().numpy(),
            'predicted_rollout': pred.cpu().numpy(),
            'ground_truth_rollout': p['truth'].cpu().numpy(),
            'node_coords': p['node_coords'].cpu().numpy(),
            'node_types': p['node_types'].cpu().numpy(),
            'node_property': p['node_property'].cpu().numpy(),
            'mean_loss': loss.mean().cpu().numpy(),
            'mean_acc_loss': None
        })
    return outputs
