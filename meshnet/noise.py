import torch
from meshnet.utils import NodeType


def get_velocity_noise(graph, noise_std, device, generator=None):
    velocity_sequence = graph.x[:, 1:3]
    type = graph.x[:, 0]
    if generator is None:
        noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    else:
        noise = torch.normal(mean=0.0, std=noise_std, size=velocity_sequence.shape,
                              generator=generator).to(device)
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    return noise.to(device)
