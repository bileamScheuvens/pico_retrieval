import torch


def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(
        mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device
    )

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples
