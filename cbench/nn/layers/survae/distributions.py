
import torch
from survae.distributions import Distribution
from survae.distributions import DiagonalNormal
from survae.distributions import ConditionalDistribution
from survae.utils import sum_except_batch
from survae.transforms import Softplus
from .transforms import integer_to_base


class BinaryEncoder(ConditionalDistribution):
    '''An encoder for BinaryProductArgmaxSurjection.'''

    def __init__(self, noise_dist, dims):
        super(BinaryEncoder, self).__init__()
        self.noise_dist = noise_dist
        self.dims = dims
        self.softplus = Softplus()

    def sample_with_log_prob(self, context):
        # Example: context.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        binary = integer_to_base(context, base=2, dims=self.dims)
        sign = binary * 2 - 1

        u, log_pu = self.noise_dist.sample_with_log_prob(context=context)
        u_positive, ldj = self.softplus(u)

        log_pu_positive = log_pu - ldj
        z = u_positive * sign

        log_pz = log_pu_positive
        return z, log_pz


class ConvNormal1d(DiagonalNormal):
    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        assert len(shape) == 2
        self.shape = torch.Size(shape)
        self.loc = torch.nn.Parameter(torch.zeros(1, shape[0], 1))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, shape[0], 1))


class StandardGumbel(Distribution):
    """A standard Gumbel distribution."""

    def __init__(self, shape):
        super(StandardGumbel, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        return sum_except_batch(- x - (-x).exp())

    def sample(self, num_samples):
        u = torch.rand(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)
        eps = torch.finfo(u.dtype).tiny # 1.18e-38 for float32
        return -torch.log(-torch.log(u + eps) + eps)