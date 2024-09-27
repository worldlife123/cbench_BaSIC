# https://raw.githubusercontent.com/vsimkus/torch-reparametrised-mixture-distribution/main/mixture_same_family.py

from numpy import prod
import warnings

import torch
import torch.distributions as distr
import torch.nn.functional as F
from torch.autograd.functional import jacobian
# from torch.distributions.distribution import MixtureSameFamily
from torch.distributions.normal import Normal

from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from typing import Dict


class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Copied from PyTorch 1.8, so that it can be used with earlier PyTorch versions.

    Examples::

        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
                     torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = \
            self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution = \
            self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(batch_shape=batch_shape,
                                               event_shape=event_shape,
                                               validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when support contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'


class ReparametrizedMixtureSameFamily(MixtureSameFamily):
    """
    Adds rsample method to the MixtureSameFamily class
    that implements implicit reparametrization.
    """
    has_rsample = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self._component_distribution.has_rsample:
            raise ValueError('Cannot reparameterize a mixture of non-reparameterizable components.')

        # NOTE: Not necessary for implicit reparametrisation.
        if not callable(getattr(self._component_distribution, '_log_cdf', None)):
            warnings.warn(message=('The component distributions do not have numerically stable '
                                   '`_log_cdf`, will use torch.log(cdf) instead, which may not '
                                   'be stable. NOTE: this will not affect implicit reparametrisation.'),
                        )

    def rsample(self, sample_shape=torch.Size()):
        """Adds reparameterization (pathwise) gradients to samples of the mixture.
        
        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L433-L498

        Implicit reparameterization gradients are

        .. math::
            dx/dphi = -(d transform(x, phi) / dx)^-1 * d transform(x, phi) / dphi,

        where transform(x, phi) is distributional transform that removes all
        parameters from samples x.

        We implement them by replacing x with
        -stop_gradient(d transform(x, phi) / dx)^-1 * transform(x, phi)]
        for the backward pass (gradient computation).
        The derivative of this quantity w.r.t. phi is then the implicit
        reparameterization gradient.
        Note that this replaces the gradients w.r.t. both the mixture
        distribution parameters and components distributions parameters.

        Limitations:
        1. Fundamental: components must be fully reparameterized.
        2. Distributional transform is currently only implemented for
            factorized components.

        Args:
            x: Sample of mixture distribution

        Returns:
            Tensor with same value as x, but with reparameterization gradients
        """
        x = self.sample(sample_shape=sample_shape)
        if not torch.is_grad_enabled():
            return x

        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            x_2d_shape = (-1, event_size)

            # Perform distributional transform of x in [S, B, E] shape,
            # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
            def reshaped_distributional_transform(x_2d):
                return torch.reshape(
                        self._distributional_transform(x_2d.reshape(x.shape)),
                        x_2d_shape)
                
            x_2d = x.reshape(x_2d_shape)
            
            # Compute transform (the gradients of this transform will be computed using autodiff)
            # transform_2d: [S*prod(B), prod(E)]
            transform_2d = reshaped_distributional_transform(x_2d)
            
            # Compute the Jacobian of the distributional transform
            def batched_jacobian_of_reshaped_distributional_transform(x_2d):
                # Used to compute the batched Jacobian for a function that takes a (B, N) and produces (B, M). 
                # NOTE: the function must be independent for each element in B. Otherwise, this would be incorrect.
                # See: https://pytorch.org/functorch/1.13/notebooks/jacobians_hessians.html#batch-jacobian-and-batch-hessian
                def reshaped_distributional_transform_summed(x_2d):
                    return torch.sum(
                            reshaped_distributional_transform(x_2d),
                            dim=0)
                return jacobian(reshaped_distributional_transform_summed, x_2d).detach().movedim(1, 0)
            # jacobian: [S*prod(B), prod(E), prod(E)]
            jac = batched_jacobian_of_reshaped_distributional_transform(x_2d)

            # We only provide the first derivative; the second derivative computed by
            # autodiff would be incorrect, so we raise an error if it is requested.
            # TODO: prevent 2nd derivative of transform_2d.

            # Compute [- stop_gradient(jacobian)^-1 * transform] by solving a linear
            # system. The Jacobian is lower triangular because the distributional
            # transform for i-th event dimension does not depend on the next
            # dimensions.
            surrogate_x_2d = -torch.triangular_solve(transform_2d[..., None], jac.detach(), upper=False)[0]
            surrogate_x = surrogate_x_2d.reshape(x.shape)
        else:
            # For univariate distributions the Jacobian/derivative of the transformation is the
            # density, so the computation is much cheaper.
            transform = self._distributional_transform(x)
            log_prob_x = self.log_prob(x)
            
            if self._event_ndims > 1:
                log_prob_x = log_prob_x.reshape(log_prob_x.shape + (1,)*self._event_ndims)

            surrogate_x = -transform*torch.exp(-log_prob_x.detach())

        # Replace gradients of x with gradients of surrogate_x, but keep the value.
        return x + (surrogate_x - surrogate_x.detach())

    def _distributional_transform(self, x):
        """Performs distributional transform of the mixture samples.

        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L500-L574

        Distributional transform removes the parameters from samples of a
        multivariate distribution by applying conditional CDFs:

        .. math::
            (F(x_1), F(x_2 | x1_), ..., F(x_d | x_1, ..., x_d-1))

        (the indexing is over the 'flattened' event dimensions).
        The result is a sample of product of Uniform[0, 1] distributions.

        We assume that the components are factorized, so the conditional CDFs become

        .. math::
          `F(x_i | x_1, ..., x_i-1) = sum_k w_i^k F_k (x_i),`

        where :math:`w_i^k` is the posterior mixture weight: for :math:`i > 0`
        :math:`w_i^k = w_k prob_k(x_1, ..., x_i-1) / sum_k' w_k' prob_k'(x_1, ..., x_i-1)`
        and :math:`w_0^k = w_k` is the mixture probability of the k-th component.

        Args:
            x: Sample of mixture distribution

        Returns:
            Result of the distributional transform
        """
        # Obtain factorized components distribution and assert that it's
        # a scalar distribution.
        if isinstance(self._component_distribution, distr.Independent):
            univariate_components = self._component_distribution.base_dist
        else:
            univariate_components = self._component_distribution

        # Expand input tensor and compute log-probs in each component
        x = self._pad(x)  # [S, B, 1, E]
        # NOTE: Only works with fully-factorised distributions!
        log_prob_x = univariate_components.log_prob(x)  # [S, B, K, E]
        
        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            # Compute exclusive cumulative sum
            # log prob_k (x_1, ..., x_i-1)
            cumsum_log_prob_x = log_prob_x.reshape(-1, event_size)  # [S*prod(B)*K, prod(E)]
            cumsum_log_prob_x = torch.cumsum(cumsum_log_prob_x, dim=-1)
            cumsum_log_prob_x = cumsum_log_prob_x.roll(1, -1)
            cumsum_log_prob_x[:, 0] = 0
            cumsum_log_prob_x = cumsum_log_prob_x.reshape(log_prob_x.shape)

            logits_mix_prob = self._pad_mixture_dimensions(self._mixture_distribution.logits)

            # Logits of the posterior weights: log w_k + log prob_k (x_1, ..., x_i-1)
            log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x
            
            # Normalise posterior weights
            component_axis = -self._event_ndims-1
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=component_axis)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)
        else:
            # For univariate distributions logits of the posterior weights = log w_k
            log_posterior_weights_x = self._mixture_distribution.logits
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=-1)
            posterior_weights_x = self._pad_mixture_dimensions(posterior_weights_x)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            component_axis = -self._event_ndims-1
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)


    def _log_cdf(self, x):
        x = self._pad(x)
        if callable(getattr(self._component_distribution, '_log_cdf', None)):
            log_cdf_x = self.component_distribution._log_cdf(x)
        else:
            # NOTE: This may be unstable
            log_cdf_x = torch.log(self.component_distribution.cdf(x))

        if isinstance(self.component_distribution, (distr.Bernoulli, distr.Binomial, distr.ContinuousBernoulli, 
                                                    distr.Geometric, distr.NegativeBinomial, distr.RelaxedBernoulli)):
            log_mix_prob = torch.sigmoid(self.mixture_distribution.logits)
        else:
            log_mix_prob = F.log_softmax(self.mixture_distribution.logits, dim=-1)

        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)


class StableNormal(Normal):
    """
    Add stable cdf for implicit reparametrization, and stable _log_cdf.
    """
    
    # Override default
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ndtr(self._standardise(value))
    
    # NOTE: This is not necessary for implicit reparam.
    def _log_cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return log_ndtr(self._standardise(value))
    
    def _standardise(self, x):
        return (x - self.loc) * self.scale.reciprocal()

#
# Below are based on the investigation in https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188
# and implementations in SciPy and Tensorflow Probability
#

def ndtr(value: torch.Tensor):
    """
    Standard Gaussian cumulative distribution function.
    Based on the SciPy implementation of ndtr
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L201-L224
    """
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=value.dtype))
    x = value * sqrt_half
    z = torch.abs(x)
    y = 0.5 * torch.erfc(z)
    output = torch.where(z < sqrt_half,
                        0.5 + 0.5 * torch.erf(x),
                        torch.where(x > 0, 1 - y, y))
    return output


# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.

def log_ndtr(value: torch.Tensor):
    """
    Standard Gaussian log-cumulative distribution function.
    This is based on the TFP and SciPy implementations.
    https://github.com/tensorflow/probability/blame/master/tensorflow_probability/python/internal/special_math.py#L156-L245
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    dtype = value.dtype
    if dtype == torch.float64:
        lower, upper = LOGNDTR_FLOAT64_LOWER, LOGNDTR_FLOAT64_UPPER
    elif dtype == torch.float32:
        lower, upper = LOGNDTR_FLOAT32_LOWER, LOGNDTR_FLOAT32_UPPER
    else:
        raise TypeError(f'dtype={value.dtype} is not supported.')
    
    # When x < lower, then we perform a fixed series expansion (asymptotic)
    # = log(cdf(x)) = log(1 - cdf(-x)) = log(1 / 2 * erfc(-x / sqrt(2)))
    # = log(-1 / sqrt(2 * pi) * exp(-x ** 2 / 2) / x * (1 + sum))
    # When x >= lower and x <= upper, then we simply perform log(cdf(x))
    # When x > upper, then we use the approximation log(cdf(x)) = log(1 - cdf(-x)) \approx -cdf(-x)
    # The above approximation comes from Taylor expansion of log(1 - y) = -y - y^2/2 - y^3/3 - y^4/4 ...
    # So for a small y the polynomial terms are even smaller and negligible.
    # And we know that for x > upper, y = cdf(x) will be very small.
    return torch.where(value > upper,
                       -ndtr(-value),
                       torch.where(value >= lower,
                                   torch.log(ndtr(value)),
                                   log_ndtr_series(value)))

def log_ndtr_series(value: torch.Tensor, num_terms=3):
    """
    Function to compute the asymptotic series expansion of the log of normal CDF
    at value.
    This is based on the SciPy implementation.
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    # sum = sum_{n=1}^{num_terms} (-1)^{n} (2n - 1)!! / x^{2n}))
    value_sq = value ** 2
    t1 = -0.5 * (np.log(2 * np.pi) + value_sq) - torch.log(-value)
    t2 = torch.zeros_like(value)
    value_even_power = value_sq.clone()
    double_fac = 1
    multiplier = -1
    for n in range(1, num_terms + 1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(value_sq)
        double_fac *= (2 * n - 1)
        multiplier *= -1
    return t1 + torch.log1p(t2)