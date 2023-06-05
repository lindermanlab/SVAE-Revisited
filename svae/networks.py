from jax import vmap
import jax.numpy as np
import jax.random as jr
key_0 = jr.PRNGKey(0)
# Flax
import flax.linen as nn
from flax.linen import Conv, ConvTranspose

# Tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

# Common math functions
from flax.linen import softplus
from jax.numpy.linalg import inv

# For typing in neural network utils
from typing import (Any, Callable, Sequence, Iterable)

import numpy as onp

from svae.utils import lie_params_to_constrained

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

class MLP(nn.Module):
    """
    Define a simple fully connected MLP with ReLU activations.
    """
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, 
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,)(x))
        x = nn.Dense(self.features[-1], 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init)(x)
        return x

class Identity(nn.Module):
    """
    A layer which passes the input through unchanged.
    """
    features: int

    def __call__(self, inputs):
        return inputs

class Static(nn.Module):
    """
    A layer which just returns some static parameters.
    """
    features: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.features, ))
        return kernel

class CNN(nn.Module):
    """A simple CNN model."""
    input_rank : int = None   
    output_dim : int = None
    layer_params : Sequence[dict] = None

    @nn.compact
    def __call__(self, x):
        for params in self.layer_params:
            x = nn.relu(Conv(**params)(x))
        # No activations at the output
        x = nn.Dense(features=self.output_dim)(x.flatten())
        return x

class TemporalCNN(nn.Module):
    """Same as CNN, but we don't flatten the output."""
    input_rank : int = None
    output_dim : int = None
    layer_params : Sequence[dict] = None

    @nn.compact
    def __call__(self, x):
        for params in self.layer_params:
            x = nn.relu(Conv(**params)(x))
        # No activations at the output
        x = nn.Dense(features=self.output_dim)(x)
        return x

class DCNN(nn.Module):
    """A simple DCNN model."""

    input_shape: Sequence[int] = None
    layer_params: Sequence[dict] = None

    @nn.compact
    def __call__(self, x):
        input_features = onp.prod(onp.array(self.input_shape))
        x = nn.Dense(features=input_features)(x)
        x = x.reshape(self.input_shape)
        # Note that the last layer doesn't have an activation
        for params in self.layer_params:
            x = ConvTranspose(**params)(nn.relu(x))
        return x

# @title Potential networks (outputs potentials on single observations)
class PotentialNetwork(nn.Module):
    def __call__(self, inputs):
        Sigma, mu = self._generate_distribution_parameters(inputs)
        return { "Sigma": Sigma, "mu": mu }

    def _generate_distribution_parameters(self, inputs):
        if (len(inputs.shape) == self.input_rank + 2):
            # We have both a batch dimension and a time dimension
            # and we have to vmap over both...!
            return vmap(vmap(self._call_single, 0), 0)(inputs)
        elif (len(inputs.shape) == self.input_rank + 1):
            return vmap(self._call_single)(inputs)
        elif (len(inputs.shape) == self.input_rank):
            return self._call_single(inputs)
        else:
            # error
            return None

    def _call_single(self, inputs):
        pass

# A new, more general implementation of the Gaussian recognition network
# Uses mean parameterization which works better empirically
class GaussianRecognition(PotentialNetwork):

    use_diag : int = None
    input_rank : int = None
    latent_dims : int = None
    trunk_fn : nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, input_rank=1, input_dim=None, output_dim=None, 
                    trunk_type="Identity", trunk_params=None, 
                    head_mean_type="MLP", head_mean_params=None,
                    head_var_type="MLP", head_var_params=None, diagonal_covariance=False,
                    cov_init=1, eps=1e-4): 

        if trunk_type == "Identity":
            trunk_params = { "features": input_dim }
        if head_mean_type == "MLP":
            head_mean_params["features"] += [output_dim]
        if head_var_type == "MLP":
            if (diagonal_covariance):
                head_var_params["features"] += [output_dim]
            else:
                head_var_params["features"] += [output_dim * (output_dim + 1) // 2]
            head_var_params["kernel_init"] = nn.initializers.zeros
            head_var_params["bias_init"] = nn.initializers.constant(cov_init)

        trunk_fn = globals()[trunk_type](**trunk_params)
        head_mean_fn = globals()[head_mean_type](**head_mean_params)
        head_log_var_fn = globals()[head_var_type](**head_var_params)

        return cls(diagonal_covariance, input_rank, output_dim, trunk_fn, 
                   head_mean_fn, head_log_var_fn, eps)

    def _call_single(self, inputs):
        # Apply the trunk.
        trunk_output = self.trunk_fn(inputs)
        # Get the mean.
        mu = self.head_mean_fn(trunk_output)
        # Get the covariance parameters and build a full matrix from it.
        var_output_flat = self.head_log_var_fn(trunk_output)
        if self.use_diag:
            Sigma = np.diag(softplus(var_output_flat) + self.eps)
        else:
            Sigma = lie_params_to_constrained(var_output_flat, self.latent_dims, self.eps)
        # h = np.linalg.solve(Sigma, mu)
        # J = np.linalg.inv(Sigma)
        # lower diagonal blocks of precision matrix
        return (Sigma, mu)

# @title Posterior networks (outputs full posterior for entire sequence)
# Outputs Gaussian distributions for the entire sequence at once
class PosteriorNetwork(PotentialNetwork):
    def __call__(self, inputs):
        As, bs, Qs = self._generate_distribution_parameters(inputs)
        return {"As": As, "bs": bs, "Qs": Qs}

    def _generate_distribution_parameters(self, inputs):
        is_batched = (len(inputs.shape) == self.input_rank+2)
        if is_batched:
            return vmap(self._call_single, in_axes=0)(inputs)
        else:
            assert(len(inputs.shape) == self.input_rank+1)
            return self._call_single(inputs)

class GaussianBiRNN(PosteriorNetwork):
    
    use_diag : int = None
    input_rank : int = None
    rnn_dim : int = None
    output_dim : int = None
    forward_RNN : nn.Module = None
    backward_RNN : nn.Module = None
    input_fn : nn.Module = None
    trunk_fn: nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    head_dyn_fn : nn.Module = None
    eps : float = None
        
    @classmethod
    def from_params(cls, input_rank=1, cell_type=nn.GRUCell,
                    input_dim=None, rnn_dim=None, output_dim=None, 
                    input_type="MLP", input_params=None,
                    trunk_type="Identity", trunk_params=None, 
                    head_mean_type="MLP", head_mean_params=None,
                    head_var_type="MLP", head_var_params=None,
                    head_dyn_type="MLP", head_dyn_params=None,
                    diagonal_covariance=False,
                    cov_init=1, eps=1e-4): 

        forward_RNN = nn.scan(cell_type, variable_broadcast="params", 
                                             split_rngs={"params": False})()
        backward_RNN = nn.scan(cell_type, variable_broadcast="params", 
                                               split_rngs={"params": False}, reverse=True)()
        if trunk_type == "Identity":
            trunk_params = { "features": rnn_dim }
        if input_type == "MLP":
            input_params["features"] += [rnn_dim]
        if head_mean_type == "MLP":
            head_mean_params["features"] += [output_dim]
        if head_var_type == "MLP":
            if (diagonal_covariance):
                head_var_params["features"] += [output_dim]
            else:
                head_var_params["features"] += [output_dim * (output_dim + 1) // 2]
            head_var_params["kernel_init"] = nn.initializers.zeros
            head_var_params["bias_init"] = nn.initializers.constant(cov_init)
        if head_dyn_type == "MLP":
            head_dyn_params["features"] += [output_dim ** 2,]
            head_dyn_params["kernel_init"] = nn.initializers.zeros
            head_dyn_params["bias_init"] = nn.initializers.zeros

        trunk_fn = globals()[trunk_type](**trunk_params)
        input_fn = globals()[input_type](**input_params)
        head_mean_fn = globals()[head_mean_type](**head_mean_params)
        head_log_var_fn = globals()[head_var_type](**head_var_params)
        head_dyn_fn = globals()[head_dyn_type](**head_dyn_params)

        return cls(diagonal_covariance, input_rank, rnn_dim, output_dim, 
                   forward_RNN, backward_RNN, 
                   input_fn, trunk_fn, 
                   head_mean_fn, head_log_var_fn, head_dyn_fn, eps)

    # Applied the BiRNN to a single sequence of inputs
    def _call_single(self, inputs):
        output_dim = self.output_dim
        
        inputs = vmap(self.input_fn)(inputs)
        init_carry_forward = np.zeros((self.rnn_dim,))
        _, out_forward = self.forward_RNN(init_carry_forward, inputs)
        init_carry_backward = np.zeros((self.rnn_dim,))
        _, out_backward = self.backward_RNN(init_carry_backward, inputs)
        # Concatenate the forward and backward outputs
        out_combined = np.concatenate([out_forward, out_backward], axis=-1)
        
        # Get the mean.
        # vmap over the time dimension
        b = vmap(self.head_mean_fn)(out_combined)

        # Get the variance output and reshape it.
        # vmap over the time dimension
        var_output_flat = vmap(self.head_log_var_fn)(out_combined)
        if self.use_diag:
            Q = vmap(np.diag)(softplus(var_output_flat) + self.eps)
        else:
            Q = vmap(lie_params_to_constrained, in_axes=(0, None, None))\
                (var_output_flat, output_dim, self.eps)
        dynamics_flat = vmap(self.head_dyn_fn)(out_combined)
        A = dynamics_flat.reshape((-1, output_dim, output_dim))

        return (A, b, Q)

class TemporalConv(PosteriorNetwork):

    input_rank : int = None
    output_dim : int = None
    input_fn : nn.Module = None
    CNN : nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    head_dyn_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, input_rank=1,
                    input_dim=None, output_dim=None,
                    input_type="Identity", input_params=None,
                    cnn_params=None,
                    head_mean_type="MLP", head_mean_params=None,
                    head_var_type="MLP", head_var_params=None,
                    head_dyn_type="MLP", head_dyn_params=None,
                    cov_init=1, eps=1e-4):
        if input_type == "Identity":
            input_params = { "features": input_dim }
        if head_mean_type == "MLP":
            head_mean_params["features"] += [output_dim]
        if head_var_type == "MLP":
            head_var_params["features"] += [output_dim * (output_dim + 1) // 2]
            head_var_params["kernel_init"] = nn.initializers.zeros
            head_var_params["bias_init"] = nn.initializers.constant(cov_init)
        if head_dyn_type == "MLP":
            head_dyn_params["features"] += [output_dim ** 2,]
            head_dyn_params["kernel_init"] = nn.initializers.zeros
            head_dyn_params["bias_init"] = nn.initializers.zeros

        cnn = TemporalCNN(**cnn_params)
        input_fn = globals()[input_type](**input_params)
        head_mean_fn = globals()[head_mean_type](**head_mean_params)
        head_log_var_fn = globals()[head_var_type](**head_var_params)
        head_dyn_fn = globals()[head_dyn_type](**head_dyn_params)

        return cls(input_rank, output_dim, input_fn, cnn,
                   head_mean_fn, head_log_var_fn, head_dyn_fn, eps)

    # Applied the BiRNN to a single sequence of inputs
    def _call_single(self, inputs):
        output_dim = self.output_dim
        
        inputs = vmap(self.input_fn)(inputs)
        out = self.CNN(inputs)
        
        # Get the mean.
        # vmap over the time dimension
        b = vmap(self.head_mean_fn)(out)

        # Get the variance output and reshape it.
        # vmap over the time dimension
        var_output_flat = vmap(self.head_log_var_fn)(out)
        Q = vmap(lie_params_to_constrained, in_axes=(0, None, None))\
            (var_output_flat, output_dim, self.eps)
        dynamics_flat = vmap(self.head_dyn_fn)(out)
        A = dynamics_flat.reshape((-1, output_dim, output_dim))

        return (A, b, Q)

# @title Special architectures for PlaNet
class PlaNetRecognitionWrapper:
    def __init__(self, rec_net):
        self.rec_net = rec_net

    def init(self, key, *inputs):
        return self.rec_net.init(key, *inputs)
    
    def apply(self, params, x):
        return {
            "network_input": self.rec_net.apply(params["rec_params"], x)["bs"],
            "network_params": params["post_params"],
        }

class StochasticRNNCell(nn.Module):

    output_dim : int = None
    rnn_cell : nn.Module = None
    trunk_fn: nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None
        
    @classmethod
    def from_params(cls, cell_type=nn.GRUCell,
                    rnn_dim=None, output_dim=None, 
                    trunk_type="Identity", trunk_params=None, 
                    head_mean_type="MLP", head_mean_params=None,
                    head_var_type="MLP", head_var_params=None,
                    cov_init=1, eps=1e-4, **kwargs): 

        rnn_cell = cell_type()

        if trunk_type == "Identity":
            trunk_params = { "features": rnn_dim }
        if head_mean_type == "MLP":
            head_mean_params["features"] += [output_dim]
        if head_var_type == "MLP":
            head_var_params["features"] += [output_dim * (output_dim + 1) // 2]
            head_var_params["kernel_init"] = nn.initializers.zeros
            head_var_params["bias_init"] = nn.initializers.constant(cov_init)

        trunk_fn = globals()[trunk_type](**trunk_params)
        head_mean_fn = globals()[head_mean_type](**head_mean_params)
        head_log_var_fn = globals()[head_var_type](**head_var_params)

        return cls(output_dim, rnn_cell, trunk_fn, head_mean_fn, head_log_var_fn, eps)

    # h: latent state that's carried to the next
    # x: last sample
    # u: input at this timestep
    def __call__(self, h, x, u):
        h, out = self.rnn_cell(h, np.concatenate([x, u]))
        out = self.trunk_fn(out)
        mean, cov_flat = self.head_mean_fn(out), self.head_log_var_fn(out)
        cov = lie_params_to_constrained(cov_flat, self.output_dim, self.eps)
        return h, (cov, mean)

# This is largely for convenience
class GaussianEmission(GaussianRecognition):
    def __call__(self, inputs):
        J, h = self._generate_distribution_parameters(inputs)
        # TODO: inverting J is pretty bad numerically, perhaps save Cholesky instead?
        if (len(J.shape) == 3):
            Sigma = vmap(inv)(J)
            mu = np.einsum("tij,tj->ti", Sigma, h)
        elif (len(J.shape) == 2):
            Sigma = inv(J)
            mu = np.linalg.solve(J, h)
        else:
            # Error
            return None
        return tfd.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=Sigma)
        
class GaussianDCNNEmission(PotentialNetwork):

    input_rank : int = None
    network : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, **params):
        network = DCNN(input_shape=params["input_shape"], 
                       layer_params=params["layer_params"])
        eps = params.get("eps") or 1e-4
        return cls(1, network, eps)

    def __call__(self, inputs):
        out = self._generate_distribution_parameters(inputs)
        mu = out["mu"]
        # Adding a constant to prevent the model from getting too crazy
        sigma = out["sigma"] + self.eps
        return tfd.Normal(loc=mu, scale=sigma)

    def _call_single(self, x):
        out_raw = self.network(x)
        mu_raw, sigma_raw = np.split(out_raw, 2, axis=-1)
        # Get rid of the Sigmoid
        # mu = sigmoid(mu_raw)
        mu = mu_raw
        sigma = softplus(sigma_raw)
        return { "mu": mu, "sigma": sigma }

class GaussianDCNNEmissionFixedCovariance(GaussianDCNNEmission):

    input_rank : int = None
    network : nn.Module = None
    output_noise_scale : float = None
    eps : float = None

    @classmethod
    def from_params(cls, **params):
        network = DCNN(input_shape=params["input_shape"], 
                       layer_params=params["layer_params"])
        return cls(1, network, params["output_noise_scale"], 0)

    def _call_single(self, x):
        out_raw = self.network(x)
        mu_raw, sigma_raw = np.split(out_raw, 2, axis=-1)
        mu = mu_raw
        sigma = np.ones_like(mu) * self.output_noise_scale
        return { "mu": mu, "sigma": sigma }

class PoissonEmissions(GaussianRecognition):
    def __call__(self, inputs):
        _, mu = self._generate_distribution_parameters(inputs)
        # Softplus rate should be more stable than directly setting log-rate
        return tfd.Poisson(rate=softplus(mu) + self.eps)