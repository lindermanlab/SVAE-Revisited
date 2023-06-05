import copy
from copy import deepcopy

import jax.numpy as np
import jax.random as jr
key_0 = jr.PRNGKey(0)

# Tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from svae.utils import random_rotation, inv_softplus, lie_params_to_constrained
from svae.distributions import LinearGaussianChain
from svae.utils import dynamics_to_tridiag

class SVAEPrior:
    def init(self, key):
        """
        Returns the initial prior parameters.
        """
        pass

    def distribution(self, prior_params):
        """
        Returns a tfp distribution object
        Takes constrained params
        """
        pass

    def m_step(self, prior_params, posterior, post_params):
        """
        Returns updated prior parameters.
        """
        pass
    
    def sample(self, params, shape, key):
        return self.distribution(
            self.get_constrained_params(params)).sample(sample_shape=shape, seed=key)

    def get_constrained_params(self, params):
        return deepcopy(params)

    @property
    def shape(self):
        raise NotImplementedError

class LinearGaussianChainPrior(SVAEPrior):

    def __init__(self, latent_dims, seq_len):
        self.latent_dims = latent_dims
        # The only annoying thing is that we have to specify the sequence length
        # ahead of time
        self.seq_len = seq_len

    @property
    def shape(self):
        return (self.seq_len, self.latent_dims)

    # Must be the full set of constrained parameters!
    def distribution(self, params):
        As, bs, Qs = params["As"], params["bs"], params["Qs"]
        Ex, ExxT, ExnxT = params["Ex"], params["ExxT"], params["ExnxT"]
        return LinearGaussianChain(As, bs, Qs, Ex, ExxT, ExnxT)

    def init(self, key):
        T, D = self.seq_len, self.latent_dims
        key_A, key = jr.split(key, 2)
        params = {
            "m1": np.zeros(D),
            "Q1": np.eye(D),
            "A": random_rotation(key_A, D, theta=np.pi/20),
            "b": np.zeros(D),
            "Q": np.eye(D)
        }
        constrained = self.get_constrained_params(params)
        return params

    def get_dynamics_params(self, params):
        return params

    def get_constrained_params(self, params):
        p = copy.deepcopy(params)
        tridiag = dynamics_to_tridiag(params, self.seq_len, self.latent_dims)
        p.update(tridiag)
        dist = LinearGaussianChain.from_stationary_dynamics(p["m1"], p["Q1"], 
                                         p["A"], p["b"], p["Q"], self.seq_len)
        p.update({
            "As": dist._dynamics_matrix,
            "bs": dist._dynamics_bias,
            "Qs": dist._noise_covariance,
            "Ex": dist.expected_states,
            "ExxT": dist.expected_states_squared,
            "ExnxT": dist.expected_states_next_states
        })
        return p

class LieParameterizedLinearGaussianChainPrior(LinearGaussianChainPrior):

    def __init__(self, latent_dims, seq_len, init_dynamics_noise_scale=1):
        super().__init__(latent_dims, seq_len)
        self.init_dynamics_noise_scale = init_dynamics_noise_scale

    def init(self, key):
        D = self.latent_dims
        key_A, key = jr.split(key, 2)
        # Equivalent to the unit matrix
        eps = min(self.init_dynamics_noise_scale / 100, 1e-4)
        Q_flat = np.concatenate([np.ones(D) 
            * inv_softplus(self.init_dynamics_noise_scale, eps=eps), np.zeros((D*(D-1)//2))])
        Q1_flat = np.concatenate([np.ones(D) * inv_softplus(1), np.zeros((D*(D-1)//2))])
        params = {
            "m1": np.zeros(D),
            "A": random_rotation(key_A, D, theta=np.pi/20),
            "Q1": Q1_flat,
            "b": np.zeros(D),
            "Q": Q_flat
        }
        return params

    def get_dynamics_params(self, params):
        return {
            "m1": params["m1"],
            "Q1": lie_params_to_constrained(params["Q1"], self.latent_dims),
            "A": params["A"],
            "b": params["b"],
            "Q": lie_params_to_constrained(params["Q"], self.latent_dims)   
        }

    def get_constrained_params(self, params):
        D = self.latent_dims
        p = self.get_dynamics_params(params)
        return super().get_constrained_params(p)