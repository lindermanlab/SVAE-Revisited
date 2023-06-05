from jax import numpy as np
from jax import random as jr

import copy

from svae.priors import SVAEPrior, LinearGaussianChainPrior
from svae.distributions import ParallelLinearGaussianSSM, LinearGaussianSSM, LinearGaussianChain
from svae.utils import random_rotation

class LDSSVAEPosterior(SVAEPrior):
    def __init__(self, latent_dims, seq_len, use_parallel=False):
        self.latent_dims = latent_dims
        # The only annoying thing is that we have to specify the sequence length
        # ahead of time
        self.seq_len = seq_len
        self.dist = ParallelLinearGaussianSSM if use_parallel else LinearGaussianSSM

    @property
    def shape(self):
        return (self.seq_len, self.latent_dims)

    # Must be the full set of constrained parameters!
    def distribution(self, p):
        m1, Q1, A, b, Q, mus, Sigmas = p["m1"], p["Q1"], p["A"], p["b"], p["Q"], p["mu"], p["Sigma"]
        log_Z, mu_filtered, Sigma_filtered = p["log_Z"], p["mu_filtered"], p["Sigma_filtered"]
        mu_smoothed, Sigma_smoothed, ExnxT = p["mu_smoothed"], p["Sigma_smoothed"], p["ExnxT"]
        return self.dist(m1, Q1, A, b, Q, mus, Sigmas, 
                             log_Z, mu_filtered, Sigma_filtered, 
                             mu_smoothed, Sigma_smoothed, ExnxT)

    def init(self, key):
        T, D = self.seq_len, self.latent_dims
        key_A, key = jr.split(key, 2)
        p = {
            "m1": np.zeros(D),
            "Q1": np.eye(D),
            "A": random_rotation(key_A, D, theta=np.pi/20),
            "b": np.zeros(D),
            "Q": np.eye(D),
            "Sigma": np.tile(np.eye(D)[None], (T, 1, 1)),
            "mu": np.zeros((T, D))
        }

        dist = self.dist.infer_from_dynamics_and_potential(p, 
                                    {"mu": p["mu"], "Sigma": p["Sigma"]})
        
        p.update({
            "log_Z": dist.log_normalizer,
            "mu_filtered": dist.filtered_means,
            "Sigma_filtered": dist.filtered_covariances,
            "mu_smoothed": dist.smoothed_means,
            "Sigma_smoothed": dist.smoothed_covariances,
            "ExnxT": dist.expected_states_next_states
        })
        return p

    def get_dynamics_params(self, params):
        return params

    def infer(self, prior_params, potential_params):
        p = {
            "m1": prior_params["m1"],
            "Q1": prior_params["Q1"],
            "A": prior_params["A"],
            "b": prior_params["b"],
            "Q": prior_params["Q"],
            "Sigma": potential_params["Sigma"],
            "mu": potential_params["mu"]
        }

        dist = self.dist.infer_from_dynamics_and_potential(prior_params, 
                                    {"mu": p["mu"], "Sigma": p["Sigma"]})
        p.update({
            "log_Z": dist.log_normalizer,
            "mu_filtered": dist.filtered_means,
            "Sigma_filtered": dist.filtered_covariances,
            "mu_smoothed": dist.smoothed_means,
            "Sigma_smoothed": dist.smoothed_covariances,
            "ExnxT": dist.expected_states_next_states
        })
        return p

    def get_constrained_params(self, params):
        p = copy.deepcopy(params)
        return p

class CDKFPosterior(LinearGaussianChainPrior):
    def init(self, key):
        T, D = self.seq_len, self.latent_dims
        params = {
            "As": np.zeros((T, D, D)), 
            "bs": np.zeros((T, D)),
            "Qs": np.tile(np.eye(D)[None], (T, 1, 1))
        }
        return self.get_constrained_params(params)

    def get_constrained_params(self, params):
        p = copy.deepcopy(params)
        dist = LinearGaussianChain.from_nonstationary_dynamics(p["As"], p["bs"], p["Qs"])
        p.update({
            "Ex": dist.expected_states,
            "ExxT": dist.expected_states_squared,
            "ExnxT": dist.expected_states_next_states
        })
        return p

    def sufficient_statistics(self, params):
        return {
            "Ex": params["Ex"],
            "ExxT": params["ExxT"],
            "ExnxT": params["ExnxT"]
        }

    def infer(self, prior_params, posterior_params):
        return self.get_constrained_params(posterior_params)

class DKFPosterior(CDKFPosterior):
    def get_constrained_params(self, params):
        p = copy.deepcopy(params)
        # The DKF produces a factored posterior\n",
        # So the dynamics matrix is zeroed out\n",
        p["As"] *= 0
        dist = LinearGaussianChain.from_nonstationary_dynamics(p["As"], p["bs"], p["Qs"])
        p.update({
            "Ex": dist.expected_states,
            "ExxT": dist.expected_states_squared,
            "ExnxT": dist.expected_states_next_states
        })
        return p

# The infer function for the DKF version just uses the posterior params 
class PlaNetPosterior(DKFPosterior):
    def __init__(self, network_params, latent_dims, seq_len):
        super().__init__(latent_dims, seq_len)
        self.network = StochasticRNNCell.from_params(**network_params)
        self.input_dim = network_params["input_dim"]      # u
        self.latent_dim = network_params["rnn_dim"]       # h
        self.output_dim = network_params["output_dim"]    # x

    def init(self, key):
        input_dummy = np.zeros((self.input_dim,))
        latent_dummy = np.zeros((self.latent_dim,))
        output_dummy = np.zeros((self.output_dim,))
        rnn_params = self.network.init(key, input_dummy, latent_dummy, output_dummy)
        return {
            "network_input": np.zeros((self.seq_len, self.input_dim)),
            "network_params": {
                "rnn_params": rnn_params,
                "input_dummy": input_dummy,
                "latent_dummy": latent_dummy,
                "output_dummy": output_dummy,
            }
        }

    def get_constrained_params(self, params):
        # All of the information is stored in the second argument already
        return params

    def distribution(self, params):
        return DeepAutoregressiveDynamics(self.network, params)
        
    def infer(self, prior_params, posterior_params):
        return self.get_constrained_params(posterior_params)

    # These are just dummies
    def sufficient_statistics(self, params):
        T, D = self.seq_len, self.latent_dims
        return {
            "Ex": np.zeros((T, D)),
            "ExxT": np.zeros((T, D, D)),
            "ExnxT": np.zeros((T-1, D, D))
        }