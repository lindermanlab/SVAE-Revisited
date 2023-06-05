from jax import vmap
import jax.numpy as np
from jax.numpy.linalg import solve
import jax.random as jr

from flax.core import frozen_dict as fd

import tensorflow_probability.substrates.jax.distributions as tfd

from svae.utils import random_rotation
from svae.priors import LinearGaussianChainPrior
from svae.posteriors import LDSSVAEPosterior

# TODO: get rid of this and its dependencies
data_dict = {}

# Takes a linear Gaussian chain as its base
class LDS(LinearGaussianChainPrior):
    def __init__(self, latent_dims, seq_len, base=None, posterior=None):
        super().__init__(latent_dims, seq_len)
        self.posterior = posterior or LDSSVAEPosterior(latent_dims, seq_len)
        self.base = base or LinearGaussianChainPrior(latent_dims, seq_len) # Slightly redundant...

    # Takes unconstrained params
    def sample(self, params, shape, key):
        latents = self.base.sample(params, shape, key)
        sample_shape = latents.shape[:-1]
        key, _ = jr.split(key)
        C, d, R = params["C"], params["d"], params["R"]
        obs_noise = tfd.MultivariateNormalFullCovariance(loc=d, covariance_matrix=R)\
            .sample(sample_shape=sample_shape, seed=key)
        obs = np.einsum("ij,...tj->...ti", C, latents) + obs_noise
        return latents, obs

    # Should work with any batch dimension
    def log_prob(self, params, states, data):
        latent_dist = self.base.distribution(self.base.get_constrained_params(params))
        latent_ll = latent_dist.log_prob(states)
        C, d, R = params["C"], params["d"], params["R"]
        # Gets around batch dimensions
        noise = tfd.MultivariateNormalFullCovariance(loc=d, covariance_matrix=R)
        obs_ll = noise.log_prob(data - np.einsum("ij,...tj->...ti", C, states))
        return latent_ll + obs_ll.sum(axis=-1)

    # Assumes single data points
    def e_step(self, params, data):
        # Shorthand names for parameters
        C, d, R = params["C"], params["d"], params["R"]

        J = np.dot(C.T, np.linalg.solve(R, C))
        J = np.tile(J[None, :, :], (self.seq_len, 1, 1))
        # linear potential
        h = np.dot(data - d, np.linalg.solve(R, C))

        Sigma = solve(J, np.eye(self.latent_dims)[None])
        mu = vmap(solve)(J, h)

        return self.posterior.infer(self.base.get_constrained_params(params), {"J": J, "h": h, 
                                                                    "mu": mu, "Sigma": Sigma})
        
    # Also assumes single data points
    def marginal_log_likelihood(self, params, data):
        posterior = self.posterior.distribution(self.e_step(params, data))
        states = posterior.mean
        prior_ll = self.log_prob(params, states, data)
        posterior_ll = posterior.log_prob(states)
        # This is numerically unstable!
        lps = prior_ll - posterior_ll
        return lps

def sample_lds_dataset(run_params):    
    d = run_params["dataset_params"]
    
    global data_dict
    if data_dict is not None \
        and "dataset_params" in data_dict \
        and str(data_dict["dataset_params"]) == str(fd.freeze(d)):
        print("Using existing data.")
        print("Data MLL: ", data_dict["marginal_log_likelihood"])        
        return data_dict

    data_dict = {}

    seed = d["seed"]
    emission_dims = d["emission_dims"]
    latent_dims = d["latent_dims"]
    emission_cov = d["emission_cov"]
    dynamics_cov = d["dynamics_cov"]
    num_timesteps = d["num_timesteps"]
    num_trials = d["num_trials"]
    seed_m1, seed_C, seed_d, seed_A, seed_sample = jr.split(seed, 5)

    R = emission_cov * np.eye(emission_dims)
    Q = dynamics_cov * np.eye(latent_dims)
    C = jr.normal(seed_C, shape=(emission_dims, latent_dims))
    d = jr.normal(seed_d, shape=(emission_dims,))

    # Here we let Q1 = Q
    lds = LDS(latent_dims, num_timesteps)
    
    params = {
            "m1": jr.normal(key=seed_m1, shape=(latent_dims,)),
            "Q1": Q,
            "Q": Q,
            "A": random_rotation(seed_A, latent_dims, theta=np.pi/20),
            "b": np.zeros(latent_dims),
            "R": R,
            "C": C,
            "d": d,
        }
    constrained = lds.get_constrained_params(params)
    params["avg_suff_stats"] = { "Ex": constrained["Ex"], 
                                "ExxT": constrained["ExxT"], 
                                "ExnxT": constrained["ExnxT"] }


    states, data = vmap(lambda key: lds.sample(params, shape=(), key=key))(jr.split(seed_sample, num_trials))
    
    mll = vmap(lds.marginal_log_likelihood, in_axes=(None, 0))(params, data)
    mll = np.sum(mll) / data.size
    print("Data MLL: ", mll)
    
    seed_val, _ = jr.split(seed_sample)
    val_states, val_data = lds.sample(params, 
                              shape=(num_trials,), 
                              key=seed_val)

    data_dict["generative_model"] = lds
    data_dict["marginal_log_likelihood"] = mll
    data_dict["train_data"] = data
    data_dict["train_states"] = states
    data_dict["val_data"] = val_data
    data_dict["val_states"] = val_states
    data_dict["dataset_params"] = fd.freeze(run_params["dataset_params"])
    data_dict["lds_params"] = params
    return data_dict

def load_pendulum(run_params, log=False):    
    d = run_params["dataset_params"]
    train_trials = d["train_trials"]
    val_trials = d["val_trials"]
    noise_scale = d["emission_cov"] ** 0.5
    key_train, key_val, key_pred = jr.split(d["seed"], 3)

    data = np.load("pendulum/pend_regression.npz")

    def _process_data(data, key):
        processed = data[:, ::2] / 255.0
        processed += jr.normal(key=key, shape=processed.shape) * noise_scale
        # return np.clip(processed, 0, 1)
        return processed # We are not cliping the data anymore!

    # Take subset, subsample every 2 frames, normalize to [0, 1]
    train_data = _process_data(data["train_obs"][:train_trials], key_train)
    train_states = data["train_targets"][:train_trials, ::2]
    # val_data = _process_data(data["test_obs"][:val_trials], key_val)
    data = np.load("pendulum/pend_regression_longer.npz")
    val_data = _process_data(data["test_obs"][:val_trials], key_pred)
    val_states = data["test_targets"][:val_trials, ::2]

    print("Full dataset:", data["train_obs"].shape)
    print("Subset:", train_data.shape)
    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_states": train_states,
        "val_states": val_states,
    }

def load_nlb(run_params, log=False):
    d = run_params["dataset_params"]
    train_trials = d["train_trials"]
    val_trials = d["val_trials"]

    train_data = np.load("nlb-for-yz/nlb-dsmc_maze-phase_trn-split_trn.p", allow_pickle=True)
    val_data = np.load("nlb-for-yz/nlb-dsmc_maze-phase_trn-split_val.p", allow_pickle=True)

    x_train = np.asarray(train_data.tensors[0], dtype=np.float32)
    y_train = np.asarray(train_data.tensors[1], dtype=np.float32)
    x_val = np.asarray(val_data.tensors[0], dtype=np.float32)
    y_val = np.asarray(val_data.tensors[1], dtype=np.float32)

    print("Full dataset:", x_train.shape, x_val.shape)

    x_train, y_train = x_train[:train_trials], y_train[:train_trials]
    x_val, y_val = x_val[:val_trials], y_val[:val_trials]

    print("Subset:", x_train.shape, x_val.shape)

    return {
        "train_data": x_train,
        "train_targets": y_train,
        "val_data": x_val,
        "val_targets": y_val,
    }
