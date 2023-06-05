from jax import lax, vmap
from jax import numpy as np
from jax import random as jr

# Tensorflow probability
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
MVN = tfd.MultivariateNormalFullCovariance

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother
from dynamax.utils.utils import psd_solve
from jax.numpy.linalg import solve

from functools import partial

from svae.inference import lgssm_log_normalizer, parallel_lgssm_smoother, _make_associative_sampling_elements
from svae.utils import dynamics_to_tridiag

class LinearGaussianChain:
    def __init__(self, dynamics_matrix, dynamics_bias, noise_covariance,
                 expected_states, expected_states_squared, expected_states_next_states):
        """
        params: dictionary containing the following keys:
            A:  (seq_len, dim, dim)
            Q:  (seq_len, dim, dim)
            b:  (seq_len, dim)
        """
        self._dynamics_matrix = dynamics_matrix
        self._dynamics_bias = dynamics_bias
        self._noise_covariance = noise_covariance
        self._expected_states = expected_states
        self._expected_states_squared = expected_states_squared
        self._expected_states_next_states = expected_states_next_states

    @classmethod
    def from_stationary_dynamics(cls, m1, Q1, A, b, Q, T):
        dynamics_matrix = np.tile(A[None], (T, 1, 1))
        dynamics_bias = np.concatenate([m1[None],
                                        np.tile(b[None], (T - 1, 1))])
        noise_covariance = np.concatenate([Q1[None],
                                           np.tile(Q[None], (T - 1, 1, 1))])
        return cls.from_nonstationary_dynamics(dynamics_matrix, dynamics_bias, noise_covariance)

    @classmethod
    def from_nonstationary_dynamics(cls, dynamics_matrix, dynamics_bias, noise_covariance):
        # Compute the means and covariances via parallel scan
        init_elems = (dynamics_matrix, dynamics_bias, noise_covariance)

        @vmap
        def assoc_op(elem1, elem2):
            A1, b1, Q1 = elem1
            A2, b2, Q2 = elem2
            return A2 @ A1, A2 @ b1 + b2, A2 @ Q1 @ A2.T + Q2

        _, Ex, covariances = lax.associative_scan(assoc_op, init_elems)
        expected_states = Ex
        expected_states_squared = covariances + np.einsum("...i,...j->...ij", Ex, Ex)
        expected_states_next_states = np.einsum("...ij,...jk->...ik",
                                                covariances[:-1], dynamics_matrix[1:]) + np.einsum("...i,...j->...ji",
                                                                                                   Ex[:-1], Ex[1:])

        return cls(dynamics_matrix, dynamics_bias, noise_covariance,
                   expected_states, expected_states_squared, expected_states_next_states)

    @property
    def mean(self):
        return self._expected_states

    @property
    def covariance(self):
        Ex = self._expected_states
        ExxT = self._expected_states_squared
        return ExxT - np.einsum("...i,...j->...ij", Ex, Ex)

    @property
    def expected_states(self):
        return self._expected_states

    @property
    def expected_states_squared(self):
        return self._expected_states_squared

    @property
    def expected_states_next_states(self):
        return self._expected_states_next_states

    # Works with batched distributions and arguments...!
    def log_prob(self, xs):

        @partial(np.vectorize, signature="(t,d,d),(t,d),(t,d,d),(t,d)->()")
        def log_prob_single(A, b, Q, x):
            ll = MVN(loc=b[0], covariance_matrix=Q[0]).log_prob(x[0])
            ll += MVN(loc=np.einsum("tij,tj->ti", A[1:], x[:-1]) + b[1:],
                      covariance_matrix=Q[1:]).log_prob(x[1:]).sum()
            return ll

        return log_prob_single(self._dynamics_matrix,
                               self._dynamics_bias,
                               self._noise_covariance, xs)

    # Only supports 0d and 1d sample shapes
    # Does not support sampling with batched object
    def sample(self, seed, sample_shape=()):

        @partial(np.vectorize, signature="(n),(t,d,d),(t,d),(t,d,d)->(t,d)")
        def sample_single(key, A, b, Q):

            biases = MVN(loc=b, covariance_matrix=Q).sample(seed=key)
            init_elems = (A, biases)

            @vmap
            def assoc_op(elem1, elem2):
                A1, b1 = elem1
                A2, b2 = elem2
                return A2 @ A1, A2 @ b1 + b2

            _, sample = lax.associative_scan(assoc_op, init_elems)
            return sample

        if (len(sample_shape) == 0):
            return sample_single(seed, self._dynamics_matrix,
                                 self._dynamics_bias,
                                 self._noise_covariance)
        elif (len(sample_shape) == 1):
            return sample_single(jr.split(seed, sample_shape[0]),
                                 self._dynamics_matrix[None],
                                 self._dynamics_bias[None],
                                 self._noise_covariance[None])
        else:
            raise Exception("More than one sample dimensions are not supported!")

    def entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ex = self.expected_states
        ExxT = self.expected_states_squared
        ExnxT = self.expected_states_next_states

        dim = Ex.shape[-1]
        Q_inv = solve(self._noise_covariance, np.eye(dim)[None])
        A = self._dynamics_matrix

        J_lower_diag = np.einsum("til,tlj->tij", -Q_inv[1:], A[1:])
        ATQinvA = np.einsum("tji,tjl,tlk->tik", A[1:], Q_inv[1:], A[1:])
        J_diag = Q_inv.at[:-1].add(ATQinvA)

        Sigmatt = ExxT - np.einsum("ti,tj->tij", Ex, Ex)
        Sigmatnt = ExnxT - np.einsum("ti,tj->tji", Ex[:-1], Ex[1:])

        trm1 = 0.5 * np.sum(J_diag * Sigmatt)
        trm2 = np.sum(J_lower_diag * Sigmatnt)

        return trm1 + trm2 - self.log_prob(Ex)

class LinearGaussianSSM(tfd.Distribution):
    def __init__(self,
                 initial_mean,
                 initial_covariance,
                 dynamics_matrix,
                 dynamics_bias,
                 dynamics_noise_covariance,
                 emissions_means,
                 emissions_covariances,
                 log_normalizer,
                 filtered_means,
                 filtered_covariances,
                 smoothed_means,
                 smoothed_covariances,
                 smoothed_cross,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="LinearGaussianSSM",
                 ) -> None:
        # Dynamics
        self._initial_mean = initial_mean
        self._initial_covariance = initial_covariance
        self._dynamics_matrix = dynamics_matrix
        self._dynamics_bias = dynamics_bias
        self._dynamics_noise_covariance = dynamics_noise_covariance
        # Emissions
        self._emissions_means = emissions_means
        self._emissions_covariances = emissions_covariances
        # Filtered
        self._log_normalizer = log_normalizer
        self._filtered_means = filtered_means
        self._filtered_covariances = filtered_covariances
        # Smoothed
        self._smoothed_means = smoothed_means
        self._smoothed_covariances = smoothed_covariances
        self._smoothed_cross = smoothed_cross

        # We would detect the dtype dynamically but that would break vmap
        # see https://github.com/tensorflow/probability/issues/1271
        dtype = np.float32
        super(LinearGaussianSSM, self).__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(initial_mean=self._initial_mean,
                            initial_covariance=self._initial_covariance,
                            dynamics_matrix=self._dynamics_matrix,
                            dynamics_bias=self._dynamics_bias,
                            dynamics_noise_covariance=self._dynamics_noise_covariance,
                            emissions_means=self._emissions_means,
                            emissions_covariances=self._emissions_covariances,
                            log_normalizer=self._log_normalizer,
                            filtered_means=self._filtered_means,
                            filtered_covariances=self._filtered_covariances,
                            smoothed_means=self._smoothed_means,
                            smoothed_covariances=self._smoothed_covariances,
                            smoothed_cross=self._smoothed_cross),
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(initial_mean=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
                    initial_covariance=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    dynamics_matrix=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    dynamics_bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
                    dynamics_noise_covariance=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    emissions_means=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    emissions_covariances=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
                    log_normalizer=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
                    filtered_means=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    filtered_covariances=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
                    smoothed_means=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
                    smoothed_covariances=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
                    smoothed_cross=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3)
                    )

    @classmethod
    def infer_from_dynamics_and_potential(cls, dynamics_params, emissions_potentials):
        p = dynamics_params
        mus, Sigmas = emissions_potentials["mu"], emissions_potentials["Sigma"]

        dim = mus.shape[-1]
        C = np.eye(dim)
        d = np.zeros(dim)

        params = make_lgssm_params(p["m1"], p["Q1"], p["A"], p["Q"], C, Sigmas,
                                   dynamics_bias=p["b"], emissions_bias=d)

        smoothed = lgssm_smoother(params, mus)._asdict()

        # Compute ExxT
        A, Q = dynamics_params["A"], dynamics_params["Q"]
        filtered_cov = smoothed["filtered_covariances"]
        filtered_mean = smoothed["smoothed_means"]
        smoothed_cov = smoothed["smoothed_covariances"]
        smoothed_mean = smoothed["smoothed_means"]
        G = vmap(lambda C: psd_solve(Q + A @ C @ A.T, A @ C).T)(filtered_cov)

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = vmap(
            lambda Gt, mean, next_mean, next_cov: Gt @ next_cov + np.outer(mean, next_mean)) \
            (G[:-1], smoothed_mean[:-1], smoothed_mean[1:], smoothed_cov[1:])

        log_Z = lgssm_log_normalizer(dynamics_params,
                                     smoothed["filtered_means"],
                                     smoothed["filtered_covariances"],
                                     emissions_potentials)

        return cls(dynamics_params["m1"],
                   dynamics_params["Q1"],
                   dynamics_params["A"],
                   dynamics_params["b"],
                   dynamics_params["Q"],
                   emissions_potentials["mu"],
                   emissions_potentials["Sigma"],
                   log_Z,  # smoothed["marginal_loglik"],
                   smoothed["filtered_means"],
                   smoothed["filtered_covariances"],
                   smoothed["smoothed_means"],
                   smoothed["smoothed_covariances"],
                   smoothed_cross)

    # Properties to get private class variables
    @property
    def log_normalizer(self):
        return self._log_normalizer

    @property
    def filtered_means(self):
        return self._filtered_means

    @property
    def filtered_covariances(self):
        return self._filtered_covariances

    @property
    def smoothed_means(self):
        return self._smoothed_means

    @property
    def smoothed_covariances(self):
        return self._smoothed_covariances

    @property
    def expected_states(self):
        return self._smoothed_means

    @property
    def expected_states_squared(self):
        Ex = self._smoothed_means
        return self._smoothed_covariances + np.einsum("...i,...j->...ij", Ex, Ex)

    @property
    def expected_states_next_states(self):
        return self._smoothed_cross

    @property
    def mean(self):
        return self.smoothed_means

    @property
    def covariance(self):
        return self.smoothed_covariances

    # TODO: currently this function does not depend on the dynamics bias
    def _log_prob(self, data, **kwargs):
        A = self._dynamics_matrix  # params["A"]
        Q = self._dynamics_noise_covariance  # params["Q"]
        Q1 = self._initial_covariance  # params["Q1"]
        m1 = self._initial_mean  # params["m1"]

        num_batch_dims = len(data.shape) - 2

        ll = np.sum(
            MVN(loc=np.einsum("ij,...tj->...ti", A, data[..., :-1, :]),
                covariance_matrix=Q).log_prob(data[..., 1:, :])
        )
        ll += MVN(loc=m1, covariance_matrix=Q1).log_prob(data[..., 0, :])

        # Add the observation potentials
        # ll += - 0.5 * np.einsum("...ti,tij,...tj->...", data, self._emissions_precisions, data) \
        #       + np.einsum("...ti,ti->...", data, self._emissions_linear_potentials)
        ll += np.sum(MVN(loc=self._emissions_means,
                         covariance_matrix=self._emissions_covariances).log_prob(data), axis=-1)
        # Add the log normalizer
        ll -= self._log_normalizer

        return ll

    def _sample_n(self, n, seed=None):

        F = self._dynamics_matrix
        b = self._dynamics_bias
        Q = self._dynamics_noise_covariance

        def sample_single(
                key,
                filtered_means,
                filtered_covariances
        ):

            initial_elements = _make_associative_sampling_elements(
                {"A": F, "b": b, "Q": Q}, key, filtered_means, filtered_covariances)

            @vmap
            def sampling_operator(elem1, elem2):
                E1, h1 = elem1
                E2, h2 = elem2

                E = E2 @ E1
                h = E2 @ h1 + h2
                return E, h

            _, sample = \
                lax.associative_scan(sampling_operator, initial_elements, reverse=True)

            return sample

        # TODO: Handle arbitrary batch shapes
        if self._filtered_covariances.ndim == 4:
            # batch mode
            samples = vmap(vmap(sample_single, in_axes=(None, 0, 0)), in_axes=(0, None, None)) \
                (jr.split(seed, n), self._filtered_means, self._filtered_covariances)
            # Transpose to be (num_samples, num_batches, num_timesteps, dim)
            # samples = np.transpose(samples, (1, 0, 2, 3))
        else:
            # non-batch mode
            samples = vmap(sample_single, in_axes=(0, None, None)) \
                (jr.split(seed, n), self._filtered_means, self._filtered_covariances)
        return samples

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ex = self.expected_states
        ExxT = self.expected_states_squared
        ExnxT = self.expected_states_next_states
        p = dynamics_to_tridiag(
            {
                "m1": self._initial_mean,
                "Q1": self._initial_covariance,
                "A": self._dynamics_matrix,
                "b": self._dynamics_bias,
                "Q": self._dynamics_noise_covariance,
            }, Ex.shape[0], Ex.shape[1]
        )
        J_diag = p["J"] + solve(self._emissions_covariances, np.eye(Ex.shape[-1])[None])
        J_lower_diag = p["L"]

        Sigmatt = ExxT - np.einsum("ti,tj->tij", Ex, Ex)
        Sigmatnt = ExnxT - np.einsum("ti,tj->tji", Ex[:-1], Ex[1:])

        entropy = 0.5 * np.sum(J_diag * Sigmatt)
        entropy += np.sum(J_lower_diag * Sigmatnt)
        return entropy - self.log_prob(Ex)


class ParallelLinearGaussianSSM(LinearGaussianSSM):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["name"] = "ParallelLinearGaussianSSM"
        super().__init__(*args, **kwargs)

    @classmethod
    def infer_from_dynamics_and_potential(cls, dynamics_params, emissions_potentials):
        # p = dynamics_params
        # mus, Sigmas = emissions_potentials["mu"], emissions_potentials["Sigma"]

        # dim = mus.shape[-1]
        # C = np.eye(dim)
        # d = np.zeros(dim)

        smoothed = parallel_lgssm_smoother(dynamics_params, emissions_potentials)

        # Compute ExxT
        A, Q = dynamics_params["A"], dynamics_params["Q"]
        filtered_cov = smoothed["filtered_covariances"]
        filtered_mean = smoothed["smoothed_means"]
        smoothed_cov = smoothed["smoothed_covariances"]
        smoothed_mean = smoothed["smoothed_means"]
        G = vmap(lambda C: psd_solve(Q + A @ C @ A.T, A @ C).T)(filtered_cov)

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = vmap(
            lambda Gt, mean, next_mean, next_cov: Gt @ next_cov + np.outer(mean, next_mean)) \
            (G[:-1], smoothed_mean[:-1], smoothed_mean[1:], smoothed_cov[1:])

        log_Z = lgssm_log_normalizer(dynamics_params,
                                     smoothed["filtered_means"],
                                     smoothed["filtered_covariances"],
                                     emissions_potentials)

        return cls(dynamics_params["m1"],
                   dynamics_params["Q1"],
                   dynamics_params["A"],
                   dynamics_params["b"],
                   dynamics_params["Q"],
                   emissions_potentials["mu"],
                   emissions_potentials["Sigma"],
                   log_Z,  # smoothed["marginal_loglik"],
                   smoothed["filtered_means"],
                   smoothed["filtered_covariances"],
                   smoothed["smoothed_means"],
                   smoothed["smoothed_covariances"],
                   smoothed_cross)