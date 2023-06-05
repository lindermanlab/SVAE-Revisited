import jax.numpy as np
from jax import scipy, vmap, lax
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from dynamax.utils.utils import psd_solve

def _make_associative_sampling_elements(params, key, filtered_means, filtered_covariances):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    F = params["A"]
    Q = params["Q"]

    def _last_sampling_element(key, m, P):
        return np.zeros_like(P), MVN(m, P).sample(seed=key)

    def _generic_sampling_element(params, key, m, P):

        eps = 1e-3
        P += np.eye(dims) * eps
        Pp = F @ P @ F.T + Q

        FP = F @ P
        E  = psd_solve(Pp, FP).T
        g  = m - E @ F @ m
        L  = P - E @ Pp @ E.T

        L = (L + L.T) * .5 + np.eye(dims) * eps # Add eps to the crucial covariance matrix

        h = MVN(g, L).sample(seed=key)
        return E, h

    num_timesteps = len(filtered_means)
    dims = filtered_means.shape[-1]
    keys = jr.split(key, num_timesteps)
    last_elems = _last_sampling_element(keys[-1], filtered_means[-1], 
                                        filtered_covariances[-1])
    generic_elems = vmap(_generic_sampling_element, (None, 0, 0, 0))(
        params, keys[:-1], filtered_means[:-1], filtered_covariances[:-1]
        )
    combined_elems = tuple(np.append(gen_elm, last_elm[None,:], axis=0)
                           for gen_elm, last_elm in zip(generic_elems, last_elems))
    return combined_elems

def _make_associative_filtering_elements(params, potentials):
    """Preprocess observations to construct input for filtering assocative scan."""

    F = params["A"]
    Q = params["Q"]
    Q1 = params["Q1"]
    P0 = Q1
    P1 = Q1
    m1 = params["m1"]
    dim = Q.shape[0]
    H = np.eye(dim)

    def _first_filtering_element(params, mu, Sigma):

        y, R = mu, Sigma

        S = H @ Q @ H.T + R
        CF, low = scipy.linalg.cho_factor(S)

        S1 = H @ P1 @ H.T + R
        K1 = psd_solve(S1, H @ P1).T

        A = np.zeros_like(F)
        b = m1 + K1 @ (y - H @ m1)
        C = P1 - K1 @ S1 @ K1.T
        eta = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), y)
        J = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), H @ F)

        logZ = -MVN(loc=np.zeros_like(y), covariance_matrix=H @ P0 @ H.T + R).log_prob(y)

        return A, b, C, J, eta, logZ


    def _generic_filtering_element(params, mu, Sigma):

        y, R = mu, Sigma

        S = H @ Q @ H.T + R
        CF, low = scipy.linalg.cho_factor(S)
        K = scipy.linalg.cho_solve((CF, low), H @ Q).T
        A = F - K @ H @ F
        b = K @ y
        C = Q - K @ H @ Q

        eta = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), y)
        J = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), H @ F)

        logZ = -MVN(loc=np.zeros_like(y), covariance_matrix=S).log_prob(y)

        return A, b, C, J, eta, logZ

    mus, Sigmas = potentials["mu"], potentials["Sigma"]

    first_elems = _first_filtering_element(params, mus[0], Sigmas[0])
    generic_elems = vmap(_generic_filtering_element, (None, 0, 0))(params, mus[1:], Sigmas[1:])
    combined_elems = tuple(np.concatenate((first_elm[None,...], gen_elm))
                           for first_elm, gen_elm in zip(first_elems, generic_elems))
    return combined_elems

def lgssm_filter(params, emissions):
    """A parallel version of the lgssm filtering algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """

    initial_elements = _make_associative_filtering_elements(params, emissions)

    @vmap
    def filtering_operator(elem1, elem2):
        A1, b1, C1, J1, eta1, logZ1 = elem1
        A2, b2, C2, J2, eta2, logZ2 = elem2
        dim = A1.shape[0]
        I = np.eye(dim)

        I_C1J2 = I + C1 @ J2
        temp = scipy.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = temp @ C1 @ A2.T + C2

        I_J2C1 = I + J2 @ C1
        temp = scipy.linalg.solve(I_J2C1.T, A1).T

        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = temp @ J2 @ A1 + J1

        # mu = scipy.linalg.solve(J2, eta2)
        # t2 = - eta2 @ mu + (b1 - mu) @ scipy.linalg.solve(I_J2C1, (J2 @ b1 - eta2))

        mu = np.linalg.solve(C1, b1)
        t1 = (b1 @ mu - (eta2 + mu) @ np.linalg.solve(I_C1J2, C1 @ eta2 + b1))

        logZ = (logZ1 + logZ2 + 0.5 * np.linalg.slogdet(I_C1J2)[1] + 0.5 * t1)

        return A, b, C, J, eta, logZ

    _, filtered_means, filtered_covs, _, _, logZ = lax.associative_scan(
                                                filtering_operator, initial_elements
                                                )

    return {
        "marginal_logliks": -logZ,
        "marginal_loglik": -logZ[-1],
        "filtered_means": filtered_means, 
        "filtered_covariances": filtered_covs
    }

def _make_associative_smoothing_elements(params, filtered_means, filtered_covariances):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    F = params["A"]
    Q = params["Q"]

    def _last_smoothing_element(m, P):
        return np.zeros_like(P), m, P

    def _generic_smoothing_element(params, m, P):

        Pp = F @ P @ F.T + Q

        E  = psd_solve(Pp, F @ P).T
        g  = m - E @ F @ m
        L  = P - E @ Pp @ E.T
        return E, g, L

    last_elems = _last_smoothing_element(filtered_means[-1], filtered_covariances[-1])
    generic_elems = vmap(_generic_smoothing_element, (None, 0, 0))(
        params, filtered_means[:-1], filtered_covariances[:-1]
        )
    combined_elems = tuple(np.append(gen_elm, last_elm[None,:], axis=0)
                           for gen_elm, last_elm in zip(generic_elems, last_elems))
    return combined_elems


def parallel_lgssm_smoother(params, emissions):
    """A parallel version of the lgssm smoothing algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """
    filtered_posterior = lgssm_filter(params, emissions)
    filtered_means = filtered_posterior["filtered_means"]
    filtered_covs = filtered_posterior["filtered_covariances"]
    initial_elements = _make_associative_smoothing_elements(params, filtered_means, filtered_covs)

    @vmap
    def smoothing_operator(elem1, elem2):
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2

        E = E2 @ E1
        g = E2 @ g1 + g2
        L = E2 @ L1 @ E2.T + L2

        return E, g, L

    _, smoothed_means, smoothed_covs, *_ = lax.associative_scan(
                                                smoothing_operator, initial_elements, reverse=True
                                                )
    return {
        "marginal_loglik": filtered_posterior["marginal_loglik"],
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "smoothed_means": smoothed_means,
        "smoothed_covariances": smoothed_covs
    }

def lgssm_log_normalizer(dynamics_params, mu_filtered, Sigma_filtered, potentials):
    p = dynamics_params
    Q, A, b = p["Q"][None], p["A"][None], p["b"][None]
    AT = (p["A"].T)[None]

    I = np.eye(Q.shape[-1])

    Sigma_filtered, mu_filtered = Sigma_filtered[:-1], mu_filtered[:-1]
    Sigma = Q + A @ Sigma_filtered @ AT
    mu = (A[0] @ mu_filtered.T).T + b
    # Append the first element
    Sigma_pred = np.concatenate([p["Q1"][None], Sigma])
    mu_pred = np.concatenate([p["m1"][None], mu])
    mu_rec, Sigma_rec = potentials["mu"], potentials["Sigma"]

    def log_Z_single(mu_pred, Sigma_pred, mu_rec, Sigma_rec):
        return MVN(loc=mu_pred, covariance_matrix=Sigma_pred+Sigma_rec).log_prob(mu_rec)

    log_Z = vmap(log_Z_single)(mu_pred, Sigma_pred, mu_rec, Sigma_rec)
    return np.sum(log_Z)