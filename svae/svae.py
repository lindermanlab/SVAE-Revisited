from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as np
import jax.random as jr

import numpy as onp

class SVAE:
    def __init__(self,
                 recognition=None, decoder=None, prior=None, posterior=None,
                 input_dummy=None, latent_dummy=None):
        """
        rec_net, dec_net, prior are all objects that take in parameters
        rec_net.apply(params, data) returns Gaussian potentials (parameters)
        dec_net.apply(params, latents) returns probability distributions
        prior : SVAEPrior
        """
        self.recognition = recognition
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.input_dummy = input_dummy
        self.latent_dummy = latent_dummy

    def init(self, key=None):
        if key is None:
            key = jr.PRNGKey(0)

        rec_key, dec_key, prior_key, post_key = jr.split(key, 4)

        return {
            "rec_params": self.recognition.init(rec_key, self.input_dummy),
            "dec_params": self.decoder.init(dec_key, self.latent_dummy),
            "prior_params": self.prior.init(prior_key),
            "post_params": self.posterior.init(post_key)
        }

    def kl_posterior_prior(self, posterior_params, prior_params, 
                           samples=None):
        posterior = self.posterior.distribution(posterior_params)
        prior = self.prior.distribution(prior_params)
        if samples is None:
            return posterior.kl_divergence(prior)
        else:
            return np.mean(posterior.log_prob(samples) - prior.log_prob(samples))

    def elbo(self, key, data, target, # Added the new target parameter 
             model_params, sample_kl=False, **params):
        rec_params = model_params["rec_params"]
        dec_params = model_params["dec_params"]
        prior_params = self.prior.get_constrained_params(model_params["prior_params"])

        # Mask out a large window of states
        mask_size = params.get("mask_size")
        T = data.shape[0]
        D = self.prior.latent_dims

        mask = onp.ones((T,))
        key, dropout_key = jr.split(key)
        if mask_size:
            # Potential dropout...!
            # Use a trick to generate the mask without indexing with a tracer
            start_id = jr.choice(dropout_key, T - mask_size + 1)
            mask = np.array(np.arange(T) >= start_id) \
                 * np.array(np.arange(T) < start_id + mask_size)
            mask = 1 - mask
            if params.get("mask_type") == "potential":
                # This only works with svaes
                potential = self.recognition.apply(rec_params, data)
                # Uninformative potential
                infinity = 1e5
                uninf_potential = {"mu": np.zeros((T, D)), 
                                   "Sigma": np.tile(np.eye(D) * infinity, (T, 1, 1))}
                # Replace masked parts with uninformative potentials
                potential = tree_map(
                    lambda t1, t2: np.einsum("i,i...->i...", mask[:t1.shape[0]], t1) 
                                 + np.einsum("i,i...->i...", 1-mask[:t2.shape[0]], t2), 
                    potential, 
                    uninf_potential)
            else:
                potential = self.recognition.apply(rec_params, 
                                                   np.einsum("t...,t->t...", data, mask))
        else:
            # Don't do any masking
            potential = self.recognition.apply(rec_params, data)

        # Update: it makes more sense that inference is done in the posterior object
        posterior_params = self.posterior.infer(prior_params, potential)
        
        # Take samples under the posterior
        num_samples = params.get("obj_samples") or 1
        samples = self.posterior.sample(posterior_params, (num_samples,), key)
        # and compute average ll
        def likelihood_outputs(latent):
            likelihood_dist = self.decoder.apply(dec_params, latent)
            return likelihood_dist.mean(), likelihood_dist.log_prob(target)

        mean, ells = vmap(likelihood_outputs)(samples)
        # Take average over samples then sum the rest
        ell = np.sum(np.mean(ells, axis=0))
        # Compute kl from posterior to prior
        if sample_kl:
            kl = self.kl_posterior_prior(posterior_params, prior_params, 
                                         samples=samples)
        else:
            kl = self.kl_posterior_prior(posterior_params, prior_params)

        kl /= target.size
        ell /= target.size
        elbo = ell - kl

        return {
            "elbo": elbo,
            "ell": ell,
            "kl": kl,
            "posterior_params": posterior_params,
            "posterior_samples": samples,
            "reconstruction": mean,
            "mask": mask
        }

    def compute_objective(self, key, data, target, model_params, **params):
        results = self.elbo(key, data, target, model_params, **params)
        results["objective"] = results["elbo"]
        return results

class DeepLDS(SVAE):
    def kl_posterior_prior(self, posterior_params, prior_params, 
                           samples=None):
        posterior = self.posterior.distribution(posterior_params)
        prior = self.prior.distribution(prior_params)
        if samples is None:
            Ex = posterior.expected_states
            ExxT = posterior.expected_states_squared
            ExnxT = posterior.expected_states_next_states
            Sigmatt = ExxT - np.einsum("ti,tj->tij", Ex, Ex)
            Sigmatnt = ExnxT - np.einsum("ti,tj->tji", Ex[:-1], Ex[1:])

            J, L = prior_params["J"], prior_params["L"]

            cross_entropy = -prior.log_prob(Ex)
            cross_entropy += 0.5 * np.einsum("tij,tij->", J, Sigmatt) 
            cross_entropy += np.einsum("tij,tij->", L, Sigmatnt)
            return cross_entropy - posterior.entropy()
            
        else:
            return np.mean(posterior.log_prob(samples) - prior.log_prob(samples))