import wandb
import matplotlib.pyplot as plt

from jax import vmap
import jax.numpy as np
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from jax.numpy.linalg import eigh, svd
from PIL import Image

from pprint import pprint
from copy import deepcopy

import pickle as pkl

import numpy as onp

from svae.distributions import LinearGaussianChain
from svae.utils import lie_params_to_constrained

def visualize_pendulum(trainer, aux):
    # This assumes single sequence has shape (100, 24, 24, 1)
    recon = aux["reconstruction"][0][0]
    # Show the sequence as a block of images
    stacked = recon.reshape(10, 24 * 10, 24)
    imgrid = stacked.swapaxes(0, 1).reshape(24 * 10, 24 * 10)
    recon_img = wandb.Image(onp.array(imgrid), caption="Sample Reconstruction")

    fig = plt.figure()
    mask = aux["mask"][0]
    post_sample = aux["posterior_samples"][0][0]
    top, bot = np.max(post_sample) + 5, np.min(post_sample) - 5
    left, right = 0, post_sample.shape[0]
    plt.imshow(mask[None], cmap="gray", alpha=.4, vmin=0, vmax=1,
               extent=(left, right, top, bot))
    plt.plot(post_sample)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    post_img = wandb.Image(img, caption="Posterior Sample")
    plt.close()
    return {
        "Reconstruction": recon_img, 
        "Posterior Sample": post_img
    }

def get_group_name(run_params):
    p = run_params
    run_type = "" if p["inference_method"] in ["EM", "GT", "SMC"] else "_" + p["run_type"]
    is_diag = "_diag" if p.get("diagonal_covariance") else ""
    if p["dataset"] == "pendulum":
        dataset_summary = "pendulum"
    elif p["dataset"] == "lds":
        d = p["dataset_params"]
        dataset_summary = "lds_dims_{}_{}_noises_{}_{}".format(
            d["latent_dims"], d["emission_dims"], 
            d["dynamics_cov"], d["emission_cov"])
    elif p["dataset"] == "nlb":
        dataset_summary = "nlb"
    else:
        dataset_summary = "???"

    model_summary = "_{}d_latent_".format(p["latent_dims"]) + p["inference_method"]

    group_tag = p.get("group_tag") or ""
    if group_tag != "": group_tag += "_"

    group_name = (group_tag +
        dataset_summary
        + model_summary
        + run_type + is_diag
    )
    return group_name

def validation_log_to_wandb(trainer, loss_out, data_dict):
    p = trainer.train_params
    if not p.get("log_to_wandb"): return
    
    project_name = p["project_name"]
    group_name = get_group_name(p)

    obj, aux = loss_out
    elbo = -obj
    kl = np.mean(aux["kl"])
    ell = np.mean(aux["ell"])

    model = trainer.model
    prior = model.prior
    D = prior.latent_dims
    prior_params = trainer.params["prior_params"]

    visualizations = {}
    if p["dataset"] == "pendulum":
        visualizations = visualize_pendulum(trainer, aux)
        pred_ll = np.mean(aux["prediction_ll"])
        visualizations = {
            "Validation reconstruction": visualizations["Reconstruction"], 
            "Validation posterior sample": visualizations["Posterior Sample"],
            "Validation prediction log likelihood": pred_ll
        }
        
    to_log = {"Validation ELBO": elbo, "Validation KL": kl, "Validation likelihood": ell,}
    to_log.update(visualizations)

    if (p["dataset"] == "nlb"): 
        to_log["Validation BPS"] = compute_bps(aux, data_dict["val_targets"])
    
    wandb.log(to_log)

def compute_bps(out, targets):
    eps = 1e-8
    mean_rate = np.nanmean(targets, axis=(0,1), keepdims=True)
    baseline = tfd.Poisson(rate=mean_rate+eps).log_prob(targets)
    num_spikes = np.sum(targets)
    return (np.mean(out["ell"]) * targets.size - np.sum(baseline)) / num_spikes / np.log(2)

def log_to_wandb(trainer, loss_out, data_dict, grads):
    p = trainer.train_params
    if not p.get("log_to_wandb"): return
    
    project_name = p["project_name"]
    group_name = get_group_name(p)

    itr = len(trainer.train_losses) - 1
    if len(trainer.train_losses) == 1:
        wandb.init(project=project_name, group=group_name, config=p,    
            dir="/scratch/users/yixiuz/")
        pprint(p)

    obj, aux = loss_out
    elbo = -obj
    kl = np.mean(aux["kl"])
    ell = np.mean(aux["ell"])

    model = trainer.model
    prior = model.prior
    D = prior.latent_dims
    prior_params = trainer.params["prior_params"]
    if (p.get("use_natural_grad")):
        Q = prior_params["Q"]
        A = prior_params["A"]
    else:
        Q = lie_params_to_constrained(prior_params["Q"], D)
        A = prior_params["A"]

    eigs = eigh(Q)[0]
    Q_cond_num = np.max(eigs) / np.min(eigs)
    svs = svd(A)[1]
    max_sv, min_sv = np.max(svs), np.min(svs)
    
    if min_sv != 0:
        A_cond_num = max_sv / min_sv
    else:
        A_cond_num = -1

    # Also log the prior params gradients
    # prior_grads = grads["prior_params"]["sgd_params"]
    # prior_grads_norm = np.linalg.norm(
    #     jax.tree_util.tree_leaves(tree_map(np.linalg.norm, prior_grads)))

    visualizations = {}
    if (itr % p["plot_interval"] == 0):
        # We have deprecated visualization of the LDS
        if p["dataset"] == "pendulum" and p.get("visualize_training"):
            visualizations = visualize_pendulum(trainer, aux)

        fig = plt.figure()
        prior_sample = prior.sample(prior_params, shape=(1,), key=jr.PRNGKey(0))[0]
        plt.plot(prior_sample)
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        prior_img = wandb.Image(img, caption="Prior Sample")
        plt.close()
        visualizations["Prior sample"] = prior_img
    # Also log the learning rates
    lr = p["learning_rate"] 
    lr = lr if isinstance(lr, float) else lr(itr)
    prior_lr = p["prior_learning_rate"] 
    prior_lr = prior_lr if isinstance(prior_lr, float) else prior_lr(itr)

    to_log = { "ELBO": elbo, "KL": kl, "Likelihood": ell, # "Prior graident norm": prior_grads_norm,
               "Max singular value of A": max_sv, "Min singular value of A": min_sv,
               "Condition number of A": A_cond_num, "Condition number of Q": Q_cond_num,
               "Learning rate": lr, "Prior learning rate": prior_lr }
    to_log.update(visualizations)

    if (p["dataset"] == "nlb"): 
        to_log["Train BPS"] = compute_bps(aux, data_dict["train_targets"])
    if (p.get("beta")):
        to_log["Beta"] = p["beta"](itr)

    wandb.log(to_log)

def save_params_to_wandb(trainer, data_dict):
    file_name = "parameters.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(trainer.params, f)
        wandb.save(file_name, policy="now")

def on_error(data_dict, model_dict):
    save_params_to_wandb(model_dict["trainer"])

# Pendulum run summaries

def predict_multiple(run_params, model_params, model, data, T, key, num_samples=100):
    """
    Returns:
        Posterior mean (T, D)
        Posterior covariance (T, D, D)
        Predictions (N, T//2, D)
        Expected prediction log likelihood (lower bound of MLL) (T//2, D)
    """
    seq_len = model.prior.seq_len
    D = model.prior.latent_dims
    model.prior.seq_len = T
    model.posterior.seq_len = T

    run_params = deepcopy(run_params)
    run_params["mask_size"] = 0

    out = model.elbo(key, data[:T], model_params, **run_params)
    post_params = out["posterior_params"]
    posterior = model.posterior.distribution(post_params)
    
    # Get the final mean and covariance
    post_mean, post_covariance = posterior.mean, posterior.covariance
    mu, Sigma = post_mean[T-1], post_covariance[T-1]
    # Build the posterior object on the future latent states 
    # ("the posterior predictive distribution")
    # Convert unconstrained params to constrained dynamics parameters
    dynamics = model.prior.get_dynamics_params(model_params["prior_params"])
    pred_posterior = LinearGaussianChain.from_stationary_dynamics(
        mu, Sigma, dynamics["A"], dynamics["b"], dynamics["Q"], seq_len-T+1) # Note the +1

    # Sample from it and evaluate the log likelihood
    x_preds = pred_posterior.sample(seed=key, sample_shape=(num_samples,))

    def pred_ll(x_pred):
        likelihood_dist = model.decoder.apply(model_params["dec_params"], x_pred)
        return likelihood_dist.log_prob(data[T:])

    pred_lls = vmap(pred_ll)(x_preds[:,1:])
    # This assumes the pendulum dataset
    # Which has 3d observations (width, height, channels)
    pred_lls = pred_lls.sum(axis=(2, 3, 4))
    
    # Revert the model sequence length
    model.prior.seq_len = seq_len
    model.posterior.seq_len = seq_len

    # Keep only the first 10 samples
    return post_mean, post_covariance, x_preds, pred_lls

def get_latents_and_predictions(run_params, model_params, model, data_dict):
    
    key = jr.PRNGKey(42)

    # Try to decode true states linearly from model encodings
    num_predictions = 200
    num_examples = 20
    num_frames = 100

    def encode(data):
        out = model.elbo(jr.PRNGKey(0), data, model_params, **run_params)
        post_params = out["posterior_params"]
        post_dist = model.posterior.distribution(post_params)
        return post_dist.mean, post_dist.covariance

    train_data = data_dict["train_data"][:,:num_frames]
    Ex, _ = vmap(encode)(train_data)
    # Figure out the linear regression weights which decodes true states
    states = data_dict["train_states"][:]
    # states = targets[:,::2] # We subsampled the data during training to make pendulum swing faster
    # Compute the true angles and angular velocities
    train_thetas = np.arctan2(states[:,:,0], states[:,:,1])
    train_omegas = train_thetas[:,1:]-train_thetas[:,:-1]
    thetas = train_thetas.flatten()
    omegas = train_omegas.flatten()
    # Fit the learned representations to the true states
    D = model.prior.latent_dims
    xs_theta = Ex.reshape((-1, D))
    xs_omega = Ex[:,1:].reshape((-1, D))
    W_theta, _, _, _ = np.linalg.lstsq(xs_theta, thetas)
    W_omega, _, _, _ = np.linalg.lstsq(xs_omega, omegas)

    # Evaluate mse on test data
    test_states = data_dict["val_states"][:, :num_frames]
    test_data = data_dict["val_data"][:, :num_frames]
    thetas = np.arctan2(test_states[:,:,0], test_states[:,:,1])
    omegas = thetas[:,1:]-thetas[:,:-1]
    test_mean, test_cov = vmap(encode)(test_data)
    pred_thetas = np.einsum("i,...i->...", W_theta, test_mean)
    theta_mse = np.mean((pred_thetas - thetas) ** 2)
    pred_omegas = np.einsum("i,...i->...", W_omega, test_mean[:,1:])
    omega_mse = np.mean((pred_omegas - omegas) ** 2)

    partial_mean = []
    partial_cov = []
    all_preds = []
    all_pred_lls = []
    for i in range(num_examples):
        print("Generating samples for test example:", i)
        mean, cov, preds, pred_lls = vmap(predict_multiple,
            in_axes=(None, None, None, 0, None, None, None))\
            (run_params, model_params, model,
             data_dict["val_data"][i:i+1,:num_frames], num_frames//2, key, num_predictions)
        key = jr.split(key)[0]
        partial_mean.append(mean)
        partial_cov.append(cov)
        all_preds.append(preds)
        all_pred_lls.append(pred_lls)

    return {
        "latent_mean": test_mean,
        "latent_covariance": test_cov,
        "latent_mean_partial": np.concatenate(partial_mean, axis=0),
        "latent_covariance_partial": np.concatenate(partial_cov, axis=0),
        "prediction_lls": np.concatenate(all_pred_lls, axis=0),
        "predictions": np.concatenate(all_preds, axis=0),
        "w_theta": W_theta,
        "w_omega": W_omega,
        "theta_mse": theta_mse,
        "omega_mse": omega_mse,
    }

def summarize_pendulum_run(trainer, data_dict):
    file_name = "parameters.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(trainer.past_params, f)
        wandb.save(file_name, policy="now")
    # Compute predictions on test set
    # Set the mask size to 0 for summary!!
    run_params = deepcopy(trainer.train_params)
    run_params["mask_size"] = 0
    results = get_latents_and_predictions(
        run_params, trainer.params, trainer.model, data_dict)
    file_name = "results.npy"
    np.save(file_name, results, allow_pickle=True)
    wandb.save(file_name, policy="now")