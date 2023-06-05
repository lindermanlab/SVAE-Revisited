import copy
from pprint import pprint
from copy import deepcopy

# for logging
import wandb
# Jax
import jax
import jax.numpy as np
import jax.random as jr
key_0 = jr.PRNGKey(0) # Convenience
# optax
import optax as opt

# Tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from svae.posteriors import DKFPosterior, CDKFPosterior, PlaNetPosterior, LDSSVAEPosterior
from svae.priors import LinearGaussianChainPrior, LieParameterizedLinearGaussianChainPrior
from svae.networks import GaussianRecognition, GaussianBiRNN, TemporalConv, \
    GaussianEmission, GaussianDCNNEmission, GaussianDCNNEmissionFixedCovariance, \
    PlaNetRecognitionWrapper
from svae.training import Trainer, experiment_scheduler, svae_pendulum_val_loss, svae_init, svae_loss, svae_update
from svae.svae import DeepLDS
from svae.datasets import sample_lds_dataset, load_nlb, load_pendulum
from svae.logging import summarize_pendulum_run, save_params_to_wandb, log_to_wandb, validation_log_to_wandb, on_error

networks = {
    "GaussianRecognition": GaussianRecognition,
    "GaussianBiRNN": GaussianBiRNN,
    "TemporalConv": TemporalConv,
    "GaussianEmission": GaussianEmission,
    "GaussianDCNNEmission": GaussianDCNNEmission,
    "GaussianDCNNEmissionFixedCovariance": GaussianDCNNEmissionFixedCovariance,
}

def init_model(run_params, data_dict):
    p = deepcopy(run_params)
    d = p["dataset_params"]
    latent_dims = p["latent_dims"]
    input_shape = data_dict["train_data"].shape[1:]
    num_timesteps = input_shape[0]
    data = data_dict["train_data"]
    seed = p["seed"]
    seed_model, seed_elbo, seed_ems, seed_rec = jr.split(seed, 4)

    run_type = p["run_type"]
    recnet_class = networks[p["recnet_class"]]
    decnet_class = networks[p["decnet_class"]]

    if p["inference_method"] == "dkf":
        posterior = DKFPosterior(latent_dims, num_timesteps)
    elif p["inference_method"] in ["cdkf", "conv"]:
        posterior = CDKFPosterior(latent_dims, num_timesteps)
    elif p["inference_method"] == "planet":
        posterior = PlaNetPosterior(p["posterior_architecture"],
                                    latent_dims, num_timesteps)
    elif p["inference_method"] == "svae":
        # The parallel Kalman stuff only applies to SVAE
        # Since RNN based methods are inherently sequential
        posterior = LDSSVAEPosterior(latent_dims, num_timesteps, 
                                     use_parallel=p.get("use_parallel_kf"))
        
    rec_net = recnet_class.from_params(**p["recnet_architecture"])
    dec_net = decnet_class.from_params(**p["decnet_architecture"])
    if p["inference_method"] == "planet":
        # Wrap the recognition network
        rec_net = PlaNetRecognitionWrapper(rec_net)

    if (p.get("use_natural_grad")):
        prior = LinearGaussianChainPrior(latent_dims, num_timesteps)
    else:
        prior = LieParameterizedLinearGaussianChainPrior(latent_dims, num_timesteps, 
                    init_dynamics_noise_scale=p.get("init_dynamics_noise_scale") or 1)

    model = DeepLDS(
        recognition=rec_net,
        decoder=dec_net,
        prior=prior,
        posterior=posterior,
        input_dummy=np.zeros(input_shape),
        latent_dummy=np.zeros((num_timesteps, latent_dims))
    )

    initial_params = None
    svae_val_loss = svae_pendulum_val_loss if run_params["dataset"] == "pendulum" else svae_loss

    # Define the trainer object here
    trainer = Trainer(model, train_params=run_params, init=svae_init, 
                      loss=svae_loss, 
                      val_loss=svae_val_loss, 
                      update=svae_update, initial_params=initial_params)

    return {
        # We don't actually need to include model here
        # 'cause it's included in the trainer object
        "model": model,
        # "emission_params": emission_params
        "trainer": trainer
    }

def start_trainer(model_dict, data_dict, run_params):
    trainer = model_dict["trainer"]
    if run_params.get("log_to_wandb"):
        if run_params["dataset"] == "pendulum":
            summary = summarize_pendulum_run
        else:
            summary = save_params_to_wandb
    else:
        summary = None
    trainer.train(data_dict,
                  max_iters=run_params["max_iters"],
                  key=run_params["seed"],
                  callback=log_to_wandb, val_callback=validation_log_to_wandb,
                  summary=summary)
    return (trainer.model, trainer.params, trainer.train_losses)

linear_recnet_architecture = {
    "diagonal_covariance": False,
    "input_rank": 1,
    "head_mean_params": { "features": [] },
    "head_var_params": { "features": [] },
    "eps": 1e-4,
    "cov_init": 1,
}

BiRNN_recnet_architecture = {
    "input_rank": 1,
    "input_type": "MLP",
    "input_params":{ "features": [20,] },
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "head_dyn_params": { "features": [20,] },
    "eps": 1e-4,
    "cov_init": 1,
}

BiRNN_recnet_architecture_32 = {
    "input_rank": 1,
    "input_type": "MLP",
    "input_params":{ "features": [64,] },
    "head_mean_params": { "features": [64, 64] },
    "head_var_params": { "features": [64, 64] },
    "head_dyn_params": { "features": [64,] },
    "eps": 1e-4,
    "cov_init": 1,
}

BiRNN_recnet_architecture_64 = {
    "input_rank": 1,
    "input_type": "MLP",
    "input_params":{ "features": [128,] },
    "head_mean_params": { "features": [128, 128] },
    "head_var_params": { "features": [128, 128] },
    "head_dyn_params": { "features": [128,] },
    "eps": 1e-4,
    "cov_init": 1,
}

# 1d convolution on the time dimension
temporal_conv_layers = [
            {"features": 32, "kernel_size": (10,), "strides": (1,),},
            {"features": 32, "kernel_size": (10,), "strides": (1,),},
            {"features": 32, "kernel_size": (10,), "strides": (1,),},
]

conv_recnet_architecture = {
    "input_rank": 1,
    "cnn_params": {
        "layer_params": temporal_conv_layers
    },
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "head_dyn_params": { "features": [20,] },
    "eps": 1e-4,
    "cov_init": 1,
}

planet_posterior_architecture = {
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "eps": 1e-4,
    "cov_init": 1,
}

linear_decnet_architecture = {
    "diagonal_covariance": True,
    "input_rank": 1,
    "head_mean_params": { "features": [] },
    "head_var_params": { "features": [] },
    "eps": 1e-4,
    "cov_init": 1,
}

CNN_layers = [
            {"features": 32, "kernel_size": (3, 3), "strides": (1, 1) },
            {"features": 64, "kernel_size": (3, 3), "strides": (2, 2) },
            {"features": 32, "kernel_size": (3, 3), "strides": (2, 2) }
]

CNN_recnet_architecture = {
    "input_rank": 3,
    "trunk_type": "CNN",
    "trunk_params": {
        "layer_params": CNN_layers
    },
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "eps": 1e-4,
    "cov_init": 1,
}

CNN_conv_recnet_architecture = {
    "input_rank": 3,
    "input_type": "CNN",
    "input_params":{
        "layer_params": CNN_layers
    },
    "cnn_params": {
        "layer_params": temporal_conv_layers
    },
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "head_dyn_params": { "features": [20,] },
    "eps": 1e-4,
    "cov_init": 1,
}

CNN_BiRNN_recnet_architecture = {
    "input_rank": 3,
    "input_type": "CNN",
    "input_params":{
        "layer_params": CNN_layers
    },
    "head_mean_params": { "features": [20, 20] },
    "head_var_params": { "features": [20, 20] },
    "head_dyn_params": { "features": [20,] },
    "eps": 1e-4,
    "cov_init": 1,
}

DCNN_decnet_architecture = {
    "input_shape": (6, 6, 32),
    "layer_params": [
        { "features": 64, "kernel_size": (3, 3), "strides": (2, 2) },
        { "features": 32, "kernel_size": (3, 3), "strides": (2, 2) },
        { "features": 2, "kernel_size": (3, 3) }
    ],
    "eps": 1e-4,
}

# @title Run parameter expanders
def get_lr(params, max_iters):
    base_lr = params["base_lr"]
    prior_base_lr = params["prior_base_lr"]
    lr = opt.constant_schedule(base_lr)
    prior_lr = opt.constant_schedule(prior_base_lr)
    pprint(params)
    if params["lr_decay"]:
        print("Using learning rate decay!")
        lr = opt.exponential_decay(init_value=base_lr, 
                                     transition_steps=max_iters,
                                     decay_rate=0.99, 
                                   transition_begin=.8*max_iters, staircase=False)
        # This is kind of a different scheme but whatever...
        if params["prior_lr_warmup"]:
            prior_lr = opt.cosine_onecycle_schedule(max_iters, prior_base_lr, 0.5)
    else:
        if params["prior_lr_warmup"]: 
            prior_lr = opt.linear_schedule(0, prior_base_lr, .2 * max_iters, 0)
    # Always use learning rate warm-up for stability in initial training
    warmup_end = 200
    lr_warmup = opt.linear_schedule(0, base_lr, warmup_end, 0)
    lr = opt.join_schedules([lr_warmup, lr], [warmup_end])
    return lr, prior_lr

def get_beta_schedule(params, max_iters):
    if params.get("beta_schedule"):
        if params["beta_schedule"] == "linear_fast":
            return opt.linear_schedule(0., 1., 1000, 0)
        elif params["beta_schedule"] == "linear_slow":
            return opt.linear_schedule(0., 1., 5000, 1000)
        else:
            print("Beta schedule undefined! Using constant instead.")
            return lambda _: 1.0
    else:
        return lambda _: 1.0

def expand_lds_parameters(params):
    num_timesteps = params.get("num_timesteps") or 200
    train_trials = { "small": 10, "medium": 100, "large": 1000 }
    batch_sizes = {"small": 10, "medium": 10, "large": 10 }
    emission_noises = { "small": 10., "medium": 1., "large": .1 }
    dynamics_noises = { "small": 0.01, "medium": .1, "large": .1 }
    latent_dims = { "small": 3, "medium": 5, "large": 10, "32": 32, "64": 64}
    emission_dims = { "small": 5, "medium": 10, "large": 20, "32": 64, "64": 128}
    max_iters = 20000

    # Modify all the architectures according to the parameters given
    # D, H, N = params["latent_dims"], params["rnn_dims"], params["emission_dims"]
    D, H, N = latent_dims[params["dimensionality"]], params["rnn_dims"], emission_dims[params["dimensionality"]]
    inf_params = {}
    if (params["inference_method"] == "svae"):
        inf_params["recnet_class"] = "GaussianRecognition"
        architecture = deepcopy(linear_recnet_architecture)
        architecture["output_dim"] = D
        architecture["diagonal_covariance"] = True if params.get("diagonal_covariance") else False
    elif (params["inference_method"] in ["dkf", "cdkf"]):
        inf_params["recnet_class"] = "GaussianBiRNN"
        architectures = {
            "small": BiRNN_recnet_architecture,
            "medium": BiRNN_recnet_architecture,
            "large": BiRNN_recnet_architecture,
            "32": BiRNN_recnet_architecture_32,
            "64": BiRNN_recnet_architecture_64,
        }
        architecture = deepcopy(architectures[params["dimensionality"]])
        architecture["output_dim"] = D
        architecture["rnn_dim"] = H
    elif (params["inference_method"] == "planet"):
        # Here we're considering the filtering setting as default
        # Most likely we're not even going to use this so it should be fine
        # If we want smoothing then we can probably just replace these with the
        # BiRNN version
        inf_params["recnet_class"] = "GaussianBiRNN"
        architecture = deepcopy(BiRNN_recnet_architecture)
        architecture["output_dim"] = H
        architecture["rnn_dim"] = H
        post_arch = deepcopy(planet_posterior_architecture)
        post_arch["input_dim"] = H
        post_arch["rnn_dim"] = H
        post_arch["output_dim"] = D
        inf_params["posterior_architecture"] = post_arch
        inf_params["sample_kl"] = True # PlaNet doesn't have built-in suff-stats
    elif (params["inference_method"] in ["conv"]):
        inf_params["recnet_class"] = "TemporalConv"
        architecture = deepcopy(conv_recnet_architecture)
        architecture["output_dim"] = D
        # The output heads will output distributions in D dimensional space
        architecture["cnn_params"]["output_dim"] = H

        # Change the convolution kernel size
        kernel_size = params.get("conv_kernel_size") or "medium"
        kernel_sizes = { "small": 10, "medium": 20, "large": 50 }
        size = kernel_sizes[kernel_size]

        for layer in architecture["cnn_params"]["layer_params"]:
            layer["kernel_size"] = (size,)
    else:
        print("Inference method not found: " + params["inference_method"])
        assert(False)
    decnet_architecture = deepcopy(linear_decnet_architecture)
    decnet_architecture["output_dim"] = N
    inf_params["decnet_class"] = "GaussianEmission"
    inf_params["decnet_architecture"] = decnet_architecture
    inf_params["recnet_architecture"] = architecture

    lr, prior_lr = get_lr(params, max_iters)

    extended_params = {
        "project_name": "SVAE-LDS-ICML-RE-1",
        "log_to_wandb": True,
        "dataset": "lds",
        # We're just doing model learning since we're lazy
        "run_type": "model_learning",
        "dataset_params": {
            "seed": key_0,
            "num_trials": train_trials[params["dataset_size"]],
            "num_timesteps": num_timesteps,
            "emission_cov": emission_noises[params["snr"]],
            "dynamics_cov": dynamics_noises[params["snr"]],
            "latent_dims": D,
            "emission_dims": N,
        },
        # Implementation choice
        "use_parallel_kf": True,
        # Training specifics
        "max_iters": max_iters,
        "elbo_samples": 1,
        "sample_kl": False,
        "batch_size": batch_sizes[params["dataset_size"]],
        "record_params": lambda i: i % 1000 == 0,
        "plot_interval": 100,
        "learning_rate": lr, 
        "prior_learning_rate": prior_lr,
        "use_validation": True,
        # Note that we do not specify this in the high-level parameters!
        "latent_dims": D,
        "lr_warmup": True
    }
    extended_params.update(inf_params)
    # This allows us to override ANY of the above...!
    extended_params.update(params)
    return extended_params

def expand_pendulum_parameters(params):
    train_trials = { "small": 20, "medium": 100, "large": 2000 }
    batch_sizes = {"small": 10, "medium": 10, "large": 40 }
    # Not a very good validation split (mostly because we're doing one full batch for val)
    val_trials = { "small": 4, "medium": 20, "large": 200 }
    noise_scales = { "small": 1., "medium": .1, "large": .01 }
    max_iters = 30000

    # Modify all the architectures according to the parameters given
    D, H = params["latent_dims"], params["rnn_dims"]
    inf_params = {}
    if (params["inference_method"] == "svae"):
        inf_params["recnet_class"] = "GaussianRecognition"
        architecture = deepcopy(CNN_recnet_architecture)
        architecture["trunk_params"]["output_dim"] = D
        architecture["output_dim"] = D
    elif (params["inference_method"] in ["dkf", "cdkf"]):
        inf_params["recnet_class"] = "GaussianBiRNN"
        architecture = deepcopy(CNN_BiRNN_recnet_architecture)
        architecture["output_dim"] = D
        architecture["rnn_dim"] = H
        architecture["input_params"]["output_dim"] = H
    elif (params["inference_method"] in ["conv"]):
        inf_params["recnet_class"] = "TemporalConv"
        architecture = deepcopy(CNN_conv_recnet_architecture)
        architecture["output_dim"] = D
        # The output heads will output distributions in D dimensional space
        architecture["cnn_params"]["output_dim"] = H
        architecture["input_params"]["output_dim"] = H
        # Change the convolution kernel size
        kernel_size = params.get("conv_kernel_size") or "medium"
        kernel_sizes = { "small": 10, "medium": 20, "large": 50 }
        size = kernel_sizes[kernel_size]

        for layer in architecture["cnn_params"]["layer_params"]:
            layer["kernel_size"] = (size,)
    elif (params["inference_method"] == "planet"):
        # Here we're considering the filtering setting as default
        # Most likely we're not even going to use this so it should be fine
        # If we want smoothing then we can probably just replace these with the
        # BiRNN version
        inf_params["recnet_class"] = "GaussianBiRNN"
        architecture = deepcopy(CNN_BiRNN_recnet_architecture)
        architecture["output_dim"] = H
        architecture["rnn_dim"] = H
        architecture["input_params"]["output_dim"] = H
        post_arch = deepcopy(planet_posterior_architecture)
        post_arch["input_dim"] = H
        post_arch["rnn_dim"] = H
        post_arch["output_dim"] = D
        inf_params["posterior_architecture"] = post_arch
        inf_params["sample_kl"] = True # PlaNet doesn't have built-in suff-stats
    else:
        print("Inference method not found: " + params["inference_method"])
        assert(False)

    decnet_architecture = deepcopy(DCNN_decnet_architecture)

    if (params.get("learn_output_covariance")):
        decnet_class = "GaussianDCNNEmission"
    else:
        decnet_class = "GaussianDCNNEmissionFixedCovariance"
        # Use the known data variance
        decnet_architecture["output_noise_scale"] = noise_scales[params["snr"]] ** 0.5

    inf_params["recnet_architecture"] = architecture
    lr, prior_lr = get_lr(params, max_iters)

    extended_params = {
        "project_name": "SVAE-Pendulum-ICML-5",
        "log_to_wandb": True,
        "dataset": "pendulum",
        # Must be model learning
        "run_type": "model_learning",
        "decnet_class": decnet_class,
        "decnet_architecture": decnet_architecture,
        "dataset_params": {
            "seed": key_0,
            "train_trials": train_trials[params["dataset_size"]],
            "val_trials": val_trials[params["dataset_size"]],
            "emission_cov": noise_scales[params["snr"]]
        },
        # Implementation choice
        "use_parallel_kf": True,
        # Training specifics
        "max_iters": max_iters,
        "elbo_samples": 1,
        "sample_kl": False,
        "batch_size": batch_sizes[params["dataset_size"]],
        "record_params": lambda i: i % 1000 == 0,
        "plot_interval": 200,
        "mask_type": "potential" if params["inference_method"] == "svae" else "data",
        "learning_rate": lr, 
        "prior_learning_rate": prior_lr,
        "use_validation": True,
        "constrain_dynamics": True,
        # Pendulum specific
        "prediction_horizon": 5,
        "learn_output_covariance": False,
        "lr_warmup": True
    }
    extended_params.update(inf_params)
    # This allows us to override ANY of the above...!
    extended_params.update(params)
    return extended_params

def expand_nlb_parameters(params):
    train_trials = { "small": 20, "medium": 100, "large": 1720 }
    batch_sizes = {"small": 10, "medium": 10, "large": 10 }
    # Not a very good validation split (mostly because we're doing one full batch for val)
    val_trials = { "small": 4, "medium": 20, "large": 570 }
    max_iters = 20000

    # Modify all the architectures according to the parameters given
    D = params["latent_dims"]
    H = D * 2 # Set rnn dims to twice the latent dims for enough capacity
    N = 45 # Number of output neurons

    inf_params = {}
    input_architecture = { "features": [128, 128, 64, 64] }

    if (params["inference_method"] == "svae"):
        inf_params["recnet_class"] = "GaussianRecognition"
        architecture = {
            "diagonal_covariance": True,
            "input_rank": 1,
            "trunk_type": "MLP",
            "trunk_params": input_architecture,
            "head_mean_params": { "features": [64] },
            "head_var_params": { "features": [64] },
            "eps": 1e-4,
            "cov_init": 1,
        }
        architecture["output_dim"] = D
    elif (params["inference_method"] in ["dkf", "cdkf"]):
        inf_params["recnet_class"] = "GaussianBiRNN"
        architecture = {
            "diagonal_covariance": True,
            "input_rank": 1,
            "input_type": "MLP",
            "input_params": input_architecture,
            "head_mean_params": { "features": [64,] },
            "head_var_params": { "features": [64,] },
            "head_dyn_params": { "features": [64,] },
            "eps": 1e-4,
            "cov_init": 1,
        }
        architecture["output_dim"] = D
        architecture["rnn_dim"] = H
    else:
        print("Inference method not found: " + params["inference_method"])
        assert(False)
    
    if (params["run_type"] == "lds_baseline"):
        architecture = deepcopy(linear_recnet_architecture)
        architecture["output_dim"] = D
        architecture["diagonal_covariance"] = True if params.get("diagonal_covariance") else False
        dec_features = []
    else:
        dec_features = [64, 64, 64]

    # Fix the decoder architecture
    decnet_architecture = {
        "diagonal_covariance": True,
        "input_rank": 1,
        "head_mean_params": { "features": dec_features },
        "head_var_params": { "features": [] },
        "output_dim": N,
        "eps": 1e-6,
        "cov_init": 1,
    }

    inf_params["recnet_architecture"] = copy.deepcopy(architecture)
    lr, prior_lr = get_lr(params, max_iters)

    extended_params = {
        "project_name": "SVAE-NLB-ICML-1",
        "log_to_wandb": True,
        "dataset": "nlb",
        # Must be model learning
        "run_type": "model_learning",
        "decnet_class": "PoissonEmissions",
        "decnet_architecture": decnet_architecture,
        "dataset_params": {
            "train_trials": train_trials[params["dataset_size"]],
            "val_trials": val_trials[params["dataset_size"]],
        },
        # Implementation choice
        "use_parallel_kf": False,
        # Training specifics
        "max_iters": max_iters,
        "elbo_samples": 1,
        "sample_kl": True,
        "batch_size": batch_sizes[params["dataset_size"]],
        "record_params": lambda i: i % 1000 == 0,
        "plot_interval": 200,
        "learning_rate": lr, 
        "prior_learning_rate": prior_lr,
        "use_validation": True,
        "constrain_dynamics": True,
        "beta": get_beta_schedule(params, max_iters),
        "init_dynamics_noise_scale": 1e-4,
        "lr_warmup": True
    }
    extended_params.update(inf_params)
    # This allows us to override ANY of the above...!
    extended_params.update(params)
    return extended_params

def run_lds(run_params, run_variations=None):
    jax.config.update("jax_debug_nans", True)
    load_lds = sample_lds_dataset

    results = experiment_scheduler(run_params, 
                     run_variations=run_variations,
                     dataset_getter=load_lds, 
                     model_getter=init_model, 
                     train_func=start_trainer,
                     params_expander=expand_lds_parameters,
                     on_error=on_error)
    wandb.finish()
    return results

def run_pendulum(run_params, run_variations=None):
    jax.config.update("jax_debug_nans", True)
    results = experiment_scheduler(run_params, 
                     run_variations=run_variations,
                     dataset_getter=load_pendulum, 
                     model_getter=init_model, 
                     train_func=start_trainer,
                     params_expander=expand_pendulum_parameters,
                     on_error=on_error)
    wandb.finish()
    return results

def run_nlb(run_params, run_variations=None):
    jax.config.update("jax_debug_nans", True)
    results = experiment_scheduler(run_params, 
                     run_variations=run_variations,
                     dataset_getter=load_nlb, 
                     model_getter=init_model, 
                     train_func=start_trainer,
                     params_expander=expand_nlb_parameters,
                     on_error=on_error)
    wandb.finish()
    return results