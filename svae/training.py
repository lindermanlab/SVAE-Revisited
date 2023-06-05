import jax
from jax import jit, vmap
from jax.lax import scan
from jax import random as jr
from jax import numpy as np
from jax.tree_util import tree_map

import optax as opt
from copy import deepcopy

from functools import partial

from tqdm import trange

from time import time
import wandb, traceback
from pprint import pprint

from svae.logging import summarize_pendulum_run, predict_multiple, save_params_to_wandb, log_to_wandb, validation_log_to_wandb
from svae.posteriors import DKFPosterior, CDKFPosterior, LDSSVAEPosterior, PlaNetPosterior
from svae.priors import LinearGaussianChainPrior, LieParameterizedLinearGaussianChainPrior
from svae.networks import PlaNetRecognitionWrapper
from svae.utils import truncate_singular_values
from svae.svae import DeepLDS



# @title Experiment scheduler
LINE_SEP = "#" * 42

def dict_len(d):
    if (type(d) == list):
        return len(d)
    else:
        return dict_len(d[list(d.keys())[0]])

def dict_map(d, func):
    if type(d) == list:
        return func(d)
    elif type(d) == dict:
        r = deepcopy(d)
        for key in d.keys():
            r[key] = dict_map(r[key], func)
            # Ignore all the Nones
            if r[key] is None:
                r.pop(key)
        if len(r.keys()) == 0:
            # There's no content
            return None
        else:
            return r
    else:
        return None

def dict_product(d1, d2):
    l1, l2 = dict_len(d1), dict_len(d2)
    def expand_list(d):
        result = []
        for item in d:
            result.append(item)
            result.extend([None] * (l2-1))
        return result
    def multiply_list(d):
        return d * l1
    result = dict_map(d1, expand_list)
    additions = dict_map(d2, multiply_list)
    return dict_update(result, additions)

def dict_get(d, id):
    return dict_map(d, lambda l: l[id])

def dict_update(d, u):
    if d is None:
        d = dict()
    for key in u.keys():
        if type(u[key]) == dict:
            d.update({
                key: dict_update(d.get(key), u[key])
            })
        else:
            d.update({key: u[key]})
    return d

# A standardized function that structures and schedules experiments
# Can chain multiple variations of experiment parameters together
def experiment_scheduler(run_params, dataset_getter, model_getter, train_func,
                         logger_func=None, err_logger_func=None,
                         run_variations=None, params_expander=None,
                         on_error=None, continue_on_error=True, use_wandb=True):
    """
    Arguments:
        run_params: dict{"dataset_params"}
            A large dictionary containing all relevant parameters to the run
        dataset_getter: run_params -> dict{"train_data", ["generative_model"]}
            A function that loads/samples a dataset
        model_getter: run_params, data_dict -> model
            A function that creates a model given parameters. Note that the model
            could depend on the specifics of the dataset/generative model as well
        train_func: model, data, run_params -> results
            A function that contains the training loop.
            TODO: later we might wanna open up this pipeline and customize further!
        (optional) logger_func: results, run_params -> ()
            A function that logs the current run.
        (optional) err_logger_func: message, run_params -> ()
            A function that is called when the run fails.
        (optional) run_variations: dict{}
            A nested dictionary where the leaves are lists of different parameters.
            None means no change from parameters of the last run.
        (optional) params_expander: dict{} -> dict{}
            Turns high level parameters into specific low level parameters.
    returns:
        all_results: List<result>
            A list containing results from all runs. Failed runs are indicated
            with a None value.
    """
    params_expander = params_expander or (lambda d: d)

    num_runs = dict_len(run_variations) if run_variations else 1
    params = deepcopy(run_params)
    print("Total number of runs: {}".format(num_runs))
    print("Base paramerters:")
    pprint(params)

    global data_dict
    all_results = []
    all_models = []

    def _single_run(data_out, model_out):
        print("Loading dataset!")
        data_dict = dataset_getter(curr_params)
        data_out.append(data_dict)
        # Make a new model
        model_dict = model_getter(curr_params, data_dict)
        model_out.append(model_dict)
        all_models.append(model_dict)
        results = train_func(model_dict, data_dict, curr_params)
        all_results.append(results)
        if logger_func:
            logger_func(results, curr_params, data_dict)

    for run in range(num_runs):
        print(LINE_SEP)
        print("Starting run #{}".format(run))
        print(LINE_SEP)
        curr_variation = dict_get(run_variations, run)
        if curr_variation is None:
            if (run != 0):
                print("Variation #{} is a duplicate, skipping run.".format(run))
                continue
            curr_params = params_expander(params)
        else:
            print("Current parameter variation:")
            pprint(curr_variation)
            curr_params = dict_update(params, curr_variation)
            curr_params = params_expander(curr_params)
            print("Current full parameters:")
            pprint(curr_params)
            if curr_variation.get("dataset_params"):
                reload_data = True
        # Hack to get the values even when they err out
        data_out = []
        model_out = []
        if not continue_on_error:
            _single_run(data_out, model_out)
        else:
            try:
                _single_run(data_out, model_out)
                if use_wandb: wandb.finish()
            except:
                all_results.append(None)
                if (on_error):
                    try:
                        on_error(data_out[0], model_out[0])
                    except:
                        pass # Oh well...
                print("Run errored out due to some the following reason:")
                traceback.print_exc()
                if use_wandb: wandb.finish(exit_code=1)
    return all_results, all_models

class Trainer:
    """
    model: a pytree node
    loss (key, params, model, data, **train_params) -> (loss, aux)
        Returns a loss (a single float) and an auxillary output (e.g. posterior)
    init (key, model, data, **train_params) -> (params, opts)
        Returns the initial parameters and optimizers to go with those parameters
    update (params, grads, opts, model, aux, **train_params) -> (params, opts)
        Returns updated parameters, optimizers
    """
    def __init__(self, model, 
                 train_params=None, 
                 init=None, 
                 loss=None, 
                 val_loss=None,
                 update=None,
                 initial_params=None):
        # Trainer state
        self.params = initial_params
        self.model = model
        self.past_params = []
        self.time_spent = []

        if train_params is None:
            train_params = dict()

        self.train_params = train_params

        if init is not None:
            self.init = init
        if loss is not None:
            self.loss = loss

        self.val_loss = val_loss or self.loss
        if update is not None: 
            self.update = update

    def train_step(self, key, params, data, target, opt_states, itr):
        model = self.model
        results = \
            jax.value_and_grad(
                lambda params: partial(self.loss, itr=itr, **self.train_params)\
                (key, model, data, target, params), has_aux=True)(params)
        (loss, aux), grads = results
        params, opts = self.update(params, grads, self.opts, opt_states, model, aux, **self.train_params)
        return params, opts, (loss, aux), grads

    def val_step(self, key, params, data, target):
        return self.val_loss(key, self.model, data, target, params)

    def val_epoch(self, key, params, data, target):

        batch_size = self.train_params.get("batch_size") or data.shape[0]

        num_batches = data.shape[0] // batch_size

        loss_sum = 0
        aux_sum = None

        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            loss, aux = self.val_step_jitted(key, params, 
                                        data[batch_start:batch_start+batch_size], 
                                        target[batch_start:batch_start+batch_size])
            loss_sum += loss
            if aux_sum is None:
                aux_sum = aux
            else:
                aux_sum = jax.tree_map(lambda a,b: a+b, aux_sum, aux)
            key, _ = jr.split(key)
        
        loss_avg = loss_sum / num_batches
        aux_avg = jax.tree_map(lambda a: a / num_batches, aux_sum)
        return loss_avg, aux_avg

    """
    Callback: a function that takes training iterations and relevant parameter
        And logs to WandB
    """
    def train(self, data_dict, max_iters, 
              callback=None, val_callback=None, 
              summary=None, key=None,
              early_stop_start=10000, 
              max_lose_streak=2000):

        if key is None:
            key = jr.PRNGKey(0)

        model = self.model
        train_data = data_dict["train_data"]
        train_targets = data_dict.get("train_targets")
        if (train_targets is None): train_targets = train_data
        val_data = data_dict.get("val_data")
        val_targets = data_dict.get("val_targets")
        if (val_targets is None): val_targets = val_targets = val_data
        batch_size = self.train_params.get("batch_size") or train_data.shape[0]
        num_batches = train_data.shape[0] // batch_size

        init_key, key = jr.split(key, 2)

        # Initialize optimizer
        self.params, self.opts, self.opt_states = self.init(init_key, model, 
                                                       train_data[:batch_size], 
                                                       self.params,
                                                       **self.train_params)
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.past_params = []

        pbar = trange(max_iters)
        pbar.set_description("[jit compling...]")
        
        mask_start = self.train_params.get("mask_start")
        if (mask_start):
            mask_size = self.train_params["mask_size"]
            self.train_params["mask_size"] = 0

        train_step = jit(self.train_step)
        self.val_step_jitted = jit(self.val_step)

        best_loss = None
        best_itr = 0
        val_loss = None

        indices = np.arange(train_data.shape[0], dtype=int)

        for itr in pbar:
            train_key, val_key, key = jr.split(key, 3)

            batch_id = itr % num_batches
            batch_start = batch_id * batch_size

            # Uncomment this to time the execution
            # t = time()
            # Training step
            # ----------------------------------------
            batch_indices = indices[batch_start:batch_start+batch_size]
            step_results = train_step(train_key, self.params, 
                           train_data[batch_indices],
                           train_targets[batch_indices], 
                           self.opt_states, itr)
            self.params, self.opt_states, loss_out, grads = step_results#\
                # jax.tree_map(lambda x: x.block_until_ready(), step_results)
            # ----------------------------------------
            # dt = time() - t
            # self.time_spent.append(dt)

            loss, aux = loss_out
            self.train_losses.append(loss)
            pbar.set_description("LP: {:.3f}".format(loss))

            if (callback): callback(self, loss_out, data_dict, grads)

            if batch_id == num_batches - 1:
                # We're at the end of an epoch
                # We could randomly shuffle the data
                indices = jr.permutation(key, indices)
                if (self.train_params.get("use_validation")):
                    val_loss_out = self.val_epoch(val_key, self.params, val_data, val_targets)
                    if (val_callback): val_callback(self, val_loss_out, data_dict)
                    val_loss, _ = val_loss_out
                    
            if not self.train_params.get("use_validation") or val_loss is None:
                curr_loss = loss
            else:
                curr_loss = val_loss

            if itr >= early_stop_start:
                if best_loss is None or curr_loss < best_loss:
                    best_itr = itr
                    best_loss = curr_loss
                if curr_loss > best_loss and itr - best_itr > max_lose_streak:
                    print("Early stopping!")
                    break

            # Record parameters
            record_params = self.train_params.get("record_params")
            if record_params and record_params(itr):
                curr_params = deepcopy(self.params)
                curr_params["iteration"] = itr
                self.past_params.append(curr_params)

            if (mask_start and itr == mask_start):
                self.train_params["mask_size"] = mask_size
                train_step = jit(self.train_step)
                val_step = jit(self.val_step)

        if summary:
            summary(self, data_dict)

def svae_init(key, model, data, initial_params=None, **train_params):
    init_params = model.init(key)
    if (initial_params): init_params.update(initial_params)
    
    if (train_params["inference_method"] == "planet"):
        init_params["rec_params"] = {
            "rec_params": init_params["rec_params"],
            "post_params": init_params["post_params"]["network_params"]
        }
    # Expand the posterior parameters by batch size
    init_params["post_params"] = vmap(lambda _: init_params["post_params"])(data)
    init_params["post_samples"] = np.zeros((data.shape[0], 
                                            train_params.get("obj_samples") or 1) 
                                             + model.posterior.shape)
    # If we are in VAE mode, set the dynamics matrix to be 0
    if (train_params.get("run_type") == "vae_baseline"):
        A = init_params["prior_params"]["A"]
        init_params["prior_params"]["A"] = np.zeros_like(A)

    learning_rate = train_params["learning_rate"]
    rec_opt = opt.adam(learning_rate=learning_rate)
    rec_opt_state = rec_opt.init(init_params["rec_params"])
    dec_opt = opt.adam(learning_rate=learning_rate)
    dec_opt_state = dec_opt.init(init_params["dec_params"])

    if (train_params.get("use_natural_grad")):
        prior_lr = None
        prior_opt = None
        prior_opt_state = None
    else:
        # Add the option of using an gradient optimizer for prior parameters
        prior_lr = train_params.get("prior_learning_rate") or learning_rate
        prior_opt = opt.adam(learning_rate=prior_lr)
        prior_opt_state = prior_opt.init(init_params["prior_params"])

    return (init_params, 
            (rec_opt, dec_opt, prior_opt), 
            (rec_opt_state, dec_opt_state, prior_opt_state))
    
def svae_loss(key, model, data_batch, target_batch, model_params, itr=0, **train_params):
    batch_size = data_batch.shape[0]
    # Axes specification for vmap
    # We're just going to ignore this for now
    params_in_axes = None
    result = vmap(partial(model.compute_objective, **train_params), 
                  in_axes=(0, 0, 0, params_in_axes))\
                  (jr.split(key, batch_size), data_batch, target_batch, model_params)
    # Need to compute sufficient stats if we want the natural gradient update
    if (train_params.get("use_natural_grad")):
        post_params = result["posterior_params"]
        post_samples = result["posterior_samples"]
        post_suff_stats = vmap(model.posterior.sufficient_statistics)(post_params)
        expected_post_suff_stats = tree_map(
            lambda l: np.mean(l,axis=0), post_suff_stats)
        result["sufficient_statistics"] = expected_post_suff_stats

    # objs = result["objective"]
    if (train_params.get("beta") is None):
        beta = 1
    else:
        beta = train_params["beta"](itr)
    objs = result["ell"] - beta * result["kl"]

    return -np.mean(objs), result

def predict_forward(x, A, b, T):
    def _step(carry, t):
        carry = A @ carry + b
        return carry, carry
    return scan(_step, x, np.arange(T))[1]

def svae_pendulum_val_loss(key, model, data_batch, target_batch, model_params, **train_params):  
    N, T = data_batch.shape[:2]
    # We only care about the first 100 timesteps
    T = T // 2
    D = model.prior.latent_dims

    # obs_data, pred_data = data_batch[:,:T//2], data_batch[:,T//2:]
    obs_data = data_batch[:,:T]
    obj, out_dict = svae_loss(key, model, obs_data, obs_data, model_params, **train_params)
    # Compute the prediction accuracy
    prior_params = model_params["prior_params"] 
    # Instead of this, we want to evaluate the expected log likelihood of the future observations
    # under the posterior given the current set of observations
    # So E_{q(x'|y)}[p(y'|x')] where the primes represent the future
    post_params = out_dict["posterior_params"]
    horizon = train_params["prediction_horizon"] or 5

    _, _, _, pred_lls = vmap(predict_multiple, in_axes=(None, None, None, 0, None, 0, None))\
        (train_params, model_params, model, obs_data, T-horizon, jr.split(key, N), 10)
    # pred_lls = vmap(_prediction_lls)(np.arange(N), jr.split(key, N))
    out_dict["prediction_ll"] = pred_lls
    return obj, out_dict

def svae_update(params, grads, opts, opt_states, model, aux, **train_params):
    rec_opt, dec_opt, prior_opt = opts
    rec_opt_state, dec_opt_state, prior_opt_state = opt_states
    rec_grad, dec_grad = grads["rec_params"], grads["dec_params"]
    updates, rec_opt_state = rec_opt.update(rec_grad, rec_opt_state)
    params["rec_params"] = opt.apply_updates(params["rec_params"], updates)
    params["post_params"] = aux["posterior_params"]
    params["post_samples"] = aux["posterior_samples"]
    if train_params["run_type"] == "model_learning":
        # Update decoder
        updates, dec_opt_state = dec_opt.update(dec_grad, dec_opt_state)
        params["dec_params"] = opt.apply_updates(params["dec_params"], updates)

        old_Q = deepcopy(params["prior_params"]["Q"])
        old_b = deepcopy(params["prior_params"]["b"])

        # Update prior parameters
        if (train_params.get("use_natural_grad")):
            # Here we interpolate the sufficient statistics instead of the parameters
            suff_stats = aux["sufficient_statistics"]
            lr = params.get("prior_learning_rate") or 1
            avg_suff_stats = params["prior_params"]["avg_suff_stats"]
            # Interpolate the sufficient statistics
            params["prior_params"]["avg_suff_stats"] = tree_map(lambda x,y : (1 - lr) * x + lr * y, 
                avg_suff_stats, suff_stats)
            params["prior_params"] = model.prior.m_step(params["prior_params"])
        else:
            updates, prior_opt_state = prior_opt.update(grads["prior_params"], prior_opt_state)
            params["prior_params"] = opt.apply_updates(params["prior_params"], updates)
        
        if (train_params.get("constrain_prior")):
            # Revert Q and b to their previous values
            params["prior_params"]["Q"] = old_Q
            params["prior_params"]["b"] = old_b
    
    A = params["prior_params"]["A"]
    if (train_params.get("run_type") == "vae_baseline"):
        # Zero out the updated dynamics matrix
        params["prior_params"]["A"] = np.zeros_like(A)
    else:
        if (train_params.get("constrain_dynamics")):
            # Scale A so that its maximum singular value does not exceed 1
            params["prior_params"]["A"] = truncate_singular_values(A)
            # params["prior_params"]["A"] = scale_singular_values(params["prior_params"]["A"])

    return params, (rec_opt_state, dec_opt_state, prior_opt_state)

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
    recnet_class = globals()[p["recnet_class"]]
    decnet_class = globals()[p["decnet_class"]]

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