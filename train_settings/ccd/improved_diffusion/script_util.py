import argparse

import torch

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .cross_model import DiT_models2
NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
        settings,
        device=torch.device("cuda:0"),
):
    train_mode = settings.train.train_mode
    tps_resolution = settings.model.tps_resolution
    grid_resolution = settings.model.grid_resolution
    learn_sigma = settings.model.learn_sigma
    sigma_small = settings.model.sigma_small
    diffusion_steps = settings.model.diffusion_steps
    noise_schedule = settings.model.noise_schedule
    timestep_respacing = settings.resume.timestep_respacing
    use_kl = settings.model.use_kl
    predict_xstart = settings.model.predict_xstart
    rescale_timesteps = settings.model.rescale_timesteps
    rescale_learned_sigmas = settings.model.rescale_learned_sigmas

    model = create_model(
        settings,
        tps_resolution,
        grid_resolution,
        train_mode=train_mode,
    )
    diffusion = create_gaussian_diffusion(
        settings=settings,
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
        settings,
        tps_resolution,
        grid_resolution,
        train_mode,
):
    if train_mode == 'Cascade':
        latent_size = tps_resolution + grid_resolution
        model = DiT_models2['DiT-S/2_new'](
            input_size=latent_size,
            in_channels=settings.model.in_channels,
            resume=settings.resume.resume_checkpoint is not None,
        )
        return model
    else:
        raise ValueError(f"unsupported train mode: {train_mode}")


def create_gaussian_diffusion(
        *,
        settings,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",    # control the noise variance of each froward diffusion process (add noise)
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        settings=settings,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args.env, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
