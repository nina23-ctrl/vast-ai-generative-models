import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import ListConfig
from PIL import Image
from torch import autocast

from sgm.util import append_dims


# =========================================================
# WATERMARK — DISABLED (INTENTIONALLY)
# =========================================================

def embed_watermark(samples: torch.Tensor) -> torch.Tensor:
    """
    Watermarking is disabled.
    This function is kept for API compatibility.
    """
    return samples


# =========================================================
# UTILS
# =========================================================

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def perform_save_locally(save_path, samples):
    os.makedirs(save_path, exist_ok=True)
    base_count = len(os.listdir(save_path))
    samples = embed_watermark(samples)

    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


# =========================================================
# DISCRETIZATION
# =========================================================

class Img2ImgDiscretizationWrapper:
    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= strength <= 1.0

    def __call__(self, *args, **kwargs):
        sigmas = self.discretization(*args, **kwargs)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        sigmas = torch.flip(sigmas, (0,))
        return sigmas


# =========================================================
# SAMPLING
# =========================================================

def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    batch2model_input: Optional[List] = None,
    return_latents=False,
    filter=None,
    device="cuda",
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    with torch.no_grad():
        with autocast(device):
            with model.ema_scope():
                num_samples = [num_samples]
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    device=device,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to(device),
                            (c, uc),
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape, device=device)

                def denoiser(x, sigma, c):
                    return model.denoiser(
                        model.model, x, sigma, c, **additional_model_inputs
                    )

                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z

                return samples


# =========================================================
# BATCHING
# =========================================================

def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        else:
            batch[key] = value_dict[key]

    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])

    return batch, batch_uc
