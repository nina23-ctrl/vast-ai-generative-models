import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample(
    input_path: str = "assets/test_image.png",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"

    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"

    elif version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_u/")
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
        cond_aug = 1e-5

    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_p/")
        model_config = "scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5

    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(model_config, device, num_frames, num_steps, verbose)
    torch.manual_seed(seed)

    path = Path(input_path)

    if path.is_file():
        all_img_paths = [input_path]

    elif path.is_dir():
        all_img_paths = sorted(
            [
                str(p)
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if not all_img_paths:
            raise ValueError("Folder does not contain any images.")

    else:
        raise ValueError("Invalid input_path")

    for input_img_path in all_img_paths:

        # =========================
        # IMAGE LOADING (FIXED)
        # =========================
        image = Image.open(input_img_path)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        w, h = image.size
        input_image = image

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(f"WARNING: resized from {w}x{h} → {width}x{height}")

        input_image = np.array(input_image)  # SAFE conversion

        # =========================
        # TENSOR PREP
        # =========================
        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to(device)

        H, W = image.shape[2:]
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        value_dict = {
            "cond_frames_without_noise": image,
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "cond_aug": cond_aug,
            "cond_frames": image + cond_aug * torch.randn_like(image),
        }

        with torch.no_grad():
            with torch.autocast(device):

                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...")

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {
                    "image_only_indicator": torch.zeros(2, num_frames).to(device),
                    "num_video_frames": batch["num_video_frames"],
                }

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)

                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)

                samples = torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_folder, f"{base_count:06d}.jpg"), input_image
                )

                samples = embed_watermark(samples)
                samples = filter(samples)

                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                imageio.mimwrite(video_path, vid)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))

        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))

        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N))

        elif key in ["cond_frames", "cond_frames_without_noise"]:
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])

        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])

    return batch, batch_uc


def load_model(config: str, device: str, num_frames: int, num_steps: int, verbose: bool = False):
    config = OmegaConf.load(config)

    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    model = instantiate_from_config(config.model).to(device).eval()
    filter = DeepFloydDataFiltering(verbose=False, device=device)

    return model, filter


if __name__ == "__main__":
    Fire(sample)
