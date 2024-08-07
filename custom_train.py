# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_sd3.py

from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import logging
from accelerate.logging import get_logger
import transformers
import diffusers
import os
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from diffusers.utils.torch_utils import is_compiled_module
import bitsandbytes as bnb
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, Value, Features
import datasets
from PIL import Image
from torchvision import transforms
from PIL.ImageOps import exif_transpose
import torch
from io import BytesIO
import numpy as np
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
import math
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import copy
import itertools
import wandb


def filter_post_dataset(row, bucket_aspects):
    tag_blocklist = ["gore", "scat", "watersports", "young", "loli", "shota"]

    if (row["is_deleted"] == "t" or
            row["is_pending"] == "t" or
            row["is_flagged"] == "t" or
            row["file_ext"] not in ["jpg", "gif", "png"]):
        return False
    
    # Must move assign_bucket function to here, and be inline, to make this method hashable
    # Modified from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py
    max_ar_error = 4  # TODO: get this from somewhere
    bucket_aspects = np.array(bucket_aspects)  # hack: cannot pass in as np.array, otherwise method hash is different
    aspect = float(row["image_width"])/float(row["image_height"])
    bucket_id = np.abs(bucket_aspects - aspect).argmin()
    error = abs(bucket_aspects[bucket_id] - aspect)
    if not error < max_ar_error:
        return False

    tags = set(row["tag_string"].split())
    aTagIsInBlocklist = bool(tags.intersection(set(tag_blocklist)))

    return not aTagIsInBlocklist


class E621Dataset(IterableDataset):
    def __init__(self, date, accelerator, num_proc, seed, dataset_buffer_size, batch_size, gradient_accumulation_steps):
        self._setup_buckets()
        self._setup_post_and_tag_dataset(date, accelerator, num_proc)
        self._setup_image_dataset(accelerator, seed, dataset_buffer_size)

        # Do images across gradient accumulation need to be of the same size? Assuming yes
        self.bucket_size = gradient_accumulation_steps * batch_size * accelerator.num_processes


    def _tag_string_to_categories(self, tag_string):
        category_id_to_column_name_dict = {
            0: "general_tags",
            1: "artist_tags",
            # 2 is unused
            3: "copyright_tags",
            4: "character_tags",
            5: "species_tags",
            6: "invalid_tags",
            7: "meta_tags",
            8: "lore_tags"
        }

        tag_categories = {
            "general_tags": list(),
            "artist_tags": list(),
            "copyright_tags": list(),
            "character_tags": list(),
            "species_tags": list(),
            "invalid_tags": list(),
            "meta_tags": list(),
            "lore_tags": list()
        }

        post_tags = list(tag_string.split())
        for tag in post_tags:
            try:
                category_id = self.tag_to_category_id_df.loc[tag]["category"]
            except KeyError:
                get_logger(__name__).info(f"The tag '{tag}' doesn't exist in the tag dataset. Skipping tag.", main_process_only=False)
                continue

            column_name = category_id_to_column_name_dict[category_id]
            tag_categories[column_name].append(tag)

        return tag_categories


    def _setup_post_and_tag_dataset(self, date, accelerator, num_proc):
        # date in format e.g. 2024-07-02
        post_dataset_url = f"https://e621.net/db_export/posts-{date}.csv.gz"
        tag_dataset_url = f"https://e621.net/db_export/tags-{date}.csv.gz"

        with accelerator.main_process_first():
            self.post_dataset = load_dataset("csv", data_files=post_dataset_url, na_filter=False)
            tag_dataset = load_dataset("csv", data_files=tag_dataset_url, na_filter=False)

            # Datasets library doesn't have a good way of querying dataset.
            # Don't use the datasets's search index 'elasticsearch'. It doesn't cache data between runs
            df = tag_dataset["train"].select_columns(["name", "category"]).to_pandas()
            self.tag_to_category_id_df = df.set_index("name", verify_integrity=True)

            self.post_dataset = self.post_dataset["train"]
            bucket_aspects = self.bucket_aspects
            self.post_dataset = self.post_dataset.filter(
                lambda row : filter_post_dataset(row, bucket_aspects),
                num_proc=num_proc,
                desc="Filtering post dataset"
            )
 
            # Make post dataset queryable
            self.post_id_to_idx_df = self.post_dataset.select_columns(["id"]).to_pandas()
            # Set post id as the index and keep the original index column
            # post id -> idx
            self.post_id_to_idx_df = self.post_id_to_idx_df.reset_index().set_index("id", verify_integrity=True)


    def _setup_image_dataset(self, accelerator, seed, dataset_buffer_size):
        features = Features({
            '__key__': Value(dtype='string', id=None),
            '__url__': Value(dtype='string', id=None),
            'image': datasets.Image(decode=False, id=None)  # Decode image in this script to handle bad files
        })

        with accelerator.main_process_first():
            self.image_dataset = load_dataset("boxingscorpionbagel/e621-2024", streaming=True, features=features)
            self.image_dataset = self.image_dataset["train"]
            self.image_dataset = self.image_dataset.shuffle(seed=seed, buffer_size=dataset_buffer_size)


    # Modified from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py
    def _setup_buckets(self, max_size=(768, 512), div=64, min_dim=256, base_res=(512, 512), dim_limit=1024):
        f = 8
        max_tokens = (max_size[0] / f) * (max_size[1] / f)

        resolutions = []
        aspects = []
        w = min_dim
        while (w/f) * (min_dim/f) <= max_tokens and w <= dim_limit:
            h = min_dim
            got_base = False
            while (w/f) * ((h+div)/f) <= max_tokens and (h+div) <= dim_limit:
                if w == base_res[0] and h == base_res[1]:
                    got_base = True
                h += div
            if (w != base_res[0] or h != base_res[1]) and got_base:
                resolutions.append(base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += div
        h = min_dim
        while (h/f) * (min_dim/f) <= max_tokens and h <= dim_limit:
            w = min_dim
            got_base = False
            while (h/f) * ((w+div)/f) <= max_tokens and (w+div) <= dim_limit:
                if w == base_res[0] and h == base_res[1]:
                    got_base = True
                w += div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]

        self.bucket_resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.bucket_aspects = list(map(lambda x: res_map[x], self.bucket_resolutions))


    # Modified from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py
    def _assign_bucket(self, image_resolution, max_ar_error=4):
        bucket_aspects = np.array(self.bucket_aspects)  # hack: cannot pass in as np.array, otherwise method hash is different
        w, h = image_resolution
        aspect = float(w)/float(h)
        bucket_id = np.abs(bucket_aspects - aspect).argmin()
        error = abs(bucket_aspects[bucket_id] - aspect)
        if error < max_ar_error:
            return bucket_id
        else:
            # Assumes e6 post data correctly specifies image size
            raise Exception("Cannot put image into a bucket, yet the dataset has already filtered out unbucketable images")


    def _get_images(self):
        buckets = { idx: list() for idx in range(0, len(self.bucket_resolutions)) }

        for index, row in enumerate(self.image_dataset):
            try:
                image = Image.open(BytesIO(row["image"]["bytes"]))
                image.load()
            except Exception as e:
                get_logger(__name__).info(f"Skipping bad file with E621 post id {row['__key__']}. Got Exception: {e}", main_process_only=False)
                continue

            post_id = int(row["__key__"])
            try:
                post_idx = self.post_id_to_idx_df.loc[post_id]
            except KeyError:
                # Post was filtered. Skip
                continue
            post_data = self.post_dataset[post_idx]

            bucket_id = self._assign_bucket(image.size)
            bucket = buckets[bucket_id]

            # Puts tags into their categories on the fly
            tag_categories = self._tag_string_to_categories(post_data["tag_string"][0])
            post_data.update(tag_categories)
            
            bucket.append({
                "post_data": post_data,
                "image": image
            })

            if len(bucket) == self.bucket_size:
                # Return and empty bucket
                buckets[bucket_id] = list()

                # Must yield buckets, not individual items otherwise there is a race condition if dataloader num_workers > 1
                bucket_resolution = self.bucket_resolutions[bucket_id]
                yield { "bucket": bucket, "bucket_resolution": bucket_resolution }


    def __iter__(self):
        return iter(self._get_images())
    

    def __len__(self):
        return len(self.post_dataset)
    

    def set_epoch(self, epoch):
        self.image_dataset.set_epoch(epoch)


# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir, accelerator, unwrap_model):
    if accelerator.is_main_process:
        for i, model in enumerate(models):
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                    hidden_size = unwrap_model(model).config.hidden_size
                    if hidden_size == 768:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                    elif hidden_size == 1280:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                else:
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()


def load_model_hook(models, input_dir, unwrap_model):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model), SD3Transformer2DModel):
            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
        elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
            try:
                load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                model(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            except Exception:
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model


def _encode_prompt_with_clip(
    text_encoder,
    text_input_ids,
    device=None,
    num_images_per_prompt: int = 1,
):
    batch_size, token_len = text_input_ids.shape

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_with_t5(
    text_encoder,
    text_input_ids,
    num_images_per_prompt=1,
    device=None,
):
    batch_size, token_len = text_input_ids.shape

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    text_input_ids,
    device=None,
    num_images_per_prompt: int = 1,
):
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []

    # Text encoder 1
    prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoder,
        text_input_ids=text_input_ids[0],
        device=device if device is not None else text_encoder.device,
        num_images_per_prompt=num_images_per_prompt,
    )
    clip_prompt_embeds_list.append(prompt_embeds)
    clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    # Text encoder 2
    prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoder_2,
        text_input_ids=text_input_ids[1],
        device=device if device is not None else text_encoder_2.device,
        num_images_per_prompt=num_images_per_prompt,
    )
    clip_prompt_embeds_list.append(prompt_embeds)
    clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    # Text encoder 3
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoder=text_encoder_3,
        text_input_ids=text_input_ids[2],
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoder_3.device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def _unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def setup_accelerator(output_dir, logging_dir, seed, gradient_accumulation_steps):
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        project_config=accelerator_project_config,
        split_batches=True  # Dataloader takes a bucket, then the bucket is split between GPUs
    )

    set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    unwrap_model = lambda model : _unwrap_model(model, accelerator)
    save_model_hook_with_accelerator = lambda models, weights, output_dir : save_model_hook(models, weights, output_dir, accelerator, unwrap_model)
    load_model_hook_with_accelerator = lambda models, input_dir : load_model_hook(models, input_dir, unwrap_model)
    accelerator.register_save_state_pre_hook(save_model_hook_with_accelerator)
    accelerator.register_load_state_pre_hook(load_model_hook_with_accelerator)

    return accelerator


def setup_logging(accelerator):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def load_model(model_name, weight_dtype):
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )

    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=weight_dtype
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    noise_scheduler = copy.deepcopy(noise_scheduler)  # https://github.com/huggingface/diffusers/issues/1606#issuecomment-1346667389

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        model_name,
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_name,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype
    )

    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_name,
        subfolder="text_encoder_3",
        torch_dtype=weight_dtype
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer"
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer_2"
    )

    tokenizer_3 = T5TokenizerFast.from_pretrained(
        model_name,
        subfolder="tokenizer_3"
    )

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)

    return transformer, vae, noise_scheduler, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3


# Modified from https://gist.github.com/Mikubill/5c9d62c28c1f2d81d82a2ed8b272540c#file-train_dreambooth-py-L1376-L1413
def get_resize_size(image_size, bucket_resolution):
    x, y = image_size
    short, long = (x, y) if x <= y else (y, x)

    w, h = bucket_resolution
    min_crop, max_crop = (w, h) if w <= h else (h, w)
    ratio_src, ratio_dst = float(long / short), float(max_crop / min_crop)

    if ratio_src > ratio_dst:
        return (min_crop, int(min_crop * ratio_src)) if x < y else (int(min_crop * ratio_src), min_crop)
    elif ratio_src < ratio_dst:
        return (max_crop, int(max_crop / ratio_src)) if x > y else (int(max_crop / ratio_src), max_crop)
    else:
        return bucket_resolution
    

def get_sigmas(accelerator, noise_scheduler_copy, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def image_preprocess(image, bucket_resolution):
    image = exif_transpose(image)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    resize_size = get_resize_size(image.size, bucket_resolution)

    # torchvision expects tuple to be (h, w)
    resize_size = resize_size[1], resize_size[0]
    bucket_resolution = bucket_resolution[1], bucket_resolution[0]
    
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(bucket_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    image = image_transforms(image)

    return image

def prompt_create(post_data):
    prompt = []
    
    prompt.extend(post_data["species_tags"])
    prompt.extend(post_data["artist_tags"])
    prompt.extend(post_data["general_tags"])
    # Anything after general_tags will probably be truncated

    prompt = ", ".join(prompt)

    return prompt


def tokenize_prompts(prompts, tokenizer, tokenizer_2, tokenizer_3):
    text_input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids_3 = tokenizer_3(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    return [text_input_ids.input_ids, text_input_ids_2.input_ids, text_input_ids_3.input_ids]


def collate_fn(batch, vae, tokenizer, tokenizer_2, tokenizer_3):
    batch = batch[0]  # Unwrap

    pixel_values = [image_preprocess(post["image"], batch["bucket_resolution"]) for post in batch["bucket"]]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(dtype=vae.dtype)
    pixel_values = pixel_values.contiguous()  # using split_batches and multi-gpu requires tensor to be contiguous

    prompts = [prompt_create(post["post_data"]) for post in batch["bucket"]]
    text_input_ids = tokenize_prompts(prompts, tokenizer, tokenizer_2, tokenizer_3)
    text_input_ids = torch.stack(text_input_ids)
    # Batch size must be the first dimension for multi-gpu training
    # Otherwise, only one gpu will receive the tensors and the others will receive empty tensors for some reason
    text_input_ids = text_input_ids.permute(1, 0, 2)
    text_input_ids = text_input_ids.contiguous()  # using split_batches and multi-gpu requires tensor to be contiguous

    return { "pixel_values": pixel_values, "text_input_ids": text_input_ids }


def main():
    output_dir = "sd3-finetune"
    logging_dir = Path(output_dir, "logs")
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
    learning_rate = 1e-06
    batch_size = 8
    first_epoch = 0
    num_train_epochs = 1
    seed = 10
    dataset_buffer_size = 100
    num_proc = 16
    gradient_accumulation_steps = 1
    weighting_scheme = "logit_normal"
    validation_steps = 200

    accelerator = setup_accelerator(output_dir, logging_dir, seed, gradient_accumulation_steps)
    setup_logging(accelerator)
    # scale learning rate
    learning_rate = learning_rate * gradient_accumulation_steps * batch_size * accelerator.num_processes

    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise Exception(f"{accelerator.mixed_precision} not supported")
    
    with accelerator.main_process_first():
        transformer, vae, noise_scheduler, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3 = load_model(model_name, weight_dtype)

    params = [
        {
            "params": transformer.parameters(),
            "lr": learning_rate
        }
    ]
    
    optimizer = bnb.optim.AdamW8bit(params)

    dataset = E621Dataset("2024-07-06", accelerator, num_proc, seed, dataset_buffer_size, batch_size, gradient_accumulation_steps)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Dataset yields a bucket for each iteration
        collate_fn=lambda batch : collate_fn(batch, vae, tokenizer, tokenizer_2, tokenizer_3),
        # Run collate_fun in thread
        # Note that cuda operations cannot be run on a thread
        num_workers=4,  # For me, 4 workers is enough to feed 2 A100s at batch size 8 without compiling models
        prefetch_factor=50,  # An arbitrary high number. Too high may run into RAM OOM
        pin_memory=True
    )

    dataloader_len = int(len(dataloader) / batch_size) * accelerator.num_processes  # assume dataloader batch_size is 1 (for avoiding race condition)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(dataloader_len / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    (
        transformer,
        vae,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
        optimizer,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        transformer,
        vae,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
        optimizer,
        dataloader,
        lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataloader_len / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd3-finetune"
        accelerator.init_trackers(tracker_name, config={"test": "test"})

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num batches each epoch = {dataloader_len}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    inference_pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        transformer=accelerator.unwrap_model(transformer),
        # Original script doesn't pass noise_scheduler
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        text_encoder_3=accelerator.unwrap_model(text_encoder_3),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3
    )

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        dataset.set_epoch(epoch)
        logger.info(f"Starting epoch {epoch}", main_process_only=False)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate([transformer]):
                text_input_ids = batch["text_input_ids"]
                text_input_ids = text_input_ids.permute(1, 0, 2)  # reverse dimension switch

                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    text_encoder_3=text_encoder_3,
                    text_input_ids=text_input_ids
                )

                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=weighting_scheme,
                    batch_size=bsz,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(accelerator, noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = latents

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_norm=1)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % validation_steps == 0:
                        logger.info("Running validation...")

                        generator = torch.Generator(device=accelerator.device).manual_seed(seed)

                        autocast_ctx = torch.autocast(accelerator.device.type)
                        with autocast_ctx:
                            def run_inference(prompt):
                                image = inference_pipeline(prompt=prompt, generator=generator).images[0]
                                image = wandb.Image(image, caption=prompt)
                                return image
                            
                            prompts = [
                                # Edited SeaArt example prompt
                                "canid, canine, fox, mammal, red_fox, true_fox, foxgirl83, photonoko, day, digitigrade, fluffy, fluffy_tail, fur, orange_body, orange_fur, orange_tail, solo, sunlight, tail, mid, 2018, digital_media_(artwork), hi_res",
                                # A fox
                                "canid, canine, fox, mammal, anthro, male, solo",
                                "canid, canine, fox, mammal, anthro, female, solo",
                                # Two foxes
                                "canid, canine, fox, mammal, anthro, male, duo",
                                "canid, canine, fox, mammal, anthro, female, duo"
                                # A wolf
                                "canid, canine, canis, mammal, wolf, anthro, male, solo"
                                "canid, canine, canis, mammal, wolf, anthro, female, solo"
                            ]

                            images = [run_inference(prompt) for prompt in prompts]
                        
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                tracker.log({ "test": images })

                        if accelerator.is_main_process:
                            accelerator.unwrap_model(transformer).save_pretrained(f"/mnt/resource_nvme/transformer-{global_step}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    logger.info("End")
    accelerator.end_training()


if __name__ == "__main__":
    main()
