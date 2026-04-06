import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
import wandb
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
)
from tqdm import tqdm

# Local module
from model_utils import *
from optim_utils import *
from aesthetic.model import aesthetic_predictor

headers = {"User-Agent": "Mozilla/5.0"}

def safe_filename(text):
    return re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", text).strip()[:100]

def preprocess_image_for_fid(image):
    """
    Convert a PIL Image to a torch.Tensor with dtype=torch.uint8
    Args:
        image (PIL.Image): Input image
    Returns:
        torch.Tensor: Preprocessed image
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),  
        ]
    )
    return transform(image).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

def calculate_open_clip_scores(image_paths, prompts, model, clip_preprocess, tokenizer, device):
    scores = []
    for image_path, prompt in zip(image_paths, prompts):
        # 이미지 로드 및 전처리
        if type(image_path) == str:
            image_path = Image.open(image_path).convert("RGB")
        img_tensor = clip_preprocess(image_path).unsqueeze(0).to(device)

        # 텍스트 임베딩
        text_tensor = tokenizer([prompt], context_length=77).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tensor)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # CLIPScore 계산
            score = (image_features @ text_features.T).item()
            scores.append(score)
    return scores

parser = argparse.ArgumentParser()
parser.add_argument("--log_to_wandb", type=bool, default=False, help="Log to wandb")
parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")

DATASET_PRESETS = {
    "ours458": "dataset/extended_mv_458_sd1.csv",
    "webster500": "dataset/sdv1_webster500.csv",
}

parser.add_argument(
    "--dataset",
    type=str,
    default="ours458",
    choices=sorted(DATASET_PRESETS.keys()),
    help="Named dataset preset. Ignored if --data is provided.",
)
parser.add_argument(
    "--data",
    type=str,
    default=None,
    help="Custom dataset CSV path. Overrides --dataset when provided.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Camera-ready seeds: 0, 1, 10, 42, 100, 441, 515, 1000, 2025, 10000",
)
parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="Model id")

# 3.1 Prompt Embeddings Play a Surprisingly Minor Role
parser.add_argument("--eot_pad_masked", type=bool, default=False)
parser.add_argument("--pr_replaced_with_eot", type=bool, default=False)
parser.add_argument("--pr_masked", type=bool, default=False)

# 3.2 Padding Embeddings Are More Influential Than Expected
parser.add_argument("--pad_replaced_with_eot", type=bool, default=False)
parser.add_argument("--pr_pad_replaced_with_eot", type=bool, default=False)
parser.add_argument("--eot_masked", type=bool, default=False)
parser.add_argument("--pr_eot_replaced_with_pad", type=bool, default=False)
parser.add_argument("--pad_masked", type=bool, default=False)

# Section 4 Mitigation
parser.add_argument("--pad_token_fix", type=bool, default=False)
parser.add_argument("--pad_token_fix_and_eot_masked", type=bool, default=False)
parser.add_argument("--pad_masked_by_ratio", type=float, default=None)

# Comparison
# Wen
parser.add_argument("--optim_target_loss", default=None, type=float)
parser.add_argument("--optim_lr", default=0.05, type=float)
parser.add_argument("--optim_iters", default=10, type=int)
parser.add_argument("--optim_target_steps", default=0, type=int)
# Ren
parser.add_argument("--rescale_attention", default=None, type=float)
parser.add_argument("--miti_mem", type=bool, default=False)
# Somepalli
parser.add_argument("--prompt_aug_style", default=None)
parser.add_argument("--repeat_num", default=1, type=int)
args = parser.parse_args()
resolved_data_path = args.data if args.data else DATASET_PRESETS[args.dataset]
args.data = resolved_data_path
if args.data and args.data not in DATASET_PRESETS.values():
    print(f"Using custom dataset CSV: {args.data}")
else:
    print(f"Using dataset preset: {args.dataset}")
    print(f"Resolved dataset CSV: {args.data}")

device = f"cuda"
if args.rescale_attention:
    print(f"REN: {args.rescale_attention}")
    from MemAttn.refactored_classes.MemAttn import MemStableDiffusionPipeline
    from MemAttn.refactored_classes.refactored_unet_2d_condition import (
        UNet2DConditionModel as MemUNet2DConditionModel,
    )

    unet = MemUNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float32
    )
    pipe = MemStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        unet=unet,
        safety_checker=None,
        torch_dtype=torch.float32,
        requires_safety_checker=False,
    ).to(device)
    
    args.c1 = args.rescale_attention
    args.cross_attn_mask = True
    args.miti_mem = True
    args.mask_length_minis1 = False
    args.save_numpy = False 
else:
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
seed = args.seed
g = torch.Generator(device=device).manual_seed(seed)

nowtime = datetime.now().strftime("%d-%H%M")

df = pd.read_csv(args.data)
# Auto-detect URL column name (for RV/TV dataset compatibility)
url_column = "retrieved_urls" if "retrieved_urls" in df.columns else "urls"
is_rv_tv_dataset = "retrieved_urls" in df.columns or "rv_tv" in args.data
prompts = df[["prompt", url_column]].values
dataset_name = os.path.splitext(os.path.basename(args.data))[0]
# 3.1 Prompt Embeddings Play a Surprisingly Minor Role
if args.eot_pad_masked:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_eot_pad_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/eot_pad_masked_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_eot_pad_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/eot_pad_masked_seed{args.seed}"
elif args.pr_replaced_with_eot:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pr_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pr_replaced_with_eot_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pr_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pr_replaced_with_eot_seed{args.seed}"
elif args.pr_masked:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pr_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pr_masked_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pr_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pr_masked_seed{args.seed}"

# 3.2 Padding Embeddings Are More Influential Than Expected
elif args.pad_replaced_with_eot:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pad_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pad_replaced_with_eot_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pad_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pad_replaced_with_eot_seed{args.seed}"
elif args.pr_pad_replaced_with_eot:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pr_pad_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pr_pad_replaced_with_eot_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pr_pad_replaced_with_eot_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pr_pad_replaced_with_eot_seed{args.seed}"
elif args.eot_masked:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_eot_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/eot_masked_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_eot_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/eot_masked_seed{args.seed}"
elif args.pr_eot_replaced_with_pad:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pr_eot_replaced_with_pad_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pr_eot_replaced_with_pad_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pr_eot_replaced_with_pad_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pr_eot_replaced_with_pad_seed{args.seed}"
elif args.pad_masked:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pad_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pad_masked_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pad_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pad_masked_seed{args.seed}"

# Mitigation
elif args.pad_token_fix:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pad_token_fix_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pad_token_fix_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pad_token_fix_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pad_token_fix_seed{args.seed}"
elif args.pad_token_fix_and_eot_masked:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pad_token_fix_and_eot_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pad_token_fix_and_eot_masked_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_pad_token_fix_and_eot_masked_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pad_token_fix_and_eot_masked_seed{args.seed}"
elif args.pad_masked_by_ratio:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_pad_masked_by_ratio_{args.pad_masked_by_ratio}_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/pad_masked_by_ratio_{args.pad_masked_by_ratio *10}_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = (
            f"{dataset_name}_sdv2_pad_masked_by_ratio_{args.pad_masked_by_ratio}_{nowtime}_seed{args.seed}"
        )
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/pad_masked_by_ratio_{args.pad_masked_by_ratio *10}_seed{args.seed}"

# Comparison to other methods
elif args.optim_target_loss:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_optim_target_loss_{args.optim_target_loss}_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/optim_target_loss_{args.optim_target_loss}_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_optim_target_loss_{args.optim_target_loss}_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/optim_target_loss_{args.optim_target_loss}_seed{args.seed}"
elif args.rescale_attention:
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_rescale_attention_{args.rescale_attention}_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/rescale_attention_{args.rescale_attention}_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_rescale_attention_{args.rescale_attention}_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/rescale_attention_{args.rescale_attention}_seed{args.seed}"
elif args.prompt_aug_style == "rand_word_add":
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_rand_word_add_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/rand_word_add_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_rand_word_add_{nowtime}_seed{args.seed}"

elif args.prompt_aug_style == "rand_numb_add":
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        wandb_run_name = f"{dataset_name}_rand_numb_add_{nowtime}_seed{args.seed}"
        folder_name = f"SD1/{dataset_name}/modified_seed{args.seed}/rand_numb_add_seed{args.seed}"
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        wandb_run_name = f"{dataset_name}_sdv2_rand_numb_add_{nowtime}_seed{args.seed}"
        folder_name = f"SD2/{dataset_name}/modified_seed{args.seed}/rand_numb_add_seed{args.seed}"


base_path = "cvpr2026/rv_tv/" if is_rv_tv_dataset else "cvpr2026/"
folder_name = os.path.join(base_path, folder_name)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
print(f"Saving images to {folder_name}")

if args.log_to_wandb:
    wandb_project = args.wandb_project
    wandb.init(
        project=wandb_project,
        name=f"{wandb_run_name}",
        config=args,
    )

# SSCD
sscd_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
skew_320 = transforms.Compose(
    [
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        sscd_normalize,
    ]
)
sscd_model = torch.jit.load("sscd_disc_large.torchscript.pt")
real_vs_original_sscd_scores = []
original_vs_modified_sscd_scores = []
real_vs_modified_sscd_scores = []
original_vs_modified_sscd_scores_above_real_original_0_5 = []
real_vs_modified_sscd_scores_above_real_original_0_5 = []
mitigate_modified_sscd_count = 0

# CLIPScore
reference_model = "ViT-H-14-378-quickgelu"
reference_model_pretrain = "dfn5b"
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(
    reference_model,
    pretrained=reference_model_pretrain,
    device=device,
)
open_clip_tokenizer = open_clip.get_tokenizer(reference_model)
real_image_open_clip_scores = []
original_image_open_clip_scores = []
modified_image_open_clip_scores = []

# Aesthetic Score
aesthetic_model = aesthetic_predictor(device)
real_image_aesthetic_scores = []
original_image_aesthetic_scores = []
modified_image_aesthetic_scores = []

original_images = []
modified_images = []
elapsed_time_list = []

# CSV result collection for each prompt
csv_results = []


for prompt, urls in tqdm(prompts, desc="Generating images", ncols=100):
    try:
        if isinstance(urls, str):
            if urls.startswith("[") and urls.endswith("]"):
                url_list = json.loads(urls.replace("'", '"'))
            else:
                url_list = [urls]  
        else:
            url_list = urls if isinstance(urls, list) else []
    except json.JSONDecodeError as e:
        print(f"[!] JSON decode error for urls: {urls} → {e}")
        continue
    original_prompt = prompt
    print(f"pipe tokenizer pad token: {pipe.tokenizer.pad_token},pad token id: {pipe.tokenizer.pad_token_id}\neot token: {pipe.tokenizer.eos_token},eot token id: {pipe.tokenizer.eos_token_id}")
    # Confirm that the tokenizer is using the correct token for padding and eos
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        pipe.tokenizer.pad_token = "<|endoftext|>"
        pipe.tokenizer.pad_token_id = 49407
        pipe.tokenizer.eos_token = "<|endoftext|>"
        pipe.tokenizer.eos_token_id = 49407
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        pipe.tokenizer.eos_token = "<|endoftext|>"
        pipe.tokenizer.eos_token_id = 49407
    # Ensure the tokenizer is using the correct token for padding and eos
    print(f"pipe tokenizer pad token: {pipe.tokenizer.pad_token},pad token id: {pipe.tokenizer.pad_token_id}\neot token: {pipe.tokenizer.eos_token},eot token id: {pipe.tokenizer.eos_token_id}")

    # Prompt Embedding
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    # Find the position of the startoftext and endoftext tokens in input_ids
    startoftext_token = "<|startoftext|>"
    startoftext_token_id = pipe.tokenizer.convert_tokens_to_ids(startoftext_token)
    endoftext_token = "<|endoftext|>"
    endoftext_token_id = pipe.tokenizer.convert_tokens_to_ids(endoftext_token)
    with torch.no_grad():
        token_ids = pipe.tokenizer(prompt, max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids[
            0
        ]
    startoftext_position = (token_ids == startoftext_token_id).nonzero(as_tuple=True)[0].item()
    endoftext_position = (token_ids == endoftext_token_id).nonzero(as_tuple=True)[0].item()
    total_length = prompt_embeds.shape[1]

    # For Efficient Inference
    ## If the original image already exists, skip the generation
    if args.model_id == "CompVis/stable-diffusion-v1-4":
        save_path = f"cvpr2026/SD1/{dataset_name}/original_seed{args.seed}/{safe_filename(prompt)}.jpg"
        if not os.path.exists(f"cvpr2026/SD1/{dataset_name}/original_seed{args.seed}"):
            os.makedirs(f"cvpr2026/SD1/{dataset_name}/original_seed{args.seed}")
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        save_path = f"cvpr2026/SD2/{dataset_name}/original_seed{args.seed}/{safe_filename(prompt)}.jpg"
        if not os.path.exists(f"cvpr2026/SD2/{dataset_name}/original_seed{args.seed}"):
            os.makedirs(f"cvpr2026/SD2/{dataset_name}/original_seed{args.seed}")
    if os.path.exists(save_path):
        print(f"Original image already exists: {save_path}")
        original_image = Image.open(save_path).convert("RGB")
    else:
        print(f"Generating original image...")
        g.manual_seed(seed)
        # For rescale_attention (MemAttn), we need to pass args but disable miti_mem for original
        if args.rescale_attention:
            if not hasattr(args, "prompt_length"):
                temp_inputs_ids = pipe.tokenizer(
                prompt,
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
                args.prompt_length = temp_inputs_ids['input_ids'].shape[1]
            original_args = argparse.Namespace(**vars(args))
            original_args.miti_mem = False
            original_image = pipe(prompt_embeds=prompt_embeds, generator=g, args=original_args).images[0]
        else:
            original_image = pipe(prompt_embeds=prompt_embeds, generator=g).images[0]
        original_image.save(save_path)
    original_images.append(original_image)
    # Ready for SSCD
    sscd_batch_original = skew_320(original_image).unsqueeze(0)
    sscd_embedding_original = sscd_model(sscd_batch_original)[0, :]

    start_time = time.time()
    if args.eot_masked:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, endoftext_position, :] = 0
    elif args.pad_token_fix:
        pipe.tokenizer.pad_token = "!"
        pipe.tokenizer.pad_token_id = 0
        modified_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    # elif args.pad_token_fix_and_eot_masked:
    #     pipe.tokenizer.pad_token = "!"
    #     pipe.tokenizer.pad_token_id = 0
    #     pipe.tokenizer.eos_token = "!"
    #     pipe.tokenizer.eos_token_id = 0
    #     modified_prompt_embeds, _ = pipe.encode_prompt(
    #         prompt=prompt,
    #         negative_prompt=None,
    #         device=device,
    #         num_images_per_prompt=1,
    #         do_classifier_free_guidance=True,
    #     )
    #     modified_prompt_embeds[:, endoftext_position, :] = 0
    elif args.pad_masked_by_ratio:
        modified_prompt_embeds = prompt_embeds.clone()
        total_pad_length = 77 - (endoftext_position + 1)
        masked_length = int(min(total_pad_length, total_pad_length * args.pad_masked_by_ratio))
        print(f"Masking {masked_length} tokens")
        modified_prompt_embeds[:, endoftext_position + 1 : endoftext_position + 1 + masked_length, :] = 0
    elif args.pad_token_fix_and_eot_masked:
        pipe.tokenizer.pad_token = "!"
        pipe.tokenizer.pad_token_id = 0
        modified_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        modified_prompt_embeds[:, endoftext_position, :] = 0

    elif args.pr_masked:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, startoftext_position + 1 : endoftext_position, :] = 0
    elif args.eot_pad_masked:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, endoftext_position:, :] = 0
    elif args.pad_masked:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, endoftext_position + 1 :, :] = 0
    elif args.pr_pad_replaced_with_eot:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, startoftext_position + 1 :, :] = prompt_embeds[:, endoftext_position, :]
    elif args.pr_replaced_with_eot:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, startoftext_position + 1 : endoftext_position, :] = prompt_embeds[
            :, endoftext_position, :
        ]
    elif args.pad_replaced_with_eot:
        modified_prompt_embeds = prompt_embeds.clone()
        modified_prompt_embeds[:, endoftext_position + 1 :, :] = prompt_embeds[:, endoftext_position, :]
    elif args.pr_eot_replaced_with_pad:
        modified_prompt_embeds = prompt_embeds.clone()
        mean_pad_embedding = torch.mean(modified_prompt_embeds[:, endoftext_position + 1 :, :], dim=1, keepdim=True)
        modified_prompt_embeds[:, startoftext_position + 1 : endoftext_position + 1, :] = mean_pad_embedding

    elif args.prompt_aug_style is not None:
        print(f"RTA or RNA: {args.prompt_aug_style}")
        prompt = prompt_augmentation(
                original_prompt,
                args.prompt_aug_style,
                tokenizer=pipe.tokenizer,
                repeat_num=args.repeat_num,
            )
        print(f"Augmented prompt: {prompt}")
        modified_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    elif args.optim_target_loss is not None:
        print(f"WEN: {args.optim_target_loss}")
        pipe = CustomStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        start_time = time.time()
        modified_prompt_embeds= pipe.aug_prompt(
            prompt,
            target_steps=[args.optim_target_steps],
            lr=args.optim_lr,
            optim_iters=args.optim_iters,
            target_loss=args.optim_target_loss,
        )
    elif args.rescale_attention is not None:
        print(f"REN: {args.rescale_attention}") 
        modified_prompt_embeds,_ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            args=args,
        )

    modified_image_filename = f"{safe_filename(prompt)}.jpg"
    modified_image_save_path = os.path.join(folder_name, modified_image_filename)
    if not os.path.exists(modified_image_save_path):
        print("Generating modified image...")
        if args.rescale_attention or args.optim_target_loss:
            if not hasattr(args, "prompt_length"):
                temp_inputs_ids = pipe.tokenizer(
                prompt,
                # padding="max_length", # This is only for print. It will not be used for generation (RJ)
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
                args.prompt_length = temp_inputs_ids['input_ids'].shape[1]
            print(f"prompt_length: {args.prompt_length}")
            g.manual_seed(seed)
            modified_image = pipe(prompt_embeds=modified_prompt_embeds, generator=g, args=args).images[0]
        else:
            g.manual_seed(seed)
            modified_image = pipe(prompt_embeds=modified_prompt_embeds, generator=g).images[0]
        modified_image.save(modified_image_save_path)
    else:
        print(f"Modified image already exists: {modified_image_save_path}")
        modified_image = Image.open(modified_image_save_path).convert("RGB")
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_list.append(elapsed_time)

    print(f"Time taken: {elapsed_time:.2f} seconds")
    modified_images.append(modified_image)
    sscd_batch_modified = skew_320(modified_image).unsqueeze(0)
    sscd_embedding_modified = sscd_model(sscd_batch_modified)[0, :]


    sscd_sim_score_original_modified = (
        torch.dot(sscd_embedding_original, sscd_embedding_modified)
        / (torch.norm(sscd_embedding_original) * torch.norm(sscd_embedding_modified))
    ).item()

    original_vs_modified_sscd_scores.append(sscd_sim_score_original_modified)

    if sscd_sim_score_original_modified < 0.5:
        mitigate_modified_sscd_count += 1

    real_images = []
    for url in url_list:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Bad status code: {response.status_code} for {url}")
                continue
            if "image" not in response.headers.get("Content-Type", ""):
                print(f"Not an image: {response.headers.get('Content-Type')} for {url}")
                continue
            image = Image.open(BytesIO(response.content))
            try:
                if image.mode in ("P", "RGBA"):
                    image = image.convert("RGB")
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                # Verify image is valid by loading it
                image.load()
                real_images.append(image)
            except (OSError, IOError) as e:
                print(f"[!] Broken image data from {url}: {e}")
                continue
        except Exception as e:
            print(f"[!] Error fetching image from {url}: {e}")
            continue
    has_real_image = len(real_images) > 0
    if has_real_image:
        if len(real_images) ==1:
            real_image = real_images[0]
        else:
            real_image = random.choice(real_images)
        if real_image.mode != "RGB":
            real_image = real_image.convert("RGB")
        aesthetic_score_real = aesthetic_model.predict(real_image).detach().cpu().numpy().item()
        real_image_aesthetic_scores.append(aesthetic_score_real)
        open_clip_score_real = calculate_open_clip_scores(
            [real_image], [prompt], open_clip_model, open_clip_preprocess, open_clip_tokenizer, device
        )
        real_image_open_clip_scores.append(open_clip_score_real)

    aesthetic_score_original = aesthetic_model.predict(original_image).detach().cpu().numpy().item()
    original_image_aesthetic_scores.append(aesthetic_score_original)
    aesthetic_score_modified = aesthetic_model.predict(modified_image).detach().cpu().numpy().item()
    modified_image_aesthetic_scores.append(aesthetic_score_modified)

    open_clip_score_original = calculate_open_clip_scores(
        [original_image], [prompt], open_clip_model, open_clip_preprocess, open_clip_tokenizer, device
    )
    original_image_open_clip_scores.append(open_clip_score_original)

    open_clip_score_modified = calculate_open_clip_scores(
        [modified_image], [prompt], open_clip_model, open_clip_preprocess, open_clip_tokenizer, device
    )
    modified_image_open_clip_scores.append(open_clip_score_modified)

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    if has_real_image:
        ax[0].imshow(real_image)
        ax[0].set_title(f"Real\nAesthetic: {aesthetic_score_real:.2f} OpenCLIP: {open_clip_score_real[0]:.4f}")
    else:
        ax[0].imshow(np.ones((256, 256, 3), dtype=np.uint8) * 255)
        ax[0].set_title("Real\n(Not available)")
    ax[0].axis("off")
    ax[1].imshow(original_image)
    ax[1].set_title(
            f"Original\nAesthetic: {aesthetic_score_original:.2f} OpenCLIP: {open_clip_score_original[0]:.4f}"
        )
    ax[1].axis("off")
    ax[2].imshow(modified_image)
    ax[2].set_title(f"Modified\nAesthetic: {aesthetic_score_modified:.2f} OpenCLIP: {open_clip_score_modified[0]:.4f}")
    ax[2].axis("off")
    plt.tight_layout()

    original_image_caption = f"{prompt}\nOpen CLIP: {open_clip_score_original[0]:.4f} Aesthetic: {aesthetic_score_original:.2f}"
    modified_image_caption = f"{prompt}\nSSCD w.Original: {sscd_sim_score_original_modified:.4f}\nOpen CLIP: {open_clip_score_modified[0]:.4f} Aesthetic: {aesthetic_score_modified:.2f}"
    if args.pad_masked_by_ratio:
        modified_image_caption += f"\nMasked Length: {masked_length}"

    # Wandb Logging
    wandb_log_dict = {
        "Metric_Original/OpenCLIP": open_clip_score_original[0],
        "Metric_Modified/OpenCLIP": open_clip_score_modified[0],
        "Metric_Original/Aesthetic": aesthetic_score_original,
        "Metric_Modified/Aesthetic": aesthetic_score_modified,
        "Metric_Modified/SSCD": sscd_sim_score_original_modified,
        "Metric_Modified/Elapsed Time": elapsed_time,
        "Image/Comparison": wandb.Image(fig, caption=f"{prompt}\nSSCD Score: {sscd_sim_score_original_modified:.4f}"),
    }
    if args.log_to_wandb:
        wandb.log(wandb_log_dict)
    else:
        print(f"\n[Prompt] {prompt}")
        print(f"  SSCD: {sscd_sim_score_original_modified:.4f}")
        print(f"  OpenCLIP (Original): {open_clip_score_original[0]:.4f}")
        print(f"  OpenCLIP (Modified): {open_clip_score_modified[0]:.4f}")
        print(f"  Aesthetic (Original): {aesthetic_score_original:.2f}")
        print(f"  Aesthetic (Modified): {aesthetic_score_modified:.2f}")
        print(f"  Time Elapsed: {elapsed_time:.2f} sec\n")

        print(f"[Running Averages]")
        print(f"  SSCD: {np.mean(original_vs_modified_sscd_scores):.4f}")
        print(f"  OpenCLIP (Original): {np.mean(original_image_open_clip_scores):.4f}")
        print(f"  OpenCLIP (Modified): {np.mean(modified_image_open_clip_scores):.4f}")
        print(f"  Aesthetic (Original): {np.mean(original_image_aesthetic_scores):.2f}")
        print(f"  Aesthetic (Modified): {np.mean(modified_image_aesthetic_scores):.2f}")
        print(f"  Time Elapsed (avg): {np.mean(elapsed_time_list):.2f} sec\n")

    # Collect results for CSV
    csv_results.append({
        "original_prompt": original_prompt,
        "effective_prompt": prompt,
        "image_filename": modified_image_filename,
        "prompt": prompt,
        "seed": args.seed,
        "sscd": round(sscd_sim_score_original_modified, 4),
        "clipscore": round(open_clip_score_modified[0], 4),
        "aesthetic_score": round(aesthetic_score_modified, 4),
        "memorization": sscd_sim_score_original_modified > 0.5
    })
    plt.close(fig)  # Close figure to save memory

# Statistics For Experiments
summary_table = wandb.Table(
    columns=[
        "Metric",
        "Real/Mean",
        "Real/Std",
        "Original/Mean",
        "Original/Std",
        "Modified/Mean",
        "Modified/Std",
    ]
)
summary_table.add_data(
    "OpenCLIP Score",
    np.round(np.mean(real_image_open_clip_scores), 4),
    np.round(np.std(real_image_open_clip_scores), 4),
    np.round(np.mean(original_image_open_clip_scores), 4),
    np.round(np.std(original_image_open_clip_scores), 4),
    np.round(np.mean(modified_image_open_clip_scores), 4),
    np.round(np.std(modified_image_open_clip_scores), 4),
)
summary_table.add_data(
    "Aesthetic Score",
    np.round(np.mean(real_image_aesthetic_scores), 4),
    np.round(np.std(real_image_aesthetic_scores), 4),
    np.round(np.mean(original_image_aesthetic_scores), 4),
    np.round(np.std(original_image_aesthetic_scores), 4),
    np.round(np.mean(modified_image_aesthetic_scores), 4),
    np.round(np.std(modified_image_aesthetic_scores), 4),
)
summary_table.add_data(
    "SSCD Score",
    np.round(np.mean(real_vs_original_sscd_scores), 4),
    np.round(np.std(real_vs_original_sscd_scores), 4),
    np.round(np.mean(original_vs_modified_sscd_scores), 4),
    np.round(np.std(original_vs_modified_sscd_scores), 4),
    np.round(np.mean(real_vs_modified_sscd_scores), 4),
    np.round(np.std(real_vs_modified_sscd_scores), 4),
)
if args.log_to_wandb:
    wandb.log({"Summary Table": summary_table})
    wandb.finish()

# Save results to CSV
is_non_mem_dataset = "non_mem" in args.data
if is_rv_tv_dataset:
    results_dir = "results/rv_tv"
elif is_non_mem_dataset:
    results_dir = "results/non_mem"
else:
    results_dir = f"results/{dataset_name}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Determine method name for CSV filename
if args.pad_token_fix_and_eot_masked:
    method_name = "pad_token_fix_and_eot_masked"
elif args.pad_masked_by_ratio:
    method_name = f"pad_masked_by_ratio_{args.pad_masked_by_ratio}"
elif args.optim_target_loss:
    method_name = "wen"
elif args.rescale_attention:
    method_name = "ren"
elif args.pad_token_fix:
    method_name = "pad_token_fix"
elif args.eot_pad_masked:
    method_name = "eot_pad_masked"
elif args.pr_replaced_with_eot:
    method_name = "pr_replaced_with_eot"
elif args.pr_masked:
    method_name = "pr_masked"
elif args.pad_replaced_with_eot:
    method_name = "pad_replaced_with_eot"
elif args.pr_pad_replaced_with_eot:
    method_name = "pr_pad_replaced_with_eot"
elif args.eot_masked:
    method_name = "eot_masked"
elif args.pr_eot_replaced_with_pad:
    method_name = "pr_eot_replaced_with_pad"
elif args.pad_masked:
    method_name = "pad_masked"
elif args.prompt_aug_style == "rand_word_add":
    method_name = "rta"
elif args.prompt_aug_style == "rand_numb_add":
    method_name = "rna"
else:
    method_name = "unknown"

csv_filename = os.path.join(results_dir, f"{method_name}_seed{args.seed}.csv")
csv_df = pd.DataFrame(csv_results)
csv_df.to_csv(csv_filename, index=False)
print(f"\nResults saved to {csv_filename}")
print(f"Total rows: {len(csv_results)}")
print(f"Memorization count (SSCD > 0.5): {csv_df['memorization'].sum()} / {len(csv_results)}")
