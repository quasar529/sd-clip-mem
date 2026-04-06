"""
LPIPS Diversity Measurement Script

Measures the output diversity of generated images across different seeds using LPIPS metric.
For each prompt, calculates the average pairwise LPIPS distance between images generated with different seeds.
Higher LPIPS values indicate greater diversity (less memorization).

Usage:
    python measure_lpips_diversity.py --dataset ours458
    python measure_lpips_diversity.py --method ours --dataset ours458
    python measure_lpips_diversity.py --method ours --dataset webster500
"""

import argparse
import csv
import os
from itertools import combinations
from pathlib import Path

import lpips
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Method configurations: method_name -> folder_name_pattern
METHODS = {
    'original': 'original_seed{seed}',  # Baseline (no modification)
    'eot_pad_masked': 'eot_pad_masked_seed{seed}',
    'pr_replaced_with_eot': 'pr_replaced_with_eot_seed{seed}',
    'pr_masked': 'pr_masked_seed{seed}',
    'pad_replaced_with_eot': 'pad_replaced_with_eot_seed{seed}',
    'pr_pad_replaced_with_eot': 'pr_pad_replaced_with_eot_seed{seed}',
    'eot_masked': 'eot_masked_seed{seed}',
    'pr_eot_replaced_with_pad': 'pr_eot_replaced_with_pad_seed{seed}',
    'pad_masked': 'pad_masked_seed{seed}',
    'ours': 'pad_token_fix_and_eot_masked_seed{seed}',  # Our method
    'ours2': 'pad_masked_by_ratio_7.0_seed{seed}',  # Our method 2 (ratio 0.7)
    'pad_token_fix': 'pad_token_fix_seed{seed}',  # Pad token fix only
    'wen': 'optim_target_loss_3.0_seed{seed}',  # Wen et al.
    'ren': 'rescale_attention_1.25_seed{seed}',  # Ren et al.
    'rna': 'rand_numb_add_seed{seed}',  # Random Number Augmentation
    'rta': 'rand_word_add_seed{seed}',  # Random Token Augmentation
}

# Seeds to evaluate
DEFAULT_SEEDS = [0, 1, 10, 100, 1000, 10000, 2025, 42, 441, 515]
REPO_ROOT = Path(__file__).resolve().parent

DATASET_CONFIG = {
    "ours458": {
        "dataset_csv": "dataset/extended_mv_458_sd1.csv",
        "original_root": "cvpr2026/SD1/extended_mv_458_sd1",
        "modified_root": "cvpr2026/SD1/extended_mv_458_sd1",
        "results_dir": "results/extended_mv_458_sd1",
    },
    "mv_458": {
        "dataset_csv": "dataset/extended_mv_458_sd1.csv",
        "original_root": "cvpr2026/SD1/extended_mv_458_sd1",
        "modified_root": "cvpr2026/SD1/extended_mv_458_sd1",
        "results_dir": "results/extended_mv_458_sd1",
    },
    "webster500": {
        "dataset_csv": "dataset/sdv1_webster500.csv",
        "original_root": "cvpr2026/SD1/sdv1_webster500",
        "modified_root": "cvpr2026/SD1/sdv1_webster500",
        "results_dir": "results/sdv1_webster500",
    },
}

RESULT_METHOD_PREFIX = {
    "eot_pad_masked": "eot_pad_masked",
    "pr_replaced_with_eot": "pr_replaced_with_eot",
    "pr_masked": "pr_masked",
    "pad_replaced_with_eot": "pad_replaced_with_eot",
    "pr_pad_replaced_with_eot": "pr_pad_replaced_with_eot",
    "eot_masked": "eot_masked",
    "pr_eot_replaced_with_pad": "pr_eot_replaced_with_pad",
    "pad_masked": "pad_masked",
    "ours": "pad_token_fix_and_eot_masked",
    "ours2": "pad_masked_by_ratio_0.7",
    "pad_token_fix": "pad_token_fix",
    "wen": "wen",
    "ren": "ren",
    "rna": "rna",
    "rta": "rta",
}

METHOD_ALIASES = {
    "pad_token_fix_and_eot_masked": "ours",
    "pad_masked_by_ratio_0.7": "ours2",
    "optim_target_loss_3.0": "wen",
    "rescale_attention_1.25": "ren",
    "rand_numb_add": "rna",
    "rand_word_add": "rta",
}


def safe_filename(text: str) -> str:
    """Convert text to a safe filename - MUST match experiments.py exactly."""
    import re
    return re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", text).strip()[:100]


def load_image_as_tensor(image_path: str, transform) -> torch.Tensor:
    """Load an image and convert to tensor for LPIPS."""
    img = Image.open(image_path).convert('RGB')
    return transform(img)


def calculate_pairwise_lpips(images: list, lpips_model: lpips.LPIPS, device: str) -> float:
    """
    Calculate the average pairwise LPIPS distance between all image pairs.
    
    Args:
        images: List of image tensors
        lpips_model: Pretrained LPIPS model
        device: Device to run on
        
    Returns:
        Average LPIPS distance (higher = more diverse)
    """
    if len(images) < 2:
        return 0.0
    
    lpips_distances = []
    pairs = list(combinations(range(len(images)), 2))
    
    for i, j in pairs:
        img1 = images[i].unsqueeze(0).to(device)
        img2 = images[j].unsqueeze(0).to(device)
        
        with torch.no_grad():
            distance = lpips_model(img1, img2)
        
        lpips_distances.append(distance.item())
    
    return sum(lpips_distances) / len(lpips_distances) if lpips_distances else 0.0


def get_dataset_info(dataset: str) -> dict:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_CONFIG[dataset]


def get_image_folder_path(base_dir: str, method: str, seed: int, dataset: str) -> str:
    """Get the image folder path for a given method and seed."""
    dataset_info = get_dataset_info(dataset)
    folder_pattern = METHODS[method]
    folder_name = folder_pattern.format(seed=seed)

    if method == 'original':
        # Original images are in a different structure
        return os.path.join(base_dir, dataset_info["original_root"], folder_name)
    else:
        # Modified images are in modified_seed{N} subfolder
        return os.path.join(
            base_dir, dataset_info["modified_root"],
            f'modified_seed{seed}', folder_name
        )


def get_results_csv_path(base_dir: str, method: str, seed: int, dataset: str) -> Path | None:
    method_prefix = RESULT_METHOD_PREFIX.get(method)
    if method_prefix is None:
        return None

    dataset_info = get_dataset_info(dataset)
    return Path(base_dir) / dataset_info["results_dir"] / f"{method_prefix}_seed{seed}.csv"


def load_filename_overrides(
    method: str,
    prompts: list[str],
    base_dir: str,
    dataset: str,
    seeds: list[int],
) -> dict[str, dict[int, str]]:
    """
    Build a mapping from original prompt -> {seed: image filename}.

    RNA/RTA save images using the augmented prompt, so we recover the image
    filename from per-seed CSV results. For current historical CSVs, fall back
    to row-order alignment when original_prompt/image_filename metadata is
    unavailable.
    """
    if method not in {"rna", "rta"}:
        return {}

    overrides = {prompt: {} for prompt in prompts}

    for seed in seeds:
        csv_path = get_results_csv_path(base_dir, method, seed, dataset)
        if csv_path is None or not csv_path.exists():
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        if "original_prompt" in rows[0]:
            for row in rows:
                original_prompt = row["original_prompt"]
                effective_prompt = row.get("effective_prompt") or row.get("prompt") or original_prompt
                image_filename = row.get("image_filename") or f"{safe_filename(effective_prompt)}.jpg"
                if original_prompt in overrides:
                    overrides[original_prompt][seed] = image_filename
            continue

        if len(rows) != len(prompts):
            raise ValueError(
                f"{csv_path} has {len(rows)} rows, but dataset has {len(prompts)} prompts. "
                "Cannot safely align RNA/RTA images for LPIPS."
            )

        for original_prompt, row in zip(prompts, rows):
            effective_prompt = row.get("prompt") or original_prompt
            overrides[original_prompt][seed] = f"{safe_filename(effective_prompt)}.jpg"

    return overrides


def measure_lpips_for_method(
    method: str,
    prompts: list,
    base_dir: str,
    dataset: str,
    seeds: list[int],
    lpips_model: lpips.LPIPS,
    transform,
    device: str
) -> pd.DataFrame:
    """Measure LPIPS diversity for a single method across all prompts."""
    results = []
    filename_overrides = load_filename_overrides(method, prompts, base_dir, dataset, seeds)
    
    for prompt in tqdm(prompts, desc=f"Processing {method}"):
        # Collect images from all seeds
        images = []
        valid_seeds = []
        
        for seed in seeds:
            filename = filename_overrides.get(prompt, {}).get(seed, safe_filename(prompt) + '.jpg')
            image_folder = get_image_folder_path(base_dir, method, seed, dataset)
            image_path = os.path.join(image_folder, filename)
            
            if os.path.exists(image_path):
                try:
                    img_tensor = load_image_as_tensor(image_path, transform)
                    images.append(img_tensor)
                    valid_seeds.append(seed)
                except Exception as e:
                    pass  # Skip failed images silently
        
        # Calculate pairwise LPIPS if we have at least 2 images
        if len(images) >= 2:
            avg_lpips = calculate_pairwise_lpips(images, lpips_model, device)
            num_pairs = len(list(combinations(range(len(images)), 2)))
            
            results.append({
                'prompt': prompt,
                'lpips_diversity': avg_lpips,
                'num_seeds': len(images),
                'num_pairs': num_pairs,
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Measure LPIPS diversity across different seeds')
    parser.add_argument('--method', type=str, default=None,
                        help='Specific method to evaluate (if not set, all methods are evaluated)')
    parser.add_argument('--dataset', type=str, default='ours458',
                        choices=sorted(DATASET_CONFIG.keys()),
                        help='Dataset to evaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Optional explicit seed subset to evaluate')
    parser.add_argument('--base_dir', type=str, default=str(REPO_ROOT),
                        help='Base directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()
    seeds = args.seeds or DEFAULT_SEEDS
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize LPIPS model (using AlexNet backbone as in the original paper)
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Transform for LPIPS (expects images in [-1, 1] range)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
    ])
    
    # Load prompts from dataset CSV
    dataset_info = get_dataset_info(args.dataset)
    csv_path = os.path.join(args.base_dir, dataset_info["dataset_csv"])
    print(f"Loading prompts from: {csv_path}")
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].tolist()
    print(f"Found {len(prompts)} prompts")
    
    # Determine output directory
    output_dir = args.output_dir or os.path.join(args.base_dir, 'results', 'lpips_diversity')
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which methods to evaluate
    if args.method:
        method_name = METHOD_ALIASES.get(args.method, args.method)
        if method_name not in METHODS:
            raise ValueError(f"Unsupported method: {args.method}")
        methods_to_eval = [method_name]
    else:
        methods_to_eval = list(METHODS.keys())
    
    print(f"\nEvaluating methods: {methods_to_eval}")
    print(f"Seeds: {seeds}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Pairs per prompt: {len(list(combinations(range(len(seeds)), 2)))}")
    print("="*60)
    
    # Store summary results
    summary_results = []
    
    for method in methods_to_eval:
        print(f"\n{'='*60}")
        print(f"Processing method: {method}")
        print(f"{'='*60}")
        
        # Measure LPIPS for this method
        results_df = measure_lpips_for_method(
            method, prompts, args.base_dir, args.dataset, seeds, lpips_model, transform, device
        )
        
        if len(results_df) == 0:
            print(f"WARNING: No results for method {method}")
            continue
        
        # Save per-method results
        output_file = os.path.join(output_dir, f'{method}_lpips_diversity.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")
        
        # Calculate and store summary
        mean_lpips = results_df['lpips_diversity'].mean()
        std_lpips = results_df['lpips_diversity'].std()
        min_lpips = results_df['lpips_diversity'].min()
        max_lpips = results_df['lpips_diversity'].max()
        
        summary_results.append({
            'method': method,
            'mean_lpips': mean_lpips,
            'std_lpips': std_lpips,
            'min_lpips': min_lpips,
            'max_lpips': max_lpips,
            'num_prompts': len(results_df)
        })
        
        print(f"\nSummary for {method}:")
        print(f"  Mean LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
        print(f"  Min: {min_lpips:.4f}, Max: {max_lpips:.4f}")
        print(f"  Prompts evaluated: {len(results_df)}")
    
    # Save summary results
    summary_file = os.path.join(output_dir, 'lpips_diversity_summary.csv')
    summary_df = pd.DataFrame(summary_results)
    if args.method and os.path.exists(summary_file):
        existing_summary_df = pd.read_csv(summary_file)
        existing_summary_df = existing_summary_df[~existing_summary_df["method"].isin(methods_to_eval)]
        summary_df = pd.concat([existing_summary_df, summary_df], ignore_index=True)

        method_order = list(METHODS.keys())
        summary_df["method"] = pd.Categorical(summary_df["method"], categories=method_order, ordered=True)
        summary_df = summary_df.sort_values("method").reset_index(drop=True)
        summary_df["method"] = summary_df["method"].astype(str)

    summary_df.to_csv(summary_file, index=False)
    
    # Print final summary table
    print("\n" + "="*60)
    print("LPIPS Diversity Summary (All Methods)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    print(f"\nSummary saved to: {summary_file}")
    
    return summary_df


if __name__ == '__main__':
    main()
