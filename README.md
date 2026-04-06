# Memorization In Stable Diffusion Is Unexpectedly Driven by CLIP Embeddings

## Setup

```bash
conda create -n MEM python=3.11 -y
conda activate MEM
pip install -r requirements.txt
```

Download `sscd_disc_large.torchscript.pt` from  
https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt  
and place it in the project root.

## Dataset Selection

Named presets:

- `ours458` -> `dataset/extended_mv_458_sd1.csv`
- `webster500` -> `dataset/sdv1_webster500.csv`

Examples:

```bash
python experiments.py --dataset ours458 --seed 0 --pad_token_fix_and_eot_masked True
python experiments.py --dataset webster500 --seed 0 --pad_token_fix_and_eot_masked True
```

Custom CSV:

```bash
python experiments.py --data dataset/Membench_sd1.csv --seed 0 --pad_token_fix_and_eot_masked True
python experiments.py --data dataset/non_mem_1500.csv --seed 0 --pad_token_fix_and_eot_masked True
```

Main experiments use these 10 seeds:

```text
0, 1, 10, 42, 100, 441, 515, 1000, 2025, 10000
```

## Mitigation Methods

```bash
sh run_mitigation.sh
```

Main method:

```bash
python experiments.py --dataset ours458 --seed 0 --pad_token_fix_and_eot_masked True
```

Other methods:

```bash
python experiments.py --dataset ours458 --seed 0 --pad_token_fix True
python experiments.py --dataset ours458 --seed 0 --pad_masked_by_ratio 0.7
python experiments.py --dataset ours458 --seed 0 --optim_target_loss 3
python experiments.py --dataset ours458 --seed 0 --rescale_attention 1.25
python experiments.py --dataset ours458 --seed 0 --prompt_aug_style rand_numb_add
python experiments.py --dataset ours458 --seed 0 --prompt_aug_style rand_word_add
```

## Section 3 Ablations

```bash
sh run_experiment.sh
```

- Section 3.1: `--eot_pad_masked`, `--pr_replaced_with_eot`, `--pr_masked`
- Section 3.2: `--pad_replaced_with_eot`, `--pr_pad_replaced_with_eot`, `--eot_masked`, `--pr_eot_replaced_with_pad`, `--pad_masked`

## Evaluation

LPIPS:

```bash
python measure_lpips_diversity.py --dataset ours458
python measure_lpips_diversity.py --dataset ours458 --method ours
python measure_lpips_diversity.py --dataset webster500 --method ours
```

## File Structure

```text
CVPR2026/
├── dataset/
│   ├── extended_mv_458_sd1.csv
│   ├── sdv1_webster500.csv
│   ├── webster_mv_sd1.csv
│   ├── Membench_sd1.csv
│   ├── non_mem_1500.csv
│   ├── mscoco.csv
│   ├── Lexica.csv
│   └── laion2B.csv
├── experiments.py
├── run_experiment.sh
├── run_mitigation.sh
├── measure_lpips_diversity.py
├── io_utils.py
├── model_utils.py
├── optim_utils.py
├── aesthetic/
├── MemAttn/
└── requirements.txt
```

## Credits

- Non-memorized prompt sources
  - [MS-COCO](https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT)
  - [Lexica](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
  - [LAION-2B](https://huggingface.co/datasets/laion/relaion2B-multi-research)
- `aesthetic/`: from [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- `sscd_disc_large.torchscript.pt`: from [SSCD (Meta AI)](https://github.com/facebookresearch/sscd-copy-detection)
- `MemAttn/`: from [MemAttn](https://github.com/renjie3/MemAttn)
- `io_utils.py`, `model_utils.py`, `optim_utils.py`: adapted from [MemBench_code](https://github.com/chunsanHong/MemBench_code)
