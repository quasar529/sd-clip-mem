# Example single-seed commands.
# The camera-ready paper averages over the following 10 seeds:
# 0 1 10 42 100 441 515 1000 2025 10000

# Ours
python experiments.py --seed 0 --dataset ours458 --pad_token_fix_and_eot_masked True
#python experiments.py --seed 0 --dataset ours458 --pad_token_fix True
#python experiments.py --seed 0 --dataset ours458 --pad_masked_by_ratio 0.7
#python experiments.py --seed 0 --dataset ours458 --pad_masked_by_ratio 0.4

# Prior Works
# Wen et al.
# python experiments.py --seed 0 --dataset ours458 --optim_target_loss 3

# Ren et al.
# python experiments.py --seed 0 --dataset ours458 --rescale_attention 1.25

# Somepalli et al.
## RNA
#python experiments.py --seed 0 --dataset ours458 --prompt_aug_style rand_numb_add
## RTA
#python experiments.py --seed 0 --dataset ours458 --prompt_aug_style rand_word_add
