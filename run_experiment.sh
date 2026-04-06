# Example single-seed commands.
# The camera-ready paper averages over the following 10 seeds:
# 0 1 10 42 100 441 515 1000 2025 10000

# 3.1 Prompt Embeddings Play a Surprisingly Minor Role
python experiments.py --seed 0 --dataset ours458 --eot_pad_masked True
#python experiments.py --seed 0 --dataset ours458 --pr_replaced_with_eot True
#python experiments.py --seed 0 --dataset ours458 --pr_masked True

# 3.2 Padding Embeddings Are More Influential Than Expected
#python experiments.py --seed 0 --dataset ours458 --pad_replaced_with_eot True
#python experiments.py --seed 0 --dataset ours458 --pr_pad_replaced_with_eot True
#python experiments.py --seed 0 --dataset ours458 --eot_masked True
#python experiments.py --seed 0 --dataset ours458 --pr_eot_replaced_with_pad True
#python experiments.py --seed 0 --dataset ours458 --pad_masked True
