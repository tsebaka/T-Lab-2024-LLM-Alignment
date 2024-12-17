export CUDA_VISIBLE_DEVICES=0


# REWARD

python -m train --config-dir config --config-name config

# WARP

python -m train \
    --config-dir config \
    --config-name config \
    +save_path="I_100_beta_01" \
    ++sft.T=100 \
    ++sft.beta=0.01

python -m train \
    --config-dir config \
    --config-name config \
    +save_path="I_500_beta_01" \
    ++sft.T=500 \
    ++sft.beta=0.1

python -m train \
    --config-dir config \
    --config-name config \
    +save_path="I_500_beta_001" \
    ++sft.T=500 \
    ++sft.beta=0.01

python -m train \
    --config-dir config \
    --config-name config \
    +save_path="I_1000_beta_01" \
    ++sft.T=1000 \
    ++sft.beta=0.1

python -m train \
    --config-dir config \
    --config-name config \
    +save_path="I_1000_beta_001" \
    ++sft.T=1000 \
    ++sft.beta=0.01

# RLOO

python -m train_warp_rloo \
    --config-dir config \
    --config-name config \
    +save_path="I_100_beta_001" \
    ++sft.T=100 \
    ++sft.beta=0.01

python -m train_warp_rloo \
    --config-dir config \
    --config-name config \
    +save_path="I_100_beta_01" \
    ++sft.T=100 \
    ++sft.beta=0.1

python -m train_warp_rloo \
    --config-dir config \
    --config-name config \
    +save_path="I_500_beta_01" \
    ++sft.T=500 \
    ++sft.beta=0.1

python -m train_warp_rloo \
    --config-dir config \
    --config-name config \
    +save_path="I_500_beta_001" \
    ++sft.T=500 \
    ++sft.beta=0.01