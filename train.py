import sys
import os
sys.path.insert(0, "/home/jovyan/.local/share/virtualenvs/ptls-experiments-w-dEu3oS/lib/python3.8/site-packages")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import transformers
import datasets
import hydra

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import RewardTrainer, RewardConfig
from torch.optim import AdamW
from src.utils import get_train_reward_pairs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_reward(
    config,
):  
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward.path, 
        num_labels=config.reward.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward.path
    )

    pairs = get_train_reward_pairs(
        config,
        tokenizer,
    )

    training_args = RewardConfig(
        output_dir=config.reward.output_dir,
        save_strategy="no",
        num_train_epochs=config.reward.num_train_epochs,  
        report_to=config.reward.report_to,  
        learning_rate=config.reward.learning_rate,
        per_device_train_batch_size=config.reward.per_device_train_batch_size,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=pairs,
    )
    
    trainer.train()

    if config.reward.save:
        model.save_pretrained("reward_model")
        
        
def train_sft(
    config,
):
    sft_model = GPT2LMHeadModel.from_pretrained(
        config.sft.path
    ).to(device)
    sft_tokenizer = GPT2Tokenizer.from_pretrained(
        config.sft.path
    )

    sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model.generation_config.pad_token_id = sft_tokenizer.pad_token_id

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward.path, 
        num_labels=config.reward.num_labels,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward.path
    )
    
    
@hydra.main(version_base=None, config_path="config")
def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    train_reward(config)
    train_sft(config)
    
    
    
if __name__ == '__main__':
    main()