import numpy as np
import pandas as pd
import torch
import transformers
import datasets
import hydra
import random

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import RewardTrainer, RewardConfig
from torch.optim import AdamW
from src.utils import (
    get_train_reward_pairs,
    generate_prompt,
    compute_reward,
    compute_kl_loss,
    slerp,
    interpolate_weights
)

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
        logging_dir="./logs",
        logging_steps=30,
        max_steps=760,
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
        model.save_pretrained("models/reward_model")


@hydra.main(version_base=None, config_path="config")
def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    train_reward(config)
    
    
if __name__ == '__main__':
    main()