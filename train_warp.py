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


def train_sft(
    config,
):
    dataset = load_dataset("stanfordnlp/imdb")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    prompts = []
    
    for comment in tqdm(np.random.choice(test_dataset, 200)):
        prompts.append(comment["text"][:config.sft.slice])
        
    train_prompts = []
    for comment in tqdm(train_dataset):
        train_prompts.append(comment["text"][:config.sft.slice])
    
    PATH_TO_MODEL = "models/reward_model"
    REWARD_PATH = "distilbert/distilbert-base-cased"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "/home/jovyan/zoloev/t-bank/reward_model",
        num_labels=1
    ).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_PATH)

    SFT_PATH = "lvwerra/gpt2-imdb"

    sft_model = GPT2LMHeadModel.from_pretrained(
        SFT_PATH
    ).to(device)
    sft_tokenizer = GPT2Tokenizer.from_pretrained(SFT_PATH)
    
    sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model.generation_config.pad_token_id = sft_tokenizer.pad_token_id

    
    reward_model.eval()
    sft_model.train()
    
    optimizer = AdamW(sft_model.parameters(), lr=1e-5)
    num_epochs = 5
    gamma = 0.99

    temp_model = GPT2LMHeadModel.from_pretrained(SFT_PATH).to(device)
    
    I = config.sft.I
    M = config.sft.M
    T = config.sft.T
    mu = config.sft.mu
    lambd = config.sft.lambd
    nu = config.sft.nu
    beta = config.sft.beta
    
    theta_init = sft_model.state_dict()
    
    for i in range(I):
        theta_m_list = []
        theta_m_ema_list = []
    
        for m in range(M):
            theta_m = {k: v.clone() for k, v in theta_init.items()}
            theta_m_ema = {k: v.clone() for k, v in theta_init.items()}
            
            for prompt in tqdm(train_prompts[:T]):
                generated_prompt = generate_prompt(prompt, sft_model, sft_tokenizer)
                reward = compute_reward(generated_prompt, reward_model, reward_tokenizer)
    
                inputs = sft_tokenizer.encode(generated_prompt, return_tensors='pt').to(device)
                outputs = sft_model(inputs, labels=inputs)
                loss = outputs.loss
                log_prob = -loss
    
                kl_loss = beta * compute_kl_loss(
                    sft_model, 
                    temp_model,
                    inputs,
                )
    
                policy_loss = - log_prob * reward - kl_loss * log_prob
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
    
                for param_name, param_value in theta_m.items():
                    theta_m_ema[param_name] = (1 - mu) * theta_m_ema[param_name] + mu * param_value
    
                temp_model.load_state_dict(theta_m_ema)
    
            theta_m_list.append(theta_m)
            theta_m_ema_list.append(theta_m_ema)
    
        theta_slerp = slerp(theta_init, theta_m_list[0], theta_m_list[1], lambd)
    
        for k in theta_init.keys():
            theta_init[k] = (1 - nu) * theta_init[k] + nu * theta_slerp[k]


    weights = interpolate_weights(sft_model.state_dict(), theta_slerp, 1 / 2)
    sft_model.load_state_dict(weights)

    if config.reward.save:
        sft_model.save_pretrained(f"models/sft_warp_{config.save_path}")
        sft_tokenizer.save_pretrained(f"models/sft_warp_{config.save_path}")

@hydra.main(version_base=None, config_path="config")
def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    train_sft(config)
    
    
if __name__ == '__main__':
    main()