import torch
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda"


def preprocess_function(pairs):
    pair = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in tqdm(pairs):

        pair["input_ids_chosen"].append(chosen["input_ids"])
        pair["attention_mask_chosen"].append(chosen["attention_mask"])
        pair["input_ids_rejected"].append(rejected["input_ids"])
        pair["attention_mask_rejected"].append(rejected["attention_mask"])

    return pair


def get_train_reward_pairs(config, tokenizer):
    dataset = load_dataset(config.dataset.path)

    train_dataset = dataset["train"]

    positive_comments = [tokenizer.encode_plus(
        comment["text"], 
        truncation=True, 
        padding="max_length",
        max_length=128) for comment in train_dataset if comment["label"] == 1][0:config.dataset.num_positive_pairs]
    
    negative_comments = [tokenizer.encode_plus(
        comment["text"],
        truncation=True, 
        padding="max_length",
        max_length=128) for comment in train_dataset if comment["label"] == 0][0:config.dataset.num_negative_pairs]

    pairs = [(pos, neg) for pos in positive_comments for neg in negative_comments]

    processed_pairs = preprocess_function(pairs)
    processed_dataset = Dataset.from_dict(processed_pairs)
    
    return processed_dataset


def compute_kl_loss(
    model1,
    model2, 
    inputs
):
    output1 = model1(inputs).logits
    output2 = model2(inputs).logits
    
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(output1, dim=-1),
        torch.nn.functional.softmax(output2, dim=-1),
        reduction='batchmean',
    )
    
    return loss


def evaluate(
    config,
    reward_model,
    reward_tokenizer,
    sft_model, 
    sft_tokenizer,
    prompts,

):

    reward_model.eval()
    sft_model.eval()
    
    rewards = []
    kls = []

    for prompt in tqdm(prompts):
        inputs = sft_tokenizer.encode_plus(
            prompt,
            return_tensors='pt', 
            truncation=True,
            padding='max_length',
            max_length=25
        )
        output = sft_model.generate(**inputs.to(device), max_length=40)
        generation = sft_tokenizer.decode(output[0], skip_special_tokens=True)

        inputs = reward_tokenizer.encode_plus(
            generation,
            return_tensors='pt',
            truncation=True, 
            padding='max_length',
            max_length=128
        )
        reward = reward_model.forward(**inputs.to(device)).logits
        
        kl = compute_kl_loss(
                sft_model, 
                init_model,
                inputs["input_ids"],
        )

        rewards.append(score.item())
        kls.append(kl.item())

    return np.mean(rewards), np.mean(kls)


def _eval(
    reward_model,
    reward_tokenizer,
    sft_model, 
    sft_tokenizer,
    prompts,
):
    init_model = GPT2LMHeadModel.from_pretrained(
            "lvwerra/gpt2-imdb"
    ).to(device)
    init_model.generation_config.pad_token_id = sft_tokenizer.pad_token_id
    
    reward_model.eval()
    sft_model.eval()
    
    reward = []
    avg_kl = []

    for prompt in tqdm(prompts):
        inputs = sft_tokenizer.encode_plus(
            prompt,
            return_tensors='pt', 
            truncation=True,
            padding='max_length',
            max_length=25
        )
        output = sft_model.generate(**inputs.to(device), max_length=40)
        generation = sft_tokenizer.decode(output[0], skip_special_tokens=True)

        inputs = reward_tokenizer.encode_plus(
            generation,
            return_tensors='pt',
            truncation=True, 
            padding='max_length',
            max_length=128
        )
        score = reward_model.forward(**inputs.to(device)).logits
        
        kl_loss = compute_kl_loss(
                sft_model, 
                init_model,
                inputs["input_ids"],
        )

        reward.append(score.item())
        
        avg_kl.append(kl_loss.item())

    return np.mean(reward), np.mean(avg_kl)


def compute_reward(prompt, reward_model, reward_tokenizer):
    inputs = reward_tokenizer.encode_plus(
        prompt,
        return_tensors='pt',
        truncation=True, 
        max_length=64).to(device)
    with torch.no_grad():
        outputs = reward_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits
    reward = logits[0]

    return reward


def generate_prompt(prompt, sft_model, sft_tokenizer):
    inputs = sft_tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = sft_model.generate(inputs, max_length=50, num_return_sequences=1)
    generated_prompt = sft_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_prompt


def slerp(theta_init, theta1, theta2, lambd):
    eps = 0.01
    interpolated_weights = {}
    for key in theta_init.keys():
        delta1 = theta1[key] - theta_init[key]
        delta2 = theta2[key] - theta_init[key]
        omega = (np.dot(theta1[key].flatten().cpu(), theta2[key].flatten().cpu()) /
                          (np.linalg.norm(theta1[key].flatten().cpu() + eps) * np.linalg.norm(theta2[key].flatten().cpu() + eps)))
        omega = np.arccos(np.min((np.max((-0.99, omega)), 0.99)))
        
        interpolated_weights[key] = (theta_init[key] +
                                     (np.sin((1 - lambd) * omega) / np.sin(omega)) * delta1 +
                                     (np.sin(lambd * omega) / np.sin(omega)) * delta2)
    return interpolated_weights

def interpolate_weights(theta_sft, theta_slerp, eta):
    interpolated_weights = {}
    for param_name in theta_sft.keys():
        interpolated_weights[param_name] = (1 - eta) * theta_sft[param_name] + eta * theta_slerp[param_name]
    return interpolated_weights