import torch
import datasets

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict


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


def get_train_prompts():
    pass

    
def get_test_prompts():
    pass
    


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