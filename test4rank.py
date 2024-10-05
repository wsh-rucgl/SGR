import os
import json
import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModelForCausalLM
from transformers import AutoTokenizer,  AutoModelForCausalLM, LlamaConfig


model_name_or_path = '/XXX/llama2-7b-hf/'
adapter_name_or_path = '/XXX'
test_aol_file = '/XXX'
output_aol_file = '/XXX'


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token_id = 0
max_input_length = 256
num_cand = 50
token_false, token_true = 694, 4874
IGNORE_INDEX = -100
test_batch_size = 50
assert test_batch_size % num_cand == 0


class TestDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        tokenizer = self.tokenizer
        input_ids, attn_mask, y_label = [], [], []
        for feature in features:
            input_ids.append(torch.tensor(feature['input_ids']))
            attn_mask.append(torch.tensor([1] * len(feature['input_ids'])))
            y_label.append(feature['label'])
                
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return batch, torch.tensor(y_label)


def preprocess4test(sample):
    prompt = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['instruction']))
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['input']))  # as input_ids
    labels = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['output'])) + [tokenizer.eos_token_id]
    # truncate the ids if the ids is too long
    ids = ids[-max_input_length+len(prompt)+len(labels):]
    input_ids = prompt + ids + labels
    return {"input_ids": input_ids, "label": sample['label']}


test_dataset = []
with open(test_aol_file, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        test_dataset.append(preprocess4test(json.loads(line)))



data_collator = TestDataCollator(tokenizer)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    collate_fn=data_collator,
)

config = LlamaConfig.from_pretrained(model_name_or_path)
config._flash_attn_2_enabled = True
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
model = PeftModelForCausalLM.from_pretrained(model, adapter_name_or_path)
model = model.merge_and_unload()
model.to(dtype=torch.bfloat16)
model.eval()

accelerator = accelerate.Accelerator()
model, test_dataloader = accelerator.prepare(model, test_dataloader)

all_scores, all_labels = [], []

for batch, batch_label in tqdm(test_dataloader):
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs["logits"]
    sequence_lengths = (torch.eq(batch['input_ids'], tokenizer.pad_token_id).long().argmax(-1) - 1).to(logits.device)
    pooled_logits = logits[torch.arange(logits.size(0), device=logits.device), sequence_lengths]
    pooled_logits = pooled_logits[:, [token_false, token_true]]
    batch_scores = F.softmax(pooled_logits, dim=-1)
    batch_scores, batch_labels = accelerator.gather_for_metrics((batch_scores, batch_label))
    batch_scores = batch_scores[:, 1].cpu().numpy()
    batch_labels = batch_labels.cpu().numpy()
    all_scores.append(batch_scores)
    all_labels.append(batch_labels)

all_scores = np.concatenate(all_scores)
all_labels = np.concatenate(all_labels)

with open(output_aol_file, 'w') as fw:
    for score, label in zip(all_scores, all_labels):
        fw.write(f'{score}\t{label}\n')

# accelerate launch test4rank.py
# python trec_eval.py --run /XXX/XXX.run  --num_cand 50


