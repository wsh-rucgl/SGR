import json
import torch
import datasets
import transformers
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModelForCausalLM
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM 
from transformers import Trainer, HfArgumentParser, TrainingArguments


model_name_or_path = '/XXX/llama2-7b-hf/'  # your model path here
pretrained_model_path = '/XXX'  # your SGR pretrained model path here
rank_aol_file = '/XXX'  # your rank train data here


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token_id = 0
max_input_length = 256
num_cand = 5
token_false, token_true = 694, 4874
IGNORE_INDEX = -100


class RankDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        tokenizer = self.tokenizer
        concatenated_input_ids, concatenated_attn_mask = [], []
        for feature in features:
            for example in feature['inputs']:
                concatenated_input_ids.append(torch.tensor(example['input_ids']))
                concatenated_attn_mask.append(torch.tensor([1] * len(example['input_ids'])))
                
        input_ids = pad_sequence(concatenated_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(concatenated_attn_mask, batch_first=True, padding_value=0)
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return batch
    

class RankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['logits']
        sequence_lengths = (torch.eq(inputs['input_ids'], tokenizer.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(logits.size(0), device=logits.device), sequence_lengths]
        pooled_logits = pooled_logits[:, token_true]
        pooled_logits = pooled_logits.view(-1, num_cand)
        nll_loss = - 1.0 * F.log_softmax(pooled_logits, dim=-1)[:, 0]  # the negative log likelihood
        loss = nll_loss.mean()
        return loss 


def preprocess4rank(example):
    inputs = []
    for sample in example:
        prompt = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['instruction']))
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['input']))  # as input_ids
        labels = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['output'])) + [tokenizer.eos_token_id]
        # truncate the ids if the ids is too long
        ids = ids[-max_input_length+len(prompt)+len(labels):]
        input_ids = prompt + ids + labels
        inputs.append({'input_ids': input_ids})
    return {'inputs': inputs}


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.remove_unused_columns = False
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format() 
    
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config._flash_attn_2_enabled = True
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
    model = PeftModelForCausalLM.from_pretrained(model, pretrained_model_path, is_trainable=True)

    model = model.to(dtype=torch.bfloat16)

    with open(rank_aol_file, 'r') as file:
        rank_data = json.load(file)
    rank_dataset = []
    for example in tqdm(rank_data):
        rank_dataset.append(preprocess4rank(example))
    data_collator = RankDataCollator(tokenizer)

    trainer = RankTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=rank_dataset,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model()
        
