""""This is the sorted out code 4 your reproduction on AOL dataset"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM 
from transformers import Trainer, HfArgumentParser, TrainingArguments


model_name_or_path = '/XXX/llama2-7b-hf/'  # your model path here
pretrain_aol_file = '/XXX'  # your pretrain file here


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token_id = 0
max_input_length = 256
num_cand = 6
token_false, token_true = 694, 4874
IGNORE_INDEX = -100


class GraphPretrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        tokenizer = self.tokenizer
        concatenated_input_ids, concatenated_labels, concatenated_y_label = [], [], []
        concatenated_attn_mask = []
        for feature in features:
            concatenated_y_label.extend(feature['y_label'])
            for example in feature['inputs']:
                concatenated_input_ids.append(torch.tensor(example['input_ids']))
                concatenated_labels.append(torch.tensor(example['labels']))
                concatenated_attn_mask.append(torch.tensor([1] * len(example['input_ids'])))
                
        input_ids = pad_sequence(concatenated_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(concatenated_labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(concatenated_attn_mask, batch_first=True, padding_value=0)
        y_label = torch.tensor(concatenated_y_label)
        batch = {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'y_label': y_label}
        
        return batch


class GraphPretrainTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        y_label = inputs.pop('y_label')
        outputs = model(**inputs)
        logits = outputs['logits']

        sequence_lengths = (torch.eq(inputs['input_ids'], tokenizer.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        # get the pooled logits
        pooled_logits = logits[torch.arange(logits.size(0), device=logits.device), sequence_lengths]
        
        # 0. calc the link prediction loss
        link_label = torch.where(y_label != 0, 1, 0)
        link_loss = loss_fct(pooled_logits[:, [token_false, token_true]].view(-1, 2), link_label.view(-1))

        # 1. calc the generation loss
        shift_labels = inputs['labels'][..., 1:]
        shift_logits = logits[..., :-1, :]
        gen_labels = shift_labels[(y_label & 1).bool(), :]
        gen_logits = shift_logits[(y_label & 1).bool(), :, :]
        generation_loss = loss_fct(gen_logits.view(-1, gen_logits.size(-1)), gen_labels.view(-1))
        
        # 2. calc the graph contrastive loss
        seq_len = shift_labels.size(1)
        shift_labels = shift_labels[y_label == 3, :].view(-1, 2, seq_len)
        shift_logits = shift_logits[y_label == 3, :, :].view(-1, 2, seq_len, shift_logits.size(-1))
        ref_logits = shift_logits[:, -1, ...].clone().detach()
        ref_labels = shift_labels[:, -1, :]
        ref_ll = self.nll(ref_logits, ref_labels)
        ga_logits = shift_logits[:, 0, ...]
        ga_labels = shift_labels[:, 0, :]
        ga_ll = self.nll(ga_logits, ga_labels)
        margin_loss = - 1.0 * F.logsigmoid(ga_ll - ref_ll).mean()
        
        loss = link_loss + generation_loss + margin_loss
        
        return loss

    @classmethod
    def nll(cls, logits, labels):
        """calc the negative log likelihood of the generated tokens"""
        loss_mask = (labels != IGNORE_INDEX)
        labels[labels == IGNORE_INDEX] = 0
        per_token_likelihood = torch.gather(F.log_softmax(logits, dim=-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return torch.div(torch.sum(per_token_likelihood * loss_mask, dim=-1), torch.sum(loss_mask, dim=-1))


def preprocess4gpt(example):
    inputs = []
    for sample in example["inputs"]:
        prompt = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['instruction']))
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['input']))  # as input_ids
        labels = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['output'])) + [tokenizer.eos_token_id]
        # truncate the ids if the ids is too long
        ids = ids[-max_input_length+len(prompt)+len(labels):]
        input_ids = prompt + ids + labels
        labels = [-100] * (len(prompt) + len(ids)) + labels
        inputs.append({'input_ids': input_ids, 'labels': labels})
    return {'inputs': inputs, 'y_label': example['y_label']}


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
    model.to(dtype=torch.bfloat16)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=['q_proj', 'v_proj'],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    with open(pretrain_aol_file, 'r') as file:
        pretrain_data = json.load(file)
    pretrain_dataset = []
    for example in tqdm(pretrain_data):
        pretrain_dataset.append(preprocess4gpt(example))
    data_collator = GraphPretrainDataCollator(tokenizer)

    trainer = GraphPretrainTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=pretrain_dataset,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model()



