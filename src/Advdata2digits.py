import argparse 
import logging 
import math 
import os
import random
import code ## code.interact(local=locals())
import datasets ## Hugging Face
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import transformers
import torch
import numpy as np
import accelerate
import time
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from modeling_roberta import RobertaForMultilabelClassification
from evaluation import all_metrics
from Advdata_args2digits import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_error() 
args = parse_args()

def main():
    output_dir = f"{args.output_prefix}top{args.num_codes}_2digs_max{args.maxtoken_length}_epochs{args.num_overalltrain_epochs}"
    os.makedirs(output_dir, exist_ok=True)
    data_files = {}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    datafiletype = "csv"
    raw_datasets = load_dataset(datafiletype, data_files=data_files) #data_files is the path to source data file(s).
    logger.info(f"Loaded raw datasets")
    #code.interact(local=locals())
    ##### Load labels
    labels = set()
    all_codes_file = args.code_file
    with open(all_codes_file, "r") as f:
        for line in f:
            if line.strip() != "":
                labels.add(line.strip())
    label_list = sorted(list(labels))
    label_to_id = {v: i for i, v in enumerate(label_list)}
    num_labels = len(label_list)

    ##### Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, output_hidden_states=True)#, finetuning_task=args.task_name)
    config.model_mode = args.model_mode ### different modes implemented by PLM-ICD, default is "laat"; args.model_type = "roberta"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,use_fast=not args.use_slow_tokenizer,do_lower_case=not args.cased)
    model_class = RobertaForMultilabelClassification ### from modeling_roberta
    model = model_class.from_pretrained(output_dir, config=config)

    ##### Tokenize the texts
    def tokenizing_function(examples):
        result = tokenizer(examples["TEXT"], padding=False, truncation=True, add_special_tokens=True, max_length=10000)
        ### examples keys are 'SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'length'
        if "LABELS" in examples:
            result["labels"] = examples["LABELS"]
            if examples["LABELS"] in label_to_id:
                result["label_ids"] = [label_to_id[examples["LABELS"]]]
            else: result["label_ids"] = result["label_ids"] = []
            #result["label_ids"] = [[
            # label_to_id.get(label.strip()) for label in labels.strip().split(';') 
            # if label.strip() in label_to_id.keys() ] if labels is not None else [] for labels in examples["LABELS"]]
        return result
    column_names = raw_datasets["validation"].column_names 
    logger.info("Tokenizing")
    tokenized_datasets = raw_datasets.map(tokenizing_function, batched=False, remove_columns=column_names)
    eval_dataset = tokenized_datasets["validation"]
    def data_collator(results):
        batch = dict()
        max_length = max([len(row["input_ids"]) for row in results]) ### max lenth of the batch; less or equal to --max_length
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        batch["input_ids"] = torch.tensor([
            row["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(row["input_ids"]))
            for row in results
        ]).contiguous().view((len(results), -1, args.chunk_size))
        if "attention_mask" in results[0]:
            batch["attention_mask"] = torch.tensor([
                row["attention_mask"] + [0] * (max_length - len(row["attention_mask"]))
                for row in results
            ]).contiguous().view((len(results), -1, args.chunk_size))
        if "token_type_ids" in results[0]:
            batch["token_type_ids"] = torch.tensor([
                row["token_type_ids"] + [0] * (max_length - len(row["token_type_ids"]))
                for row in results
            ]).contiguous().view((len(results), -1, args.chunk_size))
        label_ids = torch.zeros((len(results), len(label_list)))
        for i, row in enumerate(results):
            for label in row["label_ids"]:
                label_ids[i, label] = 1
        batch["labels"] = label_ids
        return batch
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    ##### Test!
    if args.num_train_epochs == 0:
        model.eval()
        all_preds_raw = []
        all_labels = []
        all_attn = []
        j = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            j += 1
            #if j == 20: break
            with torch.no_grad():
                outputs = model(**batch)
            preds_raw = outputs[1].sigmoid().cpu()
            all_preds_raw.extend(list(preds_raw))
            if j==1: print(all_preds_raw)
            all_labels.extend(list(batch["labels"].cpu().numpy()))
            all_attn.extend(list(outputs[2]))
        all_preds_raw = np.stack(all_preds_raw)
        all_labels = np.stack(all_labels)
        logger.info(f"evaluation finished")
        logger.info(f"model: {output_dir}")
        logger.info(f"testfile:"+args.validation_file)
        for t in [.09, .11, .13, .15, .17, .19, .21]: #these are the cutoffs of the logits
            all_preds = (all_preds_raw > t).astype(int)
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5])
            logger.info(f"metrics for threshold {t}: {metrics}")

if __name__ == "__main__":
    main()