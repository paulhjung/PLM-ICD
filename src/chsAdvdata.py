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
from chs_argsAdvdata import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_error() ## changed from info which outputs the config.json
args = parse_args()

def main():
    output_dir = f"{args.output_prefix}top{args.num_codes}_max{args.maxtoken_length}_epochs{args.num_overalltrain_epochs}"
    os.makedirs(output_dir, exist_ok=True)
    if args.seed is not None:
        set_seed(args.seed)

    ##### Load datasets
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences
    # the sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    data_files = {}
    #if args.devmode:
    #    data_files["train"+str(0)] = args.train_file+str(0)+f"_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{args.wordlimit}.csv"
    #elif args.train_file is not None:
    #    for i in range(9):
    #        logger.info(f"Loaded dataset {i}")
    #        data_files["train"+str(i)] = args.train_file+str(i)+f"_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{args.wordlimit}.csv"
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
     ## validation set is for hyperparameters; learning rate (.00005 in Edin), minibatch 4?(8 or 16 in Edin), Decision boundary cutoff theshold (default in Edin is .5), Dropout is .2 in Edin, Chunksize
    datafiletype = "csv"
    #code.interact(local=locals())
    raw_datasets = load_dataset(datafiletype, data_files=data_files) #data_files is the path to source data file(s).
    raw_datasets["validation"] = raw_datasets["validation"].filter(lambda example: example["length"] <= args.max_length)
    #raw_datasets["train0"] = raw_datasets["train0"].filter(lambda example: example["length"] <= args.max_length)
    #if not args.devmode:
    #    for i in range(1,9):
    #        raw_datasets[f"train{i}"] = raw_datasets[f"train{i}"].filter(lambda example: example["length"] <= args.max_length)
    logger.info(f"Loaded raw datasets")
    logger.info(f"Loaded data files")
    # More on loading datasets: https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/loading_methods#datasets.load_dataset

    ##### Load labels
    # A useful fast method: https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
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
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, output_hidden_states=True)#, finetuning_task=args.task_name)
    config.model_mode = args.model_mode ### different modes implemented by PLM-ICD, default is "laat"; args.model_type = "roberta"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,use_fast=not args.use_slow_tokenizer,do_lower_case=not args.cased)
    model_class = RobertaForMultilabelClassification ### from modeling_roberta

    if args.num_train_epochs > 0:
        model = model_class.from_pretrained(args.model_name_or_path,from_tf=bool(".ckpt" in args.model_name_or_path),config=config)
    else:
        model = model_class.from_pretrained(output_dir, config=config)

    ##### Tokenize the texts
    def tokenizing_function(examples):
        ### tokenizer returns a dict with keys 'input_ids' and 'attention_mask'
        result = tokenizer(examples["TEXT"], padding=False, truncation=True, add_special_tokens=True, max_length=10000)
        ### examples keys are 'SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'length'
        if "LABELS" in examples:
            result["labels"] = examples["LABELS"]
            result["label_ids"] = [[label_to_id.get(label.strip()) for label in labels.strip().split(';') if label.strip() in label_to_id.keys() ] if labels is not None else [] for labels in examples["LABELS"]]
        return result
    column_names = raw_datasets["validation"].column_names 
    if args.tokens_exist:
        logger.info("Loading tokens")
        tokenized_datasets = datasets.load_from_disk(args.tokens_dir)
    else:
        logger.info("Tokenizing")
        tokenized_datasets = raw_datasets.map(tokenizing_function, batched=True, remove_columns=column_names)
        DIRECTORY_PLM = "../data/mimic4"+str(time.strftime("%d-%H%M")) #data for PLM-ICD
        save_path = DIRECTORY_PLM + f'tokens_noDig{args.remove_digits}_noFW{args.remove_firstwords}_max{args.maxtoken_length}'
        tokenized_datasets.save_to_disk(save_path)
    eval_dataset = tokenized_datasets["validation"]
    #code.interact(local=locals())

    ### https://huggingface.co/docs/datasets/en/process
    ##### Collate data of the mini-batches https://pytorch.org/docs/stable/data.html
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
    
    ##### DataLoaders
    #!!!!
    # /opt/anaconda3/envs/plm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
    # warnings.warn(
    # 08/19/2024 16:20:36 - INFO - __main__ -   Distributed environment: NO
    # Num processes: 1
    # Process index: 0
    # Local process index: 0
    # Device: mps
    # Mixed precision type: no
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size) #default batch size is 8, should we use drop_last argument in DataLoader??
    #train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

    ##### Test!
    if args.num_train_epochs == 0:# and accelerator.is_local_main_process:
        model.eval()
        all_preds_raw = []
        all_labels = []
        all_attn = []
        j = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            j += 1
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
        logger.info(f"testfile:"+args.validation_file+f"_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{args.wordlimit}.csv")
        for t in [.2, .22, .24, .26, .28, .3, .32, .34, .36, .38, .4]: #these are the cutoffs of the logits
            all_preds = (all_preds_raw > t).astype(int)
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5,8,15])
            logger.info(f"metrics for threshold {t}: {metrics}")

if __name__ == "__main__":
    main()