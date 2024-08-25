import argparse 
import logging 
import math 
import os
import random
#import pyarrow.feather as fthr
import code ### interact inside for debugging
import datasets ### Hugging Face
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import transformers
import torch
import numpy as np
import accelerate
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
from chs_args import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_error() ###changed from info which outputs the config.json

##### Args
args = parse_args()

def main():
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    ##### Load datasets
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences
    # the sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.train_file is not None:
        for i in range(9):
            data_files["train"+str(i)] = args.train_file+str(i)+".csv"
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file 
    # validation set is for hyperparameters; learning rate (.00005 in Edin), minibatch 4?(8 or 16 in Edin), Decision boundary cutoff theshold (default in Edin is .5), Dropout is .2 in Edin, Chunksize
    datafiletype = "csv"
    
    raw_datasets = load_dataset(datafiletype, data_files=data_files) #data_files (str or Sequence or Mapping, optional) — Path(s) to source data file(s).
    # More on loading datasets: https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/loading_methods#datasets.load_dataset

    ##### Load labels
    # A useful fast method: https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    labels = set()
    all_codes_file = "../data/mimic4/top25codes.txt" #if not args.code_50 else "../data/mimic3/ALL_CODES_50.txt"

    with open(all_codes_file, "r") as f:
        for line in f:
            if line.strip() != "":
                labels.add(line.strip())
    label_list = sorted(list(labels))
    label_to_id = {v: i for i, v in enumerate(label_list)}
    num_labels = len(label_list)

    ##### Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    ### This next line outputs the Model config when verbosity is set to "info" level
    ### default model_name_or_path is "../models/roberta-mimic3-50", need to change this for ICD-10
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)#, finetuning_task=args.task_name)
    config.model_mode = args.model_mode ### different modes implemented by PLM-ICD, default is "laat"; args.model_type = "roberta"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,use_fast=not args.use_slow_tokenizer,do_lower_case=not args.cased)
    model_class = RobertaForMultilabelClassification ### from modeling_roberta
    ### This is where model is trained or loaded


    #!!!! Revisit checkpointing
    if args.num_train_epochs > 0:
        model = model_class.from_pretrained(args.model_name_or_path,from_tf=bool(".ckpt" in args.model_name_or_path),config=config)
    else:
        model = model_class.from_pretrained(args.output_dir, config=config)

    ##### Tokenize the texts
    logger.info("Tokenizing")
    def tokenizing_function(examples):
        logger.info("in tokenizer function")
        ### tokenizer returns a dict with keys 'input_ids' and 'attention_mask'
        result = tokenizer(examples["TEXT"], padding=False, max_length=args.max_length, truncation=True, add_special_tokens=True)
        ### examples keys are 'SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'length'
        if "LABELS" in examples:
            result["labels"] = examples["LABELS"]
            result["label_ids"] = [[label_to_id.get(label.strip()) for label in labels.strip().split(';') if label.strip() in label_to_id.keys() ] if labels is not None else [] for labels in examples["LABELS"]]
        return result
    column_names = raw_datasets["train0"].column_names if args.train_file is not None else raw_datasets["validation"].column_names
    tokenized_datasets = raw_datasets.map(tokenizing_function, batched=True, remove_columns=column_names)
    logger.info("Finished tokenizing")
    ### for documentation on map() see https://huggingface.co/docs/datasets/v1.1.1/processing.html default batch is 1000
    eval_dataset = tokenized_datasets["validation"]
    train_dataset = tokenized_datasets["train0"]
    for i in range(1,8):
        train_dataset = datasets.concatenate_datasets([train_dataset, tokenized_datasets[f"train{i}"]])
    #code.interact(local=locals())
    ### https://huggingface.co/docs/datasets/en/process
    tokenized_datasets.save_to_disk('tokenized_datasets')

if __name__ == "__main__":
    main()
