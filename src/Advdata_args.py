import argparse 
import logging 
import math 
import os
import json
import random
#import code #interact inside for debugging
import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import transformers
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
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
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--fromcheckpoint", type=bool, default=True, help="Set true if using checkpoint")
    parser.add_argument("--tokens_exist", type=bool, default=False, help="Set true if already tokenized")
    parser.add_argument("--tokens_dir", type=str, default="../data/mimic420-1900tokens_noDigTrue_noFWTrue_max3072", help="Path to tokens")
    parser.add_argument("--devmode", type=bool, default=False, help="Use much smaller data set for training")
    parser.add_argument("--remove_digits", type=bool, default=True, help="Remove digits in text preprocessing")
    parser.add_argument("--remove_firstwords", type=bool, default=True, help="Remove first words in text preprocessing")
    parser.add_argument("--train_file", type=str, default="../data/mimic4/CHStrain", help="Title prefix of csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default="../data/mimic4/Advdata_wordlim10000.csv", help="Title prefix of csv or a json file containing the validation data.")
    parser.add_argument("--output_prefix", type=str, default="../models/", help="Prefix for where to store the final model.")
    parser.add_argument("--maxtoken_length", type=int, default=3072, help="The maximum total input sequence length after tokenization. Seq longer than this are truncated, seq shorter are padded if `--pad_to_max_lengh` is passed.")
    parser.add_argument("--max_length", type=int, default=10000, help="Max num of words at train time .")
    parser.add_argument("--wordlimit", type=str, default=10000, help="Max num of words at data prep time.")
    parser.add_argument("--num_overalltrain_epochs", type=int, default=7, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs", type=int, default=0, help="Total number of training epochs to perform.")
    parser.add_argument("--code_file", type=str, default="../data/mimic4/top18codes.txt", help="A txt file containing all codes.")
    parser.add_argument("--num_codes", type=str, default=18, help="Number of all codes.")
    #Should only be changing above while testing
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--chunk_size", type=int, default=128, help="The size of chunks that we'll split the inputs into")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    #parser.add_argument("--pad_to_max_length", type=bool, default=False, help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--tokenizer_path", type=str, default="../models/RoBERTa-base-PM-M3-Voc-distill-hf", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_name_or_path", type=str, default="../models/RoBERTa-base-PM-M3-Voc-distill-hf", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default="roberta", help="The type of model")
    parser.add_argument("--model_mode", type=str, default="laat", help="Specify how to aggregate output in the model", choices=["cls-sum", "cls-max", "laat", "laat-split"])
    parser.add_argument("--use_slow_tokenizer", type=bool, default=False, help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--cased", type=bool, default=False, help="equivalent to do_lower_case=True")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=1800, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--mixed_precision", type=str, default="no", help="Mixed precision training.")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the GLUE task to train on.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    #parser.add_argument("--bucket", type=int, default="ai-studio-chs", help="Name of s3 bucket.")
    #Mixed precision type
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    #else:
        #if args.train_file is not None:
        #    extension = args.train_file.split(".")[-1]
        #    assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #if args.validation_file is not None:
        #    extension = args.validation_file.split(".")[-1]
        #    assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    #if args.output_dir is not None:
    #    os.makedirs(args.output_dir, exist_ok=True)

    return args

