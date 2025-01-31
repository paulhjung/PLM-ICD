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
from chs2digits_args import parse_args

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
    output_dir = f"{args.output_prefix}top{args.num_codes}_2digs_max{args.maxtoken_length}_epochs{args.num_overalltrain_epochs}"
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
    if args.devmode:
        data_files["train"+str(0)] = args.train_file+str(0)+f"_2digits_wordlim{args.wordlimit}.csv"
    elif args.train_file is not None:
        for i in range(9):
            logger.info(f"Loaded dataset {i}")
            data_files["train"+str(i)] = args.train_file+str(i)+f"_2digits_wordlim{args.wordlimit}.csv"
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file+f"_2digits_wordlim{args.wordlimit}.csv"
     ## validation set is for hyperparameters; learning rate (.00005 in Edin), minibatch 4?(8 or 16 in Edin), Decision boundary cutoff theshold (default in Edin is .5), Dropout is .2 in Edin, Chunksize
    datafiletype = "csv"
    #code.interact(local=locals())
    raw_datasets = load_dataset(datafiletype, data_files=data_files) #data_files is the path to source data file(s).
    raw_datasets["validation"] = raw_datasets["validation"].filter(lambda example: example["length"] <= args.max_length)
    raw_datasets["train0"] = raw_datasets["train0"].filter(lambda example: example["length"] <= args.max_length)
    if not args.devmode:
        for i in range(1,9):
            raw_datasets[f"train{i}"] = raw_datasets[f"train{i}"].filter(lambda example: example["length"] <= args.max_length)
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
        result = tokenizer(examples["TEXT"], padding=False, truncation=True, add_special_tokens=True, max_length=args.maxtoken_length)
        ### examples keys are 'SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'length'
        if "LABELS" in examples:
            result["labels"] = examples["LABELS"]
            result["label_ids"] = [[label_to_id.get(label.strip()) for label in labels.strip().split(';') if label.strip() in label_to_id.keys() ] if labels is not None else [] for labels in examples["LABELS"]]
        return result
    column_names = raw_datasets["train0"].column_names if args.train_file is not None else raw_datasets["validation"].column_names
    if args.tokens_exist:
        logger.info("Loading tokens")
        tokenized_datasets = datasets.load_from_disk(args.tokens_dir)
    else:
        logger.info("Tokenizing")
        tokenized_datasets = raw_datasets.map(tokenizing_function, batched=True, remove_columns=column_names)
        DIRECTORY_PLM = "../data/mimic4"+str(time.strftime("%d-%H%M")) #data for PLM-ICD
        save_path = DIRECTORY_PLM + f'tokens_2digits_max{args.maxtoken_length}'
        tokenized_datasets.save_to_disk(save_path)
    eval_dataset = tokenized_datasets["validation"]
    train_dataset = tokenized_datasets["train0"]
    if not args.devmode:
        for i in range(1,9):
            train_dataset = datasets.concatenate_datasets([train_dataset, tokenized_datasets[f"train{i}"]])

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
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)    
    
    ##### Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    logger.info("Setting optimizer")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    ##### Scheduler (and math around the number of training steps)
    num_update_steps_per_epoch =  math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    ##### Accelerator
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)], mixed_precision=args.mixed_precision, project_dir=output_dir)
    # Make one log on every process with the configuration for debugging.
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if not accelerator.is_local_main_process:
        logger.info("Accelerator NOT on")
    model, optimizer, eval_dataloader, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, eval_dataloader, train_dataloader, lr_scheduler)
    if args.fromcheckpoint:
        logger.info("Loading checkpoint")
        accelerator.load_state(output_dir)
    # Register the LR scheduler
    #accelerator.register_for_checkpointing(lr_scheduler)
    #accelerator = Accelerator(project_dir="my/save/path")
    accelerate.utils.ProjectConfiguration(automatic_checkpoint_naming=True)
    
    ### I don't know if this is necessary, but just in case
    device = accelerator.device
    model.to(device)

    ##### Train!
    # Get the metric function
    # if args.task_name is not None:
    #    metric = load_metric("glue", args.task_name)
    if args.num_train_epochs > 0:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['labels']}.")
            #logger.info(f"Original tokens: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Max lentgh = {args.max_length}")
        logger.info(f"  Remove digits = {args.remove_digits}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        i = 0
        """
        if args.fromcheckpoint:
            model.eval()
            all_preds = []
            all_preds_raw = []
            all_labels = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                preds_raw = outputs[1].sigmoid().cpu()
                preds = (preds_raw > 0.4).int()
                all_preds_raw.extend(list(preds_raw))
                all_preds.extend(list(preds))
                all_labels.extend(list(batch["labels"].cpu().numpy()))
            all_preds_raw = np.stack(all_preds_raw)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            logger.info(f"epoch {i} finished")
            logger.info(f"metrics: {metrics}")
        """
        for epoch in tqdm(range(args.num_train_epochs)):
            i += 1
            logger.info(f"Epoch = {i}")
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs[0]
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                epoch_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_postfix(loss=epoch_loss / completed_steps)

                if completed_steps >= args.max_train_steps:
                    break
            accelerator.save_state(output_dir)

            logger.info(f"Done with training for Epoch{i}")
            model.eval()
            all_preds = []
            all_preds_raw = []
            all_labels = []
            all_attn = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                preds_raw = outputs[1].sigmoid().cpu()
                all_preds_raw.extend(list(preds_raw))
                all_labels.extend(list(batch["labels"].cpu().numpy()))
                all_attn.extend(list(outputs[2])) #need to dot attn with hidden
            all_preds_raw = np.stack(all_preds_raw)
            all_labels = np.stack(all_labels)
            logger.info(f"model: {output_dir}")
            logger.info(f"testfile:"+args.validation_file+f"_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{args.wordlimit}.csv")
            for t in [.2, .22, .24, .26, .28, .3, .32, .34, .36, .38, .4]: #these are the cutoffs of the logits
                all_preds = (all_preds_raw > t).astype(int)
                metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5,8,15])
                logger.info(f"metrics for threshold {t}: {metrics}")
    
    ##### Test!
    if args.num_train_epochs == 0 and accelerator.is_local_main_process:
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

    if output_dir is not None and args.num_train_epochs > 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        logger.info("Saving model")
        unwrapped_model.save_pretrained(output_dir,is_main_process=accelerator.is_main_process, save_function=accelerator.save)

if __name__ == "__main__":
    main()