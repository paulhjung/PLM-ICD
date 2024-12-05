#import argparse 
import code ## code.interact(local=locals())
import csv
import torch
import numpy as np
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from modeling_roberta2 import RobertaForMultilabelClassification
from chs2_args import parse_args
args = parse_args()

##### Load labels
labels = set()
all_codes_file = args.code_file
all_codes_names = "../data/mimic4/top18codes_with_names.txt"
with open(all_codes_names, "r") as f:
    for line in f:
        if line.strip() != "":
            labels.add(line.strip())
label_list = sorted(list(labels))
id_to_label = {i: v for i, v in enumerate(label_list)}
num_labels = 18 #len(label_list)

##### Load model and tokenizer
config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, output_hidden_states=True)#, finetuning_task=args.task_name)
config.model_mode = args.model_mode ### different modes implemented by PLM-ICD, default is "laat"; args.model_type = "roberta"
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,use_fast=not args.use_slow_tokenizer,do_lower_case=not args.cased)
model_class = RobertaForMultilabelClassification ### from modeling_roberta
output_dir = f"{args.output_prefix}top{args.num_codes}_max{args.maxtoken_length}_epochs{args.num_overalltrain_epochs}"
model = model_class.from_pretrained(output_dir, config=config)
model.eval()

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
    label_ids = torch.zeros((len(results), num_labels))
    return batch

"""
def predict(TEXT):
    tks = tokenizer(TEXT, padding=False, truncation=True, add_special_tokens=True, max_length=args.maxtoken_length)
    d = data_collator([tks])
    wids = tks.word_ids()  
    code.interact(local=locals())
    with torch.no_grad():
        outputs = model(**d)
    preds_raw = outputs[1].sigmoid().cpu()
    return preds_raw
"""
with open('../data/mimic4/test.csv', 'r') as f:  
    reader = csv.reader(f)
    for row in reader:
        tks = tokenizer(row[1], padding=False, truncation=True, add_special_tokens=True, max_length=args.maxtoken_length)
        tokens = tokenizer.tokenize(row[1], padding=False, truncation=True, add_special_tokens=True, max_length=args.maxtoken_length)
        #tkns2 = tokenizer.convert_tokens_to_string(tokens)
        d = data_collator([tks])
        #wids = np.array(tks.word_ids())  
        with torch.no_grad():
            outputs = model(**d)
        preds_raw = np.array(outputs[0].sigmoid().cpu())[0]
        preds_cutoff = (preds_raw > .3).astype(float)
        #print(preds_raw*preds_cutoff)
        for i in range(len(preds_cutoff)):
            if preds_cutoff[i] == 1:
                print("\n", id_to_label[i], "Probability:", preds_raw[i])
                # Get indices of the top 10 tokens
                attns = np.array(outputs[2])[0][i]
                indices = np.argsort(attns[-10:])
                indices.sort()
                # Get indices2 of tokens with high attention
                high_attns = (attns> .035).astype(int)
                indices2 = np.where(high_attns == 1)[0]
                indices2.sort()
                #t = [tokens[q] for q in indices2]
                #print(tokenizer.convert_tokens_to_string(t))
                phrases = {}
                marker = 0
                for j in range(len(indices2)):
                    if indices2[j] >= marker:
                        phrases[indices2[j]] = (indices2[j]+5, attns[indices2[j]])
                    else: 
                        k=1
                        while indices2[j-k] not in phrases:
                            k += 1
                        phrases[indices2[j-k]] = (indices2[j]+5, max(attns[indices2[j]], attns[indices2[j-k]])) #make sure this is in range
                    marker = indices2[j]+5
                s_phrases = sorted(phrases.items(), key=lambda x: x[1][1], reverse = True) 
                #code.interact(local=locals())
                for k in s_phrases[:5]:
                    try:
                        t = [tokens[q-2] for q in range(k[0],k[1][0])]
                        print(round(100*k[1][1],1), tokenizer.convert_tokens_to_string(t))
                    except:
                        print("Exception", wordlist[q])
                #word_indices = wids[indices2]
                #code.interact(local=locals())
                """
                phrases = {}
                marker = 0
                for j in range(len(word_indices)):
                    if word_indices[j] >= marker:
                        phrases[word_indices[j]] = word_indices[j]+4
                    else: 
                        k=1
                        while word_indices[j-k] not in phrases:
                            k += 1
                        phrases[word_indices[j-k]] = word_indices[j]+4 #make sure this is in range
                    marker = word_indices[j]+4
                textlist = row[0].split()
                for k in phrases:
                    try:
                        print(k, phrases[k],": ",' '.join([textlist[q] for q in range(k-2, phrases[k])]),"\n")
                    except:
                        print("Exception", wordlist[q])
                phrases = {}
                marker = 0
                for j in range(len(indices)):
                    if indices[j] >= marker:
                        phrases[indices[j]] = indices[j]+4
                    else: 
                        k=1
                        while indices[j-k] not in phrases:
                            k += 1
                        phrases[indices[j-k]] = indices[j]+4 #make sure this is in range
                for k in phrases:
                    word_indices = wids[range(k,phrases[k])]
                    print(word_indices)
                    wordlist = row[0].split()
                    print([wordlist[i] for i in word_indices])
                """
"""
desired_output = []
for word_id in encoded.word_ids():
    if word_id is not None:
        start, end = encoded.word_to_tokens(word_id)
        if start == end - 1:
            tokens = [start]
        else:
            tokens = [start, end-1]
        if len(desired_output) == 0 or desired_output[-1] != tokens:
            desired_output.append(tokens)
desired_output
"""