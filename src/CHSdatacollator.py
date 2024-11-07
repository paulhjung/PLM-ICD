import torch
from chs_args import parse_args
args = parse_args()

def data_collator(features):
    batch = dict()
    max_length = max([len(f["input_ids"]) for f in features])
    if max_length % args.chunk_size != 0:
        max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

    batch["input_ids"] = torch.tensor([
        f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
        for f in features
    ]).contiguous().view((len(features), -1, args.chunk_size))
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.tensor([
            f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
    if "token_type_ids" in features[0]:
        batch["token_type_ids"] = torch.tensor([
            f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
    label_ids = torch.zeros((len(features), len(label_list)))
    for i, f in enumerate(features):
        for label in f["label_ids"]:
            label_ids[i, label] = 1
    batch["labels"] = label_ids
    return batch