import os
import pickle
import csv
import copy

dir_icd = Path(DATA_DIRECTORY_MIMICIV_ICD10)

code_dict={}
with open(dir_icd / 'icd10code2text.csv', encoding="utf8", errors='ignore') as f:
    dreader = csv.reader(f)
    headerRow = next(dreader)
    for row in dreader:
        code  = row[0]
        short  = row[1] #short description
        long = row[2] #long description
        code_dict.update({code:[short, long]})