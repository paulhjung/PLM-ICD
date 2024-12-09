import os
import pickle
import csv
import copy
from pathlib import Path

DIRECTORY_PLM = "/Users/paulj/Documents/Github/PLM-ICD/data/mimic4" #data for PLM-ICD
output_dir_icd10 = Path(DIRECTORY_PLM)
text2code_dict={}
with open(output_dir_icd10 / 'icd10code2text.csv') as f:
    dreader = csv.reader(f)
    headerRow = next(dreader)
    for row in dreader:
        codE  = row[0]
        short  = row[1] #short description
        long = row[2] #long description
        text2code_dict.update({short:codE})
t2c = text2code_dict

dic = {}
with open(output_dir_icd10/ 'Patient_Roster.csv') as f:
    dreader = csv.reader(f)
    for row in dreader:
        if row[0] in t2c:
            if t2c[row[0]][:2] in dic:
                dic[t2c[row[0]][:2]] += int(row[1])
            else: dic.update({t2c[row[0]][:2]:int(row[1])})
c = 0
for k in list(dic.keys()):
    if dic[k]<85: 
        dic.pop(k)
    else: c += dic[k]
print(c)
output_file = "top25_2digits.txt"

# Open the file in write mode
with open(output_dir_icd10 / output_file, "w") as file:
    # Write each key to a new line
    for key in dic.keys():
        file.write(f"{key}\n")


