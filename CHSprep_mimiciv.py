import random
from collections import Counter
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import code
import os
import csv
import time
import logging

from utils import (
    TextPreprocessor,
    load_gz_file_into_df,
    preprocess_documents,
    reformat_icd,
)

##### Set constants
ID_COLUMN = "HADM_ID"
TEXT = "TEXT"
SUBJECT_ID_COLUMN = "SUBJECT_ID"
DOWNLOAD_DIRECTORY_MIMICIV = "~/Desktop/mimic/mimic-iv-2.2"  #raw MIMIC-IV
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "~/Desktop/mimic/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/"  #raw MIMIC-IV-Note
DIRECTORY_PLM = "/Users/paulj/Documents/Github/PLM-ICD/data/mimic4" #data for PLM-ICD
MIN_TARGET_COUNT = 1 # Minimum number of times a code must appear to be included
random.seed(10)

download_dir_note = Path(DOWNLOAD_DIRECTORY_MIMICIV_NOTE)
download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIV)
output_dir_icd10 = Path(DIRECTORY_PLM)
output_dir_icd10.mkdir(parents=True, exist_ok=True) # if folder doesn't exist, make it

##### Create dict that converts icd-codes to text
code_dict={}
with open(output_dir_icd10 / 'icd10code2text.csv') as f:
    dreader = csv.reader(f)
    headerRow = next(dreader)
    for row in dreader:
        codE  = row[0]
        short  = row[1] #short description
        long = row[2] #long description
        code_dict.update({codE:[short, long]})

##### Functions to be used
def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.
    Example:
        Input:
                subject_id  _id     icd9_diag
        608           2   163353     V3001
        609           2   163353      V053
        610           2   163353      V290

        Output:
        icd9_diag    [V053, V290, V3001]
    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.
    Returns:
        pd.Series: Series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})

def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the codes dataframe"""
    df = df.rename(columns={"hadm_id": ID_COLUMN, "subject_id": SUBJECT_ID_COLUMN})
    df = df.dropna(subset=["icd_code"])
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, col="icd_code"))
        .reset_index()
    )
    return df

def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe"""
    df = df.rename(columns={"hadm_id": ID_COLUMN,"subject_id": SUBJECT_ID_COLUMN,"text": TEXT})
    df = df.dropna(subset=[TEXT])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT])
    return df

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
##### Load the data
mimic_notes = load_gz_file_into_df(download_dir_note / "note/discharge.csv.gz")
mimic_diag = load_gz_file_into_df(
    download_dir / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}
)
logging.info("Finished loading data")

##### Parse codes and notes
mimic_diag = parse_codes_dataframe(mimic_diag)
mimic_notes = parse_notes_dataframe(mimic_notes)
mimic_diag10 = mimic_diag[mimic_diag["icd_version"] == 10]
logging.info("Parsed codes and notes")

mimiciv_10 = mimic_notes.merge(mimic_diag10[[ID_COLUMN, "icd_code"]], on=ID_COLUMN, how="right")
logging.info("Merged notes and codes")

# remove unneeded columns
mimiciv_10 = mimiciv_10.drop(columns=["charttime","storetime","note_seq", "note_type","note_id"])

# remove notes with no codes
mimiciv_10 = mimiciv_10.dropna(subset=["icd_code"], how="all")
logging.info("Removed notes with no codes")

##### Filter Codes 
#mimiciv_10f = filter_codes(mimiciv_10, ["icd_code"], MIN_TARGET_COUNT)
#logging.info("filter_codes")

# Text preprocess the notes
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=True,
    remove_digits=False,
    remove_accents=True,
    remove_brackets=False,
    convert_danish_characters=True,
)
m10pp = preprocess_documents(df=mimiciv_10, preprocessor=preprocessor)
logging.info("Text preprocess icd10")

# remove empty text. subject_id
m10pp = m10pp[m10pp["length"].apply(lambda x: x > 0)]
m = m10pp[m10pp["length"].apply(lambda x: x < 10000)]
m = m[m["SUBJECT_ID"].apply(lambda x: x > 0)]
logging.info("Remove empty text, subject_id")

# remove icd10_diag, note_id
m = m.drop(columns=["icd10_diag"])

numrows = m.shape[0]
logging.info(f"Num rows is {numrows}")
trainrows = int(numrows*.9)

# save files to disk
# code.interact(local=locals())
train10 = m[:trainrows]
test10 = m[trainrows:]
trainfile = 'CHSmimic4icd10train'+str(time.strftime("%d-%H%M"))+'.csv'
testfile = 'CHSmimic4icd10test'+str(time.strftime("%d-%H%M"))+'.csv'
train10.to_csv(output_dir_icd10 / trainfile, index=False)#, quoting=csv.QUOTE_NONE) 
test10.to_csv(output_dir_icd10 / testfile, index=False)#, quoting=csv.QUOTE_NONE) 

def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times

    Args:
        df (pd.DataFrame): The codes dataframe
        col (str): The column name of the codes
        min_count (int): The minimum number of times a code must appear

    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([codE for codes in df[col] for codE in codes])
        codes_to_keep = set(
            codE for codE, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [codE for codE in x if codE in codes_to_keep])
        print(f"Number of unique codes in {col} before filtering: {len(code_counts)}")
        print(f"Number of unique codes in {col} after filtering: {len(codes_to_keep)}")
        outfile = "codes_atleast"+str(min_count)+".txt"
        f = open(dir_icd / outfile, "a")
        for cde in codes_to_keep:
            if cde in code_dict.keys():
                f.write(cde+','+ str(code_dict[cde][0]).replace(",",";")+'\n')
            else:
                print(cde)
        f.close()
    return df