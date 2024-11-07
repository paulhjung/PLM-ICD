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

from utils import TextPreprocessor
from typing import Optional
from chs2_args import parse_args
args = parse_args()

##### Set constants
ID_COLUMN = "HADM_ID"
TEXT = "TEXT"
SUBJECT_ID_COLUMN = "SUBJECT_ID"
DOWNLOAD_DIRECTORY_MIMICIV = "~/Desktop/mimic/mimic-iv-2.2"  #raw MIMIC-IV
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "~/Desktop/mimic/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/"  #raw MIMIC-IV-Note
DIRECTORY_PLM = "/Users/paulj/Documents/Github/PLM-ICD/data/mimic4" #data for PLM-ICD
MIN_TARGET_COUNT = 3000 # Minimum number of times a code must appear to be included
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
def load_gz_file_into_df(path: Path, dtype: Optional[dict] = None):
    """Reads the notes from a path into a dataframe. Saves the file as a feather file."""
    download_dir = path.parents[0]
    stemmed_filename = path.name.split(".")[0]
    logging.info(f"Loading data from {stemmed_filename}.csv.gz into a pandas dataframe")
    file = pd.read_csv(download_dir / f"{stemmed_filename}.csv.gz", compression="gzip", dtype=dtype)
    return file

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
        .apply(partial(reformat_code_dataframe, col="icd_code"), include_groups=False)
        .reset_index()
    )
    return df

def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe"""
    df = df.rename(columns={"hadm_id": ID_COLUMN,"subject_id": SUBJECT_ID_COLUMN,"text": TEXT})
    df = df.dropna(subset=[TEXT])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT])
    return df

def preprocesser(df: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    df = preprocessor(df)
    df.loc[:,"LABELS"] = df["icd_code"].apply(lambda x: ";".join(x))
    df.loc[:,"length"] = df.TEXT.str.count(" ") + 1
    return df

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
##### Load the data
mimic_notes = load_gz_file_into_df(download_dir_note / "note/discharge.csv.gz")
mimic_diag = load_gz_file_into_df(download_dir / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
numrows_notes = mimic_notes.shape[0]
numrows = mimic_diag.shape[0]
logging.info(f"Numrows codes is {numrows}, Numrows notes is {numrows_notes}")
logging.info("Finished loading data")

##### Process the data 
# Drop na, duplicates, rename columns, group codes, drop icd9
mimic_diag = parse_codes_dataframe(mimic_diag)
mimic_notes = parse_notes_dataframe(mimic_notes)
mimic_diag10 = mimic_diag[mimic_diag["icd_version"] == 10]
numrows_notes = mimic_notes.shape[0]
numrows = mimic_diag.shape[0]
numrows10 = mimic_diag10.shape[0]
logging.info(f"Parsed: Numrows codes is {numrows}, Numrows notes is {numrows_notes}, Numrows codes10 is {numrows10}")

# Merge the dfs
mimiciv_10 = mimic_notes.merge(mimic_diag10[[ID_COLUMN, "icd_code"]], on=ID_COLUMN, how="right")
numrows = mimiciv_10.shape[0]
logging.info(f"Merged notes+codes: Numrows is {numrows}")

# Remove unneeded columns
mimiciv_10 = mimiciv_10.drop(columns=["charttime","storetime","note_seq", "note_type","note_id"])

# Remove notes with no codes, subject_id
mimiciv_10 = mimiciv_10.dropna(subset=["icd_code"], how="all")
m = mimiciv_10[mimiciv_10["SUBJECT_ID"].apply(lambda x: x > 0)]
numrows = m.shape[0]
logging.info(f"Removed notes with no codes, subject_id: Num rows is {numrows}")

# Text preprocess the notes
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=True,
    remove_digits=args.remove_digits,
    remove_accents=True,
    remove_brackets=False,
    convert_danish_characters=True,
    remove_firstwords=args.remove_firstwords
)
m10pp = preprocesser(df=m, preprocessor=preprocessor)
# preprocesser changes icd10_code from list to string joined by semicolon
numrows = m10pp.shape[0]
logging.info(f"Text preprocess icd10: Num rows is {numrows}")

# Remove empty/long text
m10pp = m10pp[m10pp["length"].apply(lambda x: x > 0)]
numrows = m10pp.shape[0]
logging.info(f"Remove empty text: Num rows is {numrows}")
m10 = m10pp[m10pp["length"].apply(lambda x: x < 10000)]
numrows = m10.shape[0]
logging.info(f"Remove long text: Num rows is {numrows}")
LIMIT = args.wordlimit
m10 = m10pp[m10pp["length"].apply(lambda x: x < LIMIT+1)]
numrows = m10.shape[0]
logging.info(f"Limit text to {LIMIT}: Num rows is {numrows}")


# Shuffle df
m10 = m10.sample(frac = 1)

##### Old filter codes (now just find top codes) 
#codes2keep(m10, ["icd_code"], MIN_TARGET_COUNT)
#top_k_codes(m10, ["icd_code"], 120)
#logging.info(f"filter_codes: Num rows is {numrows}")
m10 = m10.drop(columns=["icd_code","SUBJECT_ID","HADM_ID"])


##### Save files to disk
k = int(numrows*.1)
for i in range(9):
    d = m10[i*k:(i+1)*k]
    fn = f'CHStrain{i}_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{LIMIT}.csv' 
    d.to_csv(output_dir_icd10 / fn, index=False)
test10 = m10[9*k:]
testfile = f'CHStest_nodigits{args.remove_digits}_nofirstwords{args.remove_firstwords}_wordlim{LIMIT}.csv'
test10.to_csv(output_dir_icd10 / testfile, index=False)#, quoting=csv.QUOTE_NONE) 
#valfile = 'CHSmimic4icd10validation'+str(time.strftime("%d-%H%M"))+'.csv'
#val10.to_csv(output_dir_icd10 / valfile, index=False)#, quoting=csv.QUOTE_NONE) 


#########################
# Functions for later use
def codes2keep(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
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
        #df[col] = df[col].apply(lambda x: [codE for codE in x if codE in codes_to_keep])
        print(f"Number of unique codes in {col} before filtering: {len(code_counts)}")
        print(f"Number of unique codes in {col} after filtering: {len(codes_to_keep)}")
        outfile = "codes_atleast"+str(min_count)+".txt"
        f = open(output_dir_icd10 / outfile, "a")
        for cde in codes_to_keep:
            if cde in code_dict.keys():
                f.write(cde+','+ str(code_dict[cde][0]).replace(",",";")+'\n')
        f.close()
    #return df
def top_k_codes(df: pd.DataFrame, column_names: list[str], k: int) -> set[str]:
    """Get the top k most frequent codes from a dataframe"""
    df = df.copy()
    counter = Counter()
    for col in column_names:
        list(map(counter.update, df[col]))
    top_k = counter.most_common(k)
    outfile = "top_k.txt"
    f = open(output_dir_icd10 / outfile, "a")
    for cde in top_k:
        if cde[0] in code_dict.keys():
            f.write(cde[0]+', '+str(cde[1])+'\t'+code_dict[cde[0]][0]+'\n')
        #else: print(str(cde)+'\n')
    f.close()