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
TEXT = "TEXT"
DOWNLOAD_DIRECTORY = "/Users/paulj/Documents/Github/PLM-ICD/data"
DIRECTORY_PLM = "/Users/paulj/Documents/Github/PLM-ICD/data/mimic4" #data for PLM-ICD
download_dir = Path(DOWNLOAD_DIRECTORY)
output_dir_icd10 = Path(DIRECTORY_PLM)
output_dir_icd10.mkdir(parents=True, exist_ok=True) # if folder doesn't exist, make it

code_dict={}
with open(output_dir_icd10 / 'icd10code2text.csv') as f:
    dreader = csv.reader(f)
    headerRow = next(dreader)
    for row in dreader:
        codE  = row[0]
        short  = row[1] #short description
        code_dict.update({short:codE})

##### Functions to be used
def load_file_into_df(path: Path, dtype: Optional[dict] = None):
    """Reads the notes from a path into a dataframe. Saves the file as a feather file."""
    download_dir = path.parents[0]
    stemmed_filename = path.name.split(".")[0]
    logging.info(f"Loading data from {stemmed_filename}.csv into a pandas dataframe")
    file = pd.read_csv(download_dir / f"{stemmed_filename}.csv", dtype=dtype)
    return file

def code_dict_func(key):
    if key in code_dict:
        return code_dict[key]
    else:
        return None
"""
def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    Takes a dataframe and a column name and returns a series with the column name and a list of codes.
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
    return pd.Series({col: row[col].sort_values().tolist()})

def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, col="icd_code"), include_groups=False)
        .reset_index()
    )
    return df
"""

def preprocessEr(df: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    df = preprocessor(df)
    #df.loc[:,"LABELS"] = df["icd_code"].apply(lambda x: try: code_dict[x])
    ####df.loc[:,"LABELS"] = df["icd_code"].apply(lambda x: ";".join(x))
    df.loc[:,"LABELS"] = df["icd_code"].apply(code_dict_func)
    df.loc[:,"LABELS"] += ";"+df["secondary"].apply(code_dict_func)
    df.loc[:,"length"] = df.TEXT.str.count(" ") + 1
    return df

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
##### Load the data
df = load_file_into_df(download_dir / "json_output.csv", dtype={"icd_code": str})
numrows = df.shape[0]
logging.info(f"Numrows codes is {numrows}")
logging.info("Finished loading data")
df = df.rename(columns={"primary": "icd_code","text": TEXT})
#df = df.drop(columns=["id","file"])
m = df.dropna(subset=["icd_code"], how="all")
m = m.dropna(subset=[TEXT], how="all")
numrows = m.shape[0]
logging.info(f"Removed notes with no codes or no text, subject_id: Num rows is {numrows}")

# Text preprocess the notes
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=True,
    remove_digits=args.remove_digits,
    remove_accents=True,
    remove_brackets=False,
    convert_danish_characters=True,
    remove_firstwords=False
)
m10pp = preprocessEr(df=m, preprocessor=preprocessor)
#code.interact(local=locals())
# preprocesser changes icd10_code from list to string joined by semicolon
numrows = m10pp.shape[0]
logging.info(f"Text preprocess icd10: Num rows is {numrows}")

# Remove empty/long text
m10pp = m10pp[m10pp["length"].apply(lambda x: x > 0)]
numrows = m10pp.shape[0]
logging.info(f"Remove empty text: Num rows is {numrows}")
m10 = m10pp[m10pp["length"].apply(lambda x: x < 15000)]
numrows = m10.shape[0]
logging.info(f"Remove long text: Num rows is {numrows}")
LIMIT = 15000 #args.wordlimit
m10 = m10pp[m10pp["length"].apply(lambda x: x < LIMIT+1)]
numrows = m10.shape[0]
logging.info(f"Limit text to {LIMIT}: Num rows is {numrows}")


# Shuffle df
m10 = m10.sample(frac = 1)
m10 = m10.drop(columns=["secondary"])#"icd_code"

file = f'Advdata_wordlim{LIMIT}_withSecond.csv'
m10.to_csv(output_dir_icd10 / file, index=False)