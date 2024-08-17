import logging
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional
import code 
import pandas as pd
import vaex
import wget
sys.path.insert(0, '/Users/paulj/Documents/GitHub/PLM-ICD')
from settings import ID_COLUMN, SUBJECT_ID_COLUMN, TARGET_COLUMN
TEXT_COLUMN = "TEXT"

class TextPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = False,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
    ) -> None:
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters

    def __call__(self, df):
        if self.lower:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.lower()
        if self.convert_danish_characters:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("å", "aa", regex=True)
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("æ", "ae", regex=True)
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("ø", "oe", regex=True)
        if self.remove_accents:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("é|è|ê", "e", regex=True)
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("á|à|â", "a", regex=True)
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("ô|ó|ò", "o", regex=True)
        if self.remove_brackets:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("\[[^]]*\]", "", regex=True)
        if self.remove_special_characters:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("\n|/|-", " ", regex=True)
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace(
                "[^a-zA-Z0-9 ]", "", regex=True
            )
        if self.remove_special_characters_mullenbach:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace(
                "[^A-Za-z0-9]+", " ", regex=True
            )
        if self.remove_digits:
            df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("(\s\d+)+\s", " ", regex=True)

        df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.replace("\s+", " ", regex=True)
        df.loc[:,(TEXT_COLUMN)] = df[TEXT_COLUMN].str.strip()
        return df

def merge_code_dataframes(code_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges all code dataframes into a single dataframe.
    Args:
        code_dfs (list[pd.DataFrame]): List of code dataframes.
    Returns:
        pd.DataFrame: Merged code dataframe.
    """
    merged_codes = code_dfs[0]
    for code_df in code_dfs[1:]:
        merged_codes = merged_codes.merge(
            code_df, how="outer", on=[SUBJECT_ID_COLUMN, ID_COLUMN]
        )
    return merged_codes

def reformat_icd10(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]

def top_k_codes(df: pd.DataFrame, column_names: list[str], k: int) -> set[str]:
    """Get the top k most frequent codes from a dataframe"""
    df = df.copy()
    counter = Counter()
    for col in column_names:
        list(map(counter.update, df[col]))
    top_k = counter.most_common(k)
    return set(map(lambda x: x[0], top_k))

# There is another filter_codes in CHSprep
def filter_codes(
    df: pd.DataFrame, column_names: list[str], codes_to_keep: set[str]
) -> pd.DataFrame:
    """Filter the codes in the dataframe to only keep the desired codes"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(
            lambda codes: list(filter(lambda x: x in codes_to_keep, codes))
        )
    return df