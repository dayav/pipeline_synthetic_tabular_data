# ordenc.py ---------------------------------------------------------------
from sklearn.preprocessing import OrdinalEncoder
import numpy as np, pandas as pd

def fit_encoder(df: pd.DataFrame, cat_cols: list[str]) -> OrdinalEncoder:
    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.int32)
    enc.fit(df[cat_cols])
    return enc

def transform_cat(df: pd.DataFrame, cat_cols: list[str], enc: OrdinalEncoder):
    return enc.transform(df[cat_cols]).astype(np.int32)
