from unidecode import unidecode
import numpy as np
import pandas as pd


def remove_diacritics(series):
    se_np = series.to_numpy()
    se_np = np.vectorize(unidecode)(se_np)
    return pd.Series(se_np)


def lowercase(series):
    return series.str.lower()


def remove_punctuation(series):
    return series.str.replace(r"[^\w\s]", "")


def normalize_whitespace(series):
    # Replace all whitespace with a single space
    s = series.str.replace(r"\s", " ")
    # Remove leading and trailing whitespace
    s = s.str.strip()
    # Remove double spaces
    return s.str.replace(r"\s+", " ")


def substring(series, start, end):
    return series.str[start:end]


def apply_normalizers(series, transforms):
    for transform in transforms:
        series = transform(series)
    return series
