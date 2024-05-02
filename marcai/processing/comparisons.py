import numpy as np
import re
import pandas as pd
from thefuzz import fuzz
import textdistance



HAND_COUNT_PAGE_PATTERN = re.compile(r"\[(?P<hand_count>\d+)\]\s*p(ages)?[^\w]")
PAGE_PATTERN = re.compile(r"(?P<pages>\d+)\s*p(ages)?[^\w]")


def equal(se0, se1, null_value):
    se0_np = se0.to_numpy(dtype=str)
    se1_np = se1.to_numpy(dtype=str)

    col = (se0_np == se1_np).astype(float)

    se0_nulls = np.argwhere(np.char.strip(se0_np, " ") == "")
    se1_nulls = np.argwhere(np.char.strip(se1_np, " ") == "")

    col[se0_nulls] = null_value
    col[se1_nulls] = null_value

    return pd.Series(col)


def maximum(df, null_value, ignore_value=np.nan):
    df_np = df.to_numpy(dtype=float)

    df_np[df_np == ignore_value] = np.nan
    
    # Mask ignore_value
    masked = np.ma.masked_invalid(df_np)

    # Get the max, ignoring NaNs
    col = np.max(masked, axis=1)

    # Replace NaNs with null_value
    col = col.filled(fill_value=null_value)

    return pd.Series(col)


def minimum(se0, se1, null_value, ignore_value=np.nan):
    se0_np = se0.to_numpy(dtype=float)
    se1_np = se1.to_numpy(dtype=float)

    # Replace ignore_value with np.nans
    se0_np[se0_np == ignore_value] = np.nan
    se1_np[se1_np == ignore_value] = np.nan

    # Get the min, ignoring NaNs
    col = np.nanmin(np.stack([se0_np, se1_np], axis=1), axis=1)

    # Replace NaNs with null_value
    col[np.isnan(col)] = null_value

    return pd.Series(col)


def pagination_match(se0, se1, null_value):
    def group_values(pat, group, s):
        return {m.groupdict()[group] for m in pat.finditer(s)}

    def compare(pag0, pag1):
        hand_counts0 = group_values(HAND_COUNT_PAGE_PATTERN, "hand_count", pag0)
        hand_counts1 = group_values(HAND_COUNT_PAGE_PATTERN, "hand_count", pag1)

        # Remove bracketed digits
        pag0 = re.sub(r"\[\d+\]", "", pag0)
        pag1 = re.sub(r"\[\d+\]", " ", pag1)

        # Remove punctuation
        pag0 = re.sub(r"[^\w\s]", " ", pag0)
        pag1 = re.sub(r"[^\w\s]", " ", pag1)

        # Extract page counts
        counts0 = group_values(PAGE_PATTERN, "pages", pag0 + " ")
        counts1 = group_values(PAGE_PATTERN, "pages", pag1 + " ")

        page_counts0 = counts0 | hand_counts0
        page_counts1 = counts1 | hand_counts1

        # Check if any pages are in common.
        if page_counts0 and page_counts1:
            for pg0 in page_counts0:
                for pg1 in page_counts1:
                    pg0 = int(pg0)
                    pg1 = int(pg1)

                    if pg0 == pg1:
                        return 1.0
            return 0.0

        return null_value

    se0_np = se0.to_numpy(dtype=str)
    se1_np = se1.to_numpy(dtype=str)

    col = np.vectorize(compare)(se0_np, se1_np)
    return pd.Series(col)


def year_similarity(se0, se1, null_value, exp_coeff):
    def compare(yr0, yr1):
        if yr0.isnumeric() and yr1.isnumeric():
            x = abs(int(yr0) - int(yr1))

            # Sigmoid where x = 0, y = 1, tail to the right
            return 2 / (1 + np.exp(exp_coeff * x))

        return null_value

    se0_np = se0.to_numpy(dtype=str)
    se1_np = se1.to_numpy(dtype=str)

    return np.vectorize(compare)(se0_np, se1_np)


def column_aggregate_similarity(df0, df1, column_weights, null_value):
    weights_dict = {k: v for k, v in zip(df0.columns, column_weights)}

    def get_word_weights(row):
        word_weights = {}
        for i, value in enumerate(row):
            column = df0.columns[i]
            if column in weights_dict:
                current_weight = weights_dict[column]
            else:
                current_weight = 0

            for w in value.split():
                if w not in word_weights:
                    word_weights[w] = current_weight
                else:
                    word_weights[w] = max(current_weight, word_weights[w])
        return word_weights

    def compare(row0, row1):
        weights0 = get_word_weights(row0)
        weights1 = get_word_weights(row1)

        total_weight = 0
        missing_weight = 0

        for w in weights0:
            weight = weights0[w]
            if w not in weights1:
                missing_weight += weights0[w]
            else:
                weight = max(weight, weights1[w])
            total_weight += weight

        for w in weights1:
            weight = weights1[w]
            if w not in weights0:
                missing_weight += weights1[w]
            else:
                weight = max(weight, weights0[w])
            total_weight += weight

        if total_weight == 0:
            return null_value

        return float((total_weight - missing_weight) / total_weight)

    if df0.columns.to_list() != df1.columns.to_list():
        raise ValueError("DataFrames must have the same columns")

    # Run compare on rows of each df
    col = np.array(
        [compare(row0, row1) for row0, row1 in zip(df0.to_numpy(), df1.to_numpy())]
    )

    return pd.Series(col)


def length_similarity(se0, se1, null_value):
    se0_np = se0.to_numpy(dtype=str)
    se1_np = se1.to_numpy(dtype=str)

    col = np.array([1 - abs(len(s0) - len(s1)) / max(len(s0), len(s1)) for s0, s1 in zip(se0_np, se1_np)])

    # If either string is empty, set similarity to null_value
    col[(se0_np == "") | (se1_np == "")] = null_value

    return pd.Series(col)


def jaccard_similarity(se0, se1, null_value):
    se0_np = se0.to_numpy(dtype=str)
    se1_np = se1.to_numpy(dtype=str)

    col = np.array([textdistance.jaccard.normalized_similarity(set(s0.split()), set(s1.split())) for s0, s1 in zip(se0_np, se1_np)])

    # If either string is empty, set similarity to null_value
    col[(se0_np == "") | (se1_np == "")] = null_value

    return pd.Series(col)


def similarity_factory(similarity_function):
    def similarity(se0, se1, null_value):
        se0_np = se0.to_numpy(dtype=str)
        se1_np = se1.to_numpy(dtype=str)

        col = np.vectorize(similarity_function)(se0_np, se1_np)

        # Replace original null values with null_value
        col[se0_np == ""] = null_value
        col[se0_np == ""] = null_value

        return pd.Series(col)

    return similarity


token_set_similarity = similarity_factory(
    lambda s0, s1: fuzz.token_set_ratio(s0, s1) / 100
)
token_sort_similarity = similarity_factory(
    lambda s0, s1: fuzz.token_sort_ratio(s0, s1) / 100
)
levenshtein_similarity = similarity_factory(lambda s0, s1: (fuzz.ratio(s0, s1) / 100))
jaro_winkler_similarity = similarity_factory(lambda s0, s1: textdistance.jaro_winkler.similarity(s0, s1))
jaro_similarity = similarity_factory(lambda s0, s1: textdistance.jaro.similarity(s0, s1))
