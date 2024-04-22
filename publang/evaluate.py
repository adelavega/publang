""" Helper functions to evaluate the performance of the entity extraction models """

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def isin(a, li):
    """Nan safe is in function,  that returns second value.
    This is useful because np.nan != np.nan
    """
    for b in li:
        if (pd.isna(a) & pd.isna(b)) or a == b:
            return b

    return False


def score_columns(true_df, predict_df, scoring="mpe"):
    """Score columns of two dataframes, aggregatingby pmcid.
    Args
    ----
    true_df: pd.DataFrame
        Dataframe with true values
    predict_df: pd.DataFrame
        Dataframe with predicted values

    Returns
    -------
    res_mean: dict
        Dictionary with score for each column, aggregated by mean
    res_sum: dict
        Dictionary with score for each column, aggregated by sum
    counts: dict
        Dictionary with percentage of pmcids with overlap for each column, 
        for studies that have both true and predicted values
    """

    if scoring == "mpe":
        scorer = mean_absolute_percentage_error
    elif scoring == "r2":
        scorer = r2_score

    res_mean = {}
    res_sum = {}
    counts = {}
    # Only look at columns if dtype is int or float
    true_df = true_df.select_dtypes(include=[np.int64, np.float64])
    predict_df = predict_df.select_dtypes(include=[np.int64, np.float64])

    true_df_sum = true_df.groupby("pmcid").sum()
    predict_df_sum = predict_df.groupby("pmcid").sum()

    true_df_mean = true_df.groupby("pmcid").mean()
    predict_df_mean = predict_df.groupby("pmcid").mean()

    for col in true_df_mean:
        # Compute overlap using mean
        # Using mean results in NAs when aggregated values are both nan
        true_df_mean_col = true_df_mean[col].dropna()
        predict_df_mean_col = predict_df_mean[col].dropna()

        n_annots = true_df_mean_col.shape[0]

        # Index of rows where both true_df and predict_df are not nan
        ix = true_df_mean_col.index.intersection(predict_df_mean_col.index)

        if len(ix) == 0:
            continue

        # Mean aggregation
        true_df_mean_col = true_df_mean_col.loc[ix]
        predict_df_mean_col = predict_df_mean_col.loc[ix]

        # Sum aggregation
        true_df_sum_col = true_df_sum[col].loc[ix]
        predict_df_sum_col = predict_df_sum[col].loc[ix]

        res_mean[col] = np.round(scorer(true_df_mean_col, predict_df_mean_col), 2)

        res_sum[col] = np.round(scorer(true_df_sum_col, predict_df_sum_col), 2)

        counts[col] = np.round(len(ix) / n_annots, 2)

    return res_mean, res_sum, counts


def hungarian_match_compare(true_df, predict_df):
    """Compare two dataframes by matching rows using the Hungarian algorithm."""
    compare_col = list(set(true_df.columns) & set(predict_df.columns) - {"pmcid"})
    res = defaultdict(float)
    for pmcid, df in true_df.groupby("pmcid"):
        for col in df:
            if col in compare_col:
                match_predict_df = predict_df[predict_df.pmcid == pmcid][col].to_list()
                score = 0
                for v in df[col]:
                    rem_val = isin(v, match_predict_df)
                    if rem_val:
                        score += 1
                        match_predict_df.remove(rem_val)
                res[col] += score

    res = {k: np.round(1 - (v / len(true_df)), 2) for k, v in res.items()}

    return res
