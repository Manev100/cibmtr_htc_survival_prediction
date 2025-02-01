import polars as pl
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

# Transform aft target for xgboost
# Note that it is not yet possible to set the ranged label using the scikit-learn interface (e.g. xgboost.XGBRegressor). For now, you should use xgboost.train with xgboost.DMatrix
def transform_target_aft(data):
    return data.with_columns(pl.col("efs_time").alias("label_lower_bound"), 
                             pl.when(pl.col("efs").cast(pl.Int64) == 1)
                                .then(pl.col("efs_time"))
                                .otherwise(pl.lit(+np.inf))
                                .alias("label_upper_bound")
    )

# transform_target(train_ds)
# objective='survival:cox',
# eval_metric='cox-nloglik',
def transform_target_cox(data):
    train = data.to_pandas()
    train["efs_time2"] = train.efs_time.copy()
    train.loc[train.efs==0,"efs_time2"] *= -1
    return data.with_columns(pl.Series(train["efs_time2"]).alias("target_cox"))


# Transform target with Kaplan Meier Fitter
# eval_metric="mae",
# objective='reg:logistic',
# (pd.Series(oof_xgb).rank(pct=True))*0.5+pd.Series(oof_cat).rank(pct=True)*0.5
def transform_target_km_filter(data):
    def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], df[event_col])
        y = kmf.survival_function_at_times(df[time_col]).values
        
        # Adjust for censoring
        # censored_mask = df[event_col] == 0
        # y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
        return y

    t = transform_survival_probability(data.to_pandas(), time_col='efs_time', event_col='efs')
    return data.with_columns(pl.Series(t).alias("target_km"))


# Transform target #1
# XGBRegressor
# eval_metric="mae",
# objective='reg:logistic'
# -pred_xgb
# https://www.kaggle.com/code/cdeotte/xgboost-catboost-baseline-cv-668-lb-668
def transform_target_1(data):
    train = data.to_pandas()
    train["y"] = train.efs_time.values
    mx = train.loc[train.efs==1,"efs_time"].max()
    mn = train.loc[train.efs==0,"efs_time"].min()
    train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
    train.y = train.y.rank()
    train.loc[train.efs==0,"y"] += len(train)//2
    train.y = train.y / train.y.max()
    return data.with_columns(pl.Series(train.y).alias("target1"))



# transform target #2
# https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676#EDA-on-Train-Targets
# pred
def transform_target_2(data):
    train = data.to_pandas()
    train["y"] = train.efs_time.values
    mx = train.loc[train.efs==1,"efs_time"].max()
    mn = train.loc[train.efs==0,"efs_time"].min()
    train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
    train.y = train.y.rank()
    train.loc[train.efs==0,"y"] += 2*len(train)
    train.y = train.y / train.y.max()
    train.y = np.log( train.y )
    train.y -= train.y.mean()
    train.y *= -1.0
    return data.with_columns(pl.Series(train.y).alias("target2"))