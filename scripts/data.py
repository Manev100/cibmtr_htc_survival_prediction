import polars as pl
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import metrics
from category_encoders import CountEncoder
from category_encoders.hashing import HashingEncoder

from target import transform_target_aft, transform_target_cox, transform_target_km_filter, transform_target_1, transform_target_2

def create_preprocessor (config: dict = {"numeric_type": None, "low_cardinality_type": None, "high_cardinality_type": None},
                         features: dict = {"passthrough_features": [], "numeric_features":[], "low_cardinality_features": [], "high_cardinality_features": []},
                         cat_feature_categories: dict = {"low_card_feat_cats": [], "high_card_feat_cats": []}, 
                         impute=False, 
                         seed=24, 
                         num_cat_cutoff=10):

    numeric_type = config["numeric_type"]
    low_cardinality_type = config["low_cardinality_type"]
    high_cardinality_type = config["high_cardinality_type"]
    assert numeric_type in [None, "quantile", "robust", "minmax", "standard"], f"Numeric type {numeric_type} undefined"
    assert low_cardinality_type in [None, "target", "ohe", "frequency"], f"low cardinality cat type {low_cardinality_type} undefined"
    assert high_cardinality_type in [None, "target", "ohe", "frequency", "hashing"], f"high cardinality cat type {high_cardinality_type} undefined"

    passthrough_features = features["passthrough_features"]
    numeric_features = features["numeric_features"]
    low_cardinality_features = features["low_cardinality_features"]
    high_cardinality_features = features["high_cardinality_features"]
    
    low_card_feat_cats = cat_feature_categories["low_card_feat_cats"]
    high_card_feat_cats = cat_feature_categories["high_card_feat_cats"]
    
    # steps for numeric features with imputation
    imputer = make_pipeline('passthrough')
    if impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True, keep_empty_features=True)
    
    num_transfomers = {"quantile": QuantileTransformer(subsample=None, random_state=seed), 
                        "robust": RobustScaler(), 
                        "minmax": MinMaxScaler(), 
                        "standard": StandardScaler()}
    num_encoder = "passthrough"
    if numeric_type is not None:
        num_encoder = num_transfomers[numeric_type]
        
    numeric_pipeline = make_pipeline(imputer, num_encoder)
    
    # Steps for low cardinality features
    low_card_transformers = {"target": TargetEncoder(categories=low_card_feat_cats, random_state=seed), 
                                "ohe": OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, categories=low_card_feat_cats), 
                                "frequency": CountEncoder()
        }
    
    low_card_encoder = "passthrough"
    if low_cardinality_type is not None: 
        low_card_encoder = low_card_transformers[low_cardinality_type]
    
    # Steps for high cardinality features
    high_card_transformers = {"target": TargetEncoder(categories=high_card_feat_cats, random_state=seed), 
                                "ohe": OneHotEncoder(max_categories=num_cat_cutoff, handle_unknown="infrequent_if_exist", sparse_output=False, categories=high_card_feat_cats), 
                                "frequency": CountEncoder(),
                                "hashing": HashingEncoder(n_components=8)
        }
    high_card_encoder = "passthrough"
    if high_cardinality_type is not None: 
        high_card_encoder = high_card_transformers[high_cardinality_type]
    
    
    
    mixed_encoded_preprocessor = ColumnTransformer(
    [
        ("special", "passthrough", passthrough_features),
        ("numerical", numeric_pipeline, numeric_features),
        (
            "high_cardinality",
            high_card_encoder,
            high_cardinality_features,
        ),
        (
            "low_cardinality",
            low_card_encoder,
            low_cardinality_features,
        ),
    ],
    verbose_feature_names_out=False,
    )
    
    mixed_encoded_preprocessor.set_output(transform="pandas")
    
    return mixed_encoded_preprocessor

def preprocess(train_data: pl.DataFrame, test_data: pl.DataFrame, target, data_config, features, cat_feature_categories, impute, seed=24):  
    
    # cast to pandas and apply category data type, need to apply categorization to test data for consistency
    train_pd = train_data.to_pandas().astype({col: CategoricalDtype() for col in features["low_cardinality_features"] + features["high_cardinality_features"]})
    test_pd = test_data.to_pandas().astype({col: CategoricalDtype(train_pd[col].cat.categories) for col in features["low_cardinality_features"] + features["high_cardinality_features"]})
    
    mixed_encoded_preprocessor = create_preprocessor(config=data_config, features=features, cat_feature_categories=cat_feature_categories, impute=impute, seed=seed)
    train_prep_ds = mixed_encoded_preprocessor.fit_transform(train_pd, train_data.select(target).to_series().to_pandas())
    test_prep_ds = mixed_encoded_preprocessor.transform(test_pd)
    
    # fix invalid columns
    invalid_cols = [col for col in train_prep_ds.columns if "[" in col or "]" in col or "<" in col]
    train_prep_ds = train_prep_ds.rename(columns={col: col.replace("<", "").replace("[", "").replace("]", "")  for col in invalid_cols})
    test_prep_ds = test_prep_ds.rename(columns={col: col.replace("<", "").replace("[", "").replace("]", "")  for col in invalid_cols})
    # get all feature columns
    feature_columns = list(train_prep_ds.columns)
    
    # Add extra columns back 
    for col in features["special_features"]:
        if col not in train_prep_ds.columns:
            train_prep_ds[col] = train_data[col]
            if col in test_data.columns:
                test_prep_ds[col] = test_data[col]                
                        
    return train_prep_ds, test_prep_ds, feature_columns

# config: dict = {"numeric_type": numeric_type, "low_cardinality_type": low_cardinality_type, "high_cardinality_type": high_cardinality_type},
def prepare_data(train_ds: pl.DataFrame, data_dict: pl.DataFrame, test_ds: pl.DataFrame | None = None, target_col="target_cox", data_config: dict = {"numeric_type": None, "low_cardinality_type": None, "high_cardinality_type": None}, seed=42):
    train_ds = train_ds.pipe(transform_target_aft).pipe(transform_target_cox).pipe(transform_target_km_filter).pipe(transform_target_1).pipe(transform_target_2)
    
    features = data_dict.select(pl.col("variable")).filter(pl.col("variable").is_in(["efs", "efs_time"]).not_()).to_series().to_list()
    
    num_features_dict = data_dict.filter(pl.col("variable").is_in(features) & (pl.col("type") == "Numerical"))
    num_features = num_features_dict.select("variable").to_series().to_list()
    cat_features_dict = data_dict.filter(pl.col("variable").is_in(features) & (pl.col("type") == "Categorical"))
    cat_features = cat_features_dict.select("variable").to_series().to_list()
    cat_features_categories = {key: train_ds.select(key).to_series().unique().to_list()  for key in cat_features if key in train_ds.columns}
    passthrough_features = []
    special_features = ["ID","efs","efs_time","race_group", target_col]
    
    NUM_CAT_CUTOFF = 10
    high_cardinality_features = [col for col in train_ds.columns if col in cat_features_categories and len(cat_features_categories[col]) > NUM_CAT_CUTOFF ]
    low_cardinality_features = [col for col in train_ds.columns if col in cat_features_categories and len(cat_features_categories[col]) <= NUM_CAT_CUTOFF]
    all_features = num_features + low_cardinality_features + high_cardinality_features

    low_card_feat_cats = [cat_features_categories[col] for col in low_cardinality_features]
    high_card_feat_cats = [cat_features_categories[col] for col in high_cardinality_features]
    
    
    features_dict = {"passthrough_features": passthrough_features, "special_features": special_features, "numeric_features": num_features, "low_cardinality_features": low_cardinality_features, "high_cardinality_features": high_cardinality_features}
    cat_feature_categories = {"low_card_feat_cats": low_card_feat_cats, "high_card_feat_cats": high_card_feat_cats}
    
    if test_ds is None:
        train_data, test_data = train_test_split(train_ds, test_size=0.1, random_state=seed)
    else:
        train_data = train_ds
        test_data = test_ds
    
    train_prep_ds, test_prep_ds, feature_cols = preprocess(train_data, test_data, target_col, data_config=data_config, features=features_dict, cat_feature_categories=cat_feature_categories, impute=False, seed=seed)
    
    return train_prep_ds, test_prep_ds, feature_cols
    