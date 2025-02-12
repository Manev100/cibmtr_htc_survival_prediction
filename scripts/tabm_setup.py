import sys, os, glob
import math
from itertools import product
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pandas.api.types
from pandas.api.types import CategoricalDtype
import polars as pl
import polars.selectors as cs
from scipy.special import softmax
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from tqdm.auto import tqdm

import rtdl_num_embeddings
from metric import score
from tabm_reference import Model, make_parameter_groups
from target import transform_target_aft, transform_target_cox, transform_target_km_filter, transform_target_1, transform_target_2, transform_target_na_filter

# Pipeline
# 1. cat features (high cardinality features? Don't care)
#       -> Impute with NAN Catgory
#       -> OrdinalEncoder
# 2. num features
# 2.1. 
#    -> Impute with mean
#     -> (1) Standard scaler, (2) Quantile Transformer w/ noise, (3) no norm
# 2.2. Missing indicator 

class RegressionLabelStats(NamedTuple):
    mean: float
    std: float

def preprocess(train, val, test, normalization='quantile', target="target_cox", norm_target: bool = False):
    assert normalization in ['quantile', 'standard', 'passthrough'], f"Unknown normalization type {normalization}, must be 'quantile', 'standard'or 'passthrough'"
    assert train is not None, "train cannot be none"
    assert target in ['target_cox', 'target_km', 'target_na', 'target1', 'target2'], f"Unknown target {target}"
    
    passthrough_cols = ['ID', 'efs', 'efs_time', 'race_group'] + ['target_cox', 'target_km', 'target_na', 'target1', 'target2']
    cat_features = train.select(pl.selectors.string()).select(pl.all().exclude(passthrough_cols)).columns
    num_features = train.select(pl.selectors.numeric()).select(pl.all().exclude(passthrough_cols)).columns

    cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value='NAN'), OrdinalEncoder(dtype=int))

    sample_size = len(train)

    # def add_noise(X):
    #     noise = (
    #         np.random.default_rng(0)
    #         .normal(0.0, 1e-5, X.shape)
    #         .astype(X.to_numpy().dtype)
    #     )
    #     return X.to_numpy() + noise
    
    normalizer = {
        'passthrough': 'passthrough',
        'standard': StandardScaler(), 
        'quantile': make_pipeline(#FunctionTransformer(add_noise, feature_names_out=lambda self, feat_in: feat_in),
                                  QuantileTransformer(n_quantiles=max(min(sample_size // 30, 1000), 10),
                                                        output_distribution='normal',
                                                        subsample=10**9,)
                                  )
    }
    num_pipeline = make_pipeline(SimpleImputer(strategy='mean'),  normalizer[normalization])

    
    
    ct = ColumnTransformer(
        [('cat', cat_pipeline, cat_features),
        ('num', num_pipeline, num_features),
        ('cat_ind', MissingIndicator(error_on_new=False, sparse=False), num_features)],
        verbose_feature_names_out=True
    )

    # passthrough some features to compute competition metric
    ct_pt = ColumnTransformer([('pass','passthrough', passthrough_cols )] ,verbose_feature_names_out=False)

    ft = FeatureUnion(
        [('other', ct),
        ('passthrough', ct_pt)],
        # verbose_feature_names_out=False
    )
    ft.set_output(transform="polars")

    # build expressions to add random noise to num features during fit
    def add_noise(cols, size):
        return [pl.col(col) + pl.Series(np.random.default_rng(idx).normal(0.0, 1e-5, size))   
                    for idx, col in enumerate(cols)]
    if normalization == 'quantile':
        ft.fit(train.with_columns(add_noise(num_features, len(train))))
    else:
        ft.fit(train)

    if norm_target:
        regression_label_stats = RegressionLabelStats(
            train.select(target).mean(), train.select(target).std()
        )

        target_transformer = FunctionTransformer(func=lambda x: (x - regression_label_stats.mean) / regression_label_stats.std,
                            inverse_func=lambda x: x * regression_label_stats.std + regression_label_stats.mean
                            )
    else:
        target_transformer = FunctionTransformer(func=lambda x: x,
                            inverse_func=lambda x: x
                            )
    
    
    cat_features = [col for col in ct.get_feature_names_out() if col.startswith("cat")]
    num_features = [col for col in ct.get_feature_names_out() if col.startswith("num")]

    data = {}
    for part_name, part in zip(["train", "val", "test"], [train, val, test]):
        if part is not None:
            prep = ft.transform(part)
            data[part_name] = {'x_cont': prep.select(num_features).to_numpy().astype(np.float32), 
                               'x_cat': prep.select(cat_features).to_numpy().astype(np.int64),
                               'y': target_transformer.transform(prep.select(target).to_numpy()).flatten(),
                               'metric': prep.select(['ID', 'efs', 'efs_time', 'race_group'] )
                               }
            
    cat_cardinalities = (data["train"]['x_cat'].max(axis=0) + 1).tolist() 
    
    # return ft.transform(train), cat_cardinalities
    return data, cat_cardinalities, target_transformer, len(num_features)



def prepare_model(data_prep, config):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) if k != 'metric' else  v for k, v in data_prep[part].items()}
        for part in data_prep
    }

    # task is regression
    for part in data:
        data[part]['y'] = data[part]['y'].float()


    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True will result in faster training on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = config["compile_model"]

    # fmt: off
    print(
        f'Device:        {device.type.upper()}'
        f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
        f'\ntorch.compile: {compile_model}'
    )
    
    # Choose one of the two configurations below.

    # TabM
    arch_type = config["arch_type"]
    bins = config["bins"]
    
    if config["bins"] == 'rtdl':
        bins = rtdl_num_embeddings.compute_bins(data['train']['x_cont'], n_bins=config["n_ple_bins"])

    # TabM-mini with the piecewise-linear embeddings.
    # arch_type = 'tabm-mini'
    # bins = rtdl_num_embeddings.compute_bins(data['train']['x_cont'])

    model = Model(
        n_num_features=config["n_cont_features"],
        cat_cardinalities=config["cat_cardinalities"],
        n_classes=config["n_classes"],
        backbone={
            'type': 'MLP',
            'n_blocks': config["n_layers"],
            'd_block': config["width"],
            'dropout': config["dropout"],
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                'type': 'PiecewiseLinearEmbeddings',
                'd_embedding': 16,
                'activation': False,
                'version': 'B',
            }
        ),
        arch_type=arch_type,
        k=config["k"],
    ).to(device)
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode
        
    model_dict = {"model": model,
                  "eval_mode": evaluation_mode,
                  "optimizer": optimizer,
                  "device": device,
                  "grad_scaler": grad_scaler,
                  "amp_enabled": amp_enabled,
                  "amp_dtype": amp_dtype,
                  "target_transformer": config["target_transformer"],
                  "target": config["target"]
                  }
        
    return model_dict, data



def train(model_dict, data, verbose=False):
    model = model_dict["model"]
    optimizer = model_dict["optimizer"]
    evaluation_mode = model_dict["eval_mode"]
    device = model_dict["device"]
    grad_scaler = model_dict["grad_scaler"]
    amp_enabled = model_dict["amp_enabled"]
    amp_dtype = model_dict["amp_dtype"]
    target_transformer = model_dict["target_transformer"]
    target = model_dict["target"]
    
    
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_cont'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )

    task_type = "regression"
    base_loss_fn = F.mse_loss if task_type == 'regression' else F.cross_entropy


    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions per object. Each of them must be trained separately.
        # (regression)     y_pred.shape == (batch_size, k)
        # (classification) y_pred.shape == (batch_size, k, n_classes)
        k = y_pred.shape[-1 if task_type == 'regression' else -2]
        return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))


    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        if task_type == 'regression':
            # Transform the predictions back to the original label space.
            assert target_transformer is not None
            y_pred = target_transformer.inverse_transform(y_pred) 

        # Compute the mean of the k predictions.
        if task_type != 'regression':
            # For classification, the mean must be computed in the probabily space.
            y_pred = softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)

        y_true = data[part]['y'].cpu().numpy()
        
        if task_type == 'regression':
            y_true = data[part]["metric"].to_pandas().copy()
            y_pred_df = data[part]["metric"].to_pandas()[["ID"]].copy()
            if target in ["target_na", "target1"]:
                y_pred_df["prediction"] = -y_pred
            else:
                y_pred_df["prediction"] = y_pred
            m = score(y_true.copy(), y_pred_df.copy(), "ID")
        
        sc = (
            # -(mean_squared_error(y_true, y_pred) ** 0.5)
            m
            if task_type == 'regression'
            else accuracy_score(y_true, y_pred.argmax(1))
        )
        return float(sc)  # The higher -- the better.

    if verbose:
        print(f'Test score before training: {evaluate("test"):.4f}')
    
    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2
    n_epochs = 1_000_000_000
    patience = 16

    batch_size = 256
    epoch_size = math.ceil(len(data["train"]["x_cont"]) / batch_size)
    best = {
        'val': -math.inf,
        'test': -math.inf,
        'epoch': -1,
    }
    # Early stopping: the training stops when
    # there are more than `patience` consequtive bad updates.
    patience = 16
    remaining_patience = patience

    if verbose:
        print('-' * 88 + '\n')
    for epoch in range(n_epochs):
        for batch_idx in tqdm(
            torch.randperm(len(data['train']['y']), device=device).split(batch_size),
            desc=f'Epoch {epoch}',
            total=epoch_size,
            disable=not verbose
        ):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), data["train"]["y"][batch_idx])
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()

        val_score = evaluate('val')
        test_score = evaluate('test')
        if verbose:
            print(f'(val) {val_score:.4f} (test) {test_score:.4f}')

        if val_score > best['val']:
            if verbose:
                print('🌸 New best epoch! 🌸')
            best = {'val': val_score, 'test': test_score, 'epoch': epoch}
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break
        
        if verbose:
            print()
    if verbose:
        print('\n\nResult:')
        print(best)
    return best

if __name__ == "__main__":
    from target import *
    COMP_DATA_BASE = os.path.join("data", "comp")
    TRAIN_PATH = os.path.join(COMP_DATA_BASE, "train.csv")
    train_ds = pl.read_csv(TRAIN_PATH)
    train_ds = (train_ds
                    # .pipe(transform_target_aft)
                    .pipe(transform_target_cox)
                    .pipe(transform_target_km_filter)
                    .pipe(transform_target_na_filter)
                    .pipe(transform_target_1)
                    .pipe(transform_target_2))
    
    train_data, test_data = train_test_split(train_ds, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

    normalization = 'quantile'
    target = "target_na"
    norm_target = False

    data_prep, cat_cardinalities, target_transformer, n_cont_features = preprocess(train_data, 
                                                                                val_data, 
                                                                                test_data, 
                                                                                normalization=normalization, 
                                                                                target=target, 
                                                                                norm_target=norm_target)
    
    CONFIG = {
        "arch_type": 'tabm', # 'tabm' or 'tabm-mini'
        "bins": 'rtdl', # None or 'rtdl'
        "compile_model": False,
        
        "n_cont_features": n_cont_features,
        "cat_cardinalities": cat_cardinalities,
        "target_transformer": target_transformer,
        "n_classes": None,
        "target": target
    }
    
    space = {
        'dropout': 0.19244279421349406, 
        'k': 32, 
        'learning_rate': 0.0030191433786342136, 
        'n_layers': 3, 
        'n_ple_bins': 11, 
        'weight_decay': 0.0, 
        'width': 204
    }
    
    model_dict, data = prepare_model(data_prep, config=CONFIG | space)
    results = train(model_dict, data)
    
    model = model_dict["model"]
    target_transformer = model_dict["target_transformer"]
    
    print(f"Run result: val {results['val']} - test {results['test']} - epochs {results['epoch']}" )
    
        