import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

def suggest_model_specific(trial, model):
    params = {}

    # Common expanded options
    d_model_options = [16, 32, 64, 128]
    n_heads_options = [1, 2, 4, 8, 16]
    d_ff_options = [16, 32, 64, 128]
    moving_avg_options = [3, 5, 7, 9, 11, 13, 15]
    patch_len_options = [8, 16, 32, 64, 96, 128]
    factor_options = [1, 2, 3, 4, 5, 6, 7]

    if model == 'Autoformer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)

        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
            'moving_avg': trial.suggest_categorical('moving_avg', moving_avg_options),
        }

    elif model == 'Informer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }
    elif model == 'Crosformer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }
    elif model == 'TiDE':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }
    elif model == 'Reformer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)


        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }

    elif model == 'Transformer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)


        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }

    elif model == 'PatchTST':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)


        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
            'patch_len': trial.suggest_categorical('patch_len', patch_len_options),
        }

    elif model == 'Nonstationary_Transformer':
        d_model = trial.suggest_categorical('d_model', d_model_options)
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)


        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', 2, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 6),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_options),
            'factor': trial.suggest_categorical('factor', factor_options),
        }

    elif model == 'LSTM':
        params = {
            'e_layers': trial.suggest_int('e_layers', 1, 6),
            'd_layers': trial.suggest_int('d_layers', 1, 4),
            'd_model': trial.suggest_categorical('d_model', d_model_options),
        }

    elif model == 'DLinear':
        params = {
            'moving_avg': trial.suggest_categorical('moving_avg', moving_avg_options),
        }

    else:
        raise ValueError(f"Model {model} not supported in suggest_model_specific!")

    return params
