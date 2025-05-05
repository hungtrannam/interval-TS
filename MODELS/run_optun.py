import argparse
import optuna
import yaml
import random
import torch
import numpy as np
from datetime import datetime
from exp.exp_main import Exp_Main
from utils.tools import set_seed
import os
from utils.params import suggest_model_specific
from utils.vis import *


# ========================
# Objective function for Optuna
# ========================
def run_experiment(args, plot=False):
    Exp = Exp_Main
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    setting = f'{args.model_id}_{args.data_path}_{args.model}_{timestamp}'

    exp = Exp(args)

    if args.is_training:
        print(f'>>>>>>> Start Training: {setting} >>>>>>>>>>>>>>>>>>>>')
        exp.train(setting, plot=plot)

    print(f'>>>>>>> Testing/Predict: {setting} <<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1, plot=plot)

    # Clean GPU Cache
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()

    print(f'>>>>>>> Finished Experiment: {setting} <<<<<<<<<<<<<<<<<<<<<<<')

    return exp, setting



def objective(trial):
    parser = argparse.ArgumentParser()

    # Basic Info
    parser.add_argument('--task_name', type=str, default='short_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='optuna_tune')
    parser.add_argument('--model', type=str, default=os.environ.get("MODEL_NAME", "Autoformer"))
    parser.add_argument('--des', type=str, default='optuna')

    # Data
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='/home/hung-tran-nam/INTERVAL/dataset')
    parser.add_argument('--data_path', type=str, default='CO2.csv')
    parser.add_argument('--target', type=str, default='lower,upper')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=trial.suggest_categorical('seq_len', [32, 48, 72]))
    parser.add_argument('--label_len', type=int, default=trial.suggest_categorical('label_len', [6, 12, 16]))
    parser.add_argument('--pred_len', type=int, default=7)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # Learning
    parser.add_argument('--dropout', type=float, default=trial.suggest_float('dropout', 0.0, 0.3))
    parser.add_argument('--lradj', type=str, default=trial.suggest_categorical('lradj', ['type1', 'type2', 'type3']), help='adjust learning rate')
    parser.add_argument('--learning_rate', type=float, default=trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True))
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--activation', type=str, default=trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh']), help='activation function')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--seg_len', type=int, default=24, help='segment length for SegRNN')


    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)


    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=True,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment") #NEED
    parser.add_argument('--seed', type=int, default=42, help="Randomization seed")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")  #NEED
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")  #NEED    
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")  #NEED
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")  #NEED
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")    

    # Fixed value
    args, _ = parser.parse_known_args()

    # Fixed param for model requirement
    if args.embed == 'timeF':
        args.enc_in = 11
        args.dec_in = 11
    elif args.embed == 'monthSine':
        args.enc_in = 6+2
        args.dec_in = 6+2
    args.c_out = 2
    args.use_amp = False
    args.inverse = True
    args.use_gpu = False
    args.gpu_type = 'cuda'
    args.gpu = 0
    args.batch_size = 16
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    # Random Seed
    set_seed(args.seed)
    args.__dict__.update(suggest_model_specific(trial, args.model))

    # Detect Device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '').split(',')
        args.device_ids = [int(id_) for id_ in args.devices]
        args.gpu = args.device_ids[0]

    set_seed(args.seed)
    exp, _ = run_experiment(args, plot=False)
    val_loss = exp.vali(*exp._get_data('val'), exp._select_criterion())
    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=2)
    )    
    
    # module = optunahub.load_module(package="samplers/auto_sampler")
    # study = optuna.create_study(
    #     direction='minimize',
    #     sampler=module.AutoSampler()
    #     pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=2)
    # )
    study.optimize(objective, n_trials=3)

    print("Best Trial Result:")
    print(study.best_trial)

    best_params = study.best_trial.params

    # ------------------ BUILD FULL ARGUMENTS ------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='best_optuna')
    parser.add_argument('--model', type=str, default=os.environ.get("MODEL_NAME", "Autoformer"))
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='/home/hung-tran-nam/SWAT_AIv2v/dataset/DataSet_raw')
    parser.add_argument('--data_path', type=str, default='pre_ChiangSaen_spi_6.csv')
    parser.add_argument('--target', type=str, default='spi_6')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='m')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--des', type=str, default='optuna')
    parser.add_argument('--train_epochs', type=int, default=50,  help='train epochs')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--seg_len', type=int, default=24, help='segment length for SegRNN')
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--use_dtw', default=True)

    # TimesBlock / FEDformer / De-stationary models
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    parser.add_argument('embed', type=str)
    # Augment
    parser.add_argument('--seed', type=int, default=42)

    args, _ = parser.parse_known_args()

    args.__dict__.update(best_params)
    args.__dict__.update(suggest_model_specific(study.best_trial, args.model))  # <- bổ sung đúng trial + model
    args.c_out = 2 

    # Các giá trị cố định
    if args.embed == 'timeF':
        args.enc_in = 11
        args.dec_in = 11
    elif args.embed == 'monthSine':
        args.enc_in = 6+2
        args.dec_in = 6+2
    args.patience = 5
    args.batch_size = 16
    args.inverse = True

    # Device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

    # Set seed
    set_seed(args.seed)

    print("Run experiment with best params and plot result")
    exp, setting = run_experiment(args, plot=True)

    dataset, _ = exp._get_data(flag='train')
    plot_input(dataset, save_path=f'./test_results/{setting}/INplot.pdf')
    plot_target_interval(dataset.df_target)

    # Vẽ các biểu đồ phân tích kết quả
    plot_metrics(study, './test_results/' + setting)
    plot_hyperparameter_importance(study, './test_results/' + setting)
    plot_optimization_history(study, './test_results/' + setting)

    # Vẽ scatter giữa ground truth và prediction
    y_true = np.load(f'./results/{setting}/true.npy')
    y_pred = np.load(f'./results/{setting}/pred.npy')
    plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/{setting}/PredScatter.pdf')

    # Lưu best config
    args_save_path = f'./test_results/{setting}/best.yaml'
    args_dict = vars(args)
    with open(args_save_path, 'w') as f:
        yaml.dump(args_dict, f, sort_keys=False)


