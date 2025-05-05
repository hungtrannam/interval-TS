import optuna.visualization.matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm

def set_plot_style():
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'mathtext.fontset': 'cm', 
        'axes.linewidth': 1,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.fontsize': 13,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

def visual_interval(true, preds=None, name='./pic/test_interval.pdf', input_len=None):
    """
    Vẽ khoảng (lower, upper) và prediction khoảng.
    """
    set_plot_style()
    plt.figure(figsize=(15, 6))

    # Tách lower và upper
    true_lower = true[:, 0]
    true_upper = true[:, 1]

    plt.plot(true_lower, label='GroundTruth Lower', color='blue', linewidth=2)
    plt.plot(true_upper, label='GroundTruth Upper', color='blue', linestyle='--', linewidth=2)
    plt.fill_between(range(len(true_lower)), true_lower, true_upper, color='blue', alpha=0.2)

    if preds is not None:
        pred_lower = preds[:, 0]
        pred_upper = preds[:, 1]

        plt.plot(pred_lower, label='Prediction Lower', color='orange', linewidth=2)
        plt.plot(pred_upper, label='Prediction Upper', color='orange', linestyle='--', linewidth=2)
        plt.fill_between(range(len(pred_lower)), pred_lower, pred_upper, color='orange', alpha=0.2)

    if input_len is not None and input_len < len(true_lower):
        plt.axvline(input_len - 1, color='gray', linestyle='--', alpha=0.7)

    plt.xlabel("Time Steps")
    plt.ylabel("Target Values")
    plt.title("GroundTruth vs Prediction (Interval)")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_loss(train_losses, val_losses, name='./pic/loss.pdf'):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.semilogy(epochs, train_losses, label='Train Loss (log)', color='blue', linestyle='--', linewidth=3)
    plt.semilogy(epochs, val_losses, label='Validation Loss (log)', color='orange', linewidth=4)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    print(f"Saved loss curve at {name}")
    plt.close()

def plot_metrics(study, save_dir):
    set_plot_style()
    trials = study.trials_dataframe()

    best_trial_idx = trials['value'].idxmin()
    best_val = trials['value'].min()

    plt.figure(figsize=(8, 5))
    plt.plot(trials['value'], marker='o', label='Validation Loss (MSE)')
    plt.scatter(best_trial_idx, best_val, color='red', zorder=5, label=f'Best Trial #{best_trial_idx} ({best_val:.4f})')
    plt.axvline(x=best_trial_idx, color='red', linestyle='--', alpha=0.6)

    plt.xlabel('Trial')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Validation Loss per Trial')
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'ValLoss.pdf'))
    plt.close()

    for metric in ['user_attrs_dtw', 'user_attrs_mae']:
        if metric in trials.columns:
            plt.figure()
            plt.plot(trials[metric], marker='o')
            plt.xlabel('Trial')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} per Trial')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric.upper()}_trial.pdf'))
            plt.close()

def plot_hyperparameter_importance(study, save_dir):
    set_plot_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.figure.savefig(os.path.join(save_dir, 'HyImpo.pdf'))
    print(f"Saved Hyperparameter Importance at {save_dir}/HyImpo.pdf")
    plt.tight_layout()
    plt.close()

def plot_optimization_history(study, save_dir):
    set_plot_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig.figure.savefig(os.path.join(save_dir, 'OptHistory.pdf'))
    print(f"Saved Hyperparameter Importance at {save_dir}/OptHistory.pdf")
    plt.tight_layout()
    plt.close()

def plot_scatter_truth_vs_pred(y_true, y_pred, save_path='./PredScatter.pdf'):
    from utils.metrics import metric
    set_plot_style()

    # Tính metrics
    mae, mse, rmse, mape, mspe, nse = metric(y_pred, y_true)

    # Xác định số batch (giả định y_true: [num_batches, batch_size, 1])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_batches = y_true.shape[0]

    # Tạo colormap để mỗi batch 1 màu
    colors = cm.get_cmap('tab10', num_batches)

    # Vẽ scatter từng batch với màu riêng
    plt.figure(figsize=(6, 6))
    for i in range(num_batches):
        plt.scatter(
            y_true[i].flatten(),
            y_pred[i].flatten(),
            alpha=0.5,
            label=f'Batch {i+1}',
            color=colors(i)
        )

    # Đường hoàn hảo
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    # Ghi chú metrics
    metrics_text = (
        f'MAE:  {mae:.4f}\n'
        f'MSE:  {mse:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAPE: {mape:.2f}%\n'
        f'MSPE: {mspe:.2f}%\n'
        f'NSE:  {nse:.4f}'
    )
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('Prediction vs. Ground Truth')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_input(dataset, save_path='./figs/input_scaled_plot.pdf'):
    file_path = os.path.join(dataset.root_path, dataset.data_path)
    df_raw = pd.read_csv(file_path)

    # Xác định input columns
    if dataset.features == 'MS':
        input_cols = list(df_raw.columns)
        input_cols.remove('date')
    elif dataset.features == 'M':
        input_cols = list(df_raw.columns)
        input_cols.remove('date')
        for t in dataset.target:
            if t in input_cols:
                input_cols.remove(t)
    elif dataset.features == 'S':
        input_cols = dataset.target if isinstance(dataset.target, list) else [dataset.target]

    # Tự động xử lý target
    target_list = dataset.target if isinstance(dataset.target, list) else [dataset.target]

    # Chuẩn bị dữ liệu
    df_input_all = df_raw[input_cols]
    data_scaled_input = dataset.scaler.transform(df_input_all.values)
    df_target = df_raw[target_list]

    # Các cột cần vẽ
    plot_cols = input_cols
    num_plot = len(plot_cols) + len(target_list)

    train_border = dataset.num_train
    val_border = train_border + dataset.num_vali
    test_border = val_border + dataset.num_test

    plt.figure(figsize=(20, 3 * num_plot))

    def draw_split_lines():
        y_top = plt.ylim()[1]
        plt.axvline(train_border, color='gray', linestyle='--')
        plt.axvline(val_border, color='gray', linestyle='--')
        plt.text(train_border / 2, y_top * 0.9, 'Train', ha='center', color='gray')
        plt.text((train_border + val_border) / 2, y_top * 0.9, 'Val', ha='center', color='gray')
        plt.text((val_border + test_border) / 2, y_top * 0.9, 'Test', ha='center', color='gray')

    # Vẽ input features gốc và scaled
    for i, col in enumerate(plot_cols):
        if col not in input_cols:
            continue
        col_idx = input_cols.index(col)

        plt.subplot(num_plot, 2, i * 2 + 1)
        plt.plot(df_raw[col], label='Original', linewidth=3)
        draw_split_lines()
        plt.title(f'Original: {col}')
        plt.grid()

        plt.subplot(num_plot, 2, i * 2 + 2)
        plt.plot(data_scaled_input[:, col_idx], label='Scaled', color='orange', linewidth=3)
        draw_split_lines()
        plt.title(f'Scaled: {col}')
        plt.grid()

    # Vẽ target (original và scaled nếu có)
    for j, t in enumerate(target_list):
        row = len(plot_cols) + j
        plt.subplot(num_plot, 2, row * 2 + 1)
        plt.plot(df_target[t], label=f'Original {t}', linewidth=3)
        draw_split_lines()
        plt.title(f'Original: {t}')
        plt.grid()

        if t in input_cols:
            target_idx = input_cols.index(t)
            plt.subplot(num_plot, 2, row * 2 + 2)
            plt.plot(data_scaled_input[:, target_idx], label=f'Scaled {t}', color='orange', linewidth=3)
            draw_split_lines()
            plt.title(f'Scaled: {t}')
            plt.grid()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_target_interval(df_target, input_cols=None, data_scaled_input=None, save_path=None, title='Target Interval (High-Low)', show=False):
    """
    Vẽ khoảng giá trị giữa High và Low (gốc và scaled nếu có)

    Parameters:
        df_target: DataFrame chứa các cột 'High' và 'Low'
        input_cols: danh sách các cột input đã dùng trong model
        data_scaled_input: numpy array chứa input đã scale (nếu có)
        save_path: đường dẫn lưu file hình (nếu có)
        title: tiêu đề chính
        show: có hiển thị hình hay không
    """
    plt.figure(figsize=(14, 6))

    # Original
    plt.subplot(1, 2, 1)
    x = df_target.index
    plt.plot(x, df_target['High'], label='High', color='red', linewidth=2)
    plt.plot(x, df_target['Low'], label='Low', color='blue', linewidth=2)
    plt.fill_between(x, df_target['Low'], df_target['High'], color='gray', alpha=0.3, label='Range')
    plt.title('Original Target (High-Low)')
    plt.legend()
    plt.grid()

    # Scaled (nếu có)
    if input_cols and data_scaled_input is not None and 'High' in input_cols and 'Low' in input_cols:
        idx_high = input_cols.index('High')
        idx_low = input_cols.index('Low')

        plt.subplot(1, 2, 2)
        x_scaled = np.arange(data_scaled_input.shape[0])
        plt.plot(x_scaled, data_scaled_input[:, idx_high], label='Scaled High', color='red', linewidth=2)
        plt.plot(x_scaled, data_scaled_input[:, idx_low], label='Scaled Low', color='blue', linewidth=2)
        plt.fill_between(x_scaled, data_scaled_input[:, idx_low], data_scaled_input[:, idx_high],
                         color='gray', alpha=0.3, label='Scaled Range')
        plt.title('Scaled Target (High-Low)')
        plt.legend()
        plt.grid()

    plt.suptitle(title, fontsize=14)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[INFO] Saved target plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
