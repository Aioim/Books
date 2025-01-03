import numpy as np
import torch
import shutil
import math
from matplotlib import pyplot as plt


# 创建训练数据集
def create_dataset(data, label_col, step=1, additional=0):
    data = data
    data_copy = data.copy()
    data_copy[:, label_col] = 0
    T, col_nums = data.shape
    features = np.zeros((step + additional, T - step - additional, col_nums))
    for i in range(step + additional):
        if i < step:
            features[i, :, :] = data[i : T - step - additional + i, :]
        else:
            features[i, :, :] = data_copy[i : T - step - additional + i, :]
    labels = data[step : T - additional, [label_col]].reshape((-1, 1))
    return features.transpose(1, 0, 2), labels


def backup_create_dataset(data, label_col, step=1):
    T, col_nums = data.shape
    features = np.zeros((step, T - step, col_nums))
    for i in range(step):
        features[i, :, :] = data[i : T - step + i, :]
    labels = data[step:T, [label_col]].reshape((-1, 1))
    return features.transpose(1, 0, 2), labels, features


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_utils`"""
    # load_array((features[:n_train], labels[:n_train]),batch_size, is_train=True)
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def check_point(model, optim, epoch, path, isBest):
    path = "/ckpt/{}_epoch.pkl".format(epoch)
    check_point = {
        "epoch": epoch,
        "optim": optim.state_dict(),
        "model": model.state_dict(),
    }
    torch.save(check_point, path)
    if isBest:
        shutil.copyfile(path, "model_best.pkl")


def plot_lr():
    num_warmup_steps = 1000
    num_training_steps = 70000
    lr = 0.01
    res_list = []
    for current_step in range(70000):
        if current_step < num_warmup_steps:
            res = float(current_step) / float(max(1, num_warmup_steps))
            res_list.append(res * lr)
            continue
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        res = 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress))
        res_list.append(res * lr)

    plt.plot(res_list)
    plt.title(
        f"Trend of Learning Rate\nnum_warmup_steps={num_warmup_steps}\nnum_training_steps={num_training_steps}"
    )
    plt.show()
