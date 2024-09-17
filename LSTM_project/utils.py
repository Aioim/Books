import numpy as np
import torch
import shutil


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


data = np.random.randint(3, 100, size=(100, 3))
# data[:, 0] = 0
# print(data)
X, y= create_dataset(data, 0, 3, additional=4)
print(X[0,:,:])


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
