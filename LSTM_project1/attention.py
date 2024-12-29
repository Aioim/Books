import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 生成示例多参数时间序列数据
# 加载数据
df = pd.read_csv(
    r"C:\Users\AIO\Desktop\Walmart.csv", parse_dates=["Date"], index_col="Date"
)
df = df[
    ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
]

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


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


time_step = 10
additional = 5
X, y = create_dataset(scaled_data, 0, time_step, additional)

print(X.shape, y.shape)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


print("X.shape:", X.shape)
dataset = TimeSeriesDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

seq_length = 15
num_samples = 1000
num_features = 6


# 定义时间序列预测模型
class MultivariateTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(MultivariateTimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc(x)
        return x

# 模型初始化
input_dim = num_features
embed_size = 64
num_heads = 4
num_layers = 3
dim_feedforward = 256
dropout = 0.1

model = MultivariateTimeSeriesTransformer(input_dim, embed_size, num_heads, num_layers, dim_feedforward, dropout)

model = MultivariateTimeSeriesTransformer(
    input_dim, embed_size, num_heads, num_layers, dim_feedforward, dropout
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.view(-1, time_step, input_dim)
        y_batch = y_batch.view(-1, 1)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 预测
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, time_step, input_dim)
    predictions = model(X_tensor).cpu().numpy()

predictions = scaler.inverse_transform(
    np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1)
)[:, 0]

# 可视化结果
plt.plot(df["Value1"], label="Original Data")
plt.plot(
    np.arange(time_step, len(predictions) + time_step),
    predictions,
    label="Self-Attention Predictions",
)
plt.legend()
plt.show()
