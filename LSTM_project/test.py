import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 生成多参数正弦波数据
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


seq_length = 15
time_step = 10
additional = 5
X, y = create_dataset(scaled_data, 0, time_step, additional)
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 使用 DataLoader 进行批量处理
batch_size = 1024
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class SalesPredictorTransformer(nn.Module):
    def __init__(
        self, input_dim, embed_size, num_heads, num_layers, dim_feedforward, dropout=0.1
    ):
        super(SalesPredictorTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_size, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)  # 池化最后的时间步
        x = self.fc(x)
        return x


# 模型初始化
input_dim = num_features = 6
embed_size = 6
num_heads = 2
num_layers = 3
dim_feedforward = 256
dropout = 0.1

model = SalesPredictorTransformer(input_dim, embed_size, num_heads, num_layers, dim_feedforward, dropout).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('outputs.shape,targets.shape',outputs.shape,targets.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)  # 累加每个 batch 的损失

    epoch_loss /= len(dataloader.dataset)  # 计算平均损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    # 评估模型
# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X).cpu().numpy().flatten()

# 反标准化
actual = y.cpu().numpy().flatten()

# 绘制结果
plt.figure(figsize=(15, 6))
plt.plot(actual, label="Actual Sales")
plt.plot(predictions, label="Predicted Sales")
plt.legend()
plt.show()
