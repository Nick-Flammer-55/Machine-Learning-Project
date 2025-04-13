
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn

# Load and prepare stats
data = pd.read_csv("stats.csv")
data = data.sort_values(by='date')  # Ensure chronological order | this may or may not be used based on csv structure
features = ['feature 1', 'feature 2', '...']
X = data[features].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequence the data for training/predicting
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X_seq, y_seq = create_sequences(X_scaled, seq_length)

# LTSM using torch
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out


model = LSTM(input_size=4, hidden_size=64)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to torch tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

# Train loop
for epoch in range(50):
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    new_input = X_tensor[-1].unsqueeze(0)  # shape (1, seq_len, input_size)
    prediction = model(new_input)
    print("Predicted feature:", prediction.item())