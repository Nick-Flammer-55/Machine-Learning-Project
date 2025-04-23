import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device("cpu")

#load in datasets
games = pd.read_csv("Games.csv", low_memory=False)
team_stats = pd.read_csv("TeamStatistics.csv")

games['gameDate'] = pd.to_datetime(games['gameDate'])
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])

# merge games and team_stats
df = pd.merge(team_stats, games[['gameId', 'gameDate', 'winner']], on=['gameId', 'gameDate'])
df['label'] = (df['teamId'] == df['winner']).astype(int)

# list of features to extract
features = [
    'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade',
    'fieldGoalsPercentage', 'threePointersAttempted', 'threePointersMade',
    'threePointersPercentage', 'freeThrowsAttempted', 'freeThrowsMade',
    'freeThrowsPercentage', 'reboundsDefensive', 'reboundsOffensive',
    'reboundsTotal', 'foulsPersonal', 'turnovers', 'plusMinusPoints',
    'benchPoints', 'biggestLead', 'biggestScoringRun', 'leadChanges',
    'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
    'pointsSecondChance', 'seasonWins', 'seasonLosses',
    'q1Points', 'q2Points', 'q3Points', 'q4Points'
]



df = df.dropna(subset=features + ['label'])
df = df.sort_values(by='gameDate')

# Normalize
X = df[features].values
y = df['label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequence creation
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 10
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

# Split and convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Define model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size=len(features)).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop
loss_history = []
accuracy_history = []

for epoch in range(100):
    model.train()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), test_preds.cpu().numpy())

    loss_history.append(loss.item())
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f} | Test Accuracy = {accuracy:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Loss", color='red')
plt.plot(accuracy_history, label="Accuracy", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("LSTM with TeamStatistics + Games")
plt.grid(True)
plt.legend()
plt.show()
