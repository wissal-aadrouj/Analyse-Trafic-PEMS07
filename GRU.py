import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx 
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. PRÉPARATION DU DATASET
# ==========================================
data_df = pd.read_csv("PEMS07.csv", index_col=0)
# Supprimer la colonne temps si elle existe pour ne garder que les capteurs
data_df = data_df.drop(columns=["timestamp"], errors="ignore")
data = data_df.values.astype(np.float32)

scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_sequences(data_norm, seq_length)

print("Shape X :", X.shape) # (Samples, Seq_len, Nodes)
print("Shape y :", y.shape) # (Samples, Nodes)

train_size = int(0.8 * len(X))
X_train_np, X_test_np = X[:train_size], X[train_size:]
y_train_np, y_test_np = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# ==========================================
# 2. MODÈLE GRU (CORRECTIF DOUBLE UNDERSCORE)
# ==========================================
class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TrafficGRU, self).__init__() # Correction ici
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # On prend le dernier timestep
        return self.fc(out)

# Initialisation avec les dimensions de vos données
input_dim = X_train.shape[2] 
model_gru = TrafficGRU(input_size=input_dim, hidden_size=64, output_size=input_dim)

# ==========================================
# 3. ENTRAÎNEMENT
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model_gru.parameters(), lr=0.001)

epochs = 30
loss_list = []

for epoch in range(epochs):
    model_gru.train()
    optimizer.zero_grad()
    
    output = model_gru(X_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Courbe de Loss
plt.figure(figsize=(6,4))
plt.plot(loss_list, label="GRU Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss - GRU (PEMS07)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_gru.png", dpi=300)
plt.show()

# ==========================================
# 4. ÉVALUATION ET VISUALISATION
# ==========================================
model_gru.eval()
with torch.no_grad():
    pred_test = model_gru(X_test).numpy()

y_test_np = y_test.numpy()
sensor_id = 0
start, end = 0, min(300, len(y_test_np)) # Sécurité sur les indices

plt.figure(figsize=(6,4))
plt.plot(y_test_np[start:end, sensor_id], label="Réel", linewidth=2)
plt.plot(pred_test[start:end, sensor_id], '--', label="GRU", linewidth=2)
plt.title(f"Réel vs GRU - Capteur {sensor_id}")
plt.xlabel("Temps")
plt.ylabel("Flux normalisé")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_real_gru.png", dpi=300)
plt.show()

# Métriques
mae_gru = mean_absolute_error(y_test_np.flatten(), pred_test.flatten())
rmse_gru = np.sqrt(mean_squared_error(y_test_np.flatten(), pred_test.flatten()))
r2_gru = r2_score(y_test_np.flatten(), pred_test.flatten())
corr_value, _ = pearsonr(y_test_np.flatten(), pred_test.flatten())

print(f"MAE : {mae_gru:.6f}")
print(f"RMSE : {rmse_gru:.6f}")
print(f"R2 : {r2_gru:.6f}")
print(f"Corrélation : {corr_value:.6f}")

# Nuage de points Corrélation
plt.figure(figsize=(5,5))
plt.scatter(y_test_np.flatten(), pred_test.flatten(), alpha=0.3)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("Valeurs Réelles")
plt.ylabel("Valeurs Prédites")
plt.title(f"Corrélation GRU (r = {corr_value:.3f})")
plt.grid(True)
plt.tight_layout()
plt.savefig("correlation_gru.png", dpi=300)
plt.show()

# Sauvegarde des résultats
results_gru = pd.DataFrame({
    "Modèle": ["GRU"],
    "MAE": [mae_gru],
    "RMSE": [rmse_gru],
    "R2": [r2_gru],
    "Corrélation": [corr_value]
})
results_gru.to_csv("metrics_gru.csv", index=False)
print(results_gru)