import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CHARGEMENT ET PREPARATION
# ==========================================
data_df = pd.read_csv("PEMS07.csv", index_col=0)
data = data_df.values.astype(np.float32)

print("Shape des données :", data.shape) 

scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

# --- CORRECTION INDICES GRAPHIQUES ---
# On s'assure que start/end ne dépassent pas la taille des données
T_max = data_norm.shape[0]
start, end = 100, min(600, T_max) 

# Matrice de corrélation
corr_matrix = np.corrcoef(data_norm.T)
threshold = 0.7
adj_corr = np.where(np.abs(corr_matrix) >= threshold, 1, 0)

np.fill_diagonal(adj_corr, 0)
adj_corr = adj_corr + np.eye(adj_corr.shape[0])
adj_corr = adj_corr.astype(np.float32)

def normalize_adj(adj):
    A = torch.from_numpy(adj).float()
    deg = torch.sum(A, dim=1)
    # Correction division par zéro au cas où
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    return D_inv_sqrt @ A @ D_inv_sqrt

A_norm = normalize_adj(adj_corr)

# Préparer les données
data_tensor = torch.from_numpy(data_norm).float()
data_tensor = data_tensor.unsqueeze(-1)  # (T, N, 1)

dataset = TensorDataset(data_tensor, data_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. DEFINITION DU MODELE
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        return self.linear(x)

class TrafficGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(1, 64)
        self.relu = nn.ReLU()
        self.gcn2 = GCNLayer(64, 1)

    def forward(self, x, adj):
        h = self.relu(self.gcn1(x, adj))
        return self.gcn2(h, adj)

# ==========================================
# 3. ENTRAINEMENT GCN CORRELATION
# ==========================================
model = TrafficGCN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list = []
for epoch in range(30):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x, A_norm)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_list.append(epoch_loss / len(loader))

# Courbe Loss Corrélation
plt.figure(figsize=(6,4))
plt.plot(loss_list, label="GCN Loss")
plt.title("Training Loss (GCN Corrélation)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_gcn_corr.png", dpi=300)
plt.show()

# Graphe Corrélation
plt.figure(figsize=(5,5))
G_corr = nx.from_numpy_array(adj_corr)
pos = nx.spring_layout(G_corr, seed=42)
nx.draw_networkx_nodes(G_corr, pos, node_size=30)
nx.draw_networkx_edges(G_corr, pos, alpha=0.3)
plt.title("Graphe GCN - Corrélation", fontsize=10)
plt.axis("off")
plt.tight_layout()
plt.savefig("graph_gcn_corr.png", dpi=300)
plt.show()

# Prédictions Corrélation
with torch.no_grad():
    pred_all = model(data_tensor, A_norm).numpy()

sensor_id = 0
plt.figure(figsize=(6,4))
plt.plot(data_norm[start:end, sensor_id], label="Réel", linewidth=2)
plt.plot(pred_all[start:end, sensor_id, 0], '--', label="GCN", linewidth=2)
plt.title(f"Réel vs GCN (Corrélation) - Capteur {sensor_id}")
plt.xlabel("Temps")
plt.ylabel("Flux normalisé")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_real_gcn_corr.png", dpi=300)
plt.show()

# Métriques Corrélation
y_true = data_norm.flatten()
y_pred_corr = pred_all.reshape(-1)
mae_corr = mean_absolute_error(y_true, y_pred_corr)
rmse_corr = np.sqrt(mean_squared_error(y_true, y_pred_corr))
r2_corr = r2_score(y_true, y_pred_corr)

# ==========================================
# 4. GCN KNN (CORRECTION ERREUR ICI)
# ==========================================
features = data_norm.T
N = features.shape[0]

# --- CORRECTION : k ne peut pas être plus grand que le nombre d'échantillons ---
k = min(5, N - 1) if N > 1 else 1 

knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(features)
distances, indices = knn.kneighbors(features)

adj_knn = np.zeros((N, N))
for i in range(N):
    for j in indices[i]:
        if i != j:
            adj_knn[i, j] = 1
            adj_knn[j, i] = 1 

adj_knn = adj_knn + np.eye(N)
adj_knn = adj_knn.astype(np.float32)

# Visualisation Graphe KNN
plt.figure(figsize=(5,5))
G_knn = nx.from_numpy_array(adj_knn)
pos = nx.spring_layout(G_knn, seed=42)
nx.draw_networkx_nodes(G_knn, pos, node_size=30)
nx.draw_networkx_edges(G_knn, pos, alpha=0.3)
plt.title("Graphe GCN - KNN", fontsize=10)
plt.axis("off")
plt.tight_layout()
plt.savefig("graph_gcn_knn.png", dpi=300)
plt.show()

# Entraînement KNN
A_knn_norm = normalize_adj(adj_knn)
model_knn = TrafficGCN()
optimizer_knn = optim.Adam(model_knn.parameters(), lr=0.001)

loss_knn = []
for epoch in range(30):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        optimizer_knn.zero_grad()
        pred = model_knn(batch_x, A_knn_norm)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer_knn.step()
        epoch_loss += loss.item()
    loss_knn.append(epoch_loss / len(loader))

# Courbe Loss KNN
plt.figure(figsize=(6,4))
plt.plot(loss_knn, label="GCN KNN Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss - GCN KNN")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("loss_gcn_knn.png", dpi=300); plt.show()

# Prédictions KNN
with torch.no_grad():
    pred_knn = model_knn(data_tensor, A_knn_norm).numpy()

plt.figure(figsize=(6,4))
plt.plot(data_norm[start:end, sensor_id], label="Réel", linewidth=2)
plt.plot(pred_knn[start:end, sensor_id, 0], '--', label="GCN KNN", linewidth=2)
plt.title(f"Réel vs GCN KNN - Capteur {sensor_id}")
plt.xlabel("Temps"); plt.ylabel("Flux normalisé")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("pred_vs_real_gcn_knn.png", dpi=300); plt.show()

# Métriques KNN et Tableau Final
y_pred_knn_flat = pred_knn.reshape(-1)
mae_knn = mean_absolute_error(y_true, y_pred_knn_flat)
rmse_knn = np.sqrt(mean_squared_error(y_true, y_pred_knn_flat))
r2_knn = r2_score(y_true, y_pred_knn_flat)

results = pd.DataFrame({
    "Méthode": ["GCN Corrélation", "GCN KNN"],
    "MAE": [mae_corr, mae_knn],
    "RMSE": [rmse_corr, rmse_knn],
    "R2": [r2_corr, r2_knn]
})

print("\n--- RESULTATS ---")
print(results)