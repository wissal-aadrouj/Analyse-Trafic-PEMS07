import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model


# ==========================================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================================

raw_data = np.load("PEMS07.npz")
data_values = raw_data['data']
num_nodes = data_values.shape[1]

# Création X et y avant normalisation
X_raw = data_values[:-1]
y_raw = data_values[1:]

split = int(0.8 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

# Normalisation (FIT UNIQUEMENT SUR TRAIN)
scaler = MinMaxScaler()
scaler.fit(X_train_raw)

X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
y_train = scaler.transform(y_train_raw)
y_test = scaler.transform(y_test_raw)

# Ajouter dimension feature (1 feature = vitesse)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

# ==========================================================
# 2. MATRICE D’ADJACENCE NORMALISÉE
# ==========================================================

def get_spatial_adj(file_path, n_nodes):
    try:
        df_dist = pd.read_csv(file_path)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _, row in df_dist.iterrows():
            i, j = int(row[0]), int(row[1])
            if i < n_nodes and j < n_nodes:
                adj[i, j] = 1
                adj[j, i] = 1  # rendre symétrique
    except:
        print("CSV non trouvé, matrice identité utilisée.")
        adj = np.eye(n_nodes)

    A_tilde = adj + np.eye(n_nodes)

    degree = np.sum(A_tilde, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))

    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_norm

A_norm = get_spatial_adj("PEMS07.csv", num_nodes)
A_norm_tf = tf.constant(A_norm, dtype=tf.float32)

# ==========================================================
# 3. DÉFINITION DU VRAI GCN
# ==========================================================

class GCNLayer(Layer):
    def __init__(self, adj_matrix, units, **kwargs):
        super().__init__(**kwargs)
        self.adj = adj_matrix
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        x = tf.matmul(self.adj, inputs)        # Â X
        x = tf.matmul(x, self.w)              # Â X W
        return tf.nn.relu(x)

# ==========================================================
# 4. CONSTRUCTION DU MODÈLE
# ==========================================================

inputs = Input(shape=(num_nodes, 1))
gcn_out = GCNLayer(A_norm_tf, units=16)(inputs)
gcn_out = GCNLayer(A_norm_tf, units=1)(gcn_out)

model_gcn = Model(inputs, gcn_out)
model_gcn.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_gcn.summary()

# ==========================================================
# 5. ENTRAÎNEMENT
# ==========================================================

history = model_gcn.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# 6. ÉVALUATION
# ==========================================================

y_pred_scaled = model_gcn.predict(X_test)

# Retirer dimension feature
y_pred_scaled = np.squeeze(y_pred_scaled, axis=-1)
y_test_scaled = np.squeeze(y_test, axis=-1)

# Retour km/h
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test_scaled)

mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
r2 = r2_score(y_true.flatten(), y_pred.flatten())

print("\n--- RÉSULTATS GCN SPATIAL CORRIGÉ ---")
print(f"MAE  : {mae:.2f} km/h")
print(f"RMSE : {rmse:.2f} km/h")
print(f"R²   : {r2:.4f}")

# ==========================================================
# 7. VISUALISATIONS
# ==========================================================

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(A_norm[:50, :50], cmap='viridis')
plt.title("Matrice d’Adjacence Normalisée (Zoom 50)")
plt.show()

# Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Évolution MSE")
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# MAE par capteur
errors_per_sensor = np.mean(np.abs(y_true - y_pred), axis=0)

plt.figure(figsize=(12, 5))
plt.bar(range(num_nodes), errors_per_sensor)
plt.axhline(y=mae, linestyle='--')
plt.title("MAE par Capteur")
plt.xlabel("ID Capteur")
plt.ylabel("Erreur (km/h)")
plt.show()

# Scatter réel vs prédit
plt.figure(figsize=(8, 8))
plt.scatter(y_true.flatten()[:5000],
            y_pred.flatten()[:5000],
            alpha=0.1,
            s=2)

plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'r--')

plt.title("Réel vs Prédit")
plt.xlabel("Vitesse Réelle")
plt.ylabel("Vitesse Prédite")
plt.show()





# ==========================================================
# 8. VISUALISATION DU GRAPHE GCN (STRUCTURE DU RÉSEAU)
# ==========================================================

import networkx as nx

# Construire le graphe depuis la matrice d'adjacence
G = nx.from_numpy_array(A_norm)

print("Nombre de noeuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())

# Pour éviter un graphe illisible si beaucoup de noeuds
MAX_NODES_TO_DISPLAY = 50

if num_nodes > MAX_NODES_TO_DISPLAY:
    print(f"Affichage limité aux {MAX_NODES_TO_DISPLAY} premiers noeuds.")
    nodes_subset = list(range(MAX_NODES_TO_DISPLAY))
    G_sub = G.subgraph(nodes_subset)
else:
    G_sub = G

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(G_sub, seed=42)  # layout physique

nx.draw_networkx_nodes(G_sub, pos, node_size=50)
nx.draw_networkx_edges(G_sub, pos, alpha=0.3)

plt.title("Graphe Spatial du GCN (structure des capteurs)")
plt.axis("off")
plt.show()

# ==========================================================
# 9. ANALYSE DE CONVERGENCE ET COMPLEXITÉ AVEC VISUALISATIONS
# ==========================================================

import time
import psutil
import os

print("\n====== ANALYSE DU MODÈLE GCN ======")

# -----------------------------
# PARAMÈTRES DU MODÈLE
# -----------------------------

N = num_nodes
F_in = 1
F_hidden = 16
F_out = 1

# Nombre d'arêtes
E = np.count_nonzero(A_norm) / 2

print("Nombre de noeuds :", N)
print("Nombre d'arêtes :", int(E))

# ==========================================================
# COMPLEXITÉ TEMPORELLE
# ==========================================================

time_complexity_layer1 = E * F_in + N * F_in * F_hidden
time_complexity_layer2 = E * F_hidden + N * F_hidden * F_out

time_complexity_total = time_complexity_layer1 + time_complexity_layer2

print("\nComplexité temporelle approximative :")
print("Layer1 :", int(time_complexity_layer1))
print("Layer2 :", int(time_complexity_layer2))
print("Total :", int(time_complexity_total))

# ==========================================================
# COMPLEXITÉ MÉMOIRE
# ==========================================================

adj_memory = N * N
feature_memory = N * F_hidden
weights_memory = (F_in * F_hidden) + (F_hidden * F_out)

memory_complexity = adj_memory + feature_memory + weights_memory

print("\nComplexité mémoire approximative :")
print("Adjacency :", adj_memory)
print("Features :", feature_memory)
print("Weights :", weights_memory)
print("Total :", memory_complexity)

# ==========================================================
# ANALYSE DU GRADIENT
# ==========================================================

with tf.GradientTape() as tape:
    predictions = model_gcn(X_train[:32])
    loss = tf.keras.losses.MSE(y_train[:32], predictions)

grads = tape.gradient(loss, model_gcn.trainable_variables)

grad_norm = 0

for g in grads:
    grad_norm += tf.reduce_sum(tf.square(g))

grad_norm = tf.sqrt(grad_norm)

print("\nNorme du gradient :", grad_norm.numpy())

# ==========================================================
# CRITÈRE DE CONVERGENCE
# ==========================================================

loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

delta_loss = abs(loss_values[-1] - loss_values[-2])

print("\nVariation de loss :", delta_loss)

if delta_loss < 1e-4:
    print("Le modèle a convergé.")
else:
    print("Le modèle peut encore apprendre.")

# ==========================================================
# TEMPS D'INFERENCE
# ==========================================================

start = time.time()
model_gcn.predict(X_test[:32])
end = time.time()

inference_time = end - start

print("\nTemps d'inférence (32 samples) :", inference_time, "secondes")

# ==========================================================
# UTILISATION MÉMOIRE
# ==========================================================

process = psutil.Process(os.getpid())
memory_used = process.memory_info().rss / 1024**2

print("Mémoire utilisée par le programme :", memory_used, "MB")

# ==========================================================
# VISUALISATIONS SCIENTIFIQUES
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. GRAPHE DE CONVERGENCE LOSS
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(loss_values,label="Train Loss")
plt.plot(val_loss_values,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Convergence du modèle GCN")
plt.legend()
plt.grid()
plt.show()

# ----------------------------------------------------------
# 2. STABILITÉ DU GRADIENT
# ----------------------------------------------------------

gradient_values = []

for batch in range(10):
    
    with tf.GradientTape() as tape:
        pred = model_gcn(X_train[batch:batch+32])
        loss = tf.keras.losses.MSE(y_train[batch:batch+32], pred)
        
    grads = tape.gradient(loss, model_gcn.trainable_variables)
    
    norm = 0
    for g in grads:
        norm += tf.reduce_sum(tf.square(g))
        
    norm = tf.sqrt(norm)
    gradient_values.append(norm.numpy())

plt.figure(figsize=(8,5))
plt.plot(gradient_values, marker='o')
plt.title("Norme du Gradient (Stabilité de l'apprentissage)")
plt.xlabel("Batch")
plt.ylabel("Gradient Norm")
plt.grid()
plt.show()

# ----------------------------------------------------------
# 3. COMPLEXITÉ TEMPORELLE PAR COUCHE
# ----------------------------------------------------------

layers = ["GCN Layer 1","GCN Layer 2"]
complexity = [time_complexity_layer1,time_complexity_layer2]

plt.figure(figsize=(7,5))
sns.barplot(x=layers,y=complexity)
plt.title("Complexité temporelle par couche")
plt.ylabel("Nombre d'opérations")
plt.show()

# ----------------------------------------------------------
# 4. COMPLEXITÉ MÉMOIRE
# ----------------------------------------------------------

labels = ["Adjacency","Features","Weights"]
sizes = [adj_memory,feature_memory,weights_memory]

plt.figure(figsize=(6,6))
plt.pie(sizes,labels=labels,autopct='%1.1f%%')
plt.title("Répartition de la mémoire du modèle")
plt.show()

# ----------------------------------------------------------
# 5. TEMPS D'INFERENCE
# ----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.bar(["Inference Time"],[inference_time])
plt.title("Temps d'inférence du modèle")
plt.ylabel("Secondes")
plt.show()


# ==========================================================
# 10. GRAPH LAPLACIAN SPECTRUM
# ==========================================================

print("\n====== ANALYSE DU SPECTRE DU GRAPHE ======")

# Calcul du Laplacien
degree_matrix = np.diag(np.sum(A_norm, axis=1))
laplacian = degree_matrix - A_norm

# Valeurs propres
eigenvalues = np.linalg.eigvals(laplacian)

# Trier
eigenvalues = np.sort(eigenvalues)

print("Nombre de valeurs propres :", len(eigenvalues))

# Graphique
plt.figure(figsize=(8,5))
plt.plot(eigenvalues)
plt.title("Spectre du Laplacien du Graphe (PEMS07)")
plt.xlabel("Index")
plt.ylabel("Valeur propre")
plt.grid()
plt.show()

# ==========================================================
# 11. ANALYSE DE SCALABILITÉ DU GCN
# ==========================================================

print("\n====== ANALYSE DE SCALABILITÉ ======")

nodes_range = np.arange(10, num_nodes, 10)

complexity_values = []

for n in nodes_range:
    
    # approx edges proportionnelles
    edges = n * 2  
    
    comp = edges * F_hidden + n * F_hidden * F_out
    complexity_values.append(comp)

plt.figure(figsize=(8,5))
plt.plot(nodes_range, complexity_values, marker='o')
plt.title("Scalabilité du GCN (Complexité vs Nombre de Capteurs)")
plt.xlabel("Nombre de Capteurs")
plt.ylabel("Complexité Approximative")
plt.grid()
plt.show()


# ==========================================================
# 12. DISTRIBUTION DES DEGRÉS DU GRAPHE
# ==========================================================

degrees = [d for n, d in G.degree()]

plt.figure(figsize=(8,5))
plt.hist(degrees, bins=20)
plt.title("Distribution du degré des capteurs")
plt.xlabel("Nombre de connexions")
plt.ylabel("Fréquence")
plt.grid()
plt.show()