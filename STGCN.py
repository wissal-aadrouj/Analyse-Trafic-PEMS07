import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout,
    Layer, LayerNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
tf.keras.backend.set_floatx('float32')
# ==========================================================
# 1. GRAPH CONVOLUTION OPTIMISÉE (SPARSE)
# ==========================================================
class GraphConv(Layer):
    def __init__(self, adj, filters, **kwargs):
        super().__init__(**kwargs)
        # On force la matrice d'adjacence en float32
        indices = np.array(np.where(adj != 0)).T
        values = adj[adj != 0].astype('float32') 
        self.adj_sparse = tf.sparse.SparseTensor(
            indices=indices, 
            values=values, 
            dense_shape=adj.shape
        )
        self.filters = filters

    def build(self, input_shape):
        # On force le poids en float32
        self.weight = self.add_weight(
            shape=(input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True,
            dtype='float32' 
        )

    def call(self, x):
        # Convertir l'entrée x systématiquement
        x = tf.cast(x, tf.float32) 

        shape = tf.shape(x)
        B, T, N, C = shape[0], shape[1], shape[2], shape[3]

        x = tf.transpose(x, perm=[0, 1, 3, 2]) 
        x = tf.reshape(x, (-1, N))             
        
        x = tf.transpose(x)                    
        # L'opération critique
        x = tf.sparse.sparse_dense_matmul(self.adj_sparse, x) 
        x = tf.transpose(x)                    
        
        x = tf.reshape(x, (B, T, C, N))
        x = tf.transpose(x, perm=[0, 1, 3, 2]) 
        
        return tf.matmul(x, self.weight)
# ==========================================================
# 2. STGCN BLOCK
# ==========================================================
class STGCNBlock(Layer):
    def __init__(self, adj, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)

        self.temp1 = Conv2D(filters, (kernel_size, 1),
                            padding='same', activation='relu')
        self.graph = GraphConv(adj, filters)
        self.temp2 = Conv2D(filters, (kernel_size, 1),
                            padding='same', activation='relu')
        self.norm = LayerNormalization()
        self.dropout = Dropout(0.2)

    def call(self, x):
        x = self.temp1(x)
        x = self.graph(x)
        x = self.temp2(x)
        x = self.norm(x)
        return self.dropout(x)


# ==========================================================
# 3. LOAD DATA
# ==========================================================
raw_data = np.load("PEMS07.npz")
data = raw_data['data'][:, :, 0]
num_nodes = data.shape[1]

# Au lieu de prendre chaque pas de temps, on saute des étapes (stride=3)
def create_sequences_fast(data, window=12, stride=3):
    X, y = [], []
    for i in range(0, len(data) - window, stride): # Stride réduit la taille du dataset
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences_fast(data, stride=5) # 5x moins de données à traiter

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===== CORRECT NORMALIZATION (NO DATA LEAKAGE) =====
scaler = MinMaxScaler()
X_train_flat = X_train.reshape(-1, num_nodes)
scaler.fit(X_train_flat)

X_train = scaler.transform(X_train.reshape(-1, num_nodes)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, num_nodes)).reshape(X_test.shape)

y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

# Add feature dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# ==========================================================
# 4. ADJACENCY MATRIX
# ==========================================================
distances_df = pd.read_csv("PeMS07.csv")

A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for _, row in distances_df.iterrows():
    i, j = int(row['from']), int(row['to'])
    A[i, j] = 1

A_tilde = A + np.eye(num_nodes)
D = np.diag(np.power(A_tilde.sum(axis=1), -0.5))
A_norm = D @ A_tilde @ D


# ==========================================================
# 5. MODEL (LIGHTER VERSION FOR CPU)
# ==========================================================
inputs = Input(shape=(12, num_nodes, 1))

x = STGCNBlock(A_norm, 32)(inputs)   # reduced filters
x = STGCNBlock(A_norm, 16)(x)

x = GlobalAveragePooling2D()(x)
outputs = Dense(num_nodes)(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()


# ==========================================================
# 6. TF.DATA PIPELINE (FASTER CPU)
# ==========================================================
batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ==========================================================
# 7. TRAIN
# ==========================================================
# 1. Configurer l'arrêt précoce pour gagner du temps
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3,                # S'arrête si l'erreur ne baisse plus pendant 3 époques
    restore_best_weights=True  # Garde les meilleurs réglages trouvés
)

# 2. Lancer l'entraînement avec le callback
print("Lancement de l'entraînement STGCN...")
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stop],    # <--- C'est ici qu'on l'ajoute
    verbose=1
)


# ==========================================================
# 8. EVALUATION
# ==========================================================
y_pred_scaled = model.predict(val_dataset)

y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Convergence du STGCN")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.savefig("01_learning_curve.png")
plt.show()



sensor_id = 100

plt.figure(figsize=(15,5))
plt.plot(y_true[:288, sensor_id], label="Réel", color='black')
plt.plot(y_pred[:288, sensor_id], label="Prédiction STGCN", linestyle='--', color='red')
plt.title(f"Prédiction sur 24h - Capteur {sensor_id}")
plt.xlabel("Temps (5 min)")
plt.ylabel("Vitesse")
plt.legend()
plt.grid()
plt.savefig("02_prediction_vs_real.png")
plt.show()



error_map = np.abs(y_true - y_pred).mean(axis=0)

plt.figure(figsize=(14,3))
sns.heatmap(
    error_map.reshape(1,-1),
    cmap="YlOrRd",
    cbar_kws={'label': 'MAE (km/h)'}
)
plt.title("Erreur moyenne par capteur (MAE spatial)")
plt.yticks([])
plt.savefig("03_spatial_error_heatmap.png")
plt.show()



errors = (y_true - y_pred).flatten()

plt.figure(figsize=(8,5))
sns.histplot(errors, bins=50, kde=True)
plt.title("Distribution des erreurs résiduelles")
plt.xlabel("Erreur (km/h)")
plt.ylabel("Fréquence")
plt.grid()
plt.savefig("04_error_distribution.png")
plt.show()


mae_time = np.mean(np.abs(y_true - y_pred), axis=1)

plt.figure(figsize=(12,5))
plt.plot(mae_time[:500])
plt.title("MAE au cours du temps")
plt.xlabel("Temps")
plt.ylabel("MAE")
plt.grid()
plt.savefig("05_temporal_mae.png")
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.2)
plt.plot([0,100],[0,100], 'r')
plt.xlabel("Valeur réelle")
plt.ylabel("Prédiction")
plt.title("Réel vs Prédiction")
plt.grid()
plt.savefig("06_scatter.png")
plt.show()


sensor_mae = np.mean(np.abs(y_true - y_pred), axis=0)

worst_idx = np.argsort(sensor_mae)[-20:]

plt.figure(figsize=(10,5))
plt.bar(range(20), sensor_mae[worst_idx])
plt.xticks(range(20), worst_idx, rotation=90)
plt.title("Top 20 capteurs les plus difficiles")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("07_worst_sensors.png")
plt.show()



node_degree = A.sum(axis=1)

plt.figure(figsize=(7,5))
plt.scatter(node_degree, sensor_mae)
plt.xlabel("Degré du noeud")
plt.ylabel("MAE")
plt.title("Erreur vs Connectivité du noeud")
plt.grid()
plt.savefig("08_degree_vs_error.png")
plt.show()


# ==========================================================
# 9. VISUALISATION INTERNE DU STGCN
# ==========================================================

print("\n===== ANALYSE DU FONCTIONNEMENT STGCN =====")

# ----------------------------------------------------------
# 1️⃣ MATRICE D’ADJACENCE SPATIALE UTILISÉE
# ----------------------------------------------------------

plt.figure(figsize=(8,6))
plt.imshow(A_norm[:50, :50], cmap='viridis')
plt.colorbar(label="Poids normalisé")
plt.title("Matrice d’Adjacence Normalisée (Spatial)")
plt.xlabel("Noeuds")
plt.ylabel("Noeuds")
plt.tight_layout()
plt.savefig("09_adjacency_matrix.png")
plt.show()

print("Shape matrice A_norm :", A_norm.shape)


# ----------------------------------------------------------
# 2️⃣ MATRICE TEMPO-SPATIALE D’ENTRÉE
# ----------------------------------------------------------

print("Shape X_train :", X_train.shape)
print("Format attendu : (batch, time_steps, nodes, features)")

sample = X_train[0, :, :, 0]  # (12, num_nodes)

plt.figure(figsize=(10,6))
plt.imshow(sample[:, :50], aspect='auto', cmap='viridis')
plt.colorbar(label="Vitesse normalisée")
plt.title("Matrice Tempo-Spatiale (Temps x Capteurs)")
plt.xlabel("Capteurs")
plt.ylabel("Temps (12 pas)")
plt.tight_layout()
plt.savefig("10_temporal_spatial_matrix.png")
plt.show()


# ----------------------------------------------------------
# 3️⃣ ÉVOLUTION TEMPORELLE D’UN CAPTEUR
# ----------------------------------------------------------

sensor_example = 50

plt.figure(figsize=(10,4))
plt.plot(sample[:, sensor_example])
plt.title(f"Evolution Temporelle - Capteur {sensor_example}")
plt.xlabel("Temps (fenêtre 12)")
plt.ylabel("Vitesse normalisée")
plt.grid()
plt.tight_layout()
plt.savefig("11_temporal_signal.png")
plt.show()


# ----------------------------------------------------------
# 4️⃣ ACTIVATION APRÈS PREMIER BLOC STGCN
# ----------------------------------------------------------

# Modèle intermédiaire pour voir la sortie du premier bloc
intermediate_model = Model(inputs=model.input,
                           outputs=model.layers[1].output)

activation = intermediate_model.predict(X_train[:1])

print("Shape activation après 1er STGCN block :", activation.shape)

# activation shape = (1, 12, nodes, filters)

activation_sample = activation[0, :, :, 0]  # premier filtre

plt.figure(figsize=(10,6))
plt.imshow(activation_sample[:, :50], aspect='auto', cmap='plasma')
plt.colorbar(label="Activation")
plt.title("Activation après 1er STGCN Block (Filtre 0)")
plt.xlabel("Capteurs")
plt.ylabel("Temps")
plt.tight_layout()
plt.savefig("12_stgcn_activation.png")
plt.show()