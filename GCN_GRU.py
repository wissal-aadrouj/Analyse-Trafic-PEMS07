import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Input, Layer, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# ⚙️ PARAMÈTRES OPTIMISÉS CPU
# ==========================================================

WINDOW = 12
GRU_UNITS = 64          # 🔥 Optimisé CPU
BATCH_SIZE = 64         # 🔥 Plus rapide sur Ryzen
EPOCHS = 20

# ==========================================================
# 1️⃣ CHARGEMENT DES DONNÉES
# ==========================================================

raw_data = np.load("PEMS07.npz")
data_values = raw_data['data'][:, :, 0]
num_nodes = data_values.shape[1]

# ==========================================================
# 2️⃣ CRÉATION DES SÉQUENCES
# ==========================================================

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(data_values, WINDOW)

# ==========================================================
# 3️⃣ SPLIT AVANT NORMALISATION
# ==========================================================

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================================
# 4️⃣ NORMALISATION (SANS DATA LEAKAGE)
# ==========================================================

scaler = MinMaxScaler()

X_train_flat = X_train.reshape(-1, num_nodes)
scaler.fit(X_train_flat)

X_train = scaler.transform(
    X_train.reshape(-1, num_nodes)
).reshape(X_train.shape)

X_test = scaler.transform(
    X_test.reshape(-1, num_nodes)
).reshape(X_test.shape)

y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

# ==========================================================
# 5️⃣ MATRICE D'ADJACENCE NORMALISÉE
# ==========================================================

def get_normalized_adj(file_path, n_nodes):

    df_dist = pd.read_csv(file_path)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for _, row in df_dist.iterrows():
        i, j = int(row.iloc[0]), int(row.iloc[1])
        if i < n_nodes and j < n_nodes:
            adj[i, j] = 1.0

    A_tilde = adj + np.eye(n_nodes, dtype=np.float32)
    degree = np.sum(A_tilde, axis=1)

    D_inv_sqrt = np.diag(np.power(degree, -0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return A_norm.astype(np.float32)

A_norm = get_normalized_adj("PEMS07.csv", num_nodes)

# ==========================================================
# 6️⃣ COUCHE GCN CORRECTE
# ==========================================================

class GCNLayer(Layer):

    def __init__(self, adj, units):
        super().__init__()
        self.adj = tf.constant(adj, dtype=tf.float32)
        self.units = units

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        # A_hat X
        x = tf.einsum('ij,btj->bti', self.adj, inputs)
        # (A_hat X) W
        x = tf.einsum('bti,ik->btk', x, self.weight)
        return tf.nn.relu(x)

# ==========================================================
# 7️⃣ MODÈLE GCN + GRU
# ==========================================================

inputs = Input(shape=(WINDOW, num_nodes))

spatial_out = GCNLayer(A_norm, num_nodes)(inputs)

temporal_out = GRU(
    GRU_UNITS,
    activation='tanh',
    return_sequences=False
)(spatial_out)

temporal_out = Dropout(0.2)(temporal_out)

outputs = Dense(num_nodes)(temporal_out)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==========================================================
# 8️⃣ PIPELINE TF.DATA (OPTIMISATION CPU)
# ==========================================================

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================================
# 9️⃣ ENTRAÎNEMENT
# ==========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(
            patience=3,
            restore_best_weights=True
        )
    ]
)

# ==========================================================
# 🔟 ÉVALUATION
# ==========================================================

y_pred_scaled = model.predict(val_ds)

y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true.flatten(), y_pred.flatten())

print("\n--- RÉSULTATS DU MODÈLE ---")
print(f"MAE : {mae:.2f} km/h")
print(f"RMSE : {rmse:.2f} km/h")
print(f"R² : {r2:.4f}")

# ==========================================================
# 1️⃣1️⃣ VISUALISATIONS
# ==========================================================

# Courbe d'apprentissage
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Validation")
plt.title("Évolution de la perte")
plt.legend()
plt.show()

# Prédiction sur un capteur
sensor_id = 10

plt.figure(figsize=(12,5))
plt.plot(y_true[:288, sensor_id], label="Réel")
plt.plot(y_pred[:288, sensor_id], '--', label="Prédiction")
plt.title(f"Prédiction sur 24h - Capteur {sensor_id}")
plt.legend()
plt.show()

# Distribution erreur
errors = np.abs(y_true - y_pred)

plt.figure(figsize=(8,4))
plt.hist(errors.flatten(), bins=50)
plt.title("Distribution de l'erreur absolue")
plt.show()

# Scatter Corrélation
plt.figure(figsize=(6,6))
plt.scatter(y_true.flatten()[:5000], y_pred.flatten()[:5000], alpha=0.3, s=2)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'r--')
plt.title(f"Corrélation R² = {r2:.2f}")
plt.show()

# ==========================================================
# 1️⃣2️⃣ VISUALISATION DÉCISION GRU (EN KM/H)
# ==========================================================

sample_idx = 100
sensor_idx = 0

past_sequence_real = scaler.inverse_transform(
    X_test[sample_idx]
)[:, sensor_idx]

actual_future_real = scaler.inverse_transform(
    y_test[sample_idx].reshape(1,-1)
)[0, sensor_idx]

predicted_future_real = scaler.inverse_transform(
    y_pred_scaled[sample_idx].reshape(1,-1)
)[0, sensor_idx]

plt.figure(figsize=(8,4))
plt.plot(range(WINDOW), past_sequence_real, marker='o', label="Passé")
plt.scatter(WINDOW, actual_future_real, label="Réalité", s=100)
plt.scatter(WINDOW, predicted_future_real, marker='X', s=100, label="Prédiction")
plt.title("Décision du GRU (valeurs réelles)")
plt.legend()
plt.show()



#PEMS07
####GCN (representation spatiale)
#map =>clique (LEF-LEt) 
#LSTM - GRU (comparaison avec GCN)
#metriques
#sensibilité (changement de dataset)
#simulation 
#svm
#filtre kallmann pour l'associer avec 
#comparaison GCN RNN
#cnn matrice 
#methode statistique arima (optmimisation avec filter de kallman)
#reseau de neuronnes    
#GRU ou LSTM
#GCN LSTM ATTENTION (combinaison entre les trois)

