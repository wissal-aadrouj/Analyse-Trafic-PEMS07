import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, GRU, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================================
# 1️⃣ NORMALISATION MATRICE ADJACENCE
# ==========================================================
def normalize_adj(A):
    A = A + np.eye(A.shape[0])  # self-loops
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt

# ==========================================================
# 2️⃣ COUCHE GCN (CORRIGÉE)
# ==========================================================
class GCNLayer(Layer):
    def __init__(self, adj, units, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.adj = tf.constant(adj, dtype=tf.float32)
        self.units = units

    def build(self, input_shape):
        # input_shape = (batch, window, num_nodes, features)
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Propagation spatiale : A_hat * X
        # inputs: [batch, window, num_nodes, features]
        x = tf.einsum('ij,btjf->btif', self.adj, inputs)   # [batch, window, num_nodes, features]

        # Projection en features
        x = tf.einsum('btif,fc->btic', x, self.W)          # [batch, window, num_nodes, units]

        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "adj": self.adj.numpy().tolist()
        })
        return config

# ==========================================================
# 3️⃣ DATA
# ==========================================================
WINDOW = 12
raw_data = np.load("PEMS07.npz")
data_values = raw_data['data']

if len(data_values.shape) == 3:
    data_values = data_values[:, :, 0]

num_nodes = data_values.shape[1]

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_raw, y_raw = create_sequences(data_values, WINDOW)
split = int(0.8 * len(X_raw))

# Scaling
scaler = MinMaxScaler()
X_train_flat = X_raw[:split].reshape(-1, num_nodes)
scaler.fit(X_train_flat)

joblib.dump(scaler, 'scaler_140.pkl')

X_train = scaler.transform(X_train_flat).reshape(-1, WINDOW, num_nodes, 1)  # ajout feature=1
X_test = scaler.transform(X_raw[split:].reshape(-1, num_nodes)).reshape(-1, WINDOW, num_nodes, 1)
y_train = scaler.transform(y_raw[:split])
y_test = scaler.transform(y_raw[split:])

# ==========================================================
# 4️⃣ MATRICE D’ADJACENCE
# ==========================================================
corr = np.corrcoef(data_values.T)
A = np.where(corr > 0.5, 1, 0)  # seuil
A_norm = normalize_adj(A)

# ==========================================================
# 5️⃣ MODÈLE GCN + GRU
# ==========================================================
inputs = Input(shape=(WINDOW, num_nodes, 1))

# GCN spatial (2 couches)
x = GCNLayer(A_norm, 64)(inputs)
x = GCNLayer(A_norm, 64)(x)

# Fusion temporelle : on a [batch, window, num_nodes, features]
# On "aplatit" les nœuds pour le GRU
x = Reshape((WINDOW, num_nodes * 64))(x)

# GRU temporel
x = GRU(64, return_sequences=False)(x)
x = Dropout(0.3)(x)

# Output
outputs = Dense(num_nodes)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==========================================================
# 6️⃣ TRAINING
# ==========================================================
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# SAVE
model.save('model_gcn_gru_pro.h5')
print("✅ Modèle PRO sauvegardé !")


# =========================
# PRÉDICTION
# =========================
prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)

print("✅ Prediction faite")

# =========================
# ENVOI VERS MONGODB
# =========================
from pymongo import MongoClient
import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["trafficDB"]
collection = db["traffic"]

# supprimer anciennes données
collection.delete_many({})

data_to_insert = []

for i in range(num_nodes):
    data_to_insert.append({
        "sensor_id": int(i),
        "lat": 34.05 + i*0.01,   # temporaire
        "lon": -118.25 + i*0.01,
        "prediction": float(prediction[0][i]),
        "timestamp": datetime.datetime.now()
    })

collection.insert_many(data_to_insert)

print("✅ Données envoyées vers MongoDB")
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

