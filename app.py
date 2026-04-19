from flask import Flask, render_template, jsonify
from pymongo import MongoClient
import numpy as np
import tensorflow as tf
import joblib
from model_utils import GCNLayer 

app = Flask(__name__)

# 1. Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["traffic07"]
collection_sensors = db["capteurs"]
# On suppose que tu as une collection pour les données historiques
collection_data = db["vitesse_historique"] 

# 2. Chargement IA
model = tf.keras.models.load_model(
    'model_gcn_gru_pro.h5', 
    custom_objects={'GCNLayer': GCNLayer}, 
    compile=False
)
scaler = joblib.load('scaler_140.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sensors')
def get_sensors():
    return jsonify(list(collection_sensors.find({}, {'_id': 0})))

@app.route('/api/predict')
def predict():
    try:
        # RÉCUPÉRATION DES VRAIES DONNÉES (12 derniers pas de temps pour les 140 nœuds)
        # On récupère les données de la DB, triées par temps
        cursor = collection_data.find().sort("timestamp", -1).limit(12)
        history = list(cursor)
        
        if len(history) < 12:
            # Si pas assez de données en DB, on utilise des données neutres (0.5) pour tester
            data_to_predict = np.ones((12, 140)) * 0.5
        else:
            # On extrait les valeurs (supposons une matrice stockée par timestamp)
            data_to_predict = np.array([h['values'] for h in reversed(history)])

        # PIPELINE DE PRÉDICTION
        input_scaled = scaler.transform(data_to_predict)
        input_final = input_scaled.reshape(1, 12, 140, 1)
        
        preds_scaled = model.predict(input_final)
        # Éviter les valeurs négatives + remettre dans une échelle réaliste
        preds_unscaled = np.clip(preds_scaled, 0, 1) * 100
        
        # Mapping des IDs
        sensor_docs = list(collection_sensors.find().sort("ID", 1))
        prediction_map = {str(s['ID']): float(v) for s, v in zip(sensor_docs, preds_unscaled[0])}
        
        return jsonify(prediction_map)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)