from pymongo import MongoClient
import folium

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["traffic07"]
collection = db["capteurs"]

# Récupérer tous les capteurs
sensors = list(collection.find())

# Créer la carte centrée sur la zone (Los Angeles)
m = folium.Map(location=[33.97, -118.18], zoom_start=11)

# Couleurs selon la direction
colors = {"N": "blue", "S": "red", "E": "green", "W": "orange"}

# Ajouter un marqueur pour chaque capteur
for sensor in sensors:
    lat = sensor.get("Latitude")
    lon = sensor.get("Longitude")
    sensor_id = sensor.get("ID")
    fwy = sensor.get("Fwy")
    direction = sensor.get("Dir", "N")
    district = sensor.get("District")

    if lat and lon:
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colors.get(direction, "gray"),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"""
                <b>Sensor ID:</b> {sensor_id}<br>
                <b>Fwy:</b> {fwy}<br>
                <b>Direction:</b> {direction}<br>
                <b>District:</b> {district}<br>
                <b>Lat:</b> {lat}<br>
                <b>Lon:</b> {lon}
                """,
                max_width=200
            ),
            tooltip=f"ID: {sensor_id} | Fwy {fwy} {direction}"
        ).add_to(m)

# Légende
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background-color: white; padding: 10px; border-radius: 8px;
     border: 2px solid grey; font-size: 13px;">
    <b>Direction</b><br>
    <i style="background:blue;width:12px;height:12px;display:inline-block;border-radius:50%"></i> North<br>
    <i style="background:red;width:12px;height:12px;display:inline-block;border-radius:50%"></i> South<br>
    <i style="background:green;width:12px;height:12px;display:inline-block;border-radius:50%"></i> East<br>
    <i style="background:orange;width:12px;height:12px;display:inline-block;border-radius:50%"></i> West
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte
m.save("sensors_map.html")
print(f"✅ Carte générée ! {len(sensors)} capteurs affichés.")
print("📂 Ouvre le fichier sensors_map.html dans ton navigateur.")
