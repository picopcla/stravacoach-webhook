import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Charger ton fichier
with open("Activities(19).json") as f:
    activities = json.load(f)

X = []
labels_true = []

for act in activities:
    points = act.get("points", [])
    laps = act.get("laps", [])
    type_sortie = act.get("type_sortie", "inconnue")

    if not points or not laps:
        continue

    # Allure en min/km
    vels = np.array([p.get("vel", 0) for p in points])
    paces = np.where(vels > 0, (1 / vels) * 16.6667, np.nan)
    paces = paces[~np.isnan(paces)]

    # FC
    fcs = np.array([p.get("hr", np.nan) for p in points])
    fcs = fcs[~np.isnan(fcs)]

    # Altitude
    alt_gains = np.array([lap.get("gain_alt", 0) for lap in laps])

    # Features
    cv_pace = np.std(paces) / np.mean(paces) if len(paces) > 5 else 0
    cv_hr = np.std(fcs) / np.mean(fcs) if len(fcs) > 5 else 0
    mean_alt_gain = np.mean(alt_gains) if len(alt_gains) > 0 else 0

    X.append([cv_pace, cv_hr, mean_alt_gain])
    labels_true.append(type_sortie)

X = np.array(X)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Affichage
print("\n📊 Comparaison clustering vs type_sortie existant :")
for i, (pred, true) in enumerate(zip(clusters, labels_true)):
    print(f"Run #{i+1:02d} ➔ cluster: {pred} | type_sortie: {true}")

# Scatter plot
plt.figure(figsize=(8,5))
scatter = plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', s=80, edgecolors='k')
plt.xlabel("CV Allure")
plt.ylabel("CV FC")
plt.title("Clustering des runs")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.show()
