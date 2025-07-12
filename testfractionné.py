import json

def detect_fractionne_glissant(points):
    ruptures = 0
    window_distance = 500  # m
    i = 0
    previous_pace = None

    while i < len(points) - 1:
        start_distance = points[i]["distance"]
        j = i
        while j < len(points) and points[j]["distance"] - start_distance < window_distance:
            j += 1

        if j >= len(points):
            break

        end_distance = points[j]["distance"]
        end_time = points[j]["time"]
        start_time = points[i]["time"]
        dist_km = (end_distance - start_distance) / 1000
        time_min = (end_time - start_time) / 60

        if dist_km > 0:
            current_pace = time_min / dist_km

            if previous_pace is not None and abs(current_pace - previous_pace) >= 0.5:
                ruptures += 1
                # saute toute cette fenêtre pour éviter de recompter sur le même intervalle
                i = j
                previous_pace = current_pace
                continue
            previous_pace = current_pace
        i += 1

    total_dist = points[-1]["distance"] / 1000
    type_sortie = "fractionné" if ruptures >= 2 else ("longue" if total_dist >= 11 else "fond")
    return type_sortie, ruptures

# -----------------------------
# Charger activities.json
# -----------------------------
with open("activities.json", "r", encoding="utf-8") as f:
    activities = json.load(f)

# -----------------------------
# Parcourir les activités
# -----------------------------
print(f"{'Date':<20}{'Distance':>10}{'Type':>15}{'Ruptures détectées'}")
print("-"*60)
for act in activities:
    points = act.get("points", [])
    if len(points) < 6:
        continue
    type_sortie, ruptures = detect_fractionne_glissant(points)
    total_dist = round(points[-1]["distance"] / 1000, 2)
    date = act.get("date", "-")[:19]
    print(f"{date:<20}{total_dist:>10} km{type_sortie:>15}   {ruptures}")
