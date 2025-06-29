import json
from datetime import datetime

# -----------------------
# Charger les données
# -----------------------
with open("activities.json") as f:
    activities = json.load(f)

with open("profile.json") as f:
    profile = json.load(f)

# -----------------------
# Dernier run
# -----------------------
last_activity = activities[-1]
laps = last_activity.get("laps", [])
date_str = last_activity.get("date")
run_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

# -----------------------
# Calcul des indicateurs
# -----------------------
total_dist = sum(lap.get("distance",0) for lap in laps) / 1000  # en km
total_time = sum(lap.get("duration",0) for lap in laps) / 60    # en min
allure_moy = total_time / total_dist if total_dist > 0 else None

fc_all = [lap.get("fc_avg") for lap in laps if lap.get("fc_avg") is not None]
fc_moy = sum(fc_all) / len(fc_all) if fc_all else None
fc_max = max(lap.get("fc_max") for lap in laps if lap.get("fc_max") is not None)

# dérive cardio : comparer 1er vs 2e moitié
half = len(laps) // 2
fc_first = [lap.get("fc_avg") for lap in laps[:half] if lap.get("fc_avg") is not None]
fc_second = [lap.get("fc_avg") for lap in laps[half:] if lap.get("fc_avg") is not None]
deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first)) / sum(fc_first)/len(fc_first)*100) if fc_first and fc_second else None

# k moyen
k_all = [(lap.get("fc_avg") / (lap.get("pace_velocity") if lap.get("pace_velocity") else 1)) 
          for lap in laps if lap.get("fc_avg") is not None and lap.get("pace_velocity")]
k_moy = sum(k_all) / len(k_all) if k_all else None

# gain altitude total
gain_alt = sum(abs(lap.get("gain_alt", 0)) for lap in laps if lap.get("gain_alt") is not None)

# -----------------------
# Événements récents
# -----------------------
events_recent = []
for event in profile.get("events", []):
    event_date = datetime.strptime(event.get("date"), "%Y-%m-%d")
    delta_days = (run_date - event_date).days
    if 0 <= delta_days <=7:
        events_recent.append({"days_ago": delta_days, "note": event.get("note")})

# -----------------------
# Générer dashboard.json
# -----------------------
dashboard = {
    "date": run_date.strftime("%Y-%m-%d"),
    "distance_km": round(total_dist,2),
    "duration_min": round(total_time,1),
    "allure": f"{int(allure_moy)}:{int((allure_moy - int(allure_moy))*60):02d}" if allure_moy else "-",
    "fc_moy": round(fc_moy,1) if fc_moy else "-",
    "fc_max": fc_max if fc_max else "-",
    "k_moy": round(k_moy,1) if k_moy else "-",
    "deriv_cardio": round(deriv_cardio,1) if deriv_cardio else "-",
    "gain_alt": round(gain_alt,1),
    "profile": {
        "age": profile.get("age"),
        "poids": profile.get("poids"),
        "objectifs": profile.get("objectifs")
    },
    "events_recent": events_recent
}

with open("dashboard.json", "w") as f:
    json.dump(dashboard, f, indent=2)

print("✅ dashboard.json généré pour le dernier run.")
