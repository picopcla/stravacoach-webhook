import sys
import json
import requests
import os

MAX_NEW_STREAMS = 5  # ← pour limiter pendant les tests

activity_id_arg = int(sys.argv[1])

# Charger access_token
with open("strava_tokens.json") as f:
    tokens = json.load(f)
access_token = tokens["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# Charger JSON existant
if os.path.exists("activities.json"):
    with open("activities.json") as f:
        activities = json.load(f)
else:
    activities = []

existing_ids = set(a["activity_id"] for a in activities)

def process_activity(activity_id):
    if activity_id in existing_ids:
        print(f"✅ Activité {activity_id} déjà présente, on skip.")
        return False

    # Récupérer les streams
    url_streams = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params_streams = {"keys": "time,distance,heartrate,cadence", "key_by_type": "true"}
    resp_streams = requests.get(url_streams, params=params_streams, headers=headers)
    streams = resp_streams.json()

    time = streams.get("time", {}).get("data", [])
    distance = streams.get("distance", {}).get("data", [])
    heartrate = streams.get("heartrate", {}).get("data", [])
    cadence = streams.get("cadence", {}).get("data", [])

    if not time or not distance:
        print(f"⚠️ Pas de données pour activité {activity_id}, on ignore.")
        return False

    # Reconstruire les laps
    laps = []
    lap_start_idx = 0
    lap_number = 1
    for i, d in enumerate(distance):
        if d - distance[lap_start_idx] >= 1000 or i == len(distance) -1:
            lap_dist = distance[i] - distance[lap_start_idx]
            lap_time = time[i] - time[lap_start_idx]
            hr_lap = heartrate[lap_start_idx:i+1] if heartrate else []
            cad_lap = cadence[lap_start_idx:i+1] if cadence else []

            fc_avg = sum(hr_lap)/len(hr_lap) if hr_lap else None
            fc_max = max(hr_lap) if hr_lap else None
            cad_avg = sum(cad_lap)/len(cad_lap) if cad_lap else None
            pace = (lap_time/60) / (lap_dist/1000) if lap_dist > 0 else None

            laps.append({
                "lap_number": lap_number,
                "distance": lap_dist,
                "duration": lap_time,
                "fc_avg": fc_avg,
                "fc_max": fc_max,
                "cadence_avg": cad_avg,
                "pace": pace
            })

            lap_start_idx = i
            lap_number +=1

    activities.append({
        "activity_id": activity_id,
        "laps": laps
    })
    existing_ids.add(activity_id)
    print(f"🚀 Activité {activity_id} ajoutée avec {len(laps)} laps.")
    return True

# ➡️ 1. Traiter l'activité passée en argument
process_activity(activity_id_arg)

# ➡️ 2. Vérifier les dernières activités avec limite pour les tests
url = "https://www.strava.com/api/v3/athlete/activities"
params = {"per_page": 100, "page": 1}
resp = requests.get(url, params=params, headers=headers)
latest_activities = resp.json()

new_count = 0
for act in latest_activities:
    if new_count >= MAX_NEW_STREAMS:
        break
    if process_activity(act["id"]):
        new_count += 1

# ➡️ Sauvegarder le JSON final
with open("activities.json", "w") as f:
    json.dump(activities, f, indent=2)

print(f"✅ Base de données mise à jour avec {len(activities)} activités consolidées.")
