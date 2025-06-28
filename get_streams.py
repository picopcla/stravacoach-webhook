import sys
import json
import os
import requests
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# -----------------------
# Auth Google Drive
# -----------------------
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # ouvre ton navigateur
drive = GoogleDrive(gauth)

# -----------------------
# Gérer le dossier Drive
# -----------------------
folder_name = "StravaData"
folder_id = None

file_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
if file_list:
    folder_id = file_list[0]['id']
else:
    folder_metadata = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    folder_id = folder['id']

# -----------------------
# Argument activité
# -----------------------
activity_id_arg = int(sys.argv[1])

# -----------------------
# Charger token Strava
# -----------------------
with open("strava_tokens.json") as f:
    tokens = json.load(f)
access_token = tokens["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# -----------------------
# Télécharger activities.json depuis Drive si existe
# -----------------------
activities = []
existing_ids = set()

file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and title='activities.json'"}).GetList()
if file_list:
    drive_file = file_list[0]
    drive_file.GetContentFile("activities.json")
    with open("activities.json") as f:
        activities = json.load(f)
    existing_ids = set(a["activity_id"] for a in activities)
    print(f"📥 Fichier activities.json récupéré de Drive avec {len(activities)} activités consolidées.")
else:
    print("⚠️ Aucun fichier activities.json sur Drive, création d'une nouvelle base.")

# -----------------------
# Fonction traitement activité
# -----------------------
def process_activity(activity_id):
    if activity_id in existing_ids:
        print(f"✅ Activité {activity_id} déjà présente, on skip.")
        return

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
        return

    # Reconstituer laps
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

# -----------------------
# ➡️ 1. Process activité argument
# -----------------------
process_activity(activity_id_arg)

# ➡️ 2. Vérifier 100 dernières
url = "https://www.strava.com/api/v3/athlete/activities"
params = {"per_page": 100, "page": 1}
resp = requests.get(url, params=params, headers=headers)
latest_activities = resp.json()

for act in latest_activities:
    process_activity(act["id"])

# -----------------------
# ➡️ Sauvegarder local & envoyer sur Drive
# -----------------------
with open("activities.json", "w") as f:
    json.dump(activities, f, indent=2)

file_drive = drive.CreateFile({'title': "activities.json", 'parents': [{'id': folder_id}]})
file_drive.SetContentFile("activities.json")
file_drive.Upload()
print(f"✅ Base mise à jour et uploadée sur Drive ({len(activities)} activités).")
