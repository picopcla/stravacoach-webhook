import json
import os
import sys
import io
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ----------------------------
# Authentification Google Drive
# ----------------------------
service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

# ----------------------------
# Config
# ----------------------------
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'  # Mets l'ID de ton dossier Drive partagé
activities = []
existing_ids = set()

# ----------------------------
# Charger activities.json depuis Drive
# ----------------------------
results = drive_service.files().list(
    q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
    spaces='drive',
    fields='files(id, name)').execute()
files = results.get('files', [])

if files:
    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open('activities.json', 'wb') as f:
        f.write(fh.read())
    with open('activities.json') as f:
        activities = json.load(f)
    existing_ids = set(a["activity_id"] for a in activities)
    print(f"✅ activities.json téléchargé depuis Drive avec {len(activities)} activités.")
else:
    print("⚠️ Aucun activities.json sur Drive, on va en créer un nouveau.")

# ----------------------------
# Préparer Strava
# ----------------------------
activity_id_arg = int(sys.argv[1])

with open("strava_tokens.json") as f:
    tokens = json.load(f)
access_token = tokens["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# ----------------------------
# Reconstruire les laps depuis les streams
# ----------------------------
def process_activity(activity_id):
    if activity_id in existing_ids:
        print(f"✅ Activité {activity_id} déjà présente, on skip.")
        return

    url_streams = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": "time,distance,heartrate,cadence", "key_by_type": "true"}
    resp = requests.get(url_streams, params=params, headers=headers)
    streams = resp.json()

    time = streams.get("time", {}).get("data", [])
    distance = streams.get("distance", {}).get("data", [])
    heartrate = streams.get("heartrate", {}).get("data", [])
    cadence = streams.get("cadence", {}).get("data", [])

    if not time or not distance:
        print(f"⚠️ Pas de données pour activité {activity_id}, on ignore.")
        return

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

# ----------------------------
# ➡️ 1. Processer activité passée en argument
# ----------------------------
process_activity(activity_id_arg)

# ➡️ 2. Vérifier les 100 dernières activités
url = "https://www.strava.com/api/v3/athlete/activities"
params = {"per_page": 100, "page": 1}
resp = requests.get(url, params=params, headers=headers)
latest_activities = resp.json()

print("📥 Réponse brute Strava activities:", latest_activities)

if isinstance(latest_activities, list):
    for act in latest_activities:
        process_activity(act["id"])
else:
    print("⚠️ Erreur Strava: ", latest_activities)


# ----------------------------
# ➡️ Sauvegarder et uploader sur Drive
# ----------------------------
with open("activities.json", "w") as f:
    json.dump(activities, f, indent=2)

if files:
    media = MediaFileUpload('activities.json', mimetype='application/json')
    drive_service.files().update(fileId=file_id, media_body=media).execute()
    print("✅ activities.json mis à jour sur Drive.")
else:
    file_metadata = {'name': 'activities.json', 'parents': [FOLDER_ID]}
    media = MediaFileUpload('activities.json', mimetype='application/json')
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print("✅ activities.json créé et uploadé sur Drive.")
