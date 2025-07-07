import json
import os
import sys
import io
import time
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from datetime import datetime

# ----------------------------
# Vérifier et rafraîchir le token Strava
# ----------------------------
with open("strava_tokens.json") as f:
    tokens = json.load(f)

access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]
expires_at = tokens["expires_at"]

time_remaining = expires_at - int(time.time())
if time_remaining < 300:
    print(f"🔄 Token expirant dans {time_remaining}s, on le renouvelle...")
    resp = requests.post(
        "https://www.strava.com/api/v3/oauth/token",
        data={
            "client_id": "162245",
            "client_secret": "0552c0e87d83493d7f6667d0570de1e8ac9e9a68",
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
    )
    new_tokens = resp.json()
    tokens["access_token"] = new_tokens["access_token"]
    tokens["refresh_token"] = new_tokens["refresh_token"]
    tokens["expires_at"] = new_tokens["expires_at"]
    with open("strava_tokens.json", "w") as f:
        json.dump(tokens, f, indent=2)
    access_token = tokens["access_token"]
    print("✅ Token Strava rafraîchi.")
else:
    print(f"✅ Token encore valide pour {time_remaining}s.")

headers = {"Authorization": f"Bearer {access_token}"}

# ----------------------------
# Auth Google Drive
# ----------------------------
service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

# ----------------------------
# Config
# ----------------------------
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'
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
# Reconstruire les laps et points
# ----------------------------
activity_id_arg = int(sys.argv[1])

def process_activity(activity_id):
    if activity_id in existing_ids:
        print(f"✅ Activité {activity_id} déjà présente, on skip.")
        return

    url_activity = f"https://www.strava.com/api/v3/activities/{activity_id}"
    resp_activity = requests.get(url_activity, headers=headers)
    activity_data = resp_activity.json()
    start_date = activity_data.get("start_date_local")

    url_streams = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": "time,distance,heartrate,cadence,velocity_smooth,altitude,temp,moving", "key_by_type": "true"}
    resp = requests.get(url_streams, params=params, headers=headers)
    streams = resp.json()

    time_data = streams.get("time", {}).get("data", [])
    distance = streams.get("distance", {}).get("data", [])
    heartrate = streams.get("heartrate", {}).get("data", [])
    cadence = streams.get("cadence", {}).get("data", [])
    velocity = streams.get("velocity_smooth", {}).get("data", [])
    altitude = streams.get("altitude", {}).get("data", [])
    temp = streams.get("temp", {}).get("data", [])
    moving = streams.get("moving", {}).get("data", [])

    if not time_data or not distance:
        print(f"⚠️ Pas de données pour activité {activity_id}, on ignore.")
        return

    # ---------------- Laps (1 km) ----------------
    laps = []
    lap_start_idx = 0
    lap_number = 1
    for i, d in enumerate(distance):
        if d - distance[lap_start_idx] >= 1000 or i == len(distance) -1:
            slice_range = range(lap_start_idx, i+1)
            lap_dist = distance[i] - distance[lap_start_idx]
            lap_time = time_data[i] - time_data[lap_start_idx]

            hr_lap = [heartrate[j] for j in slice_range if heartrate and j < len(heartrate)]
            cad_lap = [cadence[j] for j in slice_range if cadence and j < len(cadence)]
            vel_lap = [velocity[j] for j in slice_range if velocity and j < len(velocity)]
            alt_lap = [altitude[j] for j in slice_range if altitude and j < len(altitude)]
            temp_lap = [temp[j] for j in slice_range if temp and j < len(temp)]
            mov_lap = [moving[j] for j in slice_range if moving and j < len(moving)]

            moving_time = sum(1 for m in mov_lap if m)
            pace_moving = (moving_time / 60) / (lap_dist/1000) if lap_dist > 0 else None
            avg_vel = sum(vel_lap)/len(vel_lap)*3.6 if vel_lap else None
            pace_vel = 60 / avg_vel if avg_vel else None
            gain_alt = alt_lap[-1] - alt_lap[0] if alt_lap else None

            laps.append({
                "lap_number": lap_number,
                "distance": lap_dist,
                "duration": lap_time,
                "fc_avg": sum(hr_lap)/len(hr_lap) if hr_lap else None,
                "fc_max": max(hr_lap) if hr_lap else None,
                "cadence_avg": sum(cad_lap)/len(cad_lap) if cad_lap else None,
                "pace_velocity": pace_vel,
                "pace_moving": pace_moving,
                "temp_avg": sum(temp_lap)/len(temp_lap) if temp_lap else None,
                "gain_alt": gain_alt
            })
            lap_start_idx = i
            lap_number +=1

    # ---------------- Points smoothed (10 sec) ----------------
    points = []
    window = 10
    for i in range(0, len(time_data), window):
        slice_range = range(i, min(i+window, len(time_data)))
        point_time = time_data[slice_range[-1]]
        avg_dist = sum(distance[j] for j in slice_range) / len(slice_range)
        avg_hr = sum(heartrate[j] for j in slice_range if heartrate and j < len(heartrate)) / len(slice_range) if heartrate else None
        avg_vel = sum(velocity[j] for j in slice_range if velocity and j < len(velocity)) / len(slice_range) if velocity else None
        avg_alt = sum(altitude[j] for j in slice_range if altitude and j < len(altitude)) / len(slice_range) if altitude else None
        avg_cad = sum(cadence[j] for j in slice_range if cadence and j < len(cadence)) / len(slice_range) if cadence else None
        avg_temp = sum(temp[j] for j in slice_range if temp and j < len(temp)) / len(slice_range) if temp else None

        points.append({
            "time": point_time,
            "distance": avg_dist,
            "hr": avg_hr,
            "vel": avg_vel,
            "alt": avg_alt,
            "cad": avg_cad,
            "temp": avg_temp
        })

    activities.append({
        "activity_id": activity_id,
        "date": start_date,
        "laps": laps,
        "points": points
    })
    existing_ids.add(activity_id)
    print(f"🚀 Activité {activity_id} ajoutée avec {len(laps)} laps et {len(points)} points.")

# ----------------------------
# ➡️ 1. Processer activité passée en argument
# ----------------------------
process_activity(activity_id_arg)

# ➡️ 2. Vérifier les dernières activités
url = "https://www.strava.com/api/v3/athlete/activities"
params = {"per_page": 50, "page": 1}
resp = requests.get(url, params=params, headers=headers)
latest_activities = resp.json()

if isinstance(latest_activities, list):
    strava_ids = set(act["id"] for act in latest_activities)
    activities = [a for a in activities if a["activity_id"] in strava_ids]

    for act in latest_activities:
        process_activity(act["id"])
else:
    print("⚠️ Erreur Strava: ", latest_activities)

# ➡️ Sauvegarder et uploader sur Drive
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
