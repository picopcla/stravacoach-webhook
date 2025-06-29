import json
import os
import io
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# -------------------------
# Auth Google Drive : Render ou local
# -------------------------
try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    print("✅ Auth via variable d'environnement (Render)")
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)
    print("✅ Auth via fichier local (c:/StravaSecurity/service_account.json)")

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# -------------------------
# Télécharger activities.json
# -------------------------
results = drive_service.files().list(
    q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
    spaces='drive', fields='files(id, name)').execute()
files = results.get('files', [])

if not files:
    print("❌ Aucun activities.json trouvé sur Drive.")
    exit()

file_id = files[0]['id']
request = drive_service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    status, done = downloader.next_chunk()
fh.seek(0)
try:
    data = json.loads(fh.read())
    print(f"✅ Chargé {len(data)} activités depuis activities.json")
except json.decoder.JSONDecodeError:
    print("❌ Erreur JSON : activities.json est vide ou corrompu.")
    exit()

# -------------------------
# Afficher le résumé enrichi
# -------------------------
for act in data:
    activity_id = act.get("activity_id", "-")
    date_str = act.get("date")
    if date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        week = dt.isocalendar().week
        year = dt.isocalendar().year
        date_info = f"{dt.date()} | Semaine: {year}-W{week:02d}"
    else:
        date_info = "-"

    print(f"\n📝 Activité ID: {activity_id} | Date: {date_info}")
    laps = act.get('laps', [])
    print(f"  Nombre de laps: {len(laps)}")

    for lap in laps:
        fc_str = f"{lap.get('fc_avg', 0):.1f}" if lap.get('fc_avg') is not None else "-"
        fc_max_str = f"{lap.get('fc_max', 0):.1f}" if lap.get('fc_max') is not None else "-"
        cad_str = f"{lap.get('cadence_avg', 0):.1f}" if lap.get('cadence_avg') is not None else "-"

        # moving pace
        pace_mov = lap.get('pace_moving')
        if pace_mov:
            pace_min = int(pace_mov)
            pace_sec = int((pace_mov - pace_min) * 60)
            pace_mov_str = f"{pace_min}:{pace_sec:02d}"
        else:
            pace_mov_str = "-"

        # velocity_smooth pace
        pace_vel = lap.get('pace_velocity')
        if pace_vel:
            pace_min = int(pace_vel)
            pace_sec = int((pace_vel - pace_min) * 60)
            pace_vel_str = f"{pace_min}:{pace_sec:02d}"
        else:
            pace_vel_str = "-"

        temp_str = f"{lap.get('temp_avg', 0):.1f}°C" if lap.get('temp_avg') is not None else "-"
        gain_alt_str = f"{lap.get('gain_alt', 0):.1f} m" if lap.get('gain_alt') is not None else "-"

        print(f"    Lap {lap.get('lap_number', '-')}: "
              f"{lap.get('distance',0)/1000:.2f} km en {lap.get('duration',0)/60:.1f} min"
              f" | Moving: {pace_mov_str} | Vitesse: {pace_vel_str}"
              f" | FC moy: {fc_str} / max: {fc_max_str}"
              f" | Cad: {cad_str} | Δalt: {gain_alt_str} | Temp: {temp_str}")
