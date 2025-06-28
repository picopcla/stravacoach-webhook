import json
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# -------------------------
# Auth Google Drive (via service_account.json local)
# -------------------------
with open('service_account.json') as f:
    service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# -------------------------
# Télécharger activities.json
# -------------------------
results = drive_service.files().list(
    q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
    spaces='drive',
    fields='files(id, name)').execute()
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
# Afficher le résumé avec pace mn:ss
# -------------------------
for act in data:
    print(f"\n📝 Activité ID: {act.get('activity_id', '-')}")
    laps = act.get('laps', [])
    print(f"  Nombre de laps: {len(laps)}")
    for lap in laps:
        fc_str = f"{lap.get('fc_avg', 0):.1f}" if lap.get('fc_avg') is not None else "-"
        cad_str = f"{lap.get('cadence_avg', 0):.1f}" if lap.get('cadence_avg') is not None else "-"

        pace = lap.get('pace')
        if pace:
            pace_min = int(pace)
            pace_sec = int((pace - pace_min) * 60)
            pace_str = f"{pace_min}:{pace_sec:02d}"
        else:
            pace_str = "-"

        print(f"    Lap {lap.get('lap_number', '-')}: {lap.get('distance', 0)/1000:.2f} km en {lap.get('duration', 0)/60:.1f} min"
              f" | Pace: {pace_str} min/km"
              f" | FC moy: {fc_str} | Cadence: {cad_str}")
