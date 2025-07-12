import json
import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# Auth Google Drive
try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

# Charger activities.json
def load_activities_from_drive():
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
        spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    if not files:
        print("❌ Aucun activities.json trouvé sur le Drive.")
        return []
    file_id = files[0]['id']
    request_file = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request_file)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.loads(fh.read().decode("utf-8", errors="replace"))

# Réécrire sur Drive
def upload_activities_to_drive(activities):
    fh = io.BytesIO()
    fh.write(json.dumps(activities, indent=2, ensure_ascii=False).encode("utf-8"))
    fh.seek(0)
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
        spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    media = MediaIoBaseUpload(fh, mimetype='application/json')
    if files:
        file_id = files[0]['id']
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        metadata = {'name': 'activities.json', 'parents': [FOLDER_ID]}
        drive_service.files().create(body=metadata, media_body=media, fields='id').execute()

# -------------------------------
# Effacer les champs type_sortie, k_moy, deriv_cardio
activities = load_activities_from_drive()
for act in activities:
    act.pop("type_sortie", None)
    act.pop("k_moy", None)
    act.pop("deriv_cardio", None)

upload_activities_to_drive(activities)
print("✅ Toutes les activités ont été nettoyées : type_sortie, k_moy, deriv_cardio supprimés.")
