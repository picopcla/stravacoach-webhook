from flask import Flask, render_template, request, redirect, flash
import json
import io
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

app = Flask(__name__)
app.secret_key = 'ma_super_secret_key_change_moi'

FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# -------------------
# Initialisation Google Drive
# -------------------
try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)
globals()['drive_service'] = drive_service

# -------------------
# Fonctions helpers
# -------------------
def load_activities_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
    except Exception as e:
        print("😥 Erreur connexion Google Drive (activities):", e)
        return None

    files = results.get('files', [])
    if not files:
        return None

    file_id = files[0]['id']
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return json.loads(fh.read())
    except Exception as e:
        print("😥 Erreur download fichier activities.json:", e)
        return None

def load_profile_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
    except Exception as e:
        print("😥 Erreur connexion Google Drive (profile):", e)
        return {"birth_date": "", "weight": 0, "events": []}

    files = results.get('files', [])
    if not files:
        return {"birth_date": "", "weight": 0, "events": []}

    file_id = files[0]['id']
    try:
        request_dl = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_dl)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        with open('profile.json', 'wb') as f:
            f.write(fh.read())
        with open('profile.json') as f:
            return json.load(f)
    except Exception as e:
        print("😥 Erreur download profile.json:", e)
        return {"birth_date": "", "weight": 0, "events": []}

def save_profile_to_drive(profile):
    with open('profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        if files:
            file_id = files[0]['id']
            media = MediaFileUpload('profile.json', mimetype='application/json')
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': 'profile.json', 'parents': [FOLDER_ID]}
            media = MediaFileUpload('profile.json', mimetype='application/json')
            drive_service.files().create(body=file_metadata, media_body=media).execute()
    except Exception as e:
        print("😥 Erreur upload profile.json:", e)

# -------------------
# Dashboard principal
# -------------------
def compute_dashboard_data(activities, profile):
    activities.sort(key=lambda x: x.get("date"))
    last_activity = activities[-1]
    laps = last_activity.get("laps", [])
    date_str = last_activity.get("date")
    run_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

    total_dist = sum(lap.get("distance",0) for lap in laps) / 1000
    total_time = sum(lap.get("duration",0) for lap in laps) / 60
    allure_moy = total_time / total_dist if total_dist > 0 else None

    fc_all = [lap.get("fc_avg") for lap in laps if lap.get("fc_avg") is not None]
    fc_moy = sum(fc_all) / len(fc_all) if fc_all else None

    fc_max_list = [lap.get("fc_max") for lap in laps if lap.get("fc_max") is not None]
    fc_max = max(fc_max_list) if fc_max_list else "-"

    half = len(laps) // 2
    fc_first = [lap.get("fc_avg") for lap in laps[:half] if lap.get("fc_avg") is not None]
    fc_second = [lap.get("fc_avg") for lap in laps[half:] if lap.get("fc_avg") is not None]
    deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first)) / sum(fc_first)/len(fc_first)*100) if fc_first and fc_second else None

    k_all = [(lap.get("fc_avg")_]()_
