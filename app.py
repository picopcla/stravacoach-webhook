from flask import Flask, render_template, request, redirect, flash
import json
import io
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

app = Flask(__name__)
app.secret_key = 'stravacoach_secret'

FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# Google Drive auth
try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

# Fonctions helpers
def load_profile():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
    except Exception as e:
        print("Erreur Drive profile:", e)
        return {"birth_date": "", "weight": 0, "events": []}
    files = results.get('files', [])
    if not files:
        return {"birth_date": "", "weight": 0, "events": []}
    file_id = files[0]['id']
    request_dl = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request_dl)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.loads(fh.read())

def save_profile(profile):
    with open('profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
        spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    media = MediaFileUpload('profile.json', mimetype='application/json')
    if files:
        drive_service.files().update(fileId=files[0]['id'], media_body=media).execute()
    else:
        file_metadata = {'name': 'profile.json', 'parents': [FOLDER_ID]}
        drive_service.files().create(body=file_metadata, media_body=media).execute()

def load_activities():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
    except Exception as e:
        print("Erreur Drive activities:", e)
        return []
    files = results.get('files', [])
    if not files:
        return []
    file_id = files[0]['id']
    request_file = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request_file)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.loads(fh.read())

def compute_dashboard_data(activities):
    if not activities:
        return {}
    last = activities[-1]
    laps = last.get("laps", [])
    total_dist = sum(l.get("distance",0) for l in laps) / 1000
    total_time = sum(l.get("duration",0) for l in laps) / 60
    allure = total_time / total_dist if total_dist else 0
    fc_moy = sum(l.get("fc_avg",0) for l in laps if l.get("fc_avg")) / len(laps) if laps else 0
    return {
        "date": last.get("date","-"),
        "distance": round(total_dist,2),
        "duration": round(total_time,1),
        "allure": f"{int(allure)}:{int((allure - int(allure))*60):02d}" if allure else "-",
        "fc_moy": round(fc_moy,1) if fc_moy else "-"
    }

# Routes
@app.route("/")
def index():
    activities = load_activities()
    dashboard = compute_dashboard_data(activities)
    return render_template("index.html", dashboard=dashboard)

@app.route("/profile", methods=['GET', 'POST'])
def profile():
    profile = load_profile()
    if request.method == 'POST':
        profile['birth_date'] = request.form['birth_date']
        profile['weight'] = float(request.form['weight'])
        events = []
        for i in range(len(request.form.getlist('event_date'))):
            date = request.form.getlist('event_date')[i]
            name = request.form.getlist('event_name')[i]
            if date and name:
                events.append({"date": date, "name": name})
        profile['events'] = events
        save_profile(profile)
        flash("✅ Profil enregistré avec succès !")
        return redirect('/')
    return render_template("profile.html", profile=profile)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
