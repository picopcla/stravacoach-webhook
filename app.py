from flask import Flask, render_template, request, redirect
import json
import io
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import numpy as np

app = Flask(__name__)
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

def load_activities_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
    except Exception as e:
        print("😥 Erreur connexion Google Drive (activities):", e)
        return None
    files = results.get('files', [])
    if not files: return None
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
    if not files: return {"birth_date": "", "weight": 0, "events": []}
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

def compute_dashboard_data(activities, profile):
    activities.sort(key=lambda x: x.get("date"))
    last_activity = activities[-1]
    laps = last_activity.get("laps", [])
    points = last_activity.get("points", [])

    if not laps or not points:
        return {}

    # Données globales
    total_dist = sum(l.get("distance",0) for l in laps)/1000
    total_time = sum(l.get("duration",0) for l in laps)/60
    allure_moy = total_time / total_dist if total_dist > 0 else None

    hr_values = [p.get("hr") for p in points if p.get("hr") is not None]
    fc_moy = sum(hr_values) / len(hr_values) if hr_values else None
    fc_max = max(hr_values) if hr_values else "-"

    half = len(points) // 2
    fc_first = [p.get("hr") for p in points[:half] if p.get("hr") is not None]
    fc_second = [p.get("hr") for p in points[half:] if p.get("hr") is not None]
    deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first)) / sum(fc_first)/len(fc_first)*100) if fc_first and fc_second else None

    k_all = [(p.get("hr") / (p.get("vel")*3.6)) for p in points if p.get("hr") and p.get("vel")]
    k_moy = sum(k_all) / len(k_all) if k_all else None

    gain_alt = points[-1].get("alt",0) - points[0].get("alt",0) if points[0].get("alt") is not None else 0

    # Pour le graphique des laps
    laps_labels = json.dumps([f"Lap {l['lap_number']}" for l in laps])
    laps_paces = json.dumps([l.get("pace_velocity") for l in laps])

    # Pour le graphique FC sur points
    points_fc = json.dumps([p.get("hr") for p in points if p.get("hr")])

    # Elevation normalisée pour commencer à 0
    elevation_values = [p.get("alt") for p in points if p.get("alt") is not None]
    elevation_zeroed = [round(e - elevation_values[0], 2) for e in elevation_values] if elevation_values else []

    return {
        "date": datetime.strptime(last_activity.get("date"), "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
        "distance_km": round(total_dist,2),
        "duration_min": round(total_time,1),
        "allure": f"{int(allure_moy)}:{int((allure_moy - int(allure_moy))*60):02d}" if allure_moy else "-",
        "fc_moy": round(fc_moy,1) if fc_moy else "-",
        "fc_max": fc_max,
        "k_moy": round(k_moy,1) if k_moy else "-",
        "deriv_cardio": round(deriv_cardio,1) if deriv_cardio else "-",
        "gain_alt": round(gain_alt,1),
        "laps_labels": laps_labels,
        "laps_paces": laps_paces,
        "points_fc": points_fc,
        "points_elev": json.dumps(elevation_zeroed)
    }


@app.route("/")
def index():
    activities = load_activities_from_drive()
    if not activities:
        return "❌ Aucun activities.json trouvé sur ton Drive."
    profile = load_profile_from_drive()
    dashboard = compute_dashboard_data(activities, profile)
    return render_template("index.html", dashboard=dashboard)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    profile = load_profile_from_drive()
    if request.method == 'POST':
        profile['birth_date'] = request.form['birth_date']
        profile['weight'] = float(request.form['weight'])
        events = []
        event_dates = request.form.getlist('event_date')
        event_names = request.form.getlist('event_name')
        for date, name in zip(event_dates, event_names):
            if date and name:
                events.append({"date": date, "name": name})
        profile['events'] = events
        save_profile_to_drive(profile)
        return redirect('/profile')
    return render_template('profile.html', profile=profile)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
