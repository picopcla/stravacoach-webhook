from flask import Flask, render_template, request, redirect
import json
import io
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

app = Flask(__name__)
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# -------------------
# Auth Google Drive
# -------------------
try:
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    with open('c:/StravaSecurity/service_account.json') as f:
        service_account_info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

# -------------------
# Helpers pour Drive
# -------------------
def load_activities_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
    except Exception as e:
        print("Erreur connexion Drive (activities):", e)
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

def save_activities_to_drive(activities):
    with open('activities.json', 'w') as f:
        json.dump(activities, f, indent=2)
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if files:
            file_id = files[0]['id']
            media = MediaFileUpload('activities.json', mimetype='application/json')
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': 'activities.json', 'parents': [FOLDER_ID]}
            media = MediaFileUpload('activities.json', mimetype='application/json')
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    except Exception as e:
        print("Erreur upload activities.json:", e)

def load_profile_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
    except Exception as e:
        print("Erreur connexion Drive (profile):", e)
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
        print("Erreur téléchargement profile.json:", e)
        return {"birth_date": "", "weight": 0, "events": []}

def load_objectives_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='objectives.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if not files:
            return {}
        file_id = files[0]['id']
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return json.loads(fh.read())
    except Exception as e:
        print("Erreur chargement objectives:", e)
        return {}

# -------------------
# Enrichir activités avec type_sortie, k_moy, deriv_cardio
# -------------------
def enrich_activities(activities):
    for activity in activities:
        points = activity.get("points", [])
        if not points or len(points) < 5:
            activity["type_sortie"] = "inconnue"
            activity["k_moy"] = "-"
            activity["deriv_cardio"] = "-"
            continue

        total_dist = points[-1]["distance"] / 1000

        k_vals = [(p["hr"] / (p["vel"]*3.6)) for p in points if p.get("hr") and p.get("vel")]
        k_moy = sum(k_vals)/len(k_vals) if k_vals else "-"

        half = len(points)//2
        fc_first = [(p["hr"] - (p["alt"] - points[0]["alt"])*0.1) for p in points[:half] if p.get("hr")]
        fc_second= [(p["hr"] - (p["alt"] - points[0]["alt"])*0.1) for p in points[half:] if p.get("hr")]
        deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first))/(sum(fc_first)/len(fc_first))*100) if fc_first and fc_second else "-"

        blocs = []
        bloc_start_idx = 0
        next_bloc_dist = 500
        for i, p in enumerate(points):
            if p["distance"] >= next_bloc_dist or i == len(points) - 1:
                bloc_points = points[bloc_start_idx:i+1]
                if len(bloc_points) > 1:
                    bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
                    bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
                    allure = (bloc_time / 60) / (bloc_dist / 1000) if bloc_dist > 0 else None
                    if allure:
                        blocs.append(allure)
                bloc_start_idx = i+1
                next_bloc_dist += 500

        alternances = 0
        if blocs:
            avg_allure = sum(blocs) / len(blocs)
            faster = False
            for allure in blocs:
                if allure < avg_allure * 0.85:
                    if not faster:
                        alternances += 1
                        faster = True
                else:
                    faster = False

        if alternances >= 2:
            type_sortie = "fractionné"
        elif total_dist >= 11:
            type_sortie = "longue"
        else:
            type_sortie = "fond"

        activity["type_sortie"] = type_sortie
        activity["k_moy"] = round(k_moy, 2) if isinstance(k_moy, float) else "-"
        activity["deriv_cardio"] = round(deriv_cardio, 2) if isinstance(deriv_cardio, float) else "-"
    return activities

# -------------------
# Calcul dashboard complet
# -------------------
def compute_dashboard_data(activities, profile):
    activities.sort(key=lambda x: x.get("date"))
    last = activities[-1]
    points = last.get("points", [])
    if not points:
        return {}

    total_dist = points[-1]["distance"] / 1000
    total_time = (points[-1]["time"] - points[0]["time"]) / 60
    allure_moy = total_time / total_dist if total_dist > 0 else None

    hr_vals = [p["hr"] for p in points if p.get("hr")]
    fc_moy = sum(hr_vals)/len(hr_vals) if hr_vals else "-"
    fc_max = max(hr_vals) if hr_vals else "-"
    gain_alt = points[-1]["alt"] - points[0]["alt"] if points[0].get("alt") else 0

    labels = [round(p["distance"]/1000, 3) for p in points]
    points_fc = [p["hr"] for p in points]
    points_alt = [p["alt"] - points[0]["alt"] for p in points]

    allure_curve = []
    bloc_start_idx = 0
    next_bloc_dist = 500
    last_allure = None
    for i, p in enumerate(points):
        if p["distance"] >= next_bloc_dist or i == len(points)-1:
            bloc_points = points[bloc_start_idx:i+1]
            bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
            bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
            if bloc_dist > 0:
                allure = (bloc_time / 60) / (bloc_dist / 1000)
                last_allure = allure
            for _ in bloc_points:
                allure_curve.append(last_allure)
            bloc_start_idx = i+1
            next_bloc_dist += 500
    while len(allure_curve) < len(points):
        allure_curve.append(last_allure)

    return {
        "type_sortie": last.get("type_sortie", "-"),
        "date": datetime.strptime(last.get("date"), "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
        "distance_km": round(total_dist,2),
        "duration_min": round(total_time,1),
        "allure": f"{int(allure_moy)}:{int((allure_moy-int(allure_moy))*60):02d}" if allure_moy else "-",
        "fc_moy": round(fc_moy,1) if fc_moy else "-",
        "fc_max": fc_max,
        "k_moy": last.get("k_moy", "-"),
        "deriv_cardio": last.get("deriv_cardio", "-"),
        "gain_alt": round(gain_alt,1),
        "labels": json.dumps(labels),
        "allure_curve": json.dumps(allure_curve),
        "points_fc": json.dumps(points_fc),
        "points_alt": json.dumps(points_alt)
    }

# -------------------
# Routes Flask
# -------------------
@app.route("/")
def index():
    activities = load_activities_from_drive()
    activities = enrich_activities(activities)
    save_activities_to_drive(activities)
    profile = load_profile_from_drive()
    dashboard = compute_dashboard_data(activities, profile)
    objectives = load_objectives_from_drive()
    return render_template("index.html", dashboard=dashboard, objectives=objectives)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    profile = load_profile_from_drive()
    if request.method == 'POST':
        profile['birth_date'] = request.form['birth_date']
        profile['weight'] = float(request.form['weight'])
        profile['global_objective'] = request.form.get('global_objective', '')
        profile['particular_objective'] = request.form.get('particular_objective', '')
        events = [{"date": d, "name": n} 
                  for d, n in zip(request.form.getlist('event_date'), 
                                  request.form.getlist('event_name')) if d and n]
        profile['events'] = events
        with open('profile.json', 'w') as f:
            json.dump(profile, f, indent=2)
        save_profile_to_drive(profile)
        return redirect('/')
    return render_template('profile.html', profile=profile)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
