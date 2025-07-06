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
# Fonctions helpers
# -------------------
def load_activities_from_drive():
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='activities.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
    except Exception as e:
        print("Erreur connexion Drive (activities):", e)
        return None
    files = results.get('files', [])
    if not files:
        return None
    file_id = files[0]['id']
    request_file = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request_file)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.loads(fh.read())

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

def save_profile_to_drive(profile):
    with open('profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='profile.json' and trashed=false",
            spaces='drive', fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if files:
            file_id = files[0]['id']
            media = MediaFileUpload('profile.json', mimetype='application/json')
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': 'profile.json', 'parents': [FOLDER_ID]}
            media = MediaFileUpload('profile.json', mimetype='application/json')
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    except Exception as e:
        print("Erreur upload profile.json:", e)

# -------------------
# Dashboard principal
# -------------------
def compute_dashboard_data(activities, profile):
    from datetime import datetime
    import json

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
    k_vals = [(p["hr"] / (p["vel"]*3.6)) for p in points if p.get("hr") and p.get("vel")]
    k_moy = sum(k_vals)/len(k_vals) if k_vals else "-"
    half = len(points)//2
    fc_first = [p["hr"] for p in points[:half] if p.get("hr")]
    fc_second= [p["hr"] for p in points[half:] if p.get("hr")]
    deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first))/ (sum(fc_first)/len(fc_first))*100) if fc_first and fc_second else "-"
    gain_alt = points[-1]["alt"] - points[0]["alt"] if points[0].get("alt") else 0

    labels = [round(p["distance"]/1000, 3) for p in points]
    points_fc = [p["hr"] for p in points]
    points_alt = [p["alt"] - points[0]["alt"] for p in points]

    # Allure en blocs fixes de 500 m
    allure_curve = []
    bloc_start_idx = 0
    next_bloc_dist = 200  # premier bloc à 200 m
    last_allure = None

    for i, p in enumerate(points):
        if p["distance"] >= next_bloc_dist or i == len(points)-1:
            bloc_points = points[bloc_start_idx:i+1]
            bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
            bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
            if bloc_dist > 0:
                allure = (bloc_time / 60) / (bloc_dist / 1000)  # min/km
                last_allure = allure
            for _ in bloc_points:
                allure_curve.append(last_allure)
            bloc_start_idx = i+1
            next_bloc_dist += 500

    while len(allure_curve) < len(points):
        allure_curve.append(last_allure)

    # Exclure les 300 premiers mètres sur le graphique
    filtered_labels = [l for l, p in zip(labels, points) if p["distance"] >= 0]
    filtered_fc = [fc for fc, p in zip(points_fc, points) if p["distance"] >= 0]
    filtered_alt = [alt for alt, p in zip(points_alt, points) if p["distance"] >= 0]
    filtered_allure = [al for al, p in zip(allure_curve, points) if p["distance"] >= 0]

    return {
        "date": datetime.strptime(last.get("date"), "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
        "distance_km": round(total_dist,2),
        "duration_min": round(total_time,1),
        "allure": f"{int(allure_moy)}:{int((allure_moy-int(allure_moy))*60):02d}" if allure_moy else "-",
        "fc_moy": round(fc_moy,1) if fc_moy else "-",
        "fc_max": fc_max,
        "k_moy": round(k_moy,1) if k_moy else "-",
        "deriv_cardio": round(deriv_cardio,1) if deriv_cardio else "-",
        "gain_alt": round(gain_alt,1),
        "labels": json.dumps(filtered_labels),
        "allure_curve": json.dumps(filtered_allure),
        "points_fc": json.dumps(filtered_fc),
        "points_alt": json.dumps(filtered_alt)
    }


# -------------------
# Routes Flask
# -------------------
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
        events = [{"date": d, "name": n} for d,n in zip(request.form.getlist('event_date'), request.form.getlist('event_name')) if d and n]
        profile['events'] = events
        save_profile_to_drive(profile)
        return redirect('/profile')
    return render_template('profile.html', profile=profile)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
