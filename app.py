from flask import Flask, render_template, request, redirect
import json
import io
import os
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload
from openai import OpenAI

app = Flask(__name__)
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'
client = OpenAI()
print("✅ DÉMARRAGE APP.PY")

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

print("✅ Lecture Google Credentials OK")
# -------------------
# Generic helpers pour Drive
# -------------------
def load_file_from_drive(filename):
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='{filename}' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
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
        if filename.endswith('.json'):
            return json.loads(fh.read().decode("utf-8", errors="replace"))
        else:
            return fh.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"Erreur chargement {filename}:", e)
        return {} if filename.endswith('.json') else ""

def save_file_to_drive(local_file, drive_file, mime='application/json'):
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and name='{drive_file}' and trashed=false",
            spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        media = MediaFileUpload(local_file, mimetype=mime)
        if files:
            file_id = files[0]['id']
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            metadata = {'name': drive_file, 'parents': [FOLDER_ID]}
            drive_service.files().create(body=metadata, media_body=media, fields='id').execute()
    except Exception as e:
        print(f"Erreur upload {drive_file}:", e)
        
def upload_json_content_to_drive(json_data, drive_file_name):
    fh = io.BytesIO()
    fh.write(json.dumps(json_data, indent=2).encode("utf-8"))
    fh.seek(0)
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='{drive_file_name}' and trashed=false",
        spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    media = MediaIoBaseUpload(fh, mimetype='application/json')
    if files:
        file_id = files[0]['id']
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        metadata = {'name': drive_file_name, 'parents': [FOLDER_ID]}
        drive_service.files().create(body=metadata, media_body=media, fields='id').execute()

print("✅ Helpers OK")

# -------------------
# Spécifiques
# -------------------
def load_activities(): return load_file_from_drive('activities.json') or []
def load_profile(): return load_file_from_drive('profile.json') or {"birth_date": "", "weight": 0, "events": []}
def load_objectives(): return load_file_from_drive('objectives.json') or {}
def load_short_term_prompt_from_drive(): return load_file_from_drive('short_term_prompt.txt') or "Donne directement le JSON des objectifs à court terme."
def load_short_term_objectives(): return load_file_from_drive('short_term_objectives.json') or {}

def enrich_activities(activities):
    for activity in activities:
        points = activity.get("points", [])
        force = activity.get("force_recompute", False)

        # Si déjà enrichi et pas de force_recompute, on saute
        if not force and activity.get("type_sortie") not in [None, "-", "inconnue"]:
            continue

        # Calcul de l'histogramme des allures tous les ~100m
        pace_series = []
        next_distance = 100
        last_idx = 0
        for i, p in enumerate(points):
            if p["distance"] >= next_distance or i == len(points) -1:
                delta_dist = p["distance"] - points[last_idx]["distance"]
                delta_time = p["time"] - points[last_idx]["time"]
                if delta_dist > 0:
                    pace = (delta_time / 60) / (delta_dist / 1000)
                    pace_series.append(round(pace, 2))
                next_distance += 100
                last_idx = i

        # Charger le template depuis ton Drive
        prompt_template = load_file_from_drive("prompt_type.json").get("template", "")
        prompt = prompt_template.replace("{pace_series}", str(pace_series))
        upload_json_content_to_drive({"prompt": prompt}, "prompt_type_debug.json")

        # Appeler OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            detected_type = response.choices[0].message.content.strip().lower()
        except Exception as e:
            print("Erreur OpenAI:", e)
            detected_type = "inconnue"

        # Calcul k_moy et dérive cardio comme avant
        hr_vals = [p["hr"] for p in points if p.get("hr")]
        vel_vals = [p["vel"] for p in points if p.get("vel")]
        fc_moy = sum(hr_vals) / len(hr_vals) if hr_vals else 0
        vel_moy = sum(vel_vals) / len(vel_vals) if vel_vals else 0
        allure_min_km = (1 / vel_moy) * 16.6667 if vel_moy > 0 else 0
        k_moy = 0.43 * (fc_moy / allure_min_km) - 5.19 if allure_min_km > 0 else "-"

        n = len(points)
        third = n // 3
        fc_first = sum(p["hr"] for p in points[:third] if p.get("hr")) / third
        vel_first = sum(p["vel"] for p in points[:third] if p.get("vel")) / third
        allure_first = (1 / vel_first) * 16.6667 if vel_first > 0 else 0
        ratio_first = fc_first / allure_first if allure_first > 0 else 0

        fc_last = sum(p["hr"] for p in points[-third:] if p.get("hr")) / third
        vel_last = sum(p["vel"] for p in points[-third:] if p.get("vel")) / third
        allure_last = (1 / vel_last) * 16.6667 if vel_last > 0 else 0
        ratio_last = fc_last / allure_last if allure_last > 0 else 0

        deriv_cardio = (ratio_last / ratio_first) if ratio_first > 0 else "-"

        # Mettre à jour l'activité
        activity.update({
            "type_sortie": detected_type,
            "k_moy": round(k_moy, 2) if isinstance(k_moy, float) else "-",
            "deriv_cardio": round(deriv_cardio, 2) if isinstance(deriv_cardio, float) else "-"
        })

        # Nettoyer force_recompute
        activity.pop("force_recompute", None)

    return activities


print("✅ Activities OK")
# -------------------
# Dashboard principal
# -------------------
def compute_dashboard_data(activities):
    activities.sort(key=lambda x: x.get("date"))
    last = activities[-1]
    points = last.get("points", [])
    if not points:
        return {}

    total_dist = points[-1]["distance"] / 1000
    total_time = (points[-1]["time"] - points[0]["time"]) / 60
    allure_moy = total_time / total_dist if total_dist > 0 else None
    hr_vals = [p["hr"] for p in points if p.get("hr")]
    labels = [round(p["distance"]/1000, 3) for p in points]
    points_fc = [p["hr"] for p in points]
    points_alt = [p["alt"] - points[0]["alt"] for p in points]

    allure_curve, bloc_start_idx, next_bloc_dist, last_allure = [], 0, 500, None
    for i, p in enumerate(points):
        if p["distance"] >= next_bloc_dist or i == len(points)-1:
            bloc_points = points[bloc_start_idx:i+1]
            bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
            bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
            if bloc_dist > 0:
                last_allure = (bloc_time / 60) / (bloc_dist / 1000)
            allure_curve.extend([last_allure]*len(bloc_points))
            bloc_start_idx, next_bloc_dist = i+1, next_bloc_dist+500
    while len(allure_curve) < len(points):
        allure_curve.append(last_allure)

    return {
        "type_sortie": last.get("type_sortie", "-"),
        "date": datetime.strptime(last.get("date"), "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
        "distance_km": round(total_dist,2),
        "duration_min": round(total_time,1),
        "allure": f"{int(allure_moy)}:{int((allure_moy-int(allure_moy))*60):02d}" if allure_moy else "-",
        "fc_moy": round(sum(hr_vals)/len(hr_vals),1) if hr_vals else "-",
        "fc_max": max(hr_vals) if hr_vals else "-",
        "k_moy": last.get("k_moy", "-"),
        "deriv_cardio": last.get("deriv_cardio", "-"),
        "gain_alt": round(points[-1]["alt"] - points[0]["alt"],1),
        "labels": json.dumps(labels),
        "allure_curve": json.dumps(allure_curve),
        "points_fc": json.dumps(points_fc),
        "points_alt": json.dumps(points_alt),
        "history_dates": json.dumps([a["date"][:10] for a in activities if a.get("k_moy") != "-"]),
        "history_k": json.dumps([a["k_moy"] for a in activities if a.get("k_moy") != "-"]),
        "history_drift": json.dumps([a["deriv_cardio"] for a in activities if a.get("deriv_cardio") != "-"]),
    }

print("✅ Dashboard OK")

# -------------------
# Routes Flask
# -------------------
@app.route("/")
def index():
    activities = enrich_activities(load_activities())
    upload_json_content_to_drive(activities, 'activities.json')
    return render_template("index.html",
        dashboard=compute_dashboard_data(activities),
        objectives=load_objectives(),
        short_term=load_short_term_objectives())


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    profile = load_profile()
    if request.method == 'POST':
        profile.update({
            'birth_date': request.form['birth_date'],
            'weight': float(request.form['weight']),
            'global_objective': request.form.get('global_objective', ''),
            'particular_objective': request.form.get('particular_objective', ''),
            'events': [{"date": d, "name": n} for d, n in zip(request.form.getlist('event_date'), request.form.getlist('event_name')) if d and n]
        })
        upload_json_content_to_drive(profile, 'profile.json')
        return redirect('/')
    return render_template('profile.html', profile=profile)


@app.route("/generate_short_term_plan")
def generate_short_term_plan():
    profile, activities, prompt_template = load_profile(), load_activities(), load_short_term_prompt_from_drive()
    last_28_days = datetime.now() - timedelta(days=28)
    recent_events = [e for e in profile.get("events", []) if e.get("date") and datetime.strptime(e["date"], "%Y-%m-%d") > last_28_days]
    activities.sort(key=lambda x: x.get("date"))
    recent_runs = [{
        "date": act.get("date"),
        "type_sortie": act.get("type_sortie"),
        "k_moy": act.get("k_moy"),
        "deriv_cardio": act.get("deriv_cardio"),
        "laps": [{"km": round(l["distance"]/1000,2), "duration_min": round(l["duration"]/60,2), "fc_avg": l.get("fc_avg"), "fc_max": l.get("fc_max"), "gain_alt": l.get("gain_alt")} for l in act.get("laps",[])]
    } for act in activities[-10:]]
    last_run = recent_runs[-1] if recent_runs else {}
    final_prompt = prompt_template + "\n\nVoici les données JSON :\n" + json.dumps({"recent_events": recent_events, "recent_runs": recent_runs, "last_run": last_run}, indent=2, ensure_ascii=False)
    try:
        content = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": final_prompt}]).choices[0].message.content
        short_term_objectives = json.loads(content)
    except Exception as e:
        print("Erreur parsing JSON GPT:", e)
        short_term_objectives = {"error": "Impossible de parser la réponse GPT", "raw": content}
    upload_json_content_to_drive(short_term_objectives, 'short_term_objectives.json')
    return f"<pre>{json.dumps(short_term_objectives, indent=2, ensure_ascii=False)}</pre>"
print("✅ Flask OK")

if __name__ == "__main__":
    print("✅ Lancement du serveur Flask...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))