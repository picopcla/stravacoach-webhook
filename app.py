from flask import Flask, render_template, request, redirect
import json
import io
import os
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from openai import OpenAI

app = Flask(__name__)
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'
client = OpenAI()

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

# -------------------
# Spécifiques
# -------------------
def load_activities(): return load_file_from_drive('activities.json') or []
def load_profile(): return load_file_from_drive('profile.json') or {"birth_date": "", "weight": 0, "events": []}
def load_objectives(): return load_file_from_drive('objectives.json') or {}
def load_short_term_prompt_from_drive(): return load_file_from_drive('short_term_prompt.txt') or "Donne directement le JSON des objectifs à court terme."
def load_short_term_objectives(): return load_file_from_drive('short_term_objectives.json') or {}

# -------------------
# Enrichir activités avec type_sortie, k_moy, deriv_cardio
# -------------------
def enrich_activities(activities):
    for activity in activities:
        points = activity.get("points", [])
        if not points or len(points) < 5:
            activity.update({"type_sortie": "inconnue", "k_moy": "-", "deriv_cardio": "-"})
            continue

        total_dist = points[-1]["distance"] / 1000
        k_vals = [(p["hr"] / (p["vel"]*3.6)) for p in points if p.get("hr") and p.get("vel")]
        k_moy = sum(k_vals)/len(k_vals) if k_vals else "-"

        half = len(points)//2
        fc_first = [(p["hr"] - (p["alt"] - points[0]["alt"])*0.1) for p in points[:half] if p.get("hr")]
        fc_second= [(p["hr"] - (p["alt"] - points[0]["alt"])*0.1) for p in points[half:] if p.get("hr")]
        deriv_cardio = ((sum(fc_second)/len(fc_second) - sum(fc_first)/len(fc_first))/(sum(fc_first)/len(fc_first))*100) if fc_first and fc_second else "-"

        blocs, bloc_start_idx, next_bloc_dist = [], 0, 500
        for i, p in enumerate(points):
            if p["distance"] >= next_bloc_dist or i == len(points) - 1:
                bloc_points = points[bloc_start_idx:i+1]
                if len(bloc_points) > 1:
                    bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
                    bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
                    if bloc_dist > 0:
                        blocs.append((bloc_time / 60) / (bloc_dist / 1000))
                bloc_start_idx, next_bloc_dist = i+1, next_bloc_dist+500

        alternances, faster, avg_allure = 0, False, (sum(blocs)/len(blocs)) if blocs else 0
        for allure in blocs:
            if allure < avg_allure * 0.85:
                if not faster:
                    alternances += 1
                    faster = True
            else:
                faster = False

        type_sortie = "fractionné" if alternances >= 2 else ("longue" if total_dist >= 11 else "fond")
        activity.update({
            "type_sortie": type_sortie,
            "k_moy": round(k_moy, 2) if isinstance(k_moy, float) else "-",
            "deriv_cardio": round(deriv_cardio, 2) if isinstance(deriv_cardio, float) else "-"
        })
    return activities

# -------------------
# Dashboard principal
# -------------------
def compute_dashboard_data(activities):
    activities.sort(key=lambda x: x.get("date"))
    last = activities[-1]
    points = last.get("points", [])
    if not points: return {}

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
    while len(allure_curve) < len(points): allure_curve.append(last_allure)

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
        "points_alt": json.dumps(points_alt)
    }

# -------------------
# Routes Flask
# -------------------
@app.route("/")
def index():
    activities = enrich_activities(load_activities())
    with open('activities.json', 'w') as f:
        json.dump(activities, f, indent=2)
    save_file_to_drive('activities.json', 'activities.json')
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
        with open('profile.json', 'w') as f:
            json.dump(profile, f, indent=2)
        save_file_to_drive('profile.json', 'profile.json')
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
    with open('short_term_objectives.json', 'w') as f:
        json.dump(short_term_objectives, f, indent=2, ensure_ascii=False)
    save_file_to_drive('short_term_objectives.json', 'short_term_objectives.json')
    return f"<pre>{json.dumps(short_term_objectives, indent=2, ensure_ascii=False)}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
