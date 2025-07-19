from flask import Flask, render_template, request, redirect
import json
import io
import os
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload
from openai import OpenAI
import numpy as np

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
# Helpers Google Drive
# ------------------
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

def upload_json_content_to_drive(json_data, drive_file_name):
    fh = io.BytesIO()
    fh.write(json.dumps(json_data, indent=2).encode("utf-8"))
    fh.seek(0)
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and name='{drive_file_name}' and trashed=false",
        spaces='drive', fields='files(id, name)', supportsAllDrives=True
    ).execute()
    files = results.get('files', [])
    media = MediaIoBaseUpload(fh, mimetype='application/json')
    if files:
        file_id = files[0]['id']
        drive_service.files().update(
            fileId=file_id, media_body=media, supportsAllDrives=True
        ).execute()
    else:
        metadata = {
            'name': drive_file_name,
            'parents': [FOLDER_ID],
            'mimeType': 'application/json'
        }
        drive_service.files().create(
            body=metadata, media_body=media, fields='id', supportsAllDrives=True
        ).execute()

print("✅ Helpers OK")

# -------------------
# Loaders
# -------------------
def load_activities(): return load_file_from_drive('activities.json') or []
def load_profile(): return load_file_from_drive('profile.json') or {"birth_date": "", "weight": 0, "events": []}
def load_objectives(): return load_file_from_drive('objectives.json') or {}
def load_short_term_prompt_from_drive():
    return load_file_from_drive('prompt_short_term.txt') or "Donne directement le JSON des objectifs à court terme."
def load_short_term_objectives(): return load_file_from_drive('short_term_objectives.json') or {}


# -------------------
# Fonctions spécifiques
# -------------------
def get_fcmax_from_fractionnes(activities):
    fcmax = 0
    for act in activities:
        if act.get("type_sortie") == "fractionné":
            for point in act.get("points", []):
                hr = point.get("hr")
                if hr is not None and hr > fcmax:
                    fcmax = hr
    return fcmax

def enrich_single_activity(activity, fc_max_fractionnes):
    points = activity.get("points", [])
    if not points or len(points) < 5:
        return activity

    distances = np.array([p["distance"]/1000 for p in points])
    fcs = np.array([p.get("hr") if p.get("hr") is not None else np.nan for p in points])
    vels = np.array([p.get("vel", 0) for p in points])
    alts = np.array([p.get("alt", 0) for p in points])

    delta_dist = np.diff(distances, prepend=distances[0]) * 1000
    delta_alt = np.diff(alts, prepend=alts[0])
    delta_dist[delta_dist == 0] = 0.001

    pentes = (delta_alt / delta_dist) * 100
    allures_brutes = np.where(vels > 0, (1 / vels) * 16.6667, np.nan)
    allures_corrigees = np.where((allures_brutes - 0.2 * pentes) < 0, np.nan, allures_brutes - 0.2 * pentes)

    ratios = np.where(allures_corrigees > 0, fcs / allures_corrigees, np.nan)
    valid = (~np.isnan(allures_corrigees)) & (~np.isnan(fcs))
    distances, fcs, allures_corrigees, ratios = distances[valid], fcs[valid], allures_corrigees[valid], ratios[valid]

    if len(distances) < 5:
        return activity

    total_duration = points[-1]["time"] - points[0]["time"]
    slope, intercept = np.polyfit(distances, ratios, 1)
    r_squared = np.corrcoef(distances, ratios)[0,1]**2
    collapse_threshold = np.mean(allures_corrigees[:max(1,len(allures_corrigees)//3)]) * 1.10
    collapse_distance = next((d for a, d in zip(allures_corrigees, distances) if a > collapse_threshold), distances[-1])
    cv_allure = np.std(allures_corrigees) / np.mean(allures_corrigees)
    cv_cardio = np.std(ratios) / np.mean(ratios)
    seuil_90 = 0.9 * fc_max_fractionnes
    above_90_count = sum(1 for hr in fcs if hr > seuil_90)
    time_above_90 = (above_90_count / len(fcs)) * total_duration if len(fcs) else 0
    split = len(allures_corrigees)//3
    endurance_index = np.mean(allures_corrigees[-split:]) / np.mean(allures_corrigees[:split])
    fc_moy, allure_moy = np.mean(fcs), np.mean(allures_corrigees)
    k_moy = 0.43 * (fc_moy / allure_moy) - 5.19 if allure_moy > 0 else "-"
    ratio_first, ratio_last = np.mean(ratios[:split]), np.mean(ratios[-split:])
    deriv_cardio = ratio_last / ratio_first if ratio_first > 0 else "-"
    seuil_bas, seuil_haut = 0.6 * fc_max_fractionnes, 0.7 * fc_max_fractionnes
    zone2_count = sum(1 for hr in fcs if seuil_bas < hr < seuil_haut)
    pourcentage_zone2 = (zone2_count / len(fcs)) * 100 if len(fcs) else 0
    ratio_fc_allure_global = np.mean(ratios)

    activity.update({
        "drift_slope": round(slope, 4),
        "drift_r2": round(r_squared, 4),
        "collapse_distance_km": round(collapse_distance, 2),
        "cv_allure": round(cv_allure, 4),
        "cv_cardio": round(cv_cardio, 4),
        "time_above_90_pct_fcmax": round(time_above_90, 1),
        "endurance_index": round(endurance_index, 4),
        "k_moy": round(k_moy, 3) if isinstance(k_moy, float) else "-",
        "deriv_cardio": round(deriv_cardio, 3) if isinstance(deriv_cardio, float) else "-",
        "pourcentage_zone2": round(pourcentage_zone2, 1),
        "ratio_fc_allure_global": round(ratio_fc_allure_global, 3)
    })

    return activity

def enrich_activities(activities):
    fc_max_fractionnes = get_fcmax_from_fractionnes(activities)
    print(f"📈 FC max fractionnés: {fc_max_fractionnes}")
    for idx, activity in enumerate(activities):
        points = activity.get("points", [])
        force = activity.get("force_recompute", False)
        if not force and activity.get("type_sortie") not in [None, "-", "inconnue"]:
            continue
        pace_series, next_distance, last_idx = [], 100, 0
        for i, p in enumerate(points):
            if p["distance"] >= next_distance or i == len(points)-1:
                delta_dist = p["distance"] - points[last_idx]["distance"]
                delta_time = p["time"] - points[last_idx]["time"]
                if delta_dist > 0:
                    pace_series.append(round((delta_time / 60) / (delta_dist / 1000), 2))
                next_distance += 100
                last_idx = i
        prompt_template = load_file_from_drive("prompt_type.txt") or ""
        prompt = prompt_template.replace("{pace_series}", str(pace_series))
        upload_json_content_to_drive({"prompt": prompt}, "prompt_type_debug.json")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            detected_type = response.choices[0].message.content.strip().lower()
        except Exception as e:
            print("Erreur GPT:", e)
            detected_type = "inconnue"
        activity["type_sortie"] = detected_type
        activity = enrich_single_activity(activity, fc_max_fractionnes)
        print(f"🏃 Act#{idx+1} ➔ type: {activity['type_sortie']}, k_moy: {activity.get('k_moy')}")
        activity.pop("force_recompute", None)
    return activities

def allure_mmss_to_decimal(mmss):
    try:
        minutes, seconds = mmss.split(":")
        return int(minutes) + int(seconds) / 60
    except Exception:
        return 0.0
        
def convert_short_term_allures(short_term):
    if not short_term or "prochains_runs" not in short_term:
        return short_term
    for run in short_term["prochains_runs"]:
        if isinstance(run.get("allure"), str):
            run["allure_decimal"] = allure_mmss_to_decimal(run["allure"])
        else:
            run["allure_decimal"] = 0.0
    return short_term


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
    labels = [round(p["distance"]/1000,3) for p in points]
    if labels and labels[0] != 0:
        labels[0] = 0.0
    points_fc = [p["hr"] for p in points]
    points_alt = [p["alt"]-points[0]["alt"] for p in points]
    allure_curve, bloc_start_idx, next_bloc_dist, last_allure = [], 0, 500, None
    for i, p in enumerate(points):
        if p["distance"] >= next_bloc_dist or i==len(points)-1:
            bloc_points = points[bloc_start_idx:i+1]
            bloc_dist = bloc_points[-1]["distance"]-bloc_points[0]["distance"]
            bloc_time = bloc_points[-1]["time"]-bloc_points[0]["time"]
            if bloc_dist>0:
                last_allure = (bloc_time/60)/(bloc_dist/1000)
            allure_curve.extend([last_allure]*len(bloc_points))
            bloc_start_idx, next_bloc_dist = i+1, next_bloc_dist+500
    while len(allure_curve)<len(points):
        allure_curve.append(last_allure)
    return {
        "type_sortie": last.get("type_sortie","-"),
        "date": datetime.strptime(last.get("date"),"%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
        "distance_km": round(total_dist,2),
        "duration_min": round(total_time,1),
        "allure": f"{int(allure_moy)}:{int((allure_moy-int(allure_moy))*60):02d}" if allure_moy else "-",
        "fc_moy": round(sum(hr_vals)/len(hr_vals),1) if hr_vals else "-",
        "fc_max": max(hr_vals) if hr_vals else "-",
        "k_moy": last.get("k_moy","-"),
        "deriv_cardio": last.get("deriv_cardio","-"),
        "gain_alt": round(points[-1]["alt"]-points[0]["alt"],1),
        "drift_slope": last.get("drift_slope","-"),
        "cv_allure": last.get("cv_allure","-"),
        "cv_cardio": last.get("cv_cardio","-"),
        "collapse_distance_km": last.get("collapse_distance_km","-"),
        "pourcentage_zone2": last.get("pourcentage_zone2","-"),
        "time_above_90_pct_fcmax": last.get("time_above_90_pct_fcmax","-"),
        "ratio_fc_allure_global": last.get("ratio_fc_allure_global","-"),
        "labels": json.dumps(labels),
        "allure_curve": json.dumps(allure_curve),
        "points_fc": json.dumps(points_fc),
        "points_alt": json.dumps(points_alt),
        "history_dates": json.dumps([a["date"][:10] for a in activities if a.get("k_moy")!="-"]),
        "history_k": json.dumps([a["k_moy"] for a in activities if a.get("k_moy")!="-"]),
        "history_drift": json.dumps([a["deriv_cardio"] for a in activities if a.get("deriv_cardio")!="-"]),
    }

print("✅ Dashboard OK")

@app.route("/")
def index():
    activities = load_activities()
    print(f"📂 {len(activities)} activités chargées depuis Drive")
    activities = enrich_activities(activities)
    upload_json_content_to_drive(activities, 'activities.json')
    print("💾 activities.json mis à jour")
    dashboard = compute_dashboard_data(activities)
    print("📊 Dashboard calculé")
    return render_template("index.html",
        dashboard=dashboard,
        objectives=load_objectives(),
        short_term=load_short_term_objectives())

@app.route('/profile', methods=['GET','POST'])
def profile():
    profile = load_profile()
    if request.method=='POST':
        profile.update({
            'birth_date': request.form['birth_date'],
            'weight': float(request.form['weight']),
            'global_objective': request.form.get('global_objective',''),
            'particular_objective': request.form.get('particular_objective',''),
            'events': [{"date":d,"name":n} for d,n in zip(request.form.getlist('event_date'),request.form.getlist('event_name')) if d and n]
        })
        upload_json_content_to_drive(profile, 'profile.json')
        return redirect('/')
    return render_template('profile.html', profile=profile)

@app.route("/generate_short_term_plan")
def generate_short_term_plan():
    from datetime import timezone
    profile = load_profile()
    activities = load_activities()
    prompt_template = load_short_term_prompt_from_drive()
    last_30_days = datetime.now(timezone.utc)-timedelta(days=30)
    recent_events = [e for e in profile.get("events",[]) if e.get("date") and datetime.strptime(e["date"],"%Y-%m-%d").replace(tzinfo=timezone.utc)>last_30_days]
    recent_runs = [{
        "date":act.get("date"),"type_sortie":act.get("type_sortie"),"k_moy":act.get("k_moy"),
        "drift_slope":act.get("drift_slope"),"deriv_cardio":act.get("deriv_cardio"),
        "collapse_distance_km":act.get("collapse_distance_km"),"cv_allure":act.get("cv_allure"),
        "cv_cardio":act.get("cv_cardio"),"endurance_index":act.get("endurance_index")
    } for act in activities if datetime.strptime(act.get("date"),"%Y-%m-%dT%H:%M:%S%z")>last_30_days]
    last_run = recent_runs[-1] if recent_runs else {}
    same_type_runs = [r for r in recent_runs if r["type_sortie"]==last_run.get("type_sortie")]
    payload = {"recent_events":recent_events,"recent_runs_same_type":same_type_runs,"last_run":last_run}
    final_prompt = prompt_template + "\n\nVoici les données JSON :\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    print("\n=== PROMPT ENVOYÉ À GPT ===\n", final_prompt)
    try:
        content = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":final_prompt}]
        ).choices[0].message.content
        print("\n=== RÉPONSE BRUTE DE GPT ===\n", content)
        short_term_objectives = json.loads(content)
    except Exception as e:
        print("Erreur parsing JSON GPT:", e)
        short_term_objectives = {"error":"Impossible de parser la réponse GPT","raw":content}
    upload_json_content_to_drive(short_term_objectives,'short_term_objectives.json')
    return redirect("/")

if __name__=="__main__":
    print("✅ Lancement du serveur Flask...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
