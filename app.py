import os
import json
import io
from datetime import datetime, timedelta
from dateutil import parser
from flask import Flask, render_template, request, redirect
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import requests
import time
from xgboost import XGBClassifier
import pickle
import pandas as pd
import subprocess


# -------------------
# Fonction pour loguer les étapes avec durée
# -------------------
def log_step(message, start_time):
    elapsed = time.time() - start_time
    print(f"⏱️ {message} — {elapsed:.2f} sec depuis début")

app = Flask(__name__)
FOLDER_ID = '1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t'

# --- Chargement .env local si pas sur Render ---
if not os.getenv("RENDER"):
    load_dotenv(r"C:\StravaSecurity\main.env")
    print("✅ Variables d'environnement chargées depuis main.env")

# --- Init OpenAI ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY non défini")
client = OpenAI(api_key=openai_api_key)
print("✅ OpenAI client initialisé")

# --- Init Google Drive ---
try:
    # Si Render, la variable est un JSON complet en string
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
except KeyError:
    # Sinon, on lit le fichier local
    with open(r'C:\StravaSecurity\services.json', 'r', encoding='utf-8') as f:
        service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=['https://www.googleapis.com/auth/drive']
)
drive_service = build('drive', 'v3', credentials=credentials)
print("✅ Google Drive service initialisé")

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
        
def save_short_term_objectives(data):
    upload_json_content_to_drive(data, 'short_term_objectives.json')
    
# -------------------
# Détection du type de séance (règles simples par distance)
# -------------------
def detect_session_type(activity):
    """
    Règles par distance (aucune substitution par XGBoost ici) :
    - > 11 km  -> long_run
    - < 8 km   -> normal_5k
    - sinon    -> normal_10k
    """
    pts = activity.get("points", [])
    if not pts:
        return activity.get("type_sortie", "inconnue") or "inconnue"

    dist_km = pts[-1]["distance"] / 1000.0

    if dist_km > 11:
        return "long_run"
    if dist_km < 8:
        return "normal_5k"
    return "normal_10k"

    
def _features_fractionne(activity):
    """Vecteur de 4 features simples pour XGBoost: [cv_allure, cv_fc, blocs_rapides, pct_temps_90].
    - blocs_rapides détectés avec une fenêtre glissante en distance (offset libre)
    """
    pts = activity.get("points", [])
    if len(pts) < 10:
        return [0.0, 0.0, 0.0, 0.0]

    fcs = np.array([p.get("hr") for p in pts], dtype=float)
    vels = np.array([p.get("vel") for p in pts], dtype=float)
    dists = np.array([p.get("distance") for p in pts], dtype=float)

    # Allures en min/km (nan si vel <= 0)
    allures = np.where(vels > 0, (1.0 / vels) * 16.6667, np.nan)

    # 1) CV allure & 2) CV FC
    def _cv(x):
        m = np.nanmean(x)
        if not np.isfinite(m) or m == 0:
            return 0.0
        return float(np.nanstd(x) / m)

    cv_allure = _cv(allures)
    cv_fc = _cv(fcs)

    # 3) Nombre de blocs rapides détectés via fenêtre glissante
    WINDOW_M   = 500     # longueur de la fenêtre en mètres
    FAST_DELTA = 0.40    # seuil de rapidité (min/km plus rapide que la moyenne)
    COOLDOWN_M = 200     # distance minimale entre deux détections pour éviter les doublons

    mean_all = np.nanmean(allures)
    thr_fast = mean_all - FAST_DELTA if np.isfinite(mean_all) else np.nan

    blocs_rapides = 0
    i = 0
    N = len(dists)
    last_hit_end_d = -1e9  # distance fin du dernier bloc validé

    while i < N - 1:
        # trouve j tel que distance(i → j) >= WINDOW_M
        j = i
        while j < N and (dists[j] - dists[i]) < WINDOW_M:
            j += 1
        if j >= N:
            break

        bloc_all = np.nanmean(allures[i:j+1])
        dist_i, dist_j = dists[i], dists[j]

        # Critère rapide + cooldown respecté
        is_fast = (np.isfinite(bloc_all) and np.isfinite(thr_fast) and bloc_all < thr_fast)
        far_enough = (dist_i - last_hit_end_d) >= COOLDOWN_M

        if is_fast and far_enough:
            blocs_rapides += 1
            last_hit_end_d = dist_j
            # saute à la fin du bloc + cooldown
            i = j
            while i < N and (dists[i] - last_hit_end_d) < COOLDOWN_M:
                i += 1
            continue

        # sinon avance juste d'un point
        i += 1

    # 4) % temps au-dessus de 90% FCmax de la séance
    if np.all(np.isnan(fcs)) or len(fcs) == 0:
        pct_90 = 0.0
    else:
        fcmax = np.nanmax(fcs)
        thr = 0.9 * fcmax if np.isfinite(fcmax) and fcmax > 0 else np.nan
        if np.isfinite(thr):
            pct_90 = float(np.nansum(fcs > thr) / np.count_nonzero(~np.isnan(fcs)))
        else:
            pct_90 = 0.0

    feats = [cv_allure, cv_fc, float(blocs_rapides), pct_90]
    # Remplace NaN/±inf par 0 pour le modèle
    feats = [0.0 if (not np.isfinite(v)) else float(v) for v in feats]
    return feats

    
    
def tag_session_types(activities):
    changed = False
    for act in activities:
        cur = act.get("type_sortie")
        if cur in (None, "-", "inconnue") or act.get("force_recompute", False):
            new_type = detect_session_type(act)
            if new_type != cur:
                act["type_sortie"] = new_type
                changed = True
        act.pop("force_recompute", None)
    return activities, changed
def apply_fractionne_flags(activities):
    """
    Ajoute/actualise :
      - is_fractionne: bool
      - fractionne_prob: float [0..1]
    Sans jamais modifier type_sortie.
    Règle: on évalue XGB seulement si distance <= 11 km ; sinon is_fractionne=False.
    PRIORITÉ aux labels manuels (is_fractionne_label) s'ils sont présents.
    """
    changed = False
    for act in activities:
        # 1) Priorité au label manuel
        if "is_fractionne_label" in act:
            lbl = bool(act["is_fractionne_label"])
            new_flag = lbl
            new_prob = 1.0 if lbl else 0.0

        else:
            # 2) Sinon, on calcule avec le modèle si possible
            pts = act.get("points", [])
            if not pts:
                new_flag, new_prob = False, 0.0
            else:
                dist_km = pts[-1].get("distance", 0) / 1000.0
                if dist_km > 11 or fractionne_model is None:
                    new_flag, new_prob = False, 0.0
                else:
                    feats = _features_fractionne(act)
                    try:
                        proba = float(fractionne_model.predict_proba(np.array([feats]))[0][1])
                        new_prob = round(proba, 3)
                        new_flag = (proba >= 0.5)  # seuil ajustable
                    except Exception as e:
                        print("🤖 XGBoost predict error:", e)
                        new_flag, new_prob = False, 0.0

        # 3) Appliquer si changement
        if act.get("is_fractionne") != new_flag or act.get("fractionne_prob") != new_prob:
            act["is_fractionne"] = new_flag
            act["fractionne_prob"] = new_prob
            changed = True

    return activities, changed




print("✅ Helpers OK")


# -------- XGBoost fractionné (chargement modèle) --------
MODEL_PATH = "fractionne_xgb.pkl"
fractionne_model = None  # global lecture seule

# ==== Auto-réentrainement XGBoost ====
AUTO_RETRAIN_XGB = True                 # désactive en mettant False si besoin
LAST_TRAIN_META = "ml/.last_train_meta.json"  # fichier local pour mémoriser le dernier état (compte d’activités)

def _load_last_train_meta():
    try:
        with open(LAST_TRAIN_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_trained_count": 0}

def _save_last_train_meta(count):
    os.makedirs(os.path.dirname(LAST_TRAIN_META), exist_ok=True)
    with open(LAST_TRAIN_META, "w", encoding="utf-8") as f:
        json.dump({"last_trained_count": int(count)}, f)

def _count_manual_labels(activities):
    pos = neg = 0
    for a in activities:
        if "is_fractionne_label" in a:
            if bool(a["is_fractionne_label"]): pos += 1
            else: neg += 1
    return pos, neg

def _should_retrain_xgb(activities):
    """
    Vrai si:
      - on a au moins 8 fractionnés (pos) ET 8 non-fractionnés (neg), ET
      - au moins 1 nouvelle activité depuis le dernier entraînement.
    """
    meta = _load_last_train_meta()
    last_cnt = meta.get("last_trained_count", 0)
    cur_cnt = len(activities)
    if cur_cnt <= last_cnt:
        return False

    pos, neg = _count_manual_labels(activities)
    if pos < 8 or neg < 8:
        print(f"ℹ️ Pas assez de labels pour auto-train (pos={pos}, neg={neg}, min=8 chacun)")
        return False

    print(f"🔁 Auto-train éligible: new_activities={cur_cnt - last_cnt}, labels(pos={pos}, neg={neg})")
    return True

def _retrain_fractionne_model_and_reload():
    """
    Lance ml/train_fractionne_xgb.py, recharge le modèle, mémorise le nb d'activités.
    """
    try:
        print("🤖 Auto-train: lancement ml/train_fractionne_xgb.py ...")
        subprocess.run(["python", "ml/train_fractionne_xgb.py"], check=True, timeout=300)
        # Recharge le modèle
        global fractionne_model
        fractionne_model = load_fractionne_model()
        # Mémorise le nouveau compteur
        activities = load_activities()
        _save_last_train_meta(len(activities))
        print("✅ Auto-train OK et modèle rechargé.")
        return True
    except Exception as e:
        print("❌ Auto-train échoué:", e)
        return False


def load_fractionne_model(path=MODEL_PATH):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"🤖 XGBoost fractionné chargé: {path}")
        return model
    except FileNotFoundError:
        print("🤖 XGBoost fractionné introuvable → désactivé (pas de fichier .pkl).")
    except Exception as e:
        print("🤖 Erreur chargement XGBoost:", e)
    return None

# Charge à l'init
fractionne_model = load_fractionne_model()

# -------------------
# Fonction météo (Open-Meteo)
# -------------------
from datetime import datetime, timedelta, date
import requests
from collections import Counter

from dateutil import parser  # déjà importé

def get_temperature_for_run(lat, lon, start_datetime_str, duration_minutes):
    try:
        # ✅ Parse ISO 8601 (Z ou +02:00)
        start_dt = parser.isoparse(start_datetime_str)
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # ✅ Supprime le fuseau pour comparer avec les données naïves de l'API
        start_dt = start_dt.replace(tzinfo=None)
        end_dt = end_dt.replace(tzinfo=None)

        print(f"🕒 Heure début (start_dt): {start_dt}, fin (end_dt): {end_dt}")
    except Exception as e:
        print("❌ Erreur parsing datetime pour météo:", e, start_datetime_str)
        return None, None, None, None

    today = date.today()
    yesterday = today - timedelta(days=1)
    is_today = start_dt.date() == today
    is_yesterday = start_dt.date() == yesterday

   # ✅ Utilise forecast pour aujourd'hui et hier
    if is_today or is_yesterday:
        query_type = "forecast"
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,weathercode"
            f"&timezone=auto"
        )
    else:
        query_type = "archive"
        # Archive pour avant-hier et plus
        date_str = start_dt.strftime("%Y-%m-%d")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&hourly=temperature_2m,weathercode"
            f"&timezone=auto"
        )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        hours = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        weathercodes = data.get("hourly", {}).get("weathercode", [])

        if not hours or not temps:
            print("⚠️ Aucune donnée horaire trouvée.")
            return None, None, None, None

        # Convertit toutes les heures en datetime (naïves) pour comparaison
        hours_dt = [datetime.fromisoformat(h) for h in hours]

        # Trouver la température la plus proche pour début et fin
        def closest_temp(target_dt):
            diffs = [abs((dt - target_dt).total_seconds()) for dt in hours_dt]
            idx = diffs.index(min(diffs))
            return temps[idx] if temps[idx] is not None else None

        temp_debut = closest_temp(start_dt)
        temp_fin = closest_temp(end_dt)

        # Moyenne sur la fenêtre de course
        temp_values = [
            temp for dt, temp in zip(hours_dt, temps)
            if start_dt <= dt <= end_dt and temp is not None
        ]

        # ✅ Si pas de moyenne, utiliser au moins temp_debut ou temp_fin
        avg_temp = (
            round(sum(temp_values) / len(temp_values), 1)
            if temp_values else temp_debut or temp_fin
        )

        # Code météo le plus fréquent pendant la course
# ✅ Trouver le code météo dominant avec une marge de 30 min

        margin = timedelta(minutes=30)
        weather_in_window = [
            wc for dt, wc in zip(hours_dt, weathercodes)
            if (start_dt - margin) <= dt <= (end_dt + margin) and wc is not None
        ]

        if weather_in_window:
            # Si on a trouvé des codes météo dans la fenêtre élargie, on prend le plus fréquent
            most_common_code = Counter(weather_in_window).most_common(1)[0][0]
        else:
            # Sinon, on prend le code météo le plus proche du début de la course
            diffs = [abs((dt - start_dt).total_seconds()) for dt in hours_dt]
            most_common_code = weathercodes[diffs.index(min(diffs))] if diffs else None

        return avg_temp, temp_debut, temp_fin, most_common_code

    except Exception as e:
        print("❌ Erreur lors de la requête ou du traitement météo:", e)
        return None, None, None, None



def get_weather_emoji_for_activity(activity):
    weather_code_map = {
        0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
        45: "🌫️", 48: "🌫️", 51: "🌦️", 53: "🌧️",
        55: "🌧️", 61: "🌧️", 63: "🌧️", 65: "🌧️",
        71: "❄️", 73: "❄️", 75: "❄️", 80: "🌧️",
        81: "🌧️", 82: "🌧️", 95: "⛈️", 96: "⛈️",
        99: "⛈️"
    }
    points = activity.get("points", [])
    if not points:
        return "❓"
    lat, lon = None, None
    if "lat" in points[0] and "lng" in points[0]:
        lat, lon = points[0]["lat"], points[0]["lng"]
    elif "start_latlng" in activity and activity["start_latlng"]:
        lat, lon = activity["start_latlng"][0], activity["start_latlng"][1]
    date_str = activity.get("date", None)
    if not lat or not lon or not date_str:
        return "❓"
    duration_minutes = (points[-1]["time"] - points[0]["time"]) / 60
    _, _, _, weather_code = get_temperature_for_run(lat, lon, date_str, duration_minutes)
    return weather_code_map.get(weather_code, "❓")
    
def ensure_weather_data(activities):
    """Vérifie que chaque activité a les données météo et les calcule si elles sont absentes."""
    updated = False

    for act in activities:
        if act.get("avg_temperature") is None or act.get("weather_code") is None:
            points = act.get("points", [])
            if not points:
                continue

            lat, lon = points[0].get("lat"), points[0].get("lng")
            duration = (points[-1]["time"] - points[0]["time"]) / 60

            avg_temp, _, _, weather_code = get_temperature_for_run(
                lat, lon, act.get("date"), duration
            )

            act["avg_temperature"] = avg_temp
            act["weather_code"] = weather_code
            updated = True
            print(f"🌤️ Météo ajoutée pour {act.get('date')} ➜ {avg_temp}°C / code {weather_code}")

    if updated:
        upload_json_content_to_drive(activities, 'activities.json')
        print("💾 activities.json mis à jour avec la météo")

    return activities


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
# Fonctions spécifiques (inchangées sauf enrich_activities etc)
# -------------------
def get_fcmax_from_fractionnes(activities):
    fcmax = 0
    for act in activities:
        if act.get("type_sortie") == "fractionné" or act.get("is_fractionne") is True:
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
    split = max(1, len(allures_corrigees)//3)
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
        # 1) Assigner le type de séance si manquant/forcé (règles simples par distance)
        if activity.get("type_sortie") in (None, "-", "inconnue") or activity.get("force_recompute", False):
            activity["type_sortie"] = detect_session_type(activity)

        # 2) Enrichissements numériques (k, dérive cardio, etc.)
        activity = enrich_single_activity(activity, fc_max_fractionnes)

        print(f"🏃 Act#{idx+1} ➔ type: {activity.get('type_sortie')}, k_moy: {activity.get('k_moy')}")
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
    weather_code_map = {
       0: "☀️",  # Clear sky
       1: "🌤️",  # Mainly clear
       2: "⛅",   # Partly cloudy
       3: "☁️",  # Overcast
       45: "🌫️", # Fog
       48: "🌫️", # Depositing rime fog
       51: "🌦️", # Drizzle light
       53: "🌧️", # Drizzle moderate
       55: "🌧️", # Drizzle dense
       61: "🌧️", # Rain slight
       63: "🌧️", # Rain moderate
       65: "🌧️", # Rain heavy
       71: "❄️",  # Snow fall slight
       73: "❄️",  # Snow fall moderate
       75: "❄️",  # Snow fall heavy
       80: "🌧️", # Rain showers slight
       81: "🌧️", # Rain showers moderate
       82: "🌧️", # Rain showers violent
       95: "⛈️",  # Thunderstorm slight
       96: "⛈️",  # Thunderstorm with slight hail
       99: "⛈️",  # Thunderstorm with heavy hail
    }

    print("\n🔍 DEBUG --- Vérification température")

    activities.sort(key=lambda x: x.get("date"))
    last = activities[-1] if activities else {}

    points = last.get("points", [])
    if not points:
        print("⚠️ Pas de points dans la dernière activité")
        return {}

    # Date
    date_str = "-"
    try:
        date_str = datetime.strptime(last.get("date"), "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d")
    except Exception as e:
        print("❌ Erreur parsing date:", e)
        date_str = None
    print("📅 Date activité:", date_str)

    # GPS
    lat, lon = None, None
    if points and "lat" in points[0] and "lng" in points[0]:
        lat, lon = points[0]["lat"], points[0]["lng"]
    elif "start_latlng" in last and last["start_latlng"]:
        lat, lon = last["start_latlng"][0], last["start_latlng"][1]

   # Température : utiliser la météo déjà stockée si disponible
    avg_temperature = last.get("avg_temperature")
    weather_code = last.get("weather_code")
    temp_debut = avg_temperature
    temp_fin = avg_temperature

    if lat is not None and lon is not None and date_str:
        # Si météo absente, on la calcule une seule fois et on la sauvegarde
        if avg_temperature is None or weather_code is None:
            start_datetime_str = last.get("date")  # Ex: "2025-07-18T19:45:57Z"
            duration_minutes = (points[-1]["time"] - points[0]["time"]) / 60 if points else 0

            avg_temperature, temp_debut, temp_fin, weather_code = get_temperature_for_run(
                lat, lon, start_datetime_str, duration_minutes
            )

            # Sauvegarde dans l'activité
            last["avg_temperature"] = avg_temperature
            last["weather_code"] = weather_code

            # Met à jour activities.json pour éviter un recalcul futur
            upload_json_content_to_drive(activities, 'activities.json')

            print(f"🌡️ Température calculée et sauvegardée : {avg_temperature}°C")
        else:
            print(f"🌡️ Température lue depuis activities.json : {avg_temperature}°C")
    else:
        print("⚠️ Impossible d’appeler météo: coordonnées ou date manquantes.")

    # Si aucun code météo n’est disponible, on force un fallback
    if weather_code is None:
        weather_code = -1


    weather_emoji = weather_code_map.get(weather_code, "❓")

    # Metrics
    total_dist = points[-1]["distance"] / 1000
    total_time = (points[-1]["time"] - points[0]["time"]) / 60
    allure_moy = total_time / total_dist if total_dist > 0 else None
    hr_vals = [p["hr"] for p in points if p.get("hr")]
    labels = [round(p["distance"] / 1000, 3) for p in points]
    if labels and labels[0] != 0:
        labels[0] = 0.0
    points_fc = [p["hr"] for p in points if p.get("hr") is not None]
    points_alt = [p["alt"] - points[0]["alt"] for p in points if p.get("alt") is not None]

    # Allure curve
    allure_curve = []
    bloc_start_idx, next_bloc_dist, last_allure = 0, 500, None
    for i, p in enumerate(points):
        if p["distance"] >= next_bloc_dist or i == len(points) - 1:
            bloc_points = points[bloc_start_idx:i + 1]
            bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
            bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
            if bloc_dist > 0:
                last_allure = (bloc_time / 60) / (bloc_dist / 1000)
            allure_curve.extend([last_allure] * len(bloc_points))
            bloc_start_idx = i + 1
            next_bloc_dist += 500
    while len(allure_curve) < len(points):
        allure_curve.append(last_allure)

    print("📊 Dashboard calculé")

    return {
        "type_sortie": last.get("type_sortie", "-"),
        "date": date_str,
        "distance_km": round(total_dist, 2),
        "duration_min": round(total_time, 1),
        "allure": f"{int(allure_moy)}:{int((allure_moy - int(allure_moy)) * 60):02d}" if allure_moy else "-",
        "fc_moy": round(sum(hr_vals) / len(hr_vals), 1) if hr_vals else "-",
        "fc_max": max(hr_vals) if hr_vals else "-",
        "k_moy": last.get("k_moy", "-"),
        "deriv_cardio": last.get("deriv_cardio", "-"),
        "gain_alt": round(points[-1]["alt"] - points[0]["alt"], 1),
        "drift_slope": last.get("drift_slope", "-"),
        "cv_allure": last.get("cv_allure", "-"),
        "cv_cardio": last.get("cv_cardio", "-"),
        "collapse_distance_km": last.get("collapse_distance_km", "-"),
        "pourcentage_zone2": last.get("pourcentage_zone2", "-"),
        "time_above_90_pct_fcmax": last.get("time_above_90_pct_fcmax", "-"),
        "ratio_fc_allure_global": last.get("ratio_fc_allure_global", "-"),
        "avg_temperature": avg_temperature,
        "temp_debut": temp_debut,
        "temp_fin": temp_fin,
        "labels": json.dumps(labels),
        "allure_curve": json.dumps(allure_curve),
        "points_fc": json.dumps(points_fc),
        "points_alt": json.dumps(points_alt),
        "history_dates": json.dumps([a["date"][:10] for a in activities if a.get("k_moy") != "-"]),
        "history_k": json.dumps([a["k_moy"] for a in activities if a.get("k_moy") != "-"]),
        "temperature": avg_temperature,
        "weather_code": weather_code,
        "weather_emoji": weather_emoji,
        "history_drift": json.dumps([a["deriv_cardio"] for a in activities if a.get("deriv_cardio") != "-"]),
    }

@app.route("/")
def index():
    start_time = time.time()
    log_step("Début index()", start_time)

    # Charger les activités
    activities = load_activities()

    # Vérifier si certaines activités sont incomplètes
    needs_weather = any(
        act.get("avg_temperature") is None or act.get("weather_code") is None
        for act in activities
    )
    needs_enrich = any(
        act.get("k_moy") in (None, "-") or act.get("deriv_cardio") in (None, "-")
        for act in activities
    )
    needs_session = any(
    act.get("type_sortie") in (None, "-", "inconnue") for act in activities
    )


    modified = False
    if needs_weather:
        print("🌤️ Météo manquante → calcul météo")
        activities = ensure_weather_data(activities)
        modified = True
    
    if needs_session:
        print("🏷️ Session type manquant → tagging par règles")
        activities, changed = tag_session_types(activities)
        modified = modified or changed
        
        # Auto-réentrainement XGB si nouvelle activité + labels suffisants
    if AUTO_RETRAIN_XGB and _should_retrain_xgb(activities):
        _retrain_fractionne_model_and_reload()

        
    print("🤖 Marquage fractionné (is_fractionne / fractionne_prob)")
    activities, changed = apply_fractionne_flags(activities)
    modified = modified or changed

    if needs_enrich:
        print("📈 Enrichissement manquant → enrichissement")
        activities = enrich_activities(activities)
        modified = True

    if modified:
        upload_json_content_to_drive(activities, 'activities.json')
        print("💾 activities.json mis à jour après complétion")

    log_step("Activities chargées et complétées", start_time)
    print(f"📂 {len(activities)} activités prêtes")

    # Calcul du dashboard
    dashboard = compute_dashboard_data(activities)
    log_step("Dashboard calculé", start_time)

    activities_for_carousel = []

    # Construction du carrousel (inchangé)
    for act in reversed(activities[-10:]):  # 10 dernières activités
        log_step(f"Début carrousel activité {act.get('date')}", start_time)
        points = act.get("points", [])
        if not points:
            continue

        labels = [round(p["distance"] / 1000, 3) for p in points]
        points_fc = [p.get("hr", 0) for p in points]
        points_alt = [p.get("alt", 0) - points[0].get("alt", 0) for p in points]

        # Calcul allure_curve tous les 500m
        allure_curve = []
        bloc_start_idx, next_bloc_dist, last_allure = 0, 500, None
        for i, p in enumerate(points):
            if p["distance"] >= next_bloc_dist or i == len(points) - 1:
                bloc_points = points[bloc_start_idx:i + 1]
                bloc_dist = bloc_points[-1]["distance"] - bloc_points[0]["distance"]
                bloc_time = bloc_points[-1]["time"] - bloc_points[0]["time"]
                if bloc_dist > 0:
                    last_allure = (bloc_time / 60) / (bloc_dist / 1000)
                allure_curve.extend([last_allure] * len(bloc_points))
                bloc_start_idx = i + 1
                next_bloc_dist += 500
        while len(allure_curve) < len(points):
            allure_curve.append(last_allure)

        # Statistiques globales
        total_dist_km = points[-1]["distance"] / 1000
        total_time_min = (points[-1]["time"] - points[0]["time"]) / 60
        allure_moy = total_time_min / total_dist_km if total_dist_km > 0 else None
        fc_max = max(points_fc) if points_fc else None
        gain_alt = round(points[-1]["alt"] - points[0]["alt"], 1)

        # 🌡️ Météo
        avg_temperature = act.get("avg_temperature")
        weather_code = act.get("weather_code")
        weather_code_map = {
            0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
            45: "🌫️", 48: "🌫️", 51: "🌦️", 53: "🌧️",
            55: "🌧️", 61: "🌧️", 63: "🌧️", 65: "🌧️",
            71: "❄️", 73: "❄️", 75: "❄️", 80: "🌧️",
            81: "🌧️", 82: "🌧️", 95: "⛈️", 96: "⛈️", 99: "⛈️"
        }
        weather_emoji = weather_code_map.get(weather_code, "❓")

        # Date formatée
        try:
            date_formatted = datetime.strptime(
                act.get("date"), "%Y-%m-%dT%H:%M:%S%z"
            ).strftime("%Y-%m-%d")
        except:
            date_formatted = "-"

        activities_for_carousel.append({
            "date": date_formatted,
            "type_sortie": act.get("type_sortie", "-"),
            "is_fractionne": act.get("is_fractionne", False),
            "fractionne_prob": act.get("fractionne_prob", 0.0), 
            "distance_km": round(total_dist_km, 2),
            "duration_min": round(total_time_min, 1),
            "fc_moy": round(np.mean(points_fc), 1) if points_fc else "-",
            "fc_max": fc_max,
            "allure": f"{int(allure_moy)}:{int((allure_moy - int(allure_moy)) * 60):02d}" if allure_moy else "-",
            "gain_alt": gain_alt,
            "k_moy": act.get("k_moy", "-"),
            "deriv_cardio": act.get("deriv_cardio", "-"),
            "temperature": avg_temperature,
            "weather_emoji": weather_emoji,
            "labels": json.dumps(labels),
            "points_fc": json.dumps(points_fc),
            "points_alt": json.dumps(points_alt),
            "allure_curve": json.dumps(allure_curve),
        })

    # Retourne la page
    return render_template(
        "index.html",
        dashboard=dashboard,
        objectives=load_objectives(),
        short_term=load_short_term_objectives(),
        activities_for_carousel=activities_for_carousel
    )

    
@app.route("/refresh")
def refresh():
    """Recalcule et met à jour activities.json sur Drive"""
    print("♻️ Recalcul des activités...")
    activities = load_activities()
    activities = enrich_activities(activities)
    upload_json_content_to_drive(activities, 'activities.json')
    print("✅ activities.json mis à jour sur Drive")
    return "✅ Données mises à jour"
      
@app.route('/profile', methods=['GET','POST'])
def profile():
    profile = load_profile()
    if request.method == 'POST':
        profile['birth_date'] = request.form.get('birth_date', '')
        weight = request.form.get('weight', '')
        profile['weight'] = float(weight) if weight else 0.0
        profile['global_objective'] = request.form.get('global_objective', '')
        profile['particular_objective'] = request.form.get('particular_objective', '')

        # Récupérer listes des événements : dates et noms
        event_dates = request.form.getlist('event_date')
        event_names = request.form.getlist('event_name')

        events = []
        for d, n in zip(event_dates, event_names):
            d = d.strip()
            n = n.strip()
            if d and n:
                events.append({'date': d, 'name': n})
        profile['events'] = events

        upload_json_content_to_drive(profile, 'profile.json')
        return redirect('/profile')

    return render_template('profile.html', profile=profile)
    
@app.route('/generate_short_term_plan')
def generate_short_term_plan():
    profile = load_profile()
    activities = load_activities()
    prompt_template = load_short_term_prompt_from_drive()

    # Exemple simple pour construire prompt
    prompt = prompt_template
    prompt += f"\nProfil: {profile}"
    prompt += f"\nActivités récentes: {len(activities)}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        result_str = response.choices[0].message.content.strip()

        # Essayer de parser la réponse JSON (précise à OpenAI de répondre en JSON strict)
        short_term_objectives = json.loads(result_str)

        # Ajouter conversion allures au format décimal si besoin
        short_term_objectives = convert_short_term_allures(short_term_objectives)

        save_short_term_objectives(short_term_objectives)

        print("✅ Coaching court terme généré et sauvegardé.")

        return redirect('/')  # ou retourner un message / JSON si API

    except Exception as e:
        print("❌ Erreur génération coaching court terme:", e)
        return f"Erreur génération coaching: {e}", 500
        
@app.route("/recompute_session_types")
def recompute_session_types():
    """Recalcule le type_sortie de toutes les activités avec la règle par distance."""
    activities = load_activities()
    print(f"♻️ Recalcul session_type pour {len(activities)} activités")

    for act in activities:
        # Toujours recalculer, même si déjà défini
        act["type_sortie"] = detect_session_type(act)

    upload_json_content_to_drive(activities, "activities.json")
    print("✅ activities.json mis à jour avec nouveaux session_type")
    return f"✅ Recalculé pour {len(activities)} activités"
    
@app.route("/recompute_fractionne_flags")
def recompute_fractionne_flags():
    activities = load_activities()
    activities, changed = apply_fractionne_flags(activities)
    if changed:
        upload_json_content_to_drive(activities, "activities.json")
        msg = f"✅ Flags fractionné mis à jour ({len(activities)} activités)"
    else:
        msg = "ℹ️ Aucun changement sur les flags fractionné"
    print(msg)
    return msg
    
@app.route("/export_fractionne_excel")
def export_fractionne_excel():
    activities = load_activities()
    rows = []
    for a in activities:
        aid = a.get("activity_id")
        label = a.get("is_fractionne_label", "")
        rows.append({"activity_id": aid, "fractionne_label": label})
    df = pd.DataFrame(rows)
    excel_path = "fractionne_labels.xlsx"
    df.to_excel(excel_path, index=False)
    return f"✅ Fichier exporté : {excel_path}"

@app.route("/import_fractionne_excel")
def import_fractionne_excel():
    import pandas as pd
    excel_path = "fractionne_labels.xlsx"
    df = pd.read_excel(excel_path)
    label_map = dict(zip(df["activity_id"], df["fractionne_label"]))

    activities = load_activities()
    changed = 0
    for a in activities:
        aid = a.get("activity_id")
        if aid in label_map and pd.notna(label_map[aid]):
            new_val = bool(label_map[aid])
            if a.get("is_fractionne_label") != new_val:
                a["is_fractionne_label"] = new_val
                changed += 1

    if changed > 0:
        upload_json_content_to_drive(activities, "activities.json")

    return f"✅ {changed} activités mises à jour depuis Excel"
    
@app.route("/debug_autotrain_status")
def debug_autotrain_status():
    activities = load_activities()

    # mêmes calculs que les helpers
    meta = _load_last_train_meta()
    last_cnt = meta.get("last_trained_count", 0)
    cur_cnt = len(activities)

    pos = neg = 0
    for a in activities:
        if "is_fractionne_label" in a:
            if bool(a["is_fractionne_label"]): pos += 1
            else: neg += 1

    eligible = (cur_cnt > last_cnt) and (pos >= 8 and neg >= 8)

    return (
        f"AUTO_RETRAIN_XGB={AUTO_RETRAIN_XGB}<br>"
        f"last_trained_count={last_cnt}<br>"
        f"current_activities={cur_cnt}<br>"
        f"new_activities={cur_cnt - last_cnt}<br>"
        f"labels_pos={pos}, labels_neg={neg} (min 8/8)<br>"
        f"eligible={eligible}"
    )
    
@app.route("/force_autotrain_xgb")
def force_autotrain_xgb():
    ok = _retrain_fractionne_model_and_reload()
    return "✅ Auto-train OK" if ok else "❌ Auto-train échoué", (200 if ok else 500)




if __name__ == "__main__":
    app.run(debug=True)
