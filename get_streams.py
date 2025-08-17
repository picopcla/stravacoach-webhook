# get_streams.py — ingestion uniquement (cadence brute)
import os, sys, io, json, time
from bisect import bisect_left

import requests
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


# ----------------------------
# ENV bootstrap (comme app.py)
# ----------------------------
try:
    sys.path.insert(0, r"C:\StravaSecurity")
    try:
        import loadkeys
        print("✅ loadkeys.py importé")
    except Exception as e:
        print("ℹ️ loadkeys non importé:", e)
    ok = load_dotenv(r"C:\StravaSecurity\main.env", override=True)
    print("✅ main.env chargé:", ok)
except Exception as e:
    print("ℹ️ main.env non chargé:", e)

# Aliases compatibles
alias_map = {
    "GOOGLE_SERVICE_ACCOUNT_FILE": "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CREDENTIALS_FILE": "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_JSON_CREDENTIALS": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "DRIVE_FOLDER_ID": "FOLDER_ID",
}
for src, dst in alias_map.items():
    if not os.getenv(dst) and os.getenv(src):
        os.environ[dst] = os.getenv(src)

# Fallback fichier creds local
if (not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        and os.path.exists(r"C:\StravaSecurity\services.json")):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\StravaSecurity\services.json"

print("ENV FOLDER_ID          =", os.getenv("FOLDER_ID"))
print("ENV CREDS(json|path)   =", bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("ENV CREDS path exists  =", os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))


# ----------------------------
# Google Drive helpers
# ----------------------------
def build_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]
    sa_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if sa_json:
        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        print("🔐 Creds: JSON inline")
    elif sa_path and os.path.exists(sa_path):
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)
        print(f"🔐 Creds: fichier {sa_path}")
    else:
        raise RuntimeError("Aucun identifiant Google : définis GOOGLE_APPLICATION_CREDENTIALS "
                           "ou GOOGLE_APPLICATION_CREDENTIALS_JSON.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


drive_service = build_drive_service()
FOLDER_ID = os.getenv("FOLDER_ID") or "1OvCqOHHiOZoCOQtPaSwGoioR92S8-U7t"  # fallback local ok


def drive_find_file_id(name: str):
    q = "name = '{}' and '{}' in parents and trashed = false".format(
        name.replace("'", "\\'"), FOLDER_ID
    )
    res = drive_service.files().list(
        q=q, fields="files(id,name)", pageSize=5,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    # fallback insensible à la casse : lister le dossier
    res = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed = false",
        fields="files(id,name)", pageSize=1000,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    for f in res.get("files", []):
        if f["name"].lower() == name.lower():
            return f["id"]
    return None


def drive_download_json(file_id: str):
    req = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buf.seek(0)
    return json.loads(buf.read().decode("utf-8", "replace"))


def drive_upload_json(name: str, data, file_id: str | None):
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype='application/json', resumable=False)

    if file_id:
        drive_service.files().update(
            fileId=file_id, media_body=media, supportsAllDrives=True
        ).execute()
    else:
        meta = {"name": name, "parents": [FOLDER_ID], "mimeType": "application/json"}
        drive_service.files().create(
            body=meta, media_body=media, fields="id", supportsAllDrives=True
        ).execute()


# ----------------------------
# Strava token (refresh si besoin)
# ----------------------------
with open("strava_tokens.json") as f:
    tokens = json.load(f)

access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]
expires_at = tokens["expires_at"]

time_remaining = expires_at - int(time.time())
if time_remaining < 300:
    print(f"🔄 Token expirant dans {time_remaining}s, on le renouvelle...")
    resp = requests.post(
        "https://www.strava.com/api/v3/oauth/token",
        data={
            "client_id": os.getenv("STRAVA_CLIENT_ID", "162245"),
            "client_secret": os.getenv("STRAVA_CLIENT_SECRET", "0552c0e87d83493d7f6667d0570de1e8ac9e9a68"),
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        },
        timeout=30
    )
    resp.raise_for_status()
    new_tokens = resp.json()
    tokens["access_token"] = new_tokens["access_token"]
    tokens["refresh_token"] = new_tokens["refresh_token"]
    tokens["expires_at"] = new_tokens["expires_at"]
    with open("strava_tokens.json", "w") as f:
        json.dump(tokens, f, indent=2)
    access_token = tokens["access_token"]
    print("✅ Token Strava rafraîchi.")
else:
    print(f"✅ Token encore valide pour {time_remaining}s.")

headers = {"Authorization": f"Bearer {access_token}"}


# ----------------------------
# Helpers: mapping série -> points existants
# ----------------------------
def _map_series_to_points_by_time(points, time_stream, series, field_name: str, tol_sec=5):
    """
    Remplit points[i][field_name] avec la valeur la plus proche temporellement (± tol_sec).
    N'écrase PAS une valeur déjà présente.
    """
    times = time_stream or []
    vals = series or []
    if not times or not vals or not points:
        return 0

    filled = 0
    for p in points:
        if p.get(field_name) is not None:
            continue
        t = p.get("time")
        if t is None:
            continue
        idx = bisect_left(times, t)
        cand = [j for j in (idx-1, idx, idx+1) if 0 <= j < len(times)]
        best = None; best_dt = None
        for j in cand:
            v = vals[j]
            if not isinstance(v, (int, float)):
                continue
            dt = abs(times[j] - t)
            if best is None or dt < best_dt:
                best, best_dt = v, dt
        if best is not None and (best_dt is None or best_dt <= tol_sec):
            p[field_name] = best
            filled += 1
    return filled


# ----------------------------
# Charger activities.json depuis Drive
# ----------------------------
activities = []
try:
    file_id = drive_find_file_id("activities.json")
    if file_id:
        data = drive_download_json(file_id)
        if isinstance(data, list):
            activities = data
            print(f"✅ activities.json chargé ({len(activities)} activités).")
        else:
            print("⚠️ activities.json n'est pas une liste, on repart de zéro.")
    else:
        print("ℹ️ Pas d'activities.json sur Drive : on créera le fichier.")
except Exception as e:
    print("⚠️ Lecture Drive échouée, on repart de zéro:", e)


# ----------------------------
# Récupérer/mettre à jour une activité (cadence BRUTE)
# ----------------------------
def process_activity(activity_id: int):
    url_activity = f"https://www.strava.com/api/v3/activities/{activity_id}"
    ra = requests.get(url_activity, headers=headers, timeout=30)
    if ra.status_code != 200:
        print(f"❌ Erreur {ra.status_code} sur l'activité {activity_id}")
        return
    activity_data = ra.json()
    start_date = activity_data.get("start_date_local")

    # Streams
    url_streams = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {
        "keys": "time,distance,heartrate,cadence,velocity_smooth,altitude,temperature,moving,latlng",
        "key_by_type": "true"
    }
    rs = requests.get(url_streams, params=params, headers=headers, timeout=60)
    if rs.status_code != 200:
        print(f"❌ Erreur HTTP {rs.status_code} pour streams {activity_id}")
        return
    streams = rs.json()

    time_data   = (streams.get("time") or {}).get("data", []) or []
    distance    = (streams.get("distance") or {}).get("data", []) or []
    heartrate   = (streams.get("heartrate") or {}).get("data", []) or []
    velocity    = (streams.get("velocity_smooth") or {}).get("data", []) or []
    altitude    = (streams.get("altitude") or {}).get("data", []) or []
    latlng      = (streams.get("latlng") or {}).get("data", []) or []
    cadence_raw = (streams.get("cadence") or {}).get("data", []) or []

    if not time_data or not distance:
        print(f"⚠️ Pas de données time/distance pour {activity_id}, on ignore.")
        return

    # Si déjà présente: compléter uniquement 'cad_raw'
    act = next((a for a in activities if a.get("activity_id") == activity_id), None)
    if act is not None:
        print(f"👣 Activité {activity_id} déjà présente → MAJ cad_raw uniquement")
        pts = act.get("points") or []
        filled = _map_series_to_points_by_time(pts, time_data, cadence_raw, "cad_raw", tol_sec=5)
        print(f"   → cad_raw remplie sur {filled} points")
        act["points"] = pts
        return

    # Nouvelle activité → créer des points “fenêtres 10 échantillons”
    points = []
    window = 10
    n = len(time_data)
    for i in range(0, n, window):
        slice_range = range(i, min(i + window, n))
        point_time = time_data[slice_range[-1]]
        last_dist = distance[slice_range[-1]] if slice_range[-1] < len(distance) else None

        def _avg(series):
            vals = [series[j] for j in slice_range if j < len(series) and isinstance(series[j], (int, float))]
            return (sum(vals) / len(vals)) if vals else None

        avg_hr  = _avg(heartrate)
        avg_vel = _avg(velocity)
        avg_alt = _avg(altitude)

        lat_vals = [latlng[j][0] for j in slice_range if j < len(latlng) and latlng[j]]
        lng_vals = [latlng[j][1] for j in slice_range if j < len(latlng) and latlng[j]]
        avg_lat = (sum(lat_vals) / len(lat_vals)) if lat_vals else None
        avg_lng = (sum(lng_vals) / len(lng_vals)) if lng_vals else None

        cad_vals = [cadence_raw[j] for j in slice_range if j < len(cadence_raw) and isinstance(cadence_raw[j], (int, float))]
        cad_mean_raw = (sum(cad_vals) / len(cad_vals)) if cad_vals else None

        points.append({
            "time": point_time,
            "distance": last_dist,
            "hr": avg_hr,
            "vel": avg_vel,
            "alt": avg_alt,
            "lat": avg_lat,
            "lng": avg_lng,
            "cad_raw": cad_mean_raw,   # <-- BRUT uniquement, normalisé plus tard dans app.py
        })

    activities.append({
        "activity_id": activity_id,
        "date": start_date,
        "points": points
    })
    print(f"🚀 Activité {activity_id} ajoutée avec {len(points)} points.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_streams.py <ACTIVITY_ID>")
        sys.exit(1)

    activity_id_arg = int(sys.argv[1])

    # 1) Traiter l'ID demandé
    process_activity(activity_id_arg)

    # 2) Optionnel: rafraîchir les dernières activités (sans supprimer l'ancien)
    try:
        url = "https://www.strava.com/api/v3/athlete/activities"
        params = {"per_page": 30, "page": 1}
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        latest_activities = resp.json()
        if isinstance(latest_activities, list):
            for act in latest_activities:
                process_activity(int(act["id"]))
        else:
            print("⚠️ Réponse inattendue pour athlete/activities:", latest_activities)
    except Exception as e:
        print("ℹ️ Impossible de parcourir les dernières activités:", e)

    # 3) Sauvegarder local & Drive
    try:
        with open("activities.json", "w", encoding="utf-8") as f:
            json.dump(activities, f, ensure_ascii=False, indent=2)
        file_id = drive_find_file_id("activities.json")
        drive_upload_json("activities.json", activities, file_id)
        print("✅ activities.json écrit sur Drive.")
    except Exception as e:
        print("❌ Écriture Drive échouée:", e)
