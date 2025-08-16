# helpers/data_access.py
from __future__ import annotations
import io
import json
import os
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account

# ---------- Config ----------
DRIVE_FOLDER_ID = os.getenv("FOLDER_ID") or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
# Noms standards des fichiers sur Drive
F_ACTIVITIES = os.getenv("ACTIVITIES_FILENAME", "activities.json")
F_PROFILE    = os.getenv("PROFILE_FILENAME",    "profile.json")

# Fichiers de sortie générés par les modules ML/IA
F_ANALYSIS     = os.getenv("ANALYSIS_FILENAME",     "analysis.json")
F_PREDICTIONS  = os.getenv("PREDICTIONS_FILENAME",  "predictions.json")
F_WEEKLY_PLAN  = os.getenv("WEEKLY_PLAN_FILENAME",  "weekly_plan.json")
F_BENCHMARK    = os.getenv("BENCHMARK_FILENAME",    "benchmark.json")

# Debug
DEBUG = os.getenv("SC_DEBUG") == "1"
def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DA] {msg}")

class DriveUnavailableError(RuntimeError):
    pass


def _get_credentials():
    """
    Stratégies d’auth (service account) :
    1) GOOGLE_APPLICATION_CREDENTIALS_JSON (contenu JSON) dans l'env
    2) GOOGLE_APPLICATION_CREDENTIALS (path) si présent
    """
    sa_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if sa_json:
        try:
            info = json.loads(sa_json)
            scopes = ["https://www.googleapis.com/auth/drive"]
            _dbg("creds: from GOOGLE_APPLICATION_CREDENTIALS_JSON")
            return service_account.Credentials.from_service_account_info(info, scopes=scopes)
        except Exception as e:
            raise DriveUnavailableError(f"Creds JSON invalides: {e}") from e

    if sa_path and os.path.exists(sa_path):
        scopes = ["https://www.googleapis.com/auth/drive"]
        _dbg(f"creds: from file {sa_path}")
        return service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)

    raise DriveUnavailableError("Identifiants Google Drive introuvables (service account).")


def _get_drive_service():
    creds = _get_credentials()
    _dbg("building drive service v3")
    try:
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        raise DriveUnavailableError(f"Erreur build Drive API: {e}") from e


def _resolve_folder_id(service) -> str:
    """
    Si FOLDER_ID est absent, tente de le déduire en cherchant F_ACTIVITIES / F_PROFILE
    sur tout le Drive et en prenant le parent commun le plus probable.
    Met à jour os.environ['FOLDER_ID'] et la variable globale.
    """
    global DRIVE_FOLDER_ID
    if DRIVE_FOLDER_ID:
        return DRIVE_FOLDER_ID

    _dbg("FOLDER_ID absent -> découverte automatique via recherche globale…")

    def _search_file_parents(fname: str):
        q = "name = '{}' and trashed = false".format(fname.replace("'", "\\'"))
        res = service.files().list(
            q=q,
            fields="files(id,name,parents)",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        parents = []
        for f in res.get("files", []):
            parents.extend(f.get("parents", []))
        _dbg(f"parents trouvés pour {fname}: {parents}")
        return parents

    candidates = []
    for fname in (F_ACTIVITIES, F_PROFILE):
        try:
            candidates.extend(_search_file_parents(fname))
        except Exception as e:
            _dbg(f"recherche parents échouée pour {fname}: {e}")

    if not candidates:
        raise DriveUnavailableError(
            "Impossible de déduire FOLDER_ID (aucun parent trouvé pour "
            f"{F_ACTIVITIES} ou {F_PROFILE})."
        )

    # Parent le plus fréquent
    from collections import Counter
    folder_id, _ = Counter(candidates).most_common(1)[0]
    os.environ["FOLDER_ID"] = folder_id
    DRIVE_FOLDER_ID = folder_id
    _dbg(f"FOLDER_ID déduit: {folder_id}")
    return folder_id

def _find_file_id(service, name: str) -> Optional[str]:
    """
    Cherche un fichier par nom dans le dossier FOLDER_ID (auto-détecté si absent).
    1) match exact
    2) fallback insensible à la casse
    """
    folder_id = _resolve_folder_id(service)

    _dbg(f"lookup exact name='{name}' in folder={folder_id}")
    q = "name = '{}' and '{}' in parents and trashed = false".format(
        name.replace("'", "\\'"), folder_id
    )
    res = service.files().list(
        q=q,
        fields="files(id, name)",
        pageSize=5,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    if files:
        _dbg(f"found exact: {files[0]['name']} ({files[0]['id']})")
        return files[0]["id"]

    _dbg("exact not found; listing folder for case-insensitive match…")
    res = service.files().list(
        q="'{}' in parents and trashed = false".format(folder_id),
        fields="files(id, name)",
        pageSize=1000,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    for f in res.get("files", []):
        if f["name"].lower() == name.lower():
            _dbg(f"matched lower(): {f['name']} ({f['id']})")
            return f["id"]

    _dbg("not found in folder; nothing matched")
    return None


def _download_bytes(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if DEBUG and status:
            _dbg(f"download {int(status.progress() * 100)}%")
    buf.seek(0)
    return buf.read()


def _download_json(service, file_id: str) -> Any:
    data = _download_bytes(service, file_id)
    try:
        return json.loads(data.decode("utf-8", "replace"))
    except Exception as e:
        raise DriveUnavailableError(f"JSON invalide (file_id={file_id}): {e}") from e


def _upload_json(service, name: str, data: Any, file_id: Optional[str]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/json", resumable=False)

    if file_id:
        _dbg(f"update file {name} ({file_id})")
        service.files().update(
            fileId=file_id, media_body=media,
            supportsAllDrives=True
        ).execute()
        return file_id
    else:
        _dbg(f"create file {name} in folder {DRIVE_FOLDER_ID}")
        file_meta = {"name": name, "parents": [DRIVE_FOLDER_ID], "mimeType": "application/json"}
        created = service.files().create(
            body=file_meta, media_body=media, fields="id",
            supportsAllDrives=True
        ).execute()
        return created["id"]


# --------- API publique (StravaCoach) ---------

def load_activities_from_drive() -> List[Dict[str, Any]]:
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, F_ACTIVITIES)
        if not file_id:
            raise DriveUnavailableError(f"Fichier {F_ACTIVITIES} introuvable sur Drive.")
        data = _download_json(service, file_id)
        if not isinstance(data, list):
            raise DriveUnavailableError(f"{F_ACTIVITIES} n’est pas une liste JSON.")
        _dbg(f"activities loaded: {len(data)}")
        return data
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur lecture {F_ACTIVITIES} : {e}") from e


def save_activities_to_drive(activities: List[Dict[str, Any]]) -> None:
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, F_ACTIVITIES)
        _upload_json(service, F_ACTIVITIES, activities, file_id)
        _dbg(f"activities saved: {len(activities)}")
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur écriture {F_ACTIVITIES} : {e}") from e


def load_profile_from_drive() -> Dict[str, Any]:
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, F_PROFILE)
        if not file_id:
            raise DriveUnavailableError(f"Fichier {F_PROFILE} introuvable sur Drive.")
        data = _download_json(service, file_id)
        if not isinstance(data, dict):
            raise DriveUnavailableError(f"{F_PROFILE} n’est pas un objet JSON.")
        _dbg(f"profile loaded: keys={list(data.keys())}")
        return data
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur lecture {F_PROFILE} : {e}") from e


# ------- Fichiers texte optionnels (ex: prompt .txt) -------

def read_text_file(filename: str) -> Optional[str]:
    """Lit un fichier texte (UTF-8) depuis le Drive. Retourne None si absent."""
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, filename)
        if not file_id:
            return None
        data = _download_bytes(service, file_id)
        txt = data.decode("utf-8", "replace")
        _dbg(f"text read: {filename} ({len(txt)} chars)")
        return txt
    except DriveUnavailableError:
        raise
    except Exception:
        return None


# ------- Sorties ML/IA génériques -------

def read_output_json(filename: str) -> Optional[Any]:
    """
    Lit un JSON de sortie (analysis/predictions/weekly_plan/benchmark).
    Retourne None s’il n’existe pas encore.
    """
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, filename)
        if not file_id:
            _dbg(f"{filename} not found (None)")
            return None
        out = _download_json(service, file_id)
        _dbg(f"{filename} read ok")
        return out
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur lecture {filename} : {e}") from e


def write_output_json(filename: str, data: Any) -> None:
    """Écrit/Crée un JSON de sortie sur Drive."""
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, filename)
        _upload_json(service, filename, data, file_id)
        _dbg(f"{filename} write ok")
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur écriture {filename} : {e}") from e


# Helpers dédiés (facultatif, pratique)
def read_analysis() -> Optional[Dict[str, Any]]:
    return read_output_json(F_ANALYSIS)

def read_predictions() -> Optional[Dict[str, Any]]:
    return read_output_json(F_PREDICTIONS)

def read_weekly_plan() -> Optional[Dict[str, Any]]:
    return read_output_json(F_WEEKLY_PLAN)

def read_benchmark() -> Optional[Dict[str, Any]]:
    return read_output_json(F_BENCHMARK)

def write_analysis(data: Dict[str, Any]) -> None:
    write_output_json(F_ANALYSIS, data)

def write_predictions(data: Dict[str, Any]) -> None:
    write_output_json(F_PREDICTIONS, data)

def write_weekly_plan(data: Dict[str, Any]) -> None:
    write_output_json(F_WEEKLY_PLAN, data)

def write_benchmark(data: Dict[str, Any]) -> None:
    write_output_json(F_BENCHMARK, data)
