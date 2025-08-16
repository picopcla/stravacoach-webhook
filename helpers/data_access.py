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


class DriveUnavailableError(RuntimeError):
    pass


def _get_credentials():
    """
    Stratégies d’auth (service account) :
    1) GOOGLE_APPLICATION_CREDENTIALS_JSON (contenu JSON) dans l'env
    2) GOOGLE_APPLICATION_CREDENTIALS (path) si présent (optionnel)
    """
    sa_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        info = json.loads(sa_json)
        scopes = ["https://www.googleapis.com/auth/drive"]
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and os.path.exists(sa_path):
        scopes = ["https://www.googleapis.com/auth/drive"]
        return service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)

    raise DriveUnavailableError("Identifiants Google Drive introuvables (service account).")


def _get_drive_service():
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _ensure_folder():
    if not DRIVE_FOLDER_ID:
        raise DriveUnavailableError("FOLDER_ID manquant. Définis FOLDER_ID dans les variables d’environnement.")


def _find_file_id(service, name: str) -> Optional[str]:
    """
    Cherche un fichier par nom exact dans le dossier FOLDER_ID.
    """
    _ensure_folder()
    q = "name = '{}' and '{}' in parents and trashed = false".format(name.replace("'", "\\'"), DRIVE_FOLDER_ID)
    res = service.files().list(q=q, fields="files(id, name)", pageSize=5).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _download_json(service, file_id: str) -> Any:
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buf.seek(0)
    return json.load(buf)


def _upload_json(service, name: str, data: Any, file_id: Optional[str]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/json", resumable=False)

    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
        return file_id
    else:
        _ensure_folder()
        file_meta = {"name": name, "parents": [DRIVE_FOLDER_ID], "mimeType": "application/json"}
        created = service.files().create(body=file_meta, media_body=media, fields="id").execute()
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
        return data
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur lecture {F_PROFILE} : {e}") from e


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
            return None
        return _download_json(service, file_id)
    except DriveUnavailableError:
        raise
    except Exception as e:
        raise DriveUnavailableError(f"Erreur lecture {filename} : {e}") from e


def write_output_json(filename: str, data: Any) -> None:
    """
    Écrit/Crée un JSON de sortie sur Drive.
    """
    try:
        service = _get_drive_service()
        file_id = _find_file_id(service, filename)
        _upload_json(service, filename, data, file_id)
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
