import os
from dotenv import load_dotenv

def main():
    # Chemin absolu vers ton fichier main.env
    dotenv_path = r"C:\StravaSecurity\main.env"
    loaded = load_dotenv(dotenv_path)
    print("dotenv chargé :", loaded)

    # Récupération des variables d'environnement
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_credentials_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

    print("OpenAI API Key:", openai_api_key)
    print("Google Credentials file:", google_credentials_file)

    # Vérifications basiques
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY introuvable dans main.env")
    if not google_credentials_file or not os.path.isfile(google_credentials_file):
        raise FileNotFoundError(f"Fichier Google Service Account non trouvé : {google_credentials_file}")

    # Initialisation OpenAI
    import openai
    openai.api_key = openai_api_key

    # Initialisation Google Drive
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        google_credentials_file,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=credentials)

    # Test : lister 5 fichiers Drive
    results = drive_service.files().list(
        pageSize=5,
        fields="files(id, name)"
    ).execute()
    fichiers = results.get("files", [])
    print("\nListe des fichiers Drive :")
    for f in fichiers:
        print(f"- {f['name']} ({f['id']})")

if __name__ == "__main__":
    main()
