import os
from dotenv import load_dotenv, dotenv_values

dotenv_path = r"C:\StravaSecurity\main.env"

print("Fichier existe ?", os.path.exists(dotenv_path))

config = dotenv_values(dotenv_path)
print("Contenu dotenv_values :", config)

loaded = load_dotenv(dotenv_path)
print("dotenv chargé :", loaded)

print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("GOOGLE_SERVICE_ACCOUNT_FILE =", os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"))
