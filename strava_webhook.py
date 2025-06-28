from flask import Flask, request, jsonify
import requests
import json
import time

app = Flask(__name__)

# Ton config Strava
CLIENT_ID = "162245"
CLIENT_SECRET = "0552c0e87d83493d7f6667d0570de1e8ac9e9a68"
TOKEN_FILE = "strava_tokens.json"

# ---------------------
def get_access_token():
    with open(TOKEN_FILE, "r") as f:
        tokens = json.load(f)
    if time.time() > tokens["expires_at"]:
        return refresh_access_token(tokens["refresh_token"])
    return tokens["access_token"]

def refresh_access_token(refresh_token):
    resp = requests.post("https://www.strava.com/api/v3/oauth/token", data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    })
    resp.raise_for_status()
    new_tokens = resp.json()
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token": new_tokens["access_token"],
            "refresh_token": new_tokens["refresh_token"],
            "expires_at": new_tokens["expires_at"]
        }, f)
    return new_tokens["access_token"]

def get_streams(activity_id, token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": "time,distance,heartrate", "key_by_type": "true"}
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
    resp.raise_for_status()
    return resp.json()

# ---------------------
@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        # Vérification pour Strava
        return request.args.get("hub.challenge")
    elif request.method == "POST":
        data = request.get_json()
        print("🔔 Nouvelle activité:", data)
        if data["object_type"] == "activity" and data["aspect_type"] == "create":
            activity_id = data["object_id"]
            handle_activity(activity_id)
        return jsonify({"status": "received"}), 200

def handle_activity(activity_id):
    print(f"🚀 Récupération et analyse de l'activité {activity_id}")
    token = get_access_token()
    streams = get_streams(activity_id, token)
    # Ici tu peux mettre tes calculs de laps et logs
    print(f"✅ Streams récupérés pour activité {activity_id}, nb points: {len(streams['time']['data'])}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
