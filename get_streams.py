import sys
import requests
import json
import time

# -------------------
# Config : tokens Strava
TOKEN_FILE = "strava_tokens.json"
CLIENT_ID = "162245"
CLIENT_SECRET = "0552c0e87d83493d7f6667d0570de1e8ac9e9a68"

# -------------------
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

# -------------------
def get_streams(activity_id, token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": "time,distance,heartrate", "key_by_type": "true"}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()

# -------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Donne un activity_id en argument.")
        sys.exit(1)

    activity_id = sys.argv[1]
    token = get_access_token()
    streams = get_streams(activity_id, token)

    # Petit résumé
    print(f"\n✅ Streams récupérés pour activité {activity_id}")
    for key, values in streams.items():
        print(f"- {key}: {len(values)} points")
