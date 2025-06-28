from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        challenge = request.args.get("hub.challenge")
        print(f"✅ Validation webhook : challenge = {challenge}")
        return challenge, 200

    if request.method == "POST":
        data = request.json
        print("📩 Notification Strava reçue :", data)

        if data.get("object_type") == "activity" and data.get("aspect_type") == "create":
            activity_id = data.get("object_id")
            print(f"🎯 Nouvelle activité détectée : {activity_id}")

            # Lance ton script get_streams.py avec l'ID
            subprocess.Popen(["python", "get_streams.py", str(activity_id)])
            print("🚀 Script get_streams.py lancé en tâche de fond.")

        return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
