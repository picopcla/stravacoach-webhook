from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        # Pour la validation du webhook par Strava
        challenge = request.args.get("hub.challenge")
        verify_token = request.args.get("hub.verify_token")
        
        # Optionnel : vérifie le token (pour plus de sécurité)
        if verify_token != "STRAVA":
            return "Invalid verify_token", 403
        
        print(f"✅ Validation webhook : challenge = {challenge}")
        return jsonify({"hub.challenge": challenge}), 200

    if request.method == "POST":
        # Pour recevoir les notifications Strava
        data = request.json
        print("📩 Notification Strava reçue :", data)

        # Si c'est une nouvelle activité créée
        if data.get("object_type") == "activity" and data.get("aspect_type") == "create":
            activity_id = data.get("object_id")
            print(f"🎯 Nouvelle activité détectée : {activity_id}")

            # Lance ton script get_streams.py en arrière-plan
            subprocess.Popen(["python", "get_streams.py", str(activity_id)])
            print("🚀 Script get_streams.py lancé en tâche de fond.")

        return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    # Expose le serveur sur le port 5000
    app.run(host="0.0.0.0", port=5000)
