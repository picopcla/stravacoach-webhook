
# 🏃 Mon Coach Running - Strava IA Dashboard PWA

Ce projet est un **coach running IA personnalisé** qui analyse tes activités Strava, les stocke sur Google Drive, puis les affiche dans un **dashboard PWA installable sur ton téléphone**.

---

## 🚀 Fonctionnalités
✅ Récupère automatiquement tes activités Strava via webhook  
✅ Stocke toutes tes activités dans un `activities.json` sur Google Drive  
✅ Analyse tes séances (distance, allure, FC, dérive cardio, k FC/Allure)  
✅ Génère un dashboard web mobile (Flask)  
✅ App installable en tant que **PWA (Progressive Web App)** sur ton téléphone

---

## ⚙️ Architecture
```
Strava --> Webhook (Render) --> Google Drive
                                  ↓
                     Flask Dashboard (Render) --> PWA installée sur ton téléphone
```

---

## 📦 Structure du projet
```
.
├── app.py              # Dashboard Flask (PWA)
├── strava_webhook.py   # Webhook Strava (Render)
├── templates/
│   └── index.html      # HTML dashboard
├── static/
│   ├── manifest.json
│   ├── service-worker.js
│   └── icons/
├── profile.json        # Ton profil + événements
├── requirements.txt
└── .gitignore
```

---

## 🚀 Déploiement Render
### 🛰 Webhook
- Start command :
```
python strava_webhook.py
```
- Utilise `strava_tokens.json` pour appeler l’API Strava et mettre à jour Drive.

### 📱 Dashboard PWA
- Start command :
```
gunicorn app:app -b 0.0.0.0:$PORT
```
- Se connecte à Google Drive pour lire `activities.json` et générer ton dashboard.

---

## 🔥 PWA sur ton téléphone
- Le site propose automatiquement :
```
Ajouter à l'écran d'accueil
```
- Devient une vraie app mobile installée, **plein écran et offline**.

---

## ✅ Pour lancer localement
```
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt

# Pour voir ton dashboard
python app.py
```
- puis ouvre `http://127.0.0.1:5000`

---

## 📝 Variables Render
- `GOOGLE_APPLICATION_CREDENTIALS_JSON` (service account JSON pour Drive)
- `PORT` fourni automatiquement par Render
- Pour le webhook :
  - `client_id`, `client_secret` et refresh token Strava sont gérés dans `strava_tokens.json` ou en ENV.

---

## 🚀 Roadmap
✅ Milestone actuel : PWA installable + dashboard Strava  
🚀 Prochaines étapes possibles :
- + Graphiques IA avancés (progression k, zones cardio)
- + Génération plan d'entraînement IA semi <2h
- + Notifications Push

---

## ✌️ By ton-pseudo
