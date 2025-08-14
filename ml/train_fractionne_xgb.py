# ml/train_fractionne_xgb.py
import os, sys, json, pickle, numpy as np

# <-- Permet d'importer app.py (qui est à la racine du projet)
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

# Réutilise le chargeur Drive de l'app (pas besoin de fichier local)
from app import load_activities

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def features(activity):
    pts = activity.get("points") or []
    if len(pts) < 10:
        return [0.0, 0.0, 0.0, 0.0]

    fcs   = np.array([p.get("hr")       for p in pts], dtype=float)
    vels  = np.array([p.get("vel")      for p in pts], dtype=float)
    dists = np.array([p.get("distance") for p in pts], dtype=float)
    allures = np.where(vels > 0, (1.0 / vels) * 16.6667, np.nan)  # min/km

    def cv(x):
        m = np.nanmean(x)
        if not np.isfinite(m) or m == 0: return 0.0
        return float(np.nanstd(x) / m)

    cv_allure = cv(allures)
    cv_fc     = cv(fcs)

    mean_all = np.nanmean(allures)
    thr_fast = mean_all - 0.40 if np.isfinite(mean_all) else np.nan
    blocs_rapides, start = 0, 0
    for i in range(1, len(dists)):
        if (dists[i] - dists[start]) >= 500 or i == len(dists) - 1:
            bloc_all = np.nanmean(allures[start:i+1])
            if np.isfinite(bloc_all) and np.isfinite(thr_fast) and bloc_all < thr_fast:
                blocs_rapides += 1
            start = i + 1

    if np.all(np.isnan(fcs)) or len(fcs) == 0:
        pct_90 = 0.0
    else:
        fcmax = np.nanmax(fcs)
        thr = 0.9 * fcmax if np.isfinite(fcmax) and fcmax > 0 else np.nan
        if np.isfinite(thr):
            pct_90 = float(np.nansum(fcs > thr) / np.count_nonzero(~np.isnan(fcs)))
        else:
            pct_90 = 0.0

    feats = [cv_allure, cv_fc, float(blocs_rapides), pct_90]
    feats = [0.0 if (not np.isfinite(v)) else float(v) for v in feats]
    return feats

# 1) Charger les activités depuis Drive via l'app
activities = load_activities()

# 2) Construire X, y à partir des labels manuels importés depuis Excel
X, y = [], []
for a in activities:
    pts = a.get("points") or []
    if len(pts) < 10:
        continue
    # Focus: <= 11 km (on exclut les long runs)
    dist_km = pts[-1]["distance"] / 1000.0
    if dist_km > 11:
        continue
    # Label requis: is_fractionne_label (posé via ton Excel)
    if "is_fractionne_label" not in a:
        continue
    label = bool(a["is_fractionne_label"])

    X.append(features(a))
    y.append(int(label))

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

if len(X) < 10 or len(set(y)) < 2:
    raise SystemExit("❌ Pas assez de données labellisées ou une seule classe. Ajoute des 1 et des 0 via Excel, puis réimporte dans activities.json.")

# 3) Split, train
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_tr, y_tr)

# 4) Rapport rapide
y_pred = (model.predict_proba(X_te)[:,1] >= 0.5).astype(int)
print(classification_report(y_te, y_pred, digits=3))

# 5) Sauvegarde du modèle à la racine (là où app.py le charge)
save_path = os.path.join(ROOT, "fractionne_xgb.pkl")
with open(save_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Modèle sauvegardé -> {save_path}")
