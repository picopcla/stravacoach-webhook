import json
from openai import OpenAI

client = OpenAI()

# Charger les données
print("📂 Chargement des activities.json...")
with open("Activities(19).json") as f:
    activities = json.load(f)

try:
    with open("examples_correct.json") as f:
        examples = json.load(f)
    print(f"✅ examples_correct.json chargé avec {len(examples)} exemples")
except FileNotFoundError:
    examples = []
    print("⚠️ Pas de examples_correct.json trouvé, on commence vide.")

example_ids = {ex["id"] for ex in examples}

# Trier par date et garder les 15 derniers
activities.sort(key=lambda x: x["date"])
last_activities = activities[-15:]

for act in last_activities:
    activity_id = str(act["activity_id"])
    points = act.get("points", [])
    if not points:
        continue

    # Calcul pace_series et fc_series
    pace_series = []
    fc_series = []
    next_distance = 100
    last_idx = 0
    for i, p in enumerate(points):
        if p["distance"] >= next_distance or i == len(points) - 1:
            delta_dist = p["distance"] - points[last_idx]["distance"]
            delta_time = p["time"] - points[last_idx]["time"]
            if delta_dist > 0:
                pace = (delta_time / 60) / (delta_dist / 1000)
                pace_series.append(round(pace, 2))
                # moyenne FC sur ce bloc
                fc_vals = [pt["hr"] for pt in points[last_idx:i+1] if pt.get("hr") is not None]
                fc_mean = round(sum(fc_vals)/len(fc_vals), 1) if fc_vals else None
                fc_series.append(fc_mean)
            next_distance += 100
            last_idx = i

    distance_total = round(points[-1]["distance"] / 1000, 2)

    # Déjà dans examples
    if activity_id in example_ids:
        print(f"✅ Déjà dans examples : {activity_id}")
        continue

    # Construire le prompt (tu le modifieras toi-même)
    prompt = (
        f"Voici des données pour classifier une sortie :\n\n"
        f"- Distance totale : {distance_total} km\n"
        f"- Histogramme des allures tous les 100m : {pace_series}\n"
        f"- Fréquence cardiaque moyenne tous les 100m : {fc_series}\n\n"
        f"Analyse bien les examples ci-dessous qui définisse bien les fractionnés, endurance et longue pour décider selon l'aspect colline des allures mais aussi de l'augementation du rythme cardiaque lors des fractionnés (plus net que la dérive normale d'un run endurance) :\n{json.dumps(examples, indent=2)}\n\n"
        "La pente de la droite d'augmentation du rythme entre 2 points soidt être importante et l'augmentation du rythme est généralement de plus de 0,40mn / km. Donne juste le type en un mot (fractionné, longue, normale) et une explication courte (10 mots) du pourquoi."
    )

    # Appel à GPT
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        output = response.choices[0].message.content.strip()
        print(f"\n🆕 À classifier : {activity_id}")
        print(f"Distance totale : {distance_total} km")
        print(f"Pace series : {pace_series}")
        print(f"FC series : {fc_series}")
        print(f"👉 GPT propose : {output}")

    except Exception as e:
        print(f"❌ Erreur GPT pour activité {activity_id} :", e)
