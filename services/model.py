import math
import pandas as pd


def poisson_prob(goals, expected_goals):
    return (expected_goals ** goals * math.exp(-expected_goals)) / math.factorial(goals)


def calculate_matrix(team_a, team_b, xg_a, xg_b, max_goals, points_winner, points_goals, points_diff):
    rows = []
    real_scores = []

    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            prob = poisson_prob(a, xg_a) * poisson_prob(b, xg_b)

            if a > b:
                result = "Gana A"
            elif a == b:
                result = "Empate"
            else:
                result = "Gana B"

            real_scores.append({
                "a": a,
                "b": b,
                "prob": prob,
                "result": result,
                "diff": a - b
            })

    for pred_a in range(max_goals + 1):
        for pred_b in range(max_goals + 1):
            expected_points = 0

            if pred_a > pred_b:
                pred_result = "Gana A"
            elif pred_a == pred_b:
                pred_result = "Empate"
            else:
                pred_result = "Gana B"

            pred_diff = pred_a - pred_b
            exact_prob = 0

            for real in real_scores:
                points = 0

                if pred_result == real["result"]:
                    points += points_winner

                if pred_a == real["a"]:
                    points += points_goals

                if pred_b == real["b"]:
                    points += points_goals

                if pred_diff == real["diff"]:
                    points += points_diff

                expected_points += real["prob"] * points

                if pred_a == real["a"] and pred_b == real["b"]:
                    exact_prob = real["prob"]

            rows.append({
                "Marcador": f"{team_a} {pred_a} - {pred_b} {team_b}",
                "Goles A": pred_a,
                "Goles B": pred_b,
                "Prob. exacta": exact_prob,
                "Prob. exacta %": exact_prob * 100,
                "Puntos esperados": expected_points
            })

    df = pd.DataFrame(rows)
    return df.sort_values("Puntos esperados", ascending=False), pd.DataFrame(real_scores)


def summarize_probabilities(real_df):
    return {
        "Victoria A": real_df.loc[real_df["a"] > real_df["b"], "prob"].sum(),
        "Empate": real_df.loc[real_df["a"] == real_df["b"], "prob"].sum(),
        "Victoria B": real_df.loc[real_df["a"] < real_df["b"], "prob"].sum(),
        "Over 1.5": real_df.loc[(real_df["a"] + real_df["b"]) > 1.5, "prob"].sum(),
        "Over 2.5": real_df.loc[(real_df["a"] + real_df["b"]) > 2.5, "prob"].sum(),
        "Under 2.5": real_df.loc[(real_df["a"] + real_df["b"]) < 2.5, "prob"].sum(),
        "Ambos marcan Sí": real_df.loc[(real_df["a"] > 0) & (real_df["b"] > 0), "prob"].sum(),
    }


def goals_distribution(real_df, team_col):
    return {
        "0 goles": real_df.loc[real_df[team_col] == 0, "prob"].sum(),
        "1 gol": real_df.loc[real_df[team_col] == 1, "prob"].sum(),
        "2 goles": real_df.loc[real_df[team_col] == 2, "prob"].sum(),
        "3+ goles": real_df.loc[real_df[team_col] >= 3, "prob"].sum()
    }


def calculate_confidence(top_points_df):
    first = top_points_df.iloc[0]["Puntos esperados"]
    second = top_points_df.iloc[1]["Puntos esperados"]
    third = top_points_df.iloc[2]["Puntos esperados"]

    gap_1_2 = first - second
    gap_1_3 = first - third

    if gap_1_3 >= 0.50:
        confidence = 8
        risk = "Bajo"
    elif gap_1_3 >= 0.25:
        confidence = 7
        risk = "Medio"
    elif gap_1_3 >= 0.10:
        confidence = 6
        risk = "Medio-Alto"
    else:
        confidence = 5
        risk = "Alto"

    return confidence, risk, gap_1_2, gap_1_3


def get_decision_gap(top_points_df):
    """
    Mide qué tan fuerte es la recomendación principal frente a la segunda.
    Recibe un DataFrame y lo ordena por 'Puntos esperados' descendente.
    """
    if top_points_df is None or len(top_points_df) < 2:
        return None

    top_points_df = top_points_df.sort_values("Puntos esperados", ascending=False)

    first = top_points_df.iloc[0]
    second = top_points_df.iloc[1]

    gap = first["Puntos esperados"] - second["Puntos esperados"]

    if gap >= 0.50:
        label = "🟢 Decisión robusta"
        advice = "La recomendación principal tiene ventaja clara."
    elif gap >= 0.20:
        label = "🟡 Decisión competida"
        advice = "Hay una diferencia moderada. Conviene revisar contexto y alineaciones."
    else:
        label = "🔴 Decisión muy cerrada"
        advice = "No sobreinterpretar. Revisar el marcador más probable y el conservador."

    return {
        "gap": round(gap, 2),
        "label": label,
        "advice": advice,
        "best_score": first["Marcador"],
        "second_score": second["Marcador"],
        "best_points": round(first["Puntos esperados"], 2),
        "second_points": round(second["Puntos esperados"], 2),
    }


def get_conservative_score(top_points_df):
    """
    Elige un marcador conservador sin contradecir la dirección del modelo.

    Antes priorizaba una lista fija como (1,0), (0,1), etc.
    Eso podía devolver 1-0 aunque el modelo favoreciera al visitante.

    Ahora:
    1. Toma como dirección base el marcador con mayor puntaje esperado.
    2. Filtra marcadores que mantengan esa misma dirección:
       - gana local
       - empate
       - gana visitante
    3. Prioriza marcadores de baja varianza.
    4. Elige el mejor por puntos esperados dentro de esos candidatos.
    """
    if top_points_df is None or top_points_df.empty:
        return None

    df = top_points_df.sort_values("Puntos esperados", ascending=False).copy()

    best = df.iloc[0]
    best_diff = best["Goles A"] - best["Goles B"]

    if best_diff > 0:
        target_result = "home"
    elif best_diff < 0:
        target_result = "away"
    else:
        target_result = "draw"

    def get_result_type(row):
        if row["Goles A"] > row["Goles B"]:
            return "home"
        elif row["Goles A"] < row["Goles B"]:
            return "away"
        return "draw"

    df["result_type"] = df.apply(get_result_type, axis=1)
    df["total_goals"] = df["Goles A"] + df["Goles B"]

    candidates = df[df["result_type"] == target_result].copy()

    if candidates.empty:
        candidates = df.copy()

    # Baja varianza: evita marcadores demasiado altos como conservadores
    low_variance_candidates = candidates[candidates["total_goals"] <= 3].copy()

    if not low_variance_candidates.empty:
        candidates = low_variance_candidates

    row = candidates.sort_values("Puntos esperados", ascending=False).iloc[0]

    return {
        "Marcador": row["Marcador"],
        "Goles A": int(row["Goles A"]),
        "Goles B": int(row["Goles B"]),
        "Puntos esperados": round(row["Puntos esperados"], 2),
        "Prob. exacta %": round(row["Prob. exacta %"], 2),
    }