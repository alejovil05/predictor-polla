from __future__ import annotations

from functools import lru_cache

import pandas as pd

from services.model import calculate_matrix, get_conservative_score


COMMON_PUBLIC_SCORES = {
    (1, 0): 1.00,
    (2, 0): 0.95,
    (2, 1): 0.95,
    (1, 1): 0.90,
    (0, 1): 0.82,
    (0, 0): 0.75,
    (3, 0): 0.70,
    (3, 1): 0.68,
    (1, 2): 0.68,
    (0, 2): 0.62,
}


def _result_type(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "home"
    if goals_a < goals_b:
        return "away"
    return "draw"


def _result_probabilities(score_df: pd.DataFrame) -> dict:
    total_prob = float(score_df["Prob. exacta"].sum())
    if total_prob <= 0:
        total_prob = 1.0

    home_prob = float(score_df.loc[score_df["Goles A"] > score_df["Goles B"], "Prob. exacta"].sum()) / total_prob
    draw_prob = float(score_df.loc[score_df["Goles A"] == score_df["Goles B"], "Prob. exacta"].sum()) / total_prob
    away_prob = float(score_df.loc[score_df["Goles A"] < score_df["Goles B"], "Prob. exacta"].sum()) / total_prob

    return {
        "home": home_prob,
        "draw": draw_prob,
        "away": away_prob,
        "favorite": max(("home", home_prob), ("draw", draw_prob), ("away", away_prob), key=lambda item: item[1])[0],
        "favorite_prob": max(home_prob, draw_prob, away_prob),
    }


def _contextual_public_bias(row: pd.Series, result_probs: dict) -> float:
    goals_a = int(row["Goles A"])
    goals_b = int(row["Goles B"])
    total_goals = goals_a + goals_b
    result_type = _result_type(goals_a, goals_b)

    public_bias = COMMON_PUBLIC_SCORES.get((goals_a, goals_b), 0.35)

    favorite = result_probs.get("favorite")
    favorite_prob = float(result_probs.get("favorite_prob", 0))

    # En partidos con favorito claro, el público tiende a inflar marcadores del favorito.
    if favorite_prob >= 0.55 and result_type == favorite:
        public_bias *= 1.18
    elif favorite_prob >= 0.55 and result_type != favorite:
        public_bias *= 0.82

    # En partidos cerrados, los empates populares suben y las goleadas bajan.
    if favorite_prob <= 0.43:
        if result_type == "draw":
            public_bias *= 1.12
        elif total_goals >= 4:
            public_bias *= 0.78

    if total_goals >= 4:
        public_bias *= 0.55
    if abs(goals_a - goals_b) >= 3:
        public_bias *= 0.65

    return max(0.05, min(1.0, public_bias))


def _score_dict(row: pd.Series, strategy: str, reason: str, result_probs: dict | None = None) -> dict:
    goals_a = int(row["Goles A"])
    goals_b = int(row["Goles B"])
    exact_pct = float(row["Prob. exacta %"])
    expected_points = float(row["Puntos esperados"])
    result_probs = result_probs or {"favorite_prob": 0, "favorite": None}
    public_bias = _contextual_public_bias(row, result_probs)
    leverage = expected_points * (1 + (1 - public_bias) * 0.35)

    return {
        "Estrategia": strategy,
        "Marcador": row["Marcador"],
        "Goles A": goals_a,
        "Goles B": goals_b,
        "Resultado": _result_type(goals_a, goals_b),
        "Puntos esperados": round(expected_points, 2),
        "Prob. exacta %": round(exact_pct, 2),
        "Popularidad estimada": round(public_bias, 2),
        "Ventaja estratégica": round(leverage, 2),
        "Motivo": reason,
    }


def get_strategy_picks(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve picks complementarios para una polla:
    - Seguro: reduce varianza sin contradecir la dirección principal.
    - Balanceado: maximiza puntos esperados.
    - Diferencial: sacrifica un poco de EV para evitar el marcador obvio.
    - Cazador exacto: prioriza probabilidad exacta entre picks competitivos.
    - Plan B empate: expone el mejor empate sin reemplazar el pick principal.
    """
    if score_df is None or score_df.empty:
        return pd.DataFrame()

    df = score_df.sort_values("Puntos esperados", ascending=False).copy()
    result_probs = _result_probabilities(df)
    best = df.iloc[0]
    best_points = float(best["Puntos esperados"])

    conservative = get_conservative_score(df.head(12))
    conservative_row = df[
        (df["Goles A"] == conservative["Goles A"])
        & (df["Goles B"] == conservative["Goles B"])
    ].iloc[0] if conservative else best

    ev_window = max(0.10, best_points * 0.05)
    competitive = df[df["Puntos esperados"] >= best_points - ev_window].copy()
    if competitive.empty:
        competitive = df.head(8).copy()

    competitive["public_bias"] = competitive.apply(
        lambda row: _contextual_public_bias(row, result_probs),
        axis=1,
    )
    competitive["leverage"] = competitive["Puntos esperados"] * (1 + (1 - competitive["public_bias"]) * 0.35)

    differential_pool = competitive[
        ~(
            (competitive["Goles A"] == best["Goles A"])
            & (competitive["Goles B"] == best["Goles B"])
        )
    ].copy()

    if differential_pool.empty:
        differential_pool = df[
            (df["Puntos esperados"] >= best_points - (ev_window * 2))
            & ~((df["Goles A"] == best["Goles A"]) & (df["Goles B"] == best["Goles B"]))
        ].copy()
        differential_pool["public_bias"] = differential_pool.apply(
            lambda row: _contextual_public_bias(row, result_probs),
            axis=1,
        )
        differential_pool["leverage"] = (
            differential_pool["Puntos esperados"] * (1 + (1 - differential_pool["public_bias"]) * 0.35)
        )

    if differential_pool.empty:
        differential_pool = competitive

    differential = differential_pool.sort_values(
        ["leverage", "Puntos esperados", "Prob. exacta %"],
        ascending=[False, False, False],
    ).iloc[0]

    exact_hunter = competitive.sort_values(
        ["Prob. exacta %", "Puntos esperados"],
        ascending=[False, False],
    ).iloc[0]

    draw_cover = get_draw_cover_pick(df)
    draw_row = None
    if draw_cover:
        draw_row = df[
            (df["Goles A"] == draw_cover["Goles A"])
            & (df["Goles B"] == draw_cover["Goles B"])
        ].iloc[0]

    picks = [
        _score_dict(conservative_row, "Seguro", "Baja varianza manteniendo la dirección del modelo.", result_probs),
        _score_dict(best, "Balanceado", "Mayor puntaje esperado según las reglas de la polla.", result_probs),
        _score_dict(differential, "Diferencial", "Buen EV con menor popularidad estimada para ganar ventaja estratégica.", result_probs),
        _score_dict(exact_hunter, "Cazador exacto", "Mayor probabilidad exacta dentro de los picks competitivos.", result_probs),
    ]

    if draw_row is not None:
        picks.append(
            _score_dict(
                draw_row,
                "Plan B empate",
                "No es el pick principal: es cobertura táctica si querés protegerte del empate.",
                result_probs,
            )
        )

    return pd.DataFrame(picks).drop_duplicates(subset=["Estrategia"], keep="first")


def get_draw_cover_pick(score_df: pd.DataFrame) -> dict | None:
    """
    Encuentra el empate más útil como Plan B.

    No reemplaza al pick principal: lo hace visible cuando el modelo favorece
    apenas a un lado, que es justo donde una polla se puede definir.
    """
    if score_df is None or score_df.empty:
        return None

    df = score_df.sort_values("Puntos esperados", ascending=False).copy()
    best = df.iloc[0]
    draws = df[df["Goles A"] == df["Goles B"]].copy()

    if draws.empty:
        return None

    draw_row = draws.sort_values(
        ["Puntos esperados", "Prob. exacta %"],
        ascending=[False, False],
    ).iloc[0]

    draw_probability = float(draws["Prob. exacta"].sum() * 100)
    gap_to_best = float(best["Puntos esperados"] - draw_row["Puntos esperados"])

    ev_window = max(0.10, float(best["Puntos esperados"]) * 0.05)

    if gap_to_best <= ev_window:
        label = "Plan B fuerte"
        advice = "No reemplaza al pick principal; solo indica que el empate está muy cerca como cobertura."
    elif gap_to_best <= ev_window * 2.5:
        label = "Plan B jugable"
        advice = "No es el #1; consideralo únicamente si querés cubrir un partido cerrado."
    else:
        label = "Plan B lejano"
        advice = "El empate existe, pero el modelo prefiere claramente otro resultado."

    return {
        "Marcador": draw_row["Marcador"],
        "Goles A": int(draw_row["Goles A"]),
        "Goles B": int(draw_row["Goles B"]),
        "Puntos esperados": round(float(draw_row["Puntos esperados"]), 2),
        "Prob. exacta %": round(float(draw_row["Prob. exacta %"]), 2),
        "Prob. empate %": round(draw_probability, 1),
        "Gap al mejor": round(gap_to_best, 2),
        "Nivel": label,
        "Consejo": advice,
    }


def build_decision_alerts(score_df: pd.DataFrame, xg_a: float, xg_b: float) -> list[str]:
    if score_df is None or len(score_df) < 3:
        return []

    df = score_df.sort_values("Puntos esperados", ascending=False).head(8)
    first = df.iloc[0]
    second = df.iloc[1]
    third = df.iloc[2]
    alerts = []

    gap_1_2 = float(first["Puntos esperados"] - second["Puntos esperados"])
    gap_1_3 = float(first["Puntos esperados"] - third["Puntos esperados"])

    if gap_1_2 < 0.08:
        alerts.append("Decisión finísima: el primer y segundo marcador están prácticamente empatados.")
    if gap_1_3 < 0.18:
        alerts.append("Alta dispersión: no sobreinterpretar el marcador exacto; importa más la dirección.")
    if abs(float(xg_a) - float(xg_b)) < 0.25:
        alerts.append("Partido parejo por xG: los empates y marcadores de un gol de diferencia ganan peso.")
    draw_cover = get_draw_cover_pick(score_df)
    if draw_cover and draw_cover["Gap al mejor"] <= 0.45:
        alerts.append(
            f"Plan B empate: {draw_cover['Marcador']} queda a "
            f"{draw_cover['Gap al mejor']:.2f} puntos esperados del mejor pick. "
            "No reemplaza al pick principal."
        )
    if max(float(xg_a), float(xg_b)) >= 2.3 and min(float(xg_a), float(xg_b)) >= 0.9:
        alerts.append("Partido con techo alto: un pick diferencial con ambos marcando puede tener valor.")

    return alerts


def run_uncertainty_scenarios(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    max_goals: int,
    points_winner: int,
    points_goals: int,
    points_diff: int,
    spread: float = 0.12,
) -> dict:
    """
    Simula sensibilidad del pick ante incertidumbre de xG.
    No predice más partidos: mide si el pick sobrevive a pequeños cambios razonables.
    """
    cached = _run_uncertainty_scenarios_cached(
        team_a,
        team_b,
        round(float(xg_a), 2),
        round(float(xg_b), 2),
        int(max_goals),
        int(points_winner),
        int(points_goals),
        int(points_diff),
        round(float(spread), 3),
    )
    return {
        "stability": cached["stability"],
        "label": cached["label"],
        "top_score": cached["top_score"],
        "scenarios": pd.DataFrame(cached["rows"]),
    }


@lru_cache(maxsize=512)
def _run_uncertainty_scenarios_cached(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    max_goals: int,
    points_winner: int,
    points_goals: int,
    points_diff: int,
    spread: float,
) -> dict:
    factors = [1 - spread, 1.0, 1 + spread]
    rows = []

    for factor_a in factors:
        for factor_b in factors:
            scenario_xg_a = max(0.05, round(float(xg_a) * factor_a, 2))
            scenario_xg_b = max(0.05, round(float(xg_b) * factor_b, 2))
            df, _ = calculate_matrix(
                team_a,
                team_b,
                scenario_xg_a,
                scenario_xg_b,
                max_goals,
                points_winner,
                points_goals,
                points_diff,
            )
            best = df.iloc[0]
            rows.append({
                "xG A": scenario_xg_a,
                "xG B": scenario_xg_b,
                "Marcador recomendado": best["Marcador"],
                "Goles A": int(best["Goles A"]),
                "Goles B": int(best["Goles B"]),
                "Puntos esperados": round(float(best["Puntos esperados"]), 2),
                "Prob. exacta %": round(float(best["Prob. exacta %"]), 2),
            })

    scenarios = pd.DataFrame(rows)
    top_score = scenarios["Marcador recomendado"].mode().iloc[0]
    stability = (scenarios["Marcador recomendado"] == top_score).mean()

    if stability >= 0.67:
        label = "Alta"
    elif stability >= 0.45:
        label = "Media"
    else:
        label = "Baja"

    return {
        "stability": round(float(stability), 2),
        "label": label,
        "top_score": top_score,
        "rows": tuple(rows),
    }


def blend_elo_form_xg(
    elo_xg_a: float,
    elo_xg_b: float,
    form_xg_a: float,
    form_xg_b: float,
    form_games_a: int,
    form_games_b: int,
) -> tuple[float, float, float]:
    """
    Mezcla Elo y forma con peso dinámico.
    Más partidos recientes confiables => más peso a forma, sin dejar que domine.
    """
    sample_quality = min(int(form_games_a), int(form_games_b), 10) / 10
    form_weight = 0.15 + (0.20 * sample_quality)
    elo_weight = 1 - form_weight

    xg_a = round((elo_weight * float(elo_xg_a)) + (form_weight * float(form_xg_a)), 2)
    xg_b = round((elo_weight * float(elo_xg_b)) + (form_weight * float(form_xg_b)), 2)

    return xg_a, xg_b, round(form_weight, 2)
