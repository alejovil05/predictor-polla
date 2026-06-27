from __future__ import annotations

import pandas as pd


def _true_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return series.astype(str).str.lower().eq("true").mean() * 100


def summarize_backtest(evaluated_df: pd.DataFrame) -> dict:
    if evaluated_df is None or evaluated_df.empty:
        return {}

    df = evaluated_df.copy()
    df["points_recommended"] = pd.to_numeric(df["points_recommended"], errors="coerce")
    df["points_conservative"] = pd.to_numeric(df["points_conservative"], errors="coerce")

    return {
        "matches": len(df),
        "avg_recommended": round(float(df["points_recommended"].mean()), 2),
        "avg_conservative": round(float(df["points_conservative"].mean()), 2),
        "exact_rate": round(_true_rate(df["recommended_exact_hit"]), 1),
        "winner_rate": round(_true_rate(df["recommended_winner_hit"]), 1),
        "home_goal_rate": round(_true_rate(df["recommended_goal_home_hit"]), 1),
        "away_goal_rate": round(_true_rate(df["recommended_goal_away_hit"]), 1),
        "diff_rate": round(_true_rate(df["recommended_diff_hit"]), 1),
        "recommended_edge": round(float((df["points_recommended"] - df["points_conservative"]).mean()), 2),
    }


def group_backtest(evaluated_df: pd.DataFrame, by: str) -> pd.DataFrame:
    if evaluated_df is None or evaluated_df.empty or by not in evaluated_df.columns:
        return pd.DataFrame()

    df = evaluated_df.copy()
    df["points_recommended"] = pd.to_numeric(df["points_recommended"], errors="coerce")
    df["points_conservative"] = pd.to_numeric(df["points_conservative"], errors="coerce")

    grouped = df.groupby(by, dropna=False).agg(
        partidos=(by, "count"),
        prom_recomendado=("points_recommended", "mean"),
        prom_conservador=("points_conservative", "mean"),
        exactos=("recommended_exact_hit", lambda value: _true_rate(value)),
        ganador=("recommended_winner_hit", lambda value: _true_rate(value)),
    ).reset_index()

    grouped["prom_recomendado"] = grouped["prom_recomendado"].round(2)
    grouped["prom_conservador"] = grouped["prom_conservador"].round(2)
    grouped["exactos"] = grouped["exactos"].round(1)
    grouped["ganador"] = grouped["ganador"].round(1)

    return grouped.sort_values(["prom_recomendado", "partidos"], ascending=[False, False])


def find_model_learnings(evaluated_df: pd.DataFrame) -> list[str]:
    if evaluated_df is None or evaluated_df.empty:
        return []

    df = evaluated_df.copy()
    df["points_recommended"] = pd.to_numeric(df["points_recommended"], errors="coerce")
    learnings = []

    exact_rate = _true_rate(df["recommended_exact_hit"])
    winner_rate = _true_rate(df["recommended_winner_hit"])
    avg_points = df["points_recommended"].mean()

    if winner_rate >= 65 and exact_rate < 20:
        learnings.append("La dirección del partido viene bien, pero el marcador exacto necesita más calibración de goles.")
    if avg_points < 5:
        learnings.append("El promedio de puntos está bajo: conviene revisar reglas de puntuación, xG y sesgo hacia marcadores obvios.")
    if "xg_source" in df.columns:
        by_source = group_backtest(df, "xg_source")
        if len(by_source) >= 2:
            best = by_source.iloc[0]
            worst = by_source.iloc[-1]
            learnings.append(
                f"Mejor fuente xG hasta ahora: {best['xg_source']} "
                f"({best['prom_recomendado']} pts) vs {worst['xg_source']} "
                f"({worst['prom_recomendado']} pts)."
            )
    if not learnings:
        learnings.append("Todavía faltan más partidos evaluados para conclusiones fuertes. No vendas humo: muestra evidencia.")

    return learnings
