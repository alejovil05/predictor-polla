import pandas as pd

HISTORY_PATH = "data/team_match_history.csv"


def load_history_data():
    return pd.read_csv(HISTORY_PATH)


def clamp(value, min_value=0.2, max_value=3.5):
    return max(min_value, min(max_value, value))


def get_team_form(team_name, last_n=10):
    df = load_history_data()

    team_games = df[df["team"] == team_name].copy()

    if team_games.empty:
        return None

    team_games["date"] = pd.to_datetime(team_games["date"], errors="coerce")
    team_games = team_games.dropna(subset=["date"])

    # Toma únicamente los últimos 10 partidos registrados por fecha
    team_games = team_games.sort_values("date", ascending=False).head(last_n)

    weights = list(range(len(team_games), 0, -1))
    weight_sum = sum(weights)

    gf_avg = (team_games["gf"] * weights).sum() / weight_sum
    ga_avg = (team_games["ga"] * weights).sum() / weight_sum

    return {
        "team": team_name,
        "games": len(team_games),
        "gf": round(gf_avg, 2),
        "ga": round(ga_avg, 2),
        "form_index": round(gf_avg - ga_avg, 2),
    }


def estimate_xg_from_form(team_a, team_b):
    form_a = get_team_form(team_a)
    form_b = get_team_form(team_b)

    if form_a is None or form_b is None:
        return None

    # Ataque propio pesa más que defensa rival para evitar sobreajuste
    xg_a = (form_a["gf"] * 0.60) + (form_b["ga"] * 0.40)
    xg_b = (form_b["gf"] * 0.60) + (form_a["ga"] * 0.40)

    # Límites razonables para evitar resultados extremos
    xg_a = clamp(xg_a)
    xg_b = clamp(xg_b)

    return float(round(xg_a, 2)), float(round(xg_b, 2))