import pandas as pd

HISTORY_PATH = "data/team_match_history.csv"


def load_history():
    return pd.read_csv(HISTORY_PATH)


def get_recent_form_from_history(team_name, limit=10):
    df = load_history()

    team_df = df[df["team"] == team_name].copy()

    if team_df.empty:
        return None

    team_df["date"] = pd.to_datetime(team_df["date"])
    team_df = team_df.sort_values("date", ascending=False).head(limit)

    played = len(team_df)

    if played == 0:
        return None

    gf = team_df["gf"].sum()
    ga = team_df["ga"].sum()

    return {
        "team": team_name,
        "played": played,
        "gf": gf,
        "ga": ga,
        "gf_avg": round(gf / played, 2),
        "ga_avg": round(ga / played, 2)
    }


def estimate_xg_from_history(team_a, team_b, limit=10):
    form_a = get_recent_form_from_history(team_a, limit)
    form_b = get_recent_form_from_history(team_b, limit)

    if form_a is None or form_b is None:
        return None

    xg_a = (form_a["gf_avg"] + form_b["ga_avg"]) / 2
    xg_b = (form_b["gf_avg"] + form_a["ga_avg"]) / 2

    return {
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "form_a": form_a,
        "form_b": form_b
    }