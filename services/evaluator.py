import pandas as pd
from datetime import datetime


RESULTS_PATH = "data/prediction_results.csv"


def get_round_multiplier(stage):
    if stage == "Grupos":
        return 1
    return 2


def calculate_poll_points(
    pred_home,
    pred_away,
    real_home,
    real_away,
    stage="Grupos",
    points_winner=5,
    points_goals=2,
    points_diff=1
):
    multiplier = get_round_multiplier(stage)

    points = 0

    pred_result = "home" if pred_home > pred_away else "away" if pred_home < pred_away else "draw"
    real_result = "home" if real_home > real_away else "away" if real_home < real_away else "draw"

    winner_hit = pred_result == real_result
    goal_home_hit = pred_home == real_home
    goal_away_hit = pred_away == real_away
    diff_hit = (pred_home - pred_away) == (real_home - real_away)
    exact_hit = goal_home_hit and goal_away_hit

    if winner_hit:
        points += points_winner
    if goal_home_hit:
        points += points_goals
    if goal_away_hit:
        points += points_goals
    if diff_hit:
        points += points_diff

    points *= multiplier

    return {
        "points": points,
        "exact_hit": exact_hit,
        "winner_hit": winner_hit,
        "goal_home_hit": goal_home_hit,
        "goal_away_hit": goal_away_hit,
        "diff_hit": diff_hit,
        "multiplier": multiplier,
    }


def save_prediction_result(row):
    df_new = pd.DataFrame([row])

    try:
        df_old = pd.read_csv(RESULTS_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df = df_new

    df.to_csv(RESULTS_PATH, index=False)


def build_result_row(
    date,
    home,
    away,
    stage,
    xg_home,
    xg_away,
    recommended_home,
    recommended_away,
    conservative_home,
    conservative_away,
    real_home,
    real_away,
    decision_gap,
    confidence,
    risk,
    style
):
    recommended_points = calculate_poll_points(
        recommended_home,
        recommended_away,
        real_home,
        real_away,
        stage
    )

    conservative_points = calculate_poll_points(
        conservative_home,
        conservative_away,
        real_home,
        real_away,
        stage
    )

    return {
        "date": date,
        "home": home,
        "away": away,
        "stage": stage,
        "xg_home": xg_home,
        "xg_away": xg_away,
        "recommended_score": f"{recommended_home}-{recommended_away}",
        "conservative_score": f"{conservative_home}-{conservative_away}",
        "real_home": real_home,
        "real_away": real_away,
        "real_score": f"{real_home}-{real_away}",
        "points_recommended": recommended_points["points"],
        "points_conservative": conservative_points["points"],
        "recommended_exact_hit": recommended_points["exact_hit"],
        "recommended_winner_hit": recommended_points["winner_hit"],
        "recommended_goal_home_hit": recommended_points["goal_home_hit"],
        "recommended_goal_away_hit": recommended_points["goal_away_hit"],
        "recommended_diff_hit": recommended_points["diff_hit"],
        "conservative_exact_hit": conservative_points["exact_hit"],
        "conservative_winner_hit": conservative_points["winner_hit"],
        "conservative_goal_home_hit": conservative_points["goal_home_hit"],
        "conservative_goal_away_hit": conservative_points["goal_away_hit"],
        "conservative_diff_hit": conservative_points["diff_hit"],
        "decision_gap": decision_gap,
        "confidence": confidence,
        "risk": risk,
        "style": style,
        "multiplier": recommended_points["multiplier"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }