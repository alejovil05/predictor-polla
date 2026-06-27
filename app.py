import os
from datetime import datetime

import pandas as pd
import streamlit as st

from services.model import (
    calculate_matrix,
    summarize_probabilities,
    goals_distribution,
    calculate_confidence,
    get_decision_gap,
    get_conservative_score,
)
from services.elo import estimate_xg_from_elo
from services.form_model import estimate_xg_from_form, get_team_form
from services.backtesting import find_model_learnings, group_backtest, summarize_backtest
from services.strategy import (
    blend_elo_form_xg,
    build_decision_alerts,
    get_draw_cover_pick,
    get_strategy_picks,
    run_uncertainty_scenarios,
)


st.set_page_config(
    page_title="Predictor de Marcadores para Polla",
    page_icon="⚽",
    layout="wide",
)

st.title("Predictor de Marcadores para Polla")
st.caption("Modelo Poisson + Elo + forma reciente CSV + optimización por puntos esperados.")

MATCHES_PATH = "data/worldcup2026.csv"
HISTORY_PATH = "data/team_match_history.csv"
RESULTS_PATH = "data/prediction_results.csv"

RESULT_COLUMNS = [
    "date",
    "group",
    "home",
    "away",
    "stage",
    "xg_home",
    "xg_away",
    "style",
    "xg_source",
    "recommended_home",
    "recommended_away",
    "recommended_score",
    "recommended_points_expected",
    "recommended_exact_prob_pct",
    "conservative_home",
    "conservative_away",
    "conservative_score",
    "conservative_points_expected",
    "conservative_exact_prob_pct",
    "decision_gap",
    "confidence",
    "risk",
    "real_home",
    "real_away",
    "real_score",
    "points_recommended",
    "points_conservative",
    "recommended_exact_hit",
    "recommended_winner_hit",
    "recommended_goal_home_hit",
    "recommended_goal_away_hit",
    "recommended_diff_hit",
    "conservative_exact_hit",
    "conservative_winner_hit",
    "conservative_goal_home_hit",
    "conservative_goal_away_hit",
    "conservative_diff_hit",
    "multiplier",
    "status",
    "source",
    "created_at",
    "evaluated_at",
]


def ensure_results_file():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    if not os.path.exists(RESULTS_PATH):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULTS_PATH, index=False)


@st.cache_data
def load_matches():
    return pd.read_csv(MATCHES_PATH)


@st.cache_data
def load_prediction_results():
    try:
        ensure_results_file()
        df = pd.read_csv(RESULTS_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=RESULT_COLUMNS)
        df.to_csv(RESULTS_PATH, index=False)

    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[RESULT_COLUMNS]

    df = normalize_results_dtypes(df)

    return df


def normalize_results_dtypes(df):
    text_columns = [
        "date", "group", "home", "away", "stage", "style", "xg_source",
        "recommended_score", "conservative_score", "risk", "real_score",
        "status", "source", "created_at", "evaluated_at",
    ]

    bool_columns = [
        "recommended_exact_hit",
        "recommended_winner_hit",
        "recommended_goal_home_hit",
        "recommended_goal_away_hit",
        "recommended_diff_hit",
        "conservative_exact_hit",
        "conservative_winner_hit",
        "conservative_goal_home_hit",
        "conservative_goal_away_hit",
        "conservative_diff_hit",
    ]

    object_columns = text_columns + bool_columns

    for col in object_columns:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df


def save_prediction_results(results_df):
    for col in RESULT_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = pd.NA

    results_df = results_df[RESULT_COLUMNS]
    results_df = normalize_results_dtypes(results_df)
    results_df.to_csv(RESULTS_PATH, index=False)
    st.cache_data.clear()


def get_round_multiplier(stage):
    return 1 if str(stage).strip().lower() == "grupos" else 2


def match_key(date, home, away):
    return f"{str(date)}|{home}|{away}"


def get_match_result(pred_home, pred_away):
    if pred_home > pred_away:
        return "home"
    if pred_home < pred_away:
        return "away"
    return "draw"


def calculate_poll_points(
    pred_home,
    pred_away,
    real_home,
    real_away,
    stage="Grupos",
    points_winner=5,
    points_goals=2,
    points_diff=1,
):
    pred_home = int(pred_home)
    pred_away = int(pred_away)
    real_home = int(real_home)
    real_away = int(real_away)

    multiplier = get_round_multiplier(stage)
    points = 0

    winner_hit = get_match_result(pred_home, pred_away) == get_match_result(real_home, real_away)
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


def parse_score_from_row(row):
    return int(row["Goles A"]), int(row["Goles B"])


def parse_score_text(score_text):
    left, right = str(score_text).split("-")
    return int(left.strip()), int(right.strip())


if st.button("Limpiar caché"):
    st.cache_data.clear()
    st.rerun()


matches = load_matches()
ensure_results_file()

st.sidebar.header("Datos del partido")

available_dates = sorted(matches["date"].astype(str).unique())
available_groups = sorted(matches["group"].astype(str).unique())

selected_date = st.sidebar.radio("Filtrar por fecha", ["Todas"] + available_dates, index=0)
selected_group = st.sidebar.radio("Filtrar por grupo", ["Todos"] + available_groups, index=0)

filtered_matches = matches.copy()

if selected_date != "Todas":
    filtered_matches = filtered_matches[filtered_matches["date"].astype(str) == selected_date]

if selected_group != "Todos":
    filtered_matches = filtered_matches[filtered_matches["group"].astype(str) == selected_group]

if filtered_matches.empty:
    st.warning("No hay partidos con esos filtros.")
    st.stop()

match_labels = (
    filtered_matches["date"].astype(str)
    + " | Grupo "
    + filtered_matches["group"].astype(str)
    + " | "
    + filtered_matches["home"]
    + " vs "
    + filtered_matches["away"]
)

selected_label = st.sidebar.selectbox("Selecciona partido", match_labels)
selected_match = filtered_matches.loc[match_labels == selected_label].iloc[0]

team_a = selected_match["home"]
team_b = selected_match["away"]
match_date = str(selected_match["date"])
match_group = selected_match["group"]
match_stage = selected_match.get("stage", "Grupos")

elo_xg_a, elo_xg_b, elo_a, elo_b = estimate_xg_from_elo(team_a, team_b)

st.sidebar.write("### Elo estimado")
st.sidebar.write(f"{team_a}: {elo_a}")
st.sidebar.write(f"{team_b}: {elo_b}")

st.sidebar.header("Ajustes avanzados")

match_style = st.sidebar.selectbox(
    "Tipo de partido",
    ["Muy Cerrado", "Cerrado", "Normal", "Abierto", "Muy Abierto"],
    index=2,
)

attack_a = st.sidebar.slider(f"Ataque {team_a}", -20, 20, 0)
attack_b = st.sidebar.slider(f"Ataque {team_b}", -20, 20, 0)

use_auto_xg = st.sidebar.checkbox("Usar xG automático por Elo", value=True)
use_form_xg = st.sidebar.checkbox("Usar forma reciente CSV para xG", value=False)

style_factor = {
    "Muy Cerrado": 0.80,
    "Cerrado": 0.90,
    "Normal": 1.00,
    "Abierto": 1.12,
    "Muy Abierto": 1.25,
}

xg_source = "Manual"
auto_xg_a, auto_xg_b = elo_xg_a, elo_xg_b

if use_form_xg:
    form_xg = estimate_xg_from_form(team_a, team_b)

    if form_xg:
        form_xg_a, form_xg_b = form_xg

        form_a = get_team_form(team_a)
        form_b = get_team_form(team_b)
        auto_xg_a, auto_xg_b, form_weight = blend_elo_form_xg(
            elo_xg_a,
            elo_xg_b,
            form_xg_a,
            form_xg_b,
            form_a["games"] if form_a else 0,
            form_b["games"] if form_b else 0,
        )
        xg_source = "Elo + Forma CSV"

        st.sidebar.success(
            f"xG Elo + Forma CSV: {auto_xg_a} - {auto_xg_b} "
            f"(peso forma {form_weight:.0%})"
        )
        if form_a and form_b:
            with st.sidebar.expander("Forma reciente usada"):
                st.write(
                    f"**{team_a}**: {form_a['games']} partidos | "
                    f"GF {form_a['gf']} | GA {form_a['ga']} | Índice {form_a['form_index']}"
                )
                st.write(
                    f"**{team_b}**: {form_b['games']} partidos | "
                    f"GF {form_b['gf']} | GA {form_b['ga']} | Índice {form_b['form_index']}"
                )
    else:
        xg_source = "Elo"
        st.sidebar.warning("Equipo no encontrado en team_match_history.csv. Usando Elo.")

    default_xg_a = auto_xg_a * style_factor[match_style]
    default_xg_b = auto_xg_b * style_factor[match_style]

elif use_auto_xg:
    xg_source = "Elo"
    default_xg_a = auto_xg_a * style_factor[match_style]
    default_xg_b = auto_xg_b * style_factor[match_style]

else:
    xg_source = "Manual CSV"
    default_xg_a = float(selected_match["xg_home"])
    default_xg_b = float(selected_match["xg_away"])

# Ajustes manuales de ataque
default_xg_a *= (1 + attack_a / 100)
default_xg_b *= (1 + attack_b / 100)

# Ajuste leve de superioridad clara
xg_gap = default_xg_a - default_xg_b
if xg_gap >= 0.75:
    default_xg_a *= 1.12
elif xg_gap <= -0.75:
    default_xg_b *= 1.12

xg_a = st.sidebar.number_input(
    f"Goles esperados de {team_a}",
    min_value=0.00,
    max_value=6.00,
    value=float(round(default_xg_a, 2)),
    step=0.05,
)

xg_b = st.sidebar.number_input(
    f"Goles esperados de {team_b}",
    min_value=0.00,
    max_value=6.00,
    value=float(round(default_xg_b, 2)),
    step=0.05,
)

max_goals = st.sidebar.slider("Máximo de goles a simular", 3, 8, 6)

st.sidebar.header("Reglas de la polla")

points_winner = st.sidebar.number_input("Puntos por acertar ganador/empate", 0, 20, 5)
points_goals = st.sidebar.number_input("Puntos por acertar goles de cada equipo", 0, 20, 2)
points_diff = st.sidebar.number_input("Puntos por acertar diferencia de gol", 0, 20, 1)


df, real_df = calculate_matrix(
    team_a,
    team_b,
    xg_a,
    xg_b,
    max_goals,
    points_winner,
    points_goals,
    points_diff,
)

summary = summarize_probabilities(real_df)
goals_dist_a = goals_distribution(real_df, "a")
goals_dist_b = goals_distribution(real_df, "b")

best_expected = df.iloc[0]
most_likely = df.sort_values("Prob. exacta", ascending=False).iloc[0]

confidence, risk, gap_1_2, gap_1_3 = calculate_confidence(df.head(8))
decision_gap = get_decision_gap(df)
conservative_score = get_conservative_score(df.head(8))
strategy_picks = get_strategy_picks(df)
draw_cover = get_draw_cover_pick(df)
decision_alerts = build_decision_alerts(df, xg_a, xg_b)
uncertainty = run_uncertainty_scenarios(
    team_a,
    team_b,
    xg_a,
    xg_b,
    max_goals,
    points_winner,
    points_goals,
    points_diff,
)

conservative_home, conservative_away = parse_score_text(conservative_score["Marcador"].split(team_a)[-1].split(team_b)[0]) if False else (conservative_score.get("Goles A"), conservative_score.get("Goles B"))

if pd.isna(conservative_home) or pd.isna(conservative_away):
    conservative_row = df[df["Marcador"] == conservative_score["Marcador"]].iloc[0]
    conservative_home, conservative_away = parse_score_from_row(conservative_row)
else:
    conservative_home, conservative_away = int(conservative_home), int(conservative_away)

smart_row = best_expected
smart_pick = smart_row["Marcador"]
smart_reason = "Mayor puntaje esperado."

top3 = df.head(3).copy()
close_gap = top3.iloc[0]["Puntos esperados"] - top3.iloc[2]["Puntos esperados"]

# Regla 1: si los 3 primeros están casi empatados y el rival tiene xG suficiente,
# preferimos un marcador con ambos equipos marcando.
if close_gap <= 0.05:
    both_score_candidates = top3[(top3["Goles A"] >= 2) & (top3["Goles B"] >= 1)]

    if not both_score_candidates.empty and xg_a >= 1.80 and xg_b >= 0.85:
        smart_row = both_score_candidates.iloc[0]
        smart_pick = smart_row["Marcador"]
        smart_reason = "Empate técnico entre marcadores. Se prioriza ambos equipos marcan por xG del rival."

# Regla 2: si el favorito es fuerte, pero el rival tiene xG bajo,
# mantenemos portería a cero.
elif xg_a >= 1.80 and xg_b < 0.75:
    clean_sheet_candidates = top3[(top3["Goles A"] >= 1) & (top3["Goles B"] == 0)]

    if not clean_sheet_candidates.empty:
        smart_row = clean_sheet_candidates.iloc[0]
        smart_pick = smart_row["Marcador"]
        smart_reason = "Favorito fuerte y rival con bajo xG. Se prioriza portería a cero."

recommended_home, recommended_away = parse_score_from_row(smart_row)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Marcador recomendado para la polla")
    st.success(smart_pick)
    st.metric("Puntos esperados", f"{smart_row['Puntos esperados']:.2f}")
    st.metric("Probabilidad exacta", f"{smart_row['Prob. exacta %']:.2f}%")
    st.metric("Confianza", f"{confidence}/10")
    st.metric("Riesgo", risk)

with col2:
    st.subheader("Marcador exacto más probable")
    st.info(most_likely["Marcador"])
    st.metric("Probabilidad exacta", f"{most_likely['Prob. exacta %']:.2f}%")
    st.metric("Puntos esperados", f"{most_likely['Puntos esperados']:.2f}")

st.warning(f"Recomendación que se guardará: {smart_pick}")
st.caption(f"Motivo: {smart_reason}")

st.subheader("Mesa de decisión del tipster")
st.caption("No todos los picks sirven para lo mismo: seguridad, EV, diferencial y exacto son decisiones distintas.")
strategy_columns = [
    "Estrategia",
    "Marcador",
    "Puntos esperados",
    "Prob. exacta %",
    "Popularidad estimada",
    "Ventaja estratégica",
    "Motivo",
]

if strategy_picks.empty or not set(strategy_columns).issubset(strategy_picks.columns):
    st.info("No se pudieron calcular estrategias alternativas. Se mantiene el marcador recomendado.")
else:
    st.dataframe(
        strategy_picks[strategy_columns],
        hide_index=True,
        use_container_width=True,
    )

if draw_cover:
    st.info(
        f"Plan B empate, no pick principal: {draw_cover['Marcador']} | "
        f"{draw_cover['Nivel']} | "
        f"{draw_cover['Puntos esperados']:.2f} pts esperados | "
        f"Prob. empate total {draw_cover['Prob. empate %']:.1f}% | "
        f"{draw_cover['Consejo']}"
    )

unc_col1, unc_col2, unc_col3 = st.columns(3)
unc_col1.metric("Estabilidad del pick", uncertainty["label"])
unc_col2.metric("Consenso escenarios", f"{uncertainty['stability'] * 100:.0f}%")
unc_col3.metric("Marcador más estable", uncertainty["top_score"])

if decision_alerts:
    for alert in decision_alerts:
        st.warning(f"{alert}")

with st.expander("Ver sensibilidad por escenarios de xG"):
    st.dataframe(uncertainty["scenarios"], hide_index=True, use_container_width=True)

st.info(
    f"Diferencia con el segundo mejor marcador: {gap_1_2:.2f} puntos esperados. "
    f"Diferencia con el tercero: {gap_1_3:.2f}. "
    "Mientras más pequeña sea la diferencia, más abierto está el pronóstico."
)

st.divider()
st.subheader("Robustez y marcador conservador")

rob_col1, rob_col2 = st.columns(2)

with rob_col1:
    if decision_gap:
        st.write(decision_gap["label"])
        st.metric("Diferencia de decisión", f"{decision_gap['gap']:.2f}")
        st.caption(decision_gap["advice"])
        st.write(f"Mejor: {decision_gap['best_score']} ({decision_gap['best_points']})")
        st.write(f"Segundo: {decision_gap['second_score']} ({decision_gap['second_points']})")

with rob_col2:
    if conservative_score:
        st.info(conservative_score["Marcador"])
        st.metric("Puntos esperados", f"{conservative_score['Puntos esperados']:.2f}")
        st.metric("Probabilidad exacta", f"{conservative_score['Prob. exacta %']:.2f}%")

st.divider()

st.subheader("A. Probabilidad 1X2")

col1, col2, col3 = st.columns(3)
col1.metric(f"Victoria {team_a}", f"{summary['Victoria A'] * 100:.1f}%")
col2.metric("Empate", f"{summary['Empate'] * 100:.1f}%")
col3.metric(f"Victoria {team_b}", f"{summary['Victoria B'] * 100:.1f}%")

st.subheader("B. Probabilidad de goles por equipo")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**{team_a}**")
    st.dataframe(
        pd.DataFrame({
            "Goles": list(goals_dist_a.keys()),
            "Probabilidad": [f"{v * 100:.1f}%" for v in goals_dist_a.values()],
        }),
        hide_index=True,
        use_container_width=True,
    )

with col2:
    st.write(f"**{team_b}**")
    st.dataframe(
        pd.DataFrame({
            "Goles": list(goals_dist_b.keys()),
            "Probabilidad": [f"{v * 100:.1f}%" for v in goals_dist_b.values()],
        }),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("C. Líneas clave")

lines_df = pd.DataFrame({
    "Línea": ["Over 1.5", "Under 2.5", "Over 2.5", "Ambos marcan Sí", "Ambos marcan No"],
    "Probabilidad": [
        f"{summary['Over 1.5'] * 100:.1f}%",
        f"{summary['Under 2.5'] * 100:.1f}%",
        f"{summary['Over 2.5'] * 100:.1f}%",
        f"{summary['Ambos marcan Sí'] * 100:.1f}%",
        f"{(1 - summary['Ambos marcan Sí']) * 100:.1f}%",
    ],
})

st.dataframe(lines_df, hide_index=True, use_container_width=True)

st.subheader("D. Top 8 marcadores exactos más probables")

top_exact = df.sort_values("Prob. exacta", ascending=False).head(8).copy()
top_exact["Prob. exacta %"] = top_exact["Prob. exacta %"].map(lambda x: f"{x:.2f}%")
top_exact["Puntos esperados"] = top_exact["Puntos esperados"].map(lambda x: f"{x:.2f}")

st.dataframe(top_exact[["Marcador", "Prob. exacta %", "Puntos esperados"]], hide_index=True, use_container_width=True)

st.subheader("E. Mejores 8 marcadores por puntos esperados")

top_points = df.head(8).copy()
top_points["Prob. exacta %"] = top_points["Prob. exacta %"].map(lambda x: f"{x:.2f}%")
top_points["Puntos esperados"] = top_points["Puntos esperados"].map(lambda x: f"{x:.2f}")

st.dataframe(top_points[["Marcador", "Puntos esperados", "Prob. exacta %"]], hide_index=True, use_container_width=True)

st.divider()
st.subheader("Guardar pronóstico antes del partido")

current_key = match_key(match_date, team_a, team_b)
results_df = load_prediction_results()
if not results_df.empty:
    saved_mask = (
        results_df["date"].astype(str).eq(match_date)
        & results_df["home"].astype(str).eq(team_a)
        & results_df["away"].astype(str).eq(team_b)
        & results_df["source"].astype(str).eq("live_prediction")
    )
else:
    saved_mask = pd.Series(dtype=bool)

already_saved = bool(saved_mask.any()) if len(saved_mask) else False

if already_saved:
    st.success("Este partido ya tiene un pronóstico guardado en prediction_results.csv.")
else:
    st.caption("Guarda el pronóstico antes de que se juegue. Luego, cuando termine el partido, podrás calcular los puntos reales.")

if st.button("Guardar pronóstico actual", disabled=already_saved):
    new_prediction = {
        "date": match_date,
        "group": match_group,
        "home": team_a,
        "away": team_b,
        "stage": match_stage,
        "xg_home": round(float(xg_a), 2),
        "xg_away": round(float(xg_b), 2),
        "style": match_style,
        "xg_source": xg_source,
        "recommended_home": recommended_home,
        "recommended_away": recommended_away,
        "recommended_score": f"{recommended_home}-{recommended_away}",
        "recommended_points_expected": round(float(smart_row["Puntos esperados"]), 2),
        "recommended_exact_prob_pct": round(float(smart_row["Prob. exacta %"]), 2),
        "conservative_home": conservative_home,
        "conservative_away": conservative_away,
        "conservative_score": f"{conservative_home}-{conservative_away}",
        "conservative_points_expected": round(float(conservative_score["Puntos esperados"]), 2),
        "conservative_exact_prob_pct": round(float(conservative_score["Prob. exacta %"]), 2),
        "decision_gap": decision_gap["gap"] if decision_gap else None,
        "confidence": confidence,
        "risk": risk,
        "real_home": pd.NA,
        "real_away": pd.NA,
        "real_score": pd.NA,
        "points_recommended": pd.NA,
        "points_conservative": pd.NA,
        "recommended_exact_hit": pd.NA,
        "recommended_winner_hit": pd.NA,
        "recommended_goal_home_hit": pd.NA,
        "recommended_goal_away_hit": pd.NA,
        "recommended_diff_hit": pd.NA,
        "conservative_exact_hit": pd.NA,
        "conservative_winner_hit": pd.NA,
        "conservative_goal_home_hit": pd.NA,
        "conservative_goal_away_hit": pd.NA,
        "conservative_diff_hit": pd.NA,
        "multiplier": get_round_multiplier(match_stage),
        "status": "pending",
        "source": "live_prediction",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluated_at": pd.NA,
    }

    results_df = pd.concat([results_df, pd.DataFrame([new_prediction])], ignore_index=True)
    save_prediction_results(results_df)
    st.success("Pronóstico guardado en prediction_results.csv")
    st.rerun()

st.divider()
st.subheader("Registrar resultado real y calcular puntos")

results_df = load_prediction_results()
pending_results = results_df[results_df["status"].astype(str).eq("pending")].copy() if not results_df.empty else pd.DataFrame()

if pending_results.empty:
    st.info("No hay pronósticos pendientes para evaluar.")
else:
    pending_results["label"] = (
        pending_results["date"].astype(str)
        + " | "
        + pending_results["home"].astype(str)
        + " vs "
        + pending_results["away"].astype(str)
        + " | Pronóstico: "
        + pending_results["recommended_score"].astype(str)
    )

    selected_pending_label = st.selectbox("Pronóstico pendiente", pending_results["label"].tolist())
    selected_pending = pending_results[pending_results["label"] == selected_pending_label].iloc[0]

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        real_home = st.number_input(f"Goles reales {selected_pending['home']}", min_value=0, max_value=20, step=1, key="real_home_eval")
    with res_col2:
        real_away = st.number_input(f"Goles reales {selected_pending['away']}", min_value=0, max_value=20, step=1, key="real_away_eval")

    also_save_history = st.checkbox("También guardar este resultado en team_match_history.csv", value=True)

    if st.button("Calcular puntos y cerrar partido"):
        idx = selected_pending.name

        rec_points = calculate_poll_points(
            selected_pending["recommended_home"],
            selected_pending["recommended_away"],
            real_home,
            real_away,
            selected_pending["stage"],
            points_winner,
            points_goals,
            points_diff,
        )

        cons_points = calculate_poll_points(
            selected_pending["conservative_home"],
            selected_pending["conservative_away"],
            real_home,
            real_away,
            selected_pending["stage"],
            points_winner,
            points_goals,
            points_diff,
        )

        results_df = normalize_results_dtypes(results_df)

        results_df.loc[idx, "real_home"] = int(real_home)
        results_df.loc[idx, "real_away"] = int(real_away)
        results_df.loc[idx, "real_score"] = f"{int(real_home)}-{int(real_away)}"
        results_df.loc[idx, "points_recommended"] = rec_points["points"]
        results_df.loc[idx, "points_conservative"] = cons_points["points"]
        results_df.loc[idx, "recommended_exact_hit"] = rec_points["exact_hit"]
        results_df.loc[idx, "recommended_winner_hit"] = rec_points["winner_hit"]
        results_df.loc[idx, "recommended_goal_home_hit"] = rec_points["goal_home_hit"]
        results_df.loc[idx, "recommended_goal_away_hit"] = rec_points["goal_away_hit"]
        results_df.loc[idx, "recommended_diff_hit"] = rec_points["diff_hit"]
        results_df.loc[idx, "conservative_exact_hit"] = cons_points["exact_hit"]
        results_df.loc[idx, "conservative_winner_hit"] = cons_points["winner_hit"]
        results_df.loc[idx, "conservative_goal_home_hit"] = cons_points["goal_home_hit"]
        results_df.loc[idx, "conservative_goal_away_hit"] = cons_points["goal_away_hit"]
        results_df.loc[idx, "conservative_diff_hit"] = cons_points["diff_hit"]
        results_df.loc[idx, "multiplier"] = rec_points["multiplier"]
        results_df.loc[idx, "status"] = "evaluated"
        results_df.loc[idx, "evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_prediction_results(results_df)

        if also_save_history:
            old_history = pd.read_csv(HISTORY_PATH)
            old_history["date"] = pd.to_datetime(old_history["date"], errors="coerce").dt.date
            result_date = pd.to_datetime(selected_pending["date"]).date()

            duplicate_history = (
                (
                    (old_history["team"] == selected_pending["home"])
                    & (old_history["opponent"] == selected_pending["away"])
                    & (old_history["date"] == result_date)
                )
                |
                (
                    (old_history["team"] == selected_pending["away"])
                    & (old_history["opponent"] == selected_pending["home"])
                    & (old_history["date"] == result_date)
                )
            ).any()

            if not duplicate_history:
                new_rows = pd.DataFrame([
                    {
                        "team": selected_pending["home"],
                        "date": result_date,
                        "opponent": selected_pending["away"],
                        "gf": int(real_home),
                        "ga": int(real_away),
                    },
                    {
                        "team": selected_pending["away"],
                        "date": result_date,
                        "opponent": selected_pending["home"],
                        "gf": int(real_away),
                        "ga": int(real_home),
                    },
                ])
                updated_history = pd.concat([old_history, new_rows], ignore_index=True)
                updated_history.to_csv(HISTORY_PATH, index=False)
                st.success("Resultado guardado también en team_match_history.csv")
            else:
                st.warning("El partido ya existía en team_match_history.csv. No se duplicó.")

        st.success(
            f"Partido evaluado. Recomendado: {rec_points['points']} pts | "
            f"Conservador: {cons_points['points']} pts."
        )
        st.rerun()

st.divider()
st.subheader("Rendimiento del predictor")

results_df = load_prediction_results()
evaluated_df = results_df[results_df["status"].astype(str).eq("evaluated")].copy() if not results_df.empty else pd.DataFrame()

if evaluated_df.empty:
    st.info("Todavía no hay partidos evaluados en prediction_results.csv.")
else:
    evaluated_df["points_recommended"] = pd.to_numeric(evaluated_df["points_recommended"], errors="coerce")
    evaluated_df["points_conservative"] = pd.to_numeric(evaluated_df["points_conservative"], errors="coerce")
    backtest_summary = summarize_backtest(evaluated_df)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Partidos evaluados", backtest_summary["matches"])
    metric_col2.metric("Prom. recomendado", f"{backtest_summary['avg_recommended']:.2f}")
    metric_col3.metric("Prom. conservador", f"{backtest_summary['avg_conservative']:.2f}")
    metric_col4.metric("Exactos recomendado", f"{backtest_summary['exact_rate']:.1f}%")

    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
    quality_col1.metric("Ganador/empate", f"{backtest_summary['winner_rate']:.1f}%")
    quality_col2.metric("Gol local", f"{backtest_summary['home_goal_rate']:.1f}%")
    quality_col3.metric("Gol visitante", f"{backtest_summary['away_goal_rate']:.1f}%")
    quality_col4.metric("Diferencia", f"{backtest_summary['diff_rate']:.1f}%")

    st.write("**Lecturas del modelo**")
    for learning in find_model_learnings(evaluated_df):
        st.info(learning)

    st.write("**Detalle evaluado**")
    st.dataframe(
        evaluated_df[[
            "date",
            "home",
            "away",
            "stage",
            "recommended_score",
            "conservative_score",
            "real_score",
            "points_recommended",
            "points_conservative",
            "decision_gap",
            "confidence",
            "risk",
            "style",
            "xg_source",
        ]],
        hide_index=True,
        use_container_width=True,
    )

    st.write("**Desempeño por estilo**")
    st.dataframe(group_backtest(evaluated_df, "style"), hide_index=True, use_container_width=True)

    st.write("**Desempeño por fuente xG**")
    st.dataframe(group_backtest(evaluated_df, "xg_source"), hide_index=True, use_container_width=True)

    st.write("**Desempeño por riesgo**")
    st.dataframe(group_backtest(evaluated_df, "risk"), hide_index=True, use_container_width=True)

results_csv = load_prediction_results().to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Descargar prediction_results.csv",
    data=results_csv,
    file_name="prediction_results.csv",
    mime="text/csv",
)

with st.expander("Zona de pruebas"):
    st.caption("Úsalo solo si estás haciendo ensayos y quieres borrar los resultados de prueba.")
    if st.button("Reiniciar prediction_results.csv"):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULTS_PATH, index=False)
        st.cache_data.clear()
        st.success("prediction_results.csv reiniciado.")
        st.rerun()

st.divider()
st.subheader("Pronósticos para todos los partidos cargados")

all_rows = []

for _, match in matches.iterrows():
    home = match["home"]
    away = match["away"]

    elo_use_home, elo_use_away, _, _ = estimate_xg_from_elo(home, away)
    form_xg = estimate_xg_from_form(home, away)

    if form_xg:
        form_use_home, form_use_away = form_xg
        form_home = get_team_form(home)
        form_away = get_team_form(away)
        use_xg_home, use_xg_away, batch_form_weight = blend_elo_form_xg(
            elo_use_home,
            elo_use_away,
            form_use_home,
            form_use_away,
            form_home["games"] if form_home else 0,
            form_away["games"] if form_away else 0,
        )
        source = f"Elo + Forma CSV ({batch_form_weight:.0%})"
    else:
        use_xg_home, use_xg_away = elo_use_home, elo_use_away
        source = "Elo"

    df_all, real_df_all = calculate_matrix(
        home,
        away,
        use_xg_home,
        use_xg_away,
        max_goals,
        points_winner,
        points_goals,
        points_diff,
    )

    best = df_all.iloc[0]
    likely = df_all.sort_values("Prob. exacta", ascending=False).iloc[0]
    summary_all = summarize_probabilities(real_df_all)
    conf_all, risk_all, _, _ = calculate_confidence(df_all.head(8))
    gap_all = get_decision_gap(df_all)
    conservative_all = get_conservative_score(df_all)
    strategies_all = get_strategy_picks(df_all)
    differential_all = {
        "Marcador": best["Marcador"],
        "Ventaja estratégica": pd.NA,
    }
    if not strategies_all.empty and "Estrategia" in strategies_all.columns:
        differential_candidates = strategies_all[strategies_all["Estrategia"] == "Diferencial"]
        if not differential_candidates.empty:
            differential_all = differential_candidates.iloc[0]
    draw_cover_all = get_draw_cover_pick(df_all)
    uncertainty_all = run_uncertainty_scenarios(
        home,
        away,
        use_xg_home,
        use_xg_away,
        max_goals,
        points_winner,
        points_goals,
        points_diff,
    )

    all_rows.append({
        "Fecha": match["date"],
        "Grupo": match["group"],
        "Partido": f"{home} vs {away}",
        "Fuente xG": source,
        "xG A": use_xg_home,
        "xG B": use_xg_away,
        "Marcador recomendado": best["Marcador"],
        "Marcador conservador": conservative_all["Marcador"] if conservative_all else "",
        "Pick diferencial": differential_all["Marcador"],
        "Ventaja estratégica diferencial": differential_all["Ventaja estratégica"],
        "Plan B empate": draw_cover_all["Marcador"] if draw_cover_all else "",
        "Nivel empate": draw_cover_all["Nivel"] if draw_cover_all else "",
        "Gap empate": draw_cover_all["Gap al mejor"] if draw_cover_all else None,
        "Prob. empate": f"{draw_cover_all['Prob. empate %']:.1f}%" if draw_cover_all else "",
        "Estabilidad": uncertainty_all["label"],
        "Consenso escenarios": f"{uncertainty_all['stability'] * 100:.0f}%",
        "Puntos esperados": round(best["Puntos esperados"], 2),
        "Marcador más probable": likely["Marcador"],
        "Prob. exacta": f"{likely['Prob. exacta %']:.2f}%",
        "Diferencia de decisión": gap_all["gap"] if gap_all else None,
        "Confianza": f"{conf_all}/10",
        "Riesgo": risk_all,
        "Victoria A": f"{summary_all['Victoria A'] * 100:.1f}%",
        "Empate": f"{summary_all['Empate'] * 100:.1f}%",
        "Victoria B": f"{summary_all['Victoria B'] * 100:.1f}%",
    })

all_predictions = pd.DataFrame(all_rows)
st.dataframe(all_predictions, hide_index=True, use_container_width=True)

csv_export = all_predictions.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Descargar pronósticos en CSV",
    data=csv_export,
    file_name="pronosticos_polla_mundial.csv",
    mime="text/csv",
)

st.caption(
    "Nota: el modelo depende de los xG estimados. Puedes usar Elo, forma reciente CSV o valores manuales según el contexto del partido."
)

st.divider()
st.subheader("Registrar resultado para historial")

with st.expander("Agregar partido jugado al historial"):
    teams_list = sorted(set(matches["home"]).union(set(matches["away"])))

    col1, col2 = st.columns(2)

    with col1:
        hist_team_a = st.selectbox("Equipo A", teams_list, key="hist_team_a")
        hist_goals_a = st.number_input("Goles Equipo A", min_value=0, max_value=20, step=1, key="hist_goals_a")

    with col2:
        hist_team_b = st.selectbox("Equipo B", teams_list, key="hist_team_b")
        hist_goals_b = st.number_input("Goles Equipo B", min_value=0, max_value=20, step=1, key="hist_goals_b")

    hist_date = st.date_input("Fecha del partido", key="hist_date")

    if st.button("Guardar en historial"):
        if hist_team_a == hist_team_b:
            st.error("Equipo A y Equipo B no pueden ser el mismo.")
        else:
            old_history = pd.read_csv(HISTORY_PATH)
            old_history["date"] = pd.to_datetime(old_history["date"], errors="coerce").dt.date

            already_exists = (
                (
                    (old_history["team"] == hist_team_a)
                    & (old_history["opponent"] == hist_team_b)
                    & (old_history["date"] == hist_date)
                )
                |
                (
                    (old_history["team"] == hist_team_b)
                    & (old_history["opponent"] == hist_team_a)
                    & (old_history["date"] == hist_date)
                )
            ).any()

            if already_exists:
                st.warning("Este partido ya parece estar registrado en el historial. No se duplicó.")
            else:
                new_rows = pd.DataFrame([
                    {
                        "team": hist_team_a,
                        "date": hist_date,
                        "opponent": hist_team_b,
                        "gf": int(hist_goals_a),
                        "ga": int(hist_goals_b),
                    },
                    {
                        "team": hist_team_b,
                        "date": hist_date,
                        "opponent": hist_team_a,
                        "gf": int(hist_goals_b),
                        "ga": int(hist_goals_a),
                    },
                ])

                updated_history = pd.concat([old_history, new_rows], ignore_index=True)
                updated_history.to_csv(HISTORY_PATH, index=False)

                st.cache_data.clear()
                st.success("Resultado guardado en team_match_history.csv")

