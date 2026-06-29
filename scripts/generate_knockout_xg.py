import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP = ROOT / "data" / "worldcup2026.csv"
HISTORY = ROOT / "data" / "team_match_history.csv"

# Ajuste conservador para eliminación directa
KO_FACTOR = 0.92

def clamp(value, low=0.35, high=2.75):
    return max(low, min(high, value))

def estimate_xg(elo_home, elo_away, base_total=2.35):
    diff = elo_home - elo_away
    share_home = 1 / (1 + 10 ** (-diff / 400))
    xg_home = base_total * share_home
    xg_away = base_total * (1 - share_home)
    return clamp(xg_home * KO_FACTOR), clamp(xg_away * KO_FACTOR)

# Ajusta este diccionario con tus Elo actuales
TEAM_ELO = {
    "Argentina": 1850,
    "Brasil": 1810,
    "Francia": 1830,
    "Inglaterra": 1800,
    "Portugal": 1780,
    "España": 1790,
    "Alemania": 1760,
    "Países Bajos": 1740,
    "Colombia": 1710,
    "Croacia": 1690,
    "Marruecos": 1680,
    "Japón": 1660,
    "Canadá": 1630,
    "Estados Unidos": 1660,
    "Bosnia-Herzegovina": 1580,
    "Cabo Verde": 1500,
    "Sudáfrica": 1510,
    "Paraguay": 1600,
    "Costa de Marfil": 1610,
    "Noruega": 1660,
    "Suecia": 1650,
}

def main():
    df = pd.read_csv(WORLD_CUP)

    mask = df["stage"].str.lower().eq("16avos")

    for idx, row in df[mask].iterrows():
        home = row["home"]
        away = row["away"]

        elo_home = TEAM_ELO.get(home, 1550)
        elo_away = TEAM_ELO.get(away, 1550)

        xg_home, xg_away = estimate_xg(elo_home, elo_away)

        df.loc[idx, "xg_home"] = round(xg_home, 2)
        df.loc[idx, "xg_away"] = round(xg_away, 2)

    df.to_csv(WORLD_CUP, index=False, encoding="utf-8")
    print("worldcup2026.csv actualizado con xG automáticos para 16avos.")

if __name__ == "__main__":
    main()