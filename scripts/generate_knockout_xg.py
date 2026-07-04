import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP = ROOT / "data" / "worldcup2026.csv"

KO_FACTORS = {
    "16avos": 0.92,
    "Octavos": 0.88,
    "Cuartos": 0.86,
    "Semis": 0.84,
    "Final": 0.82,
}

def clamp(value, low=0.35, high=2.75):
    return max(low, min(high, value))

def estimate_xg(elo_home, elo_away, ko_factor=0.92, base_total=2.35):
    diff = elo_home - elo_away
    share_home = 1 / (1 + 10 ** (-diff / 400))

    xg_home = base_total * share_home * ko_factor
    xg_away = base_total * (1 - share_home) * ko_factor

    return clamp(xg_home), clamp(xg_away)

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
    "Egipto": 1645,
    "Suiza": 1685,
    "México": 1715,
    "Bélgica": 1775,
}

def main():
    df = pd.read_csv(WORLD_CUP)

    for stage, factor in KO_FACTORS.items():
        mask = df["stage"].str.lower().eq(stage.lower())

        for idx, row in df[mask].iterrows():
            home = row["home"]
            away = row["away"]

            elo_home = TEAM_ELO.get(home, 1550)
            elo_away = TEAM_ELO.get(away, 1550)

            xg_home, xg_away = estimate_xg(
                elo_home,
                elo_away,
                ko_factor=factor
            )

            df.loc[idx, "xg_home"] = round(xg_home, 2)
            df.loc[idx, "xg_away"] = round(xg_away, 2)

    df.to_csv(WORLD_CUP, index=False, encoding="utf-8")
    print("worldcup2026.csv actualizado con xG automáticos para fases KO.")

if __name__ == "__main__":
    main()