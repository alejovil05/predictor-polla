import pandas as pd

df = pd.read_csv("data/team_match_history.csv")

conteo = df.groupby("team").size().sort_values()

print("\n=== RESUMEN ===")
print(f"Equipos únicos: {len(conteo)}")

print("\n=== CONTEO POR EQUIPO ===")

for team, n in conteo.items():
    marca = "✅" if n == 10 else "⚠️"
    print(f"{marca} {team}: {n}")