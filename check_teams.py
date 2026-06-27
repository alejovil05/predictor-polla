import pandas as pd

wc = pd.read_csv("data/worldcup2026.csv")
hist = pd.read_csv("data/team_match_history.csv")

worldcup_teams = set(wc["home"]).union(set(wc["away"]))
history_teams = set(hist["team"])

missing_in_history = sorted(worldcup_teams - history_teams)
extra_in_history = sorted(history_teams - worldcup_teams)

print("\nEquipos en World Cup pero NO en historial:\n")
if missing_in_history:
    for t in missing_in_history:
        print("-", t)
else:
    print("✅ Todos los equipos del World Cup están en historial")

print("\nEquipos en historial pero NO en World Cup:\n")
if extra_in_history:
    for t in extra_in_history:
        print("-", t)
else:
    print("✅ Todos los equipos del historial coinciden con World Cup")

covered = sorted(worldcup_teams & history_teams)

print("\nEquipos ya cubiertos:\n")
for t in covered:
    print("-", t)

total = len(worldcup_teams)
loaded = len(worldcup_teams & history_teams)

print(
    f"\nCobertura: {loaded}/{total} equipos "
    f"({loaded/total:.0%})"
)