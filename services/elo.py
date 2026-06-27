import math

TEAM_ELO = {
    "Argentina": 2100,
    "Francia": 2045,
    "Inglaterra": 2020,
    "Portugal": 2010,
    "España": 1995,
    "Brasil": 1990,
    "Alemania": 1940,
    "Países Bajos": 1920,
    "Bélgica": 1885,
    "Croacia": 1860,
    "Noruega": 1850,
    "Colombia": 1835,
    "Uruguay": 1830,
    "Austria": 1810,
    "Senegal": 1775,
    "Marruecos": 1765,
    "Suiza": 1760,
    "Estados Unidos": 1745,
    "Japón": 1735,
    "Argelia": 1720,
    "México": 1715,
    "Costa de Marfil": 1705,
    "Turquía": 1695,
    "Suecia": 1685,
    "Irán": 1680,
    "Australia": 1660,
    "Ghana": 1650,
    "Egipto": 1645,
    "Corea del Sur": 1640,
    "Chequia": 1635,
    "Paraguay": 1625,
    "Bosnia y Herzegovina": 1615,
    "Ecuador": 1610,
    "Túnez": 1605,
    "RD Congo": 1580,
    "Uzbekistán": 1560,
    "Canadá": 1555,
    "Arabia Saudita": 1545,
    "Panamá": 1500,
    "Sudáfrica": 1495,
    "Cabo Verde": 1485,
    "Irak": 1460,
    "Qatar": 1455,
    "Jordania": 1440,
    "Nueva Zelanda": 1425,
    "Haití": 1390,
    "Curazao": 1360,
    "Escocia": 1710,
}


def clamp(value, min_value=0.25, max_value=3.25):
    return max(min_value, min(max_value, value))


def estimate_xg_from_elo(team_a, team_b):
    elo_a = TEAM_ELO.get(team_a, 1600)
    elo_b = TEAM_ELO.get(team_b, 1600)

    diff = elo_a - elo_b

    # Promedio razonable para partidos internacionales de torneo
    base_total_goals = 2.45

    # Escala más suave: evita inflar demasiado al favorito
    share_a = 1 / (1 + math.exp(-diff / 520))
    share_b = 1 - share_a

    xg_a = base_total_goals * share_a
    xg_b = base_total_goals * share_b

    # Corrección leve: equipos muy superiores sí deben tener piso ofensivo alto
    if diff >= 300:
        xg_a += 0.15
        xg_b -= 0.05
    elif diff <= -300:
        xg_b += 0.15
        xg_a -= 0.05

    xg_a = clamp(xg_a)
    xg_b = clamp(xg_b)

    return round(xg_a, 2), round(xg_b, 2), elo_a, elo_b