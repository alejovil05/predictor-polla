import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

TEAM_ALIASES = {
    "Irak": "Iraq",
    "Noruega": "Norway",
    "Francia": "France",
    "Senegal": "Senegal",
    "Inglaterra": "England",
    "España": "Spain",
    "Alemania": "Germany",
    "Países Bajos": "Netherlands",
    "Estados Unidos": "USA",
    "Corea del Sur": "South Korea",
    "Arabia Saudita": "Saudi Arabia",
    "Japón": "Japan",
    "Marruecos": "Morocco",
    "Suiza": "Switzerland",
    "Suecia": "Sweden",
    "Turquía": "Turkey",
    "Portugal": "Portugal",
    "Argentina": "Argentina",
    "Brasil": "Brazil",
    "Colombia": "Colombia",
    "Uruguay": "Uruguay",
    "Bélgica": "Belgium",
    "Austria": "Austria",
    "Egipto": "Egypt",
    "Irán": "Iran",
    "Canadá": "Canada",
    "Qatar": "Qatar",
    "México": "Mexico",
    "Paraguay": "Paraguay",
    "Australia": "Australia",
    "Ghana": "Ghana",
    "Panamá": "Panama",
    "Túnez": "Tunisia",
    "Argelia": "Algeria",
    "Ecuador": "Ecuador",
    "Jordania": "Jordan",
    "Nueva Zelanda": "New Zealand",
    "Costa de Marfil": "Ivory Coast",
    "Bosnia y Herzegovina": "Bosnia",
    "RD Congo": "Congo DR",
    "Cabo Verde": "Cape Verde",
    "Curazao": "Curacao",
    "Haití": "Haiti",
    "Chequia": "Czech Republic",
    "Uzbekistán": "Uzbekistan"
}



def search_team(team_name):
    search_name = TEAM_ALIASES.get(team_name, team_name)

    url = f"{BASE_URL}/teams"
    params = {
        "search": search_name
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    data = response.json()

    teams = data.get("response", [])

    if not teams:
        print("No encontrado:", search_name)
        print(data)
        return None

    national_teams = [
        item["team"] for item in teams
        if item["team"].get("national") is True
    ]

    if national_teams:
        return national_teams[0]

    print("Encontró equipos, pero ninguno nacional:", search_name)
    print(teams[:5])
    return teams[0]["team"]


def get_last_fixtures(team_id, limit=10):
    url = f"{BASE_URL}/fixtures"
    params = {
        "team": team_id,
        "last": limit
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    data = response.json()

    return data.get("response", [])


def summarize_recent_form(team_name, limit=10):
    team = search_team(team_name)

    if team is None:
        return {
            "team": team_name,
            "found": False,
            "message": "Equipo no encontrado en API-Football"
        }

    fixtures = get_last_fixtures(team["id"], limit)

    goals_for = 0
    goals_against = 0
    wins = 0
    draws = 0
    losses = 0
    played = 0

    for fixture in fixtures:
        home = fixture["teams"]["home"]
        away = fixture["teams"]["away"]
        goals = fixture["goals"]

        if goals["home"] is None or goals["away"] is None:
            continue

        played += 1

        is_home = home["id"] == team["id"]

        if is_home:
            gf = goals["home"]
            ga = goals["away"]
        else:
            gf = goals["away"]
            ga = goals["home"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    if played == 0:
        return {
            "team": team_name,
            "found": True,
            "team_id": team["id"],
            "played": 0,
            "message": "No hay partidos recientes con marcador"
        }

    return {
        "team": team_name,
        "found": True,
        "team_id": team["id"],
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "avg_goals_for": round(goals_for / played, 2),
        "avg_goals_against": round(goals_against / played, 2)
    }

def adjusted_xg_from_form(team_a, team_b):

    form_a = summarize_recent_form(team_a, limit=10)
    form_b = summarize_recent_form(team_b, limit=10)

    if not form_a.get("found") or not form_b.get("found"):
        return None

    gf_a = form_a["avg_goals_for"]
    ga_a = form_a["avg_goals_against"]

    gf_b = form_b["avg_goals_for"]
    ga_b = form_b["avg_goals_against"]

    xg_a = (gf_a + ga_b) / 2
    xg_b = (gf_b + ga_a) / 2

    return {
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "form_a": form_a,
        "form_b": form_b
    }