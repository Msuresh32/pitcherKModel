import re
import unicodedata


TEAM_ALIASES = {
    "ARI": "AZ",
    "ARIZONA DIAMONDBACKS": "AZ",
    "AZ": "AZ",
    "ATL": "ATL",
    "ATLANTA BRAVES": "ATL",
    "BAL": "BAL",
    "BALTIMORE ORIOLES": "BAL",
    "BOS": "BOS",
    "BOSTON RED SOX": "BOS",
    "CHC": "CHC",
    "CHICAGO CUBS": "CHC",
    "CWS": "CWS",
    "CHW": "CWS",
    "CHICAGO WHITE SOX": "CWS",
    "CIN": "CIN",
    "CINCINNATI REDS": "CIN",
    "CLE": "CLE",
    "CLEVELAND GUARDIANS": "CLE",
    "COL": "COL",
    "COLORADO ROCKIES": "COL",
    "DET": "DET",
    "DETROIT TIGERS": "DET",
    "HOU": "HOU",
    "HOUSTON ASTROS": "HOU",
    "KC": "KC",
    "KCR": "KC",
    "KANSAS CITY ROYALS": "KC",
    "LAA": "LAA",
    "LOS ANGELES ANGELS": "LAA",
    "LAD": "LAD",
    "LOS ANGELES DODGERS": "LAD",
    "MIA": "MIA",
    "MIAMI MARLINS": "MIA",
    "MIL": "MIL",
    "MILWAUKEE BREWERS": "MIL",
    "MIN": "MIN",
    "MINNESOTA TWINS": "MIN",
    "NYM": "NYM",
    "NEW YORK METS": "NYM",
    "NYY": "NYY",
    "NEW YORK YANKEES": "NYY",
    "ATH": "ATH",
    "OAK": "ATH",
    "ATHLETICS": "ATH",
    "OAKLAND ATHLETICS": "ATH",
    "SACRAMENTO ATHLETICS": "ATH",
    "A'S": "ATH",
    "AS": "ATH",
    "PHI": "PHI",
    "PHILADELPHIA PHILLIES": "PHI",
    "PIT": "PIT",
    "PITTSBURGH PIRATES": "PIT",
    "SD": "SD",
    "SDP": "SD",
    "SAN DIEGO PADRES": "SD",
    "SF": "SF",
    "SFG": "SF",
    "SAN FRANCISCO GIANTS": "SF",
    "SEA": "SEA",
    "SEATTLE MARINERS": "SEA",
    "STL": "STL",
    "ST. LOUIS CARDINALS": "STL",
    "ST LOUIS CARDINALS": "STL",
    "TB": "TB",
    "TBR": "TB",
    "TAMPA BAY RAYS": "TB",
    "TEX": "TEX",
    "TEXAS RANGERS": "TEX",
    "TOR": "TOR",
    "TORONTO BLUE JAYS": "TOR",
    "WSH": "WSH",
    "WAS": "WSH",
    "WASHINGTON NATIONALS": "WSH",
}


TEAM_FULL_NAME = {
    "AZ": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}


VENUE_META = {
    "American Family Field": {"lat": 43.0280, "lon": -87.9712, "park_factor": 100.0, "elevation_ft": 593, "roof": "retractable", "shadow_key": "American Family Field"},
    "Angel Stadium": {"lat": 33.8003, "lon": -117.8827, "park_factor": 98.0, "elevation_ft": 160, "roof": "open", "shadow_key": "Angel Stadium"},
    "Busch Stadium": {"lat": 38.6226, "lon": -90.1928, "park_factor": 100.0, "elevation_ft": 455, "roof": "open", "shadow_key": "Busch Stadium"},
    "Chase Field": {"lat": 33.4455, "lon": -112.0667, "park_factor": 100.0, "elevation_ft": 1086, "roof": "retractable", "shadow_key": "Chase Field"},
    "Citi Field": {"lat": 40.7571, "lon": -73.8458, "park_factor": 100.0, "elevation_ft": 13, "roof": "open", "shadow_key": "Citi Field"},
    "Citizens Bank Park": {"lat": 39.9061, "lon": -75.1665, "park_factor": 97.0, "elevation_ft": 15, "roof": "open", "shadow_key": "Citizens Bank Park"},
    "Comerica Park": {"lat": 42.3390, "lon": -83.0485, "park_factor": 100.0, "elevation_ft": 600, "roof": "open", "shadow_key": "Comerica Park"},
    "Coors Field": {"lat": 39.7559, "lon": -104.9942, "park_factor": 85.0, "elevation_ft": 5200, "roof": "open", "shadow_key": "Coors Field"},
    "Daikin Park": {"lat": 29.7569, "lon": -95.3555, "park_factor": 100.0, "elevation_ft": 38, "roof": "retractable", "shadow_key": "Daikin Park"},
    "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "park_factor": 100.0, "elevation_ft": 267, "roof": "open", "shadow_key": "Dodger Stadium"},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "park_factor": 99.0, "elevation_ft": 20, "roof": "open", "shadow_key": "Fenway Park"},
    "Globe Life Field": {"lat": 32.7511, "lon": -97.0825, "park_factor": 99.0, "elevation_ft": 600, "roof": "retractable", "shadow_key": "Globe Life Field"},
    "Great American Ball Park": {"lat": 39.0979, "lon": -84.5072, "park_factor": 96.0, "elevation_ft": 683, "roof": "open", "shadow_key": "Great American Ball Park"},
    "Guaranteed Rate Field": {"lat": 41.8299, "lon": -87.6338, "park_factor": 99.0, "elevation_ft": 592, "roof": "open", "shadow_key": "Guaranteed Rate Field"},
    "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803, "park_factor": 100.0, "elevation_ft": 750, "roof": "open", "shadow_key": "Kauffman Stadium"},
    "loanDepot park": {"lat": 25.7781, "lon": -80.2197, "park_factor": 100.0, "elevation_ft": 10, "roof": "retractable", "shadow_key": "loanDepot park"},
    "Minute Maid Park": {"lat": 29.7569, "lon": -95.3555, "park_factor": 100.0, "elevation_ft": 38, "roof": "retractable", "shadow_key": "Daikin Park"},
    "Nationals Park": {"lat": 38.8730, "lon": -77.0074, "park_factor": 100.0, "elevation_ft": 25, "roof": "open", "shadow_key": "Nationals Park"},
    "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "park_factor": 102.0, "elevation_ft": 15, "roof": "open", "shadow_key": "Oracle Park"},
    "Oriole Park at Camden Yards": {"lat": 39.2840, "lon": -76.6215, "park_factor": 100.0, "elevation_ft": 130, "roof": "open", "shadow_key": "Oriole Park at Camden Yards"},
    "Petco Park": {"lat": 32.7076, "lon": -117.1570, "park_factor": 105.0, "elevation_ft": 13, "roof": "open", "shadow_key": "Petco Park"},
    "PNC Park": {"lat": 40.4469, "lon": -80.0057, "park_factor": 100.0, "elevation_ft": 743, "roof": "open", "shadow_key": "PNC Park"},
    "Progressive Field": {"lat": 41.4962, "lon": -81.6852, "park_factor": 100.0, "elevation_ft": 682, "roof": "open", "shadow_key": "Progressive Field"},
    "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "park_factor": 100.0, "elevation_ft": 250, "roof": "retractable", "shadow_key": "Rogers Centre"},
    "Sutter Health Park": {"lat": 38.5804, "lon": -121.5138, "park_factor": 100.0, "elevation_ft": 30, "roof": "open", "shadow_key": "Sutter Health Park"},
    "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "park_factor": 105.0, "elevation_ft": 15, "roof": "retractable", "shadow_key": "T-Mobile Park"},
    "Target Field": {"lat": 44.9817, "lon": -93.2778, "park_factor": 100.0, "elevation_ft": 840, "roof": "open", "shadow_key": "Target Field"},
    "Truist Park": {"lat": 33.8907, "lon": -84.4677, "park_factor": 100.0, "elevation_ft": 975, "roof": "open", "shadow_key": "Truist Park"},
    "Tropicana Field": {"lat": 27.7682, "lon": -82.6534, "park_factor": 100.0, "elevation_ft": 15, "roof": "fixed", "shadow_key": "Tropicana Field"},
    "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "park_factor": 100.0, "elevation_ft": 598, "roof": "open", "shadow_key": "Wrigley Field"},
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "park_factor": 100.0, "elevation_ft": 53, "roof": "open", "shadow_key": "Yankee Stadium"},
}


TEAM_HOME_VENUE = {
    "AZ": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CWS": "Guaranteed Rate Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "DET": "Comerica Park",
    "HOU": "Daikin Park",
    "KC": "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "ATH": "Sutter Health Park",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD": "Petco Park",
    "SF": "Oracle Park",
    "SEA": "T-Mobile Park",
    "STL": "Busch Stadium",
    "TB": "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
}


def _ascii(value):
    value = "" if value is None else str(value)
    value = value.replace("’", "'").replace("`", "'")
    return unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("utf-8")


def normalize_name(name):
    value = _ascii(name).strip().lower()
    if "," in value:
        last, first = [part.strip() for part in value.split(",", 1)]
        value = f"{first} {last}"
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    parts = value.split("_") if value else []
    if parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts = parts[:-1]
    collapsed = []
    i = 0
    while i < len(parts):
        if parts[i].isalpha() and len(parts[i]) == 1:
            j = i
            initials = []
            while j < len(parts) and parts[j].isalpha() and len(parts[j]) == 1:
                initials.append(parts[j])
                j += 1
            if len(initials) > 1:
                collapsed.append("".join(initials))
            else:
                collapsed.append(parts[i])
            i = j
        else:
            collapsed.append(parts[i])
            i += 1
    parts = collapsed
    return "_".join(parts)


def normalize_team(team):
    if team is None:
        return None
    raw = _ascii(team).strip()
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.upper().replace(".", ""))
    return TEAM_ALIASES.get(key, TEAM_ALIASES.get(raw.upper(), raw.upper()))


def venue_for_home_team(team_code):
    return TEAM_HOME_VENUE.get(normalize_team(team_code))


def venue_meta(venue_name=None, home_team=None):
    if venue_name in VENUE_META:
        return venue_name, VENUE_META[venue_name]
    fallback = venue_for_home_team(home_team)
    if fallback in VENUE_META:
        return fallback, VENUE_META[fallback]
    return None, None
