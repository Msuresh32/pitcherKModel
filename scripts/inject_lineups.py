"""
Inject confirmed daily lineups into today_lineups.csv.
Usage: python scripts/inject_lineups.py --date 2026-06-15
Lineups are passed as a hardcoded dict keyed by numeric team_id -> list of names in batting order.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Name corrections applied AFTER accent-stripping: screenshot name -> name in batter_game_logs (both accent-stripped)
NAME_FIXES = {
    "bobby witt": "bobby witt jr.",
    "adam bogaerts": "xander bogaerts",
    "donnie walton": "donovan walton",
    "c.j. abrams": "cj abrams",
    "t.j. rumfield": "tj rumfield",
    "j.j. bleday": "jj bleday",
}


def normalize(name: str) -> str:
    """Lowercase, strip, and remove accent marks so 'Suárez' == 'Suarez'."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def build_lookup(batter_logs_path: str) -> pd.DataFrame:
    df = pd.read_csv(batter_logs_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    lookup = df.drop_duplicates(subset="batter_name", keep="last")[
        ["batter_name", "batter_id", "bat_side"]
    ].copy()
    lookup["name_key"] = lookup["batter_name"].apply(normalize)
    return lookup


def resolve_id(name: str, lookup: pd.DataFrame) -> tuple:
    key = normalize(name)
    key = NAME_FIXES.get(key, key)
    row = lookup[lookup["name_key"] == key]
    if not row.empty:
        r = row.iloc[0]
        return int(r["batter_id"]), r["bat_side"]
    # Partial match on last name + first initial
    parts = key.split()
    if len(parts) >= 2:
        partial = lookup[
            lookup["name_key"].str.startswith(parts[0])
            & lookup["name_key"].str.contains(parts[-1])
        ]
        if len(partial) == 1:
            r = partial.iloc[0]
            return int(r["batter_id"]), r["bat_side"]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--batter-logs", default="data/raw/batter_game_logs.csv")
    parser.add_argument("--output", default="data/raw/today_lineups.csv")
    args = parser.parse_args()

    # team_id (numeric, matches probable_pitchers) -> batting order list
    LINEUPS: dict[int, list[str]] = {
        108: ["Zach Neto", "Mike Trout", "Wade Meckler", "Jo Adell", "Nolan Schanuel",
              "Oswald Peraza", "Denzer Guzman", "Donnie Walton", "Logan O'Hoppe"],          # LAA vs ARI
        109: ["Ketel Marte", "Corbin Carroll", "Gabriel Moreno", "Nolan Arenado", "Pavin Smith",
              "Geraldo Perdomo", "Tommy Troy", "Adrian Del Castillo", "Ryan Waldschmidt"],   # ARI vs LAA
        112: ["Pete Crow-Armstrong", "Alex Bregman", "Michael Busch", "Seiya Suzuki", "Ian Happ",
              "Nico Hoerner", "Pedro Ramirez", "Carson Kelly", "Dansby Swanson"],            # CHC vs COL
        113: ["Edwin Arroyo", "JJ Bleday", "Sal Stewart", "Nathaniel Lowe", "Spencer Steer",
              "Eugenio Suarez", "Noelvi Marte", "Tyler Stephenson", "Matt McLain"],          # CIN vs NYM
        115: ["Willi Castro", "Kyle Karros", "T.J. Rumfield", "Hunter Goodman", "Troy Johnston",
              "Cole Carrigg", "Sterlin Thompson", "Braxton Fulford", "Edouard Julien"],      # COL vs CHC
        116: ["Kevin McGonigle", "Gleyber Torres", "Riley Greene", "Dillon Dingler", "Kerry Carpenter",
              "Colt Keith", "Spencer Torkelson", "Zach McKinstry", "James Outman"],          # DET vs HOU
        117: ["Jeremy Pena", "Yordan Alvarez", "Christian Walker", "Isaac Paredes", "Jose Altuve",
              "Joey Loperfido", "Cam Smith", "Taylor Trammell", "Christian Vazquez"],        # HOU vs DET
        118: ["Lane Thomas", "Bobby Witt", "Maikel Garcia", "Carter Jensen", "John Rave",
              "Starling Marte", "Jac Caglianone", "Nick Loftin", "Isaac Collins"],           # KCR vs WAS
        119: ["Shohei Ohtani", "Andy Pages", "Freddie Freeman", "Mookie Betts", "Max Muncy",
              "Kyle Tucker", "Ryan Ward", "Dalton Rushing", "Alex Freeland"],               # LAD batting vs TBR
        120: ["James Wood", "Luis Garcia Jr.", "Curtis Mead", "C.J. Abrams", "Daylen Lile",
              "Dylan Crews", "Jose Tena", "Keibert Ruiz", "Nasim Nunez"],                   # WAS vs KCR
        121: ["Carson Benge", "Bo Bichette", "Juan Soto", "Jared Young", "A.J. Ewing",
              "Marcus Semien", "Brett Baty", "MJ Melendez", "Francisco Alvarez"],            # NYM vs CIN
        133: ["Nick Kurtz", "Tyler Soderstrom", "Shea Langeliers", "Carlos Cortes", "Zack Gelof",
              "Lawrence Butler", "Henry Bolte", "Jeff McNeil", "Alika Williams"],            # ATH vs PIT
        134: ["Spencer Horwitz", "Brandon Lowe", "Bryan Reynolds", "Ryan O'Hearn", "Nick Gonzales",
              "Henry Davis", "Tyler Callihan", "Jake Mangum", "Jared Triolo"],               # PIT vs ATH
        135: ["Fernando Tatis", "Jackson Merrill", "Manny Machado", "Adam Bogaerts", "Gavin Sheets",
              "Samad Taylor", "Will Wagner", "Jase Bowen", "Rodolfo Duran"],                 # SDP vs STL
        138: ["JJ Wetherholt", "Ivan Herrera", "Alec Burleson", "Jordan Walker", "Lars Nootbaar",
              "Masyn Winn", "Jimmy Crooks", "Blaze Jordan", "Nathan Church"],                # STL vs SDP
        139: ["Yandy Diaz", "Jonathan Aranda", "Junior Caminero", "Ryan Vilade", "Austin Slater",
              "Ben Williamson", "Chandler Simpson", "Nick Fortes", "Taylor Walls"],          # TBR batting vs LAD
        140: ["Joc Pederson", "Josh Jung", "Brandon Nimmo", "Wyatt Langford", "Ezequiel Duran",
              "Alejandro Osuna", "Jake Burger", "Kyle Higashioka", "Nicky Lopez"],           # TEX batting vs MIN
        142: ["Austin Martin", "Byron Buxton", "Kody Clemens", "Royce Lewis", "Brooks Lee",
              "Orlando Arcia", "Ryan Kreidler", "Luke Keaschall", "Victor Caratini"],        # MIN batting vs TEX
        143: ["Kyle Schwarber", "Trea Turner", "Bryce Harper", "Brandon Marsh", "Alec Bohm",
              "Bryson Stott", "Gabriel Rincones", "J.T. Realmuto", "Justin Crawford"],       # PHI vs MIA
        146: ["Joe Mack", "Otto Lopez", "Xavier Edwards", "Kyle Stowers", "Heriberto Hernandez",
              "Leo Jimenez", "Jakob Marsee", "Javier Sanoja", "Connor Norby"],              # MIA vs PHI
    }

    lookup = build_lookup(args.batter_logs)
    rows = []
    misses = []

    for team_id, names in LINEUPS.items():
        for slot, name in enumerate(names, start=1):
            batter_id, bat_side = resolve_id(name, lookup)
            if batter_id is None:
                misses.append(f"  team={team_id} slot={slot}: '{name}'")
                continue
            rows.append({
                "game_date": args.date,
                "batter_id": batter_id,
                "team": team_id,
                "batting_order": slot * 100,
                "bat_side": bat_side,
                "plate_appearances": 0,
                "hits": 0,
                "walks": 0,
                "strikeouts": 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} lineup rows for {args.date} -> {args.output}")
    if misses:
        print(f"Could not resolve {len(misses)} names:")
        for m in misses:
            print(m)
    else:
        print("All names resolved.")


if __name__ == "__main__":
    main()
