import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import statistics
import itertools
import collections
import re
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import numba as nb
from random import shuffle

@nb.jit(nopython=True)  # nopython mode ensures the function is fully optimized
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2
    
def create_dummy_pitcher():
    # Create a default pitcher with average or placeholder stats
    return {
        "ID": "dummy",
        "Fpts": 1, 
        "StdDev": 1,  
        "Name": "Justin Case",
        "Position": ["P"],
    }

class MLB_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    gen_lineup_list = []
    roster_construction = []
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    cut_event = False
    entry_fee = None
    use_lineup_input = None
    max_hitters_per_team = None
    matchups = set()
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    seen_lineups = {}
    seen_lineups_ix = {}
    game_info = {}
    allow_opps = False
    id_name_dict = {}
    stacks_dict = {}

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        match_lineup_input_to_field_size,
    ):
        self.site = site
        self.max_stack_len = 5 if site == 'dk' else 4
        self.use_lineup_input = use_lineup_input
        self.match_lineup_input_to_field_size = match_lineup_input_to_field_size
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

        stacks_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["team_stacks_path"]),
        )
        self.load_team_stacks(stacks_path)

        #       batting_order_path = os.path.join(
        #           os.path.dirname(__file__),
        #            "../{}_data/{}".format(site, self.config["batting_order_path"]),
        #        )
        #        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.roster_construction = [
                "P",
                "P",
                "C",
                "1B",
                "2B",
                "3B",
                "SS",
                "OF",
                "OF",
                "OF",
            ]
            self.salary = 50000
            self.max_hitters_per_team = 5
            self.roster_positions = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']

        elif site == "fd":
            self.roster_construction = [
                "P",
                "C/1B",
                "2B",
                "3B",
                "SS",
                "OF",
                "OF",
                "OF",
                "UTIL",
            ]
            self.salary = 35000
            self.max_hitters_per_team = 4
            self.roster_positions = ['P', 'C/1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3', 'UTIL']

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0

        self.adjust_default_stdev()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        self.fill_teams_dict() # Fill the teams_dict with player data
        self.player_dict = {str(v["ID"]): v for v in self.player_dict.values()}
        self.get_opposing_pitcher_id()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
            self.generate_field_lineups()

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config["pct_field_using_stacks"])
        self.default_hitter_var = float(self.config["default_hitter_var"])
        self.default_pitcher_var = float(self.config["default_pitcher_var"])
        self.pct_max_stack_len = float(self.config["pct_max_stack_len"])
        self.overlap_limit = float(self.config["num_hitters_vs_pitcher"])
        self.pct_field_using_secondary_stacks = float(
            self.config["pct_field_using_secondary_stacks"]
        )

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        # for p, s in self.player_dict.items():
        #     print(p, s["ID"])

        problem = plp.LpProblem("MLB", plp.LpMaximize)
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }

        # for key, v in lp_variables.items():
        #     print(key, v)

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["fieldFpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Need 2 pitchers
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "P" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 2
            )
            # Need 1 catcher
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "C" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 1 first baseman
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "1B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "2B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "3B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "SS" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 3 outfielders
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "OF" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 3
            )
            # Can only roster 8 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 10
            )

            # Max 5 hitters per team
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if (
                            self.player_dict[(player, pos_str, team)]["Team"]
                            == team
                            & self.player_dict[(player, pos_str, team)]["Position"]
                            != "P"
                        )
                    )
                    <= 5
                )

        else:
            # Need 1 pitchers
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "P" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 1 catcher or first baseman
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "C/1B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 1 second baseman
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "2B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 1 third baseman
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "3B" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "SS" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need 3 outfielders
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "OF" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 3
            )
            # Need 1 UTIL
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "UTIL" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )

            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )

            # Max 4 hitters per team
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if (
                            self.player_dict[(player, pos_str, team)]["Team"]
                            == team
                            & self.player_dict[(player, pos_str, team)]["Position"]
                            != "P"
                        )
                    )
                    <= 4
                )

        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )

        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"

                player_name = row[name_key].replace("-", "#").lower()

                if "P" in row["position"]:
                    row["position"] = "P"
                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                position = sorted([pos for pos in row["position"].split("/")])
                if self.site == "fd":
                    if "P" not in position:
                        position.append("UTIL")
                    if "1B" in position:
                        position[position.index("1B")] = "C/1B"
                    elif "C" in position:
                        position[position.index("C")] = "C/1B"
                team_key = "teamabbrev" if self.site == "dk" else "team"
                if row[team_key] == "WSH":
                    team = "WAS"
                else:
                    team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                adjusted_pattern = r"(\b\w+@\w+\b)"
                match = re.search(pattern=adjusted_pattern, string=row[game_info])
                opp = ""
                match = match.groups()[0].split("@")
                if match:
                    #match = match.groups()
                    processed_match = []

                    for m in match:
                        # Standardize team abbreviation
                        if m == "WSH":
                            m = "WAS"
                        processed_match.append(m)

                    # Determine the opponent based on the primary team
                    if processed_match[0] == team:
                        opp = processed_match[1]
                    else:
                        opp = processed_match[0]

                    # Add to matchups; ensuring tuple is always in a consistent order
                    if team in processed_match:
                        matchup_tuple = (team, opp) if processed_match.index(team) == 0 else (opp, team)
                    else:
                        # If neither team matches, add as is
                        matchup_tuple = tuple(processed_match)
                        #print(f'matchup: {processed_match}')

                    # Assuming 'matchups' is a set where you store all matchups
                    self.matchups.add(matchup_tuple)
                pos_str = str(position)
                #print(player_name, opp, team, pos_str)

                if (player_name, pos_str, team) in self.player_dict:
                    self.player_dict[(player_name, pos_str, team)]["ID"] = str(
                        row["id"]
                    )
                    self.player_dict[(player_name, pos_str, team)]["Team"] = team
                    self.player_dict[(player_name, pos_str, team)]["Opp"] = opp
                    # Initialize variables for the highest-scoring pitcher's details
                    highest_fpts = float('-inf')  # Start with the lowest possible value for comparison
                    opp_pitcher_key = None  # To store the best pitcher's key (name, pos_str, team)

                    # Iterate over the player_dict
                    for key, player_data in self.player_dict.items():
                        opp_name, opp_pos_str, opp_team = key  # Extract key components
                        # Check if this player is a pitcher for the specified opponent team and has the highest Fpts so far
                        if "P" in pos_str and team == opp and player_data["Fpts"] > highest_fpts:
                            highest_fpts = player_data["Fpts"]  # Update the highest Fpts found
                            opp_pitcher_key = (opp_name, opp_pos_str, opp_team)  # Update the key of the best pitcher found
                            print("Best pitcher found: ", opp_pitcher_key)

                    # best_pitcher_key now contains the key for the pitcher with the highest Fpts for the specified team, or None if no pitcher was found
                    #print(self.player_dict)
                    if opp_pitcher_key:
                        self.player_dict[(player_name, pos_str, team)][
                            "Opp Pitcher ID"
                        ] = self.player_dict[opp_pitcher_key]["ID"]
                        self.player_dict[(player_name, pos_str, team)][
                            "Opp Pitcher Name"
                        ] = self.player_dict[opp_pitcher_key]["Name"]

                    self.id_name_dict[str(row["id"])] = row[name_key]

    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["field size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["entry fee"])
                # multi-position payouts
                if "-" in row["place"]:
                    indices = row["place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        self.teams_dict = collections.defaultdict(list)  # Initialize teams_dict

        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()
                fpts = row['fpts']
                if fpts == '':
                    fpts = 0
                else:
                    fpts = float(fpts)
                if fpts < self.projection_minimum:
                    continue
                if "P" in row["pos"]:
                    row["pos"] = "P"
                position = sorted([pos for pos in row["pos"].split("/")])
                if self.site == "fd":
                    if "P" not in position:
                        position.append("UTIL")
                    if "1B" in position:
                        position[position.index("1B")] = "C/1B"
                    elif "C" in position:
                        position[position.index("C")] = "C/1B"
                if row["team"] == "WSH":
                    team = "WAS"
                else:
                    team = row["team"]
                pos_str = str(position)
                if row["ord"] == "-":
                    order = None
                else:
                    order = int(row["ord"])
                own = float(row['own%'])
                if own == 0:
                    own = 0.000001
                stddev = float(row["stddev"])
                if stddev == 0:
                    if 'P' in position:
                        stddev = fpts * self.default_pitcher_var
                    else:
                        stddev = fpts * self.default_hitter_var
                try:
                    fieldfpts = float(row["fieldfpts"])
                except:
                    fieldfpts = ""
                if fieldfpts == "":
                    fieldfpts = fpts
                player_data = {
                    "Fpts": float(row["fpts"]),
                    "Position": position,
                    "Name": player_name,
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "Opp Pitcher ID": "",
                    "Opp Pitcher Name": "",
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": stddev,
                    "Ceiling": 0,
                    "Ownership": own,
                    "battingOrder": order,  # Handle blank orders
                    "In Lineup": False,
                    "fieldFpts": fieldfpts
                }

                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")
                    player_data["Opp Pitcher ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp Pitcher ID", "")
                    player_data["Opp Pitcher Name"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp Pitcher Name", "")

                self.player_dict[(player_name, pos_str, team)] = player_data

    def fill_teams_dict(self):
        for p in self.player_dict:
            team = self.player_dict[p]["Team"]
            if team not in self.teams_dict:
                self.teams_dict[team] = []
            self.teams_dict[team].append(self.player_dict[p])   

    def adjust_default_stdev(self):
        for player_name, pos, team in self.player_dict.keys():
            if self.player_dict[(player_name, pos, team)]["StdDev"] == 0:
                if self.player_dict[(player_name, pos, team)]["Position"] == ["P"]:
                    print(
                        player_name
                        + " has no stddev, defaulting to "
                        + str(self.default_pitcher_var)
                        + "*projection"
                    )
                    self.player_dict[(player_name, pos, team)]["StdDev"] = (
                        self.player_dict[(player_name, pos, team)]["Fpts"]
                        * self.default_pitcher_var
                    )
                else:
                    print(
                        player_name
                        + " has no stddev, defaulting to "
                        + str(self.default_hitter_var)
                        + "*projection"
                    )
                    self.player_dict[(player_name, pos, team)]["StdDev"] = (
                        self.player_dict[(player_name, pos, team)]["Fpts"]
                        * self.default_hitter_var
                    )

    def load_team_stacks(self, path):
        with open(path) as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if row["team"] == "WSH":
                    team = "WAS"
                else:
                    team = row["team"]
                self.stacks_dict[team] = float(row["own%"]) / 100
    
    def find_pitcher(self, team):
        # Assuming self.teams_dict[team] is a list of player_data dictionaries
        pitchers = [player_data['ID'] for player_data in self.teams_dict[team] if 'SP' in player_data['Position'] or 'RP' in player_data['Position'] or 'P' in player_data['Position']]
        if len(pitchers) == 0:
            print(f'No pitchers found for {team}, {self.teams_dict[team]}')
            return None
        # Return the pitcher ID with the highest Fpts, if there are any pitchers
        if pitchers:
            return max(pitchers, key=lambda id: self.player_dict[id]['Fpts'])
        else:
            #if we can't find a pitcher, we need to randomly assign one and give the user a warning
            print(f'No pitchers found for {team}, {self.teams_dict[team]}')
            msg = f'Unable to find pitcher for {team}. Creating dummy pitcher to simulate against.'
            print(msg)

    def get_opposing_pitcher_id(self):
        self.opp_pitcher_ids = {}  # Initialize the resulting dictionary

        # Go through each matchup
        for home_team, away_team in self.matchups:
            # Check if we have player data for both teams in the current matchup
            if home_team in self.teams_dict and away_team in self.teams_dict:
                # Find the opposing pitcher for the home team (pitcher from the away team)
                away_pitcher_id = self.find_pitcher(away_team)
                if away_pitcher_id:
                    self.opp_pitcher_ids[home_team] = away_pitcher_id
                
                # Find the opposing pitcher for the away team (pitcher from the home team)
                home_pitcher_id = self.find_pitcher(home_team)
                if home_pitcher_id:
                    self.opp_pitcher_ids[away_team] = home_pitcher_id
        
        print("Opposing pitchers' IDs:", self.opp_pitcher_ids)

    def remap(self, fieldnames):
        return ["P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            if self.site == "dk":
                reader = pd.read_csv(file)
                lineup = []
                for i, row in reader.iterrows():
                    # print(row)
                    if i == self.field_size:
                        break
                    lineup = [
                        str(row[0].split("(")[1].replace(")", "")),
                        str(row[1].split("(")[1].replace(")", "")),
                        str(row[2].split("(")[1].replace(")", "")),
                        str(row[3].split("(")[1].replace(")", "")),
                        str(row[4].split("(")[1].replace(")", "")),
                        str(row[5].split("(")[1].replace(")", "")),
                        str(row[6].split("(")[1].replace(")", "")),
                        str(row[7].split("(")[1].replace(")", "")),
                        str(row[8].split("(")[1].replace(")", "")),
                        str(row[9].split("(")[1].replace(")", "")),
                    ]
                    # storing if this lineup was made by an optimizer or with the generation process in this script
                    self.field_lineups[i] = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "opto",
                        "Count": 0
                    }
                    i += 1
            elif self.site == "fd":
                reader = pd.read_csv(file)
                lineup = []
                for i, row in reader.iterrows():
                    # print(row)
                    if i == self.field_size:
                        break
                    lineup = [
                        str(row[0].split(":")[0]),
                        str(row[1].split(":")[0]),
                        str(row[2].split(":")[0]),
                        str(row[3].split(":")[0]),
                        str(row[4].split(":")[0]),
                        str(row[5].split(":")[0]),
                        str(row[6].split(":")[0]),
                        str(row[7].split(":")[0]),
                        str(row[8].split(":")[0]),
                    ]
                    # storing if this lineup was made by an optimizer or with the generation process in this script
                    self.field_lineups[i] = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "opto",
                        "Count": 0
                    }
                    i += 1
        # print(self.field_lineups)

    @staticmethod
    def select_player(position, ids, in_lineup, pos_matrix, ownership, salaries, projections, remaining_salary, salary_floor, rng, roster_positions, team_counts, max_hitters_per_team, teams, overlap_limit, opponents, pitcher_opps, salary_ceiling, num_players_remaining):
        position_index = roster_positions.index(position)  # Determine the matrix index for the position
        valid_indices = np.where((pos_matrix[:, position_index] > 0) & (in_lineup == 0) & (salaries <= remaining_salary))[0]

        if position in ['P1', 'P2', 'P']:  # Selecting a pitcher
            # Check that the opposing team does not exceed overlap_limit in pitcher_opps
            valid_indices = [index for index in valid_indices if team_counts.get(opponents[index], 0) <= overlap_limit]
        else:  # Selecting a hitter
            # Ensure the hitter's team doesn't exceed max hitters per team and doesn't allow hitting against a pitcher exceeding overlap limits
            valid_indices = [
                index for index in valid_indices 
                if team_counts[teams[index]] < max_hitters_per_team 
                and all(team_counts[team] < overlap_limit for team in pitcher_opps if team == teams[index])
            ]

        if not valid_indices:
            return None  # No valid player found

        probabilities = ownership[valid_indices]
        probabilities /= probabilities.sum()  # Normalize probabilities

        chosen_index = rng.choice(valid_indices, p=probabilities)
        chosen_id = ids[chosen_index]

        return chosen_id, salaries[chosen_index], projections[chosen_index]
        
    @staticmethod
    def is_valid_lineup(lineup, salary, projection, salary_floor, salary_ceiling, optimal_score, max_pct_off_optimal, isStack):
        minimum_projection = optimal_score * (1 - max_pct_off_optimal)
        # if isStack:
        #     #allow stacks to go 25% below min projection
        #     minimum_projection = minimum_projection * 1 #0.75
        #     salary_floor = 1 * salary_floor #0.75 * salary_floor
        if salary < salary_floor or salary > salary_ceiling:
            return False
        if projection < minimum_projection:
            return False
        #check if Nones are in lineup
        if None in lineup.values():
            return False
        return True
    
    @staticmethod
    def adjust_probabilities(salaries, ownership, salary_ceiling):
        boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries])
        boosted_probabilities = ownership * boosted_salaries
        boosted_probabilities /= boosted_probabilities.sum()
        return boosted_probabilities

    @staticmethod
    def build_stack(ids, pos_matrix, teams, stack_team, ownership, stack_positions, rng, roster_positions, in_lineup, stack_len, isPrimary=False):
        team_indices = np.where(teams == stack_team)[0]
        selected_players = []
        selected_positions = []  # Tracks which positions have been filled

        # Filter for eligible indices
        eligible_indices = [i for i in team_indices if in_lineup[i] == 0 and any(pos_matrix[i][roster_positions.index(pos)] > 0 for pos in stack_positions)]
        
        if not eligible_indices:
            #print(f"No eligible players available for team {stack_team}.")
            return [], []

        probabilities = ownership[eligible_indices]
        probabilities /= probabilities.sum()  # Normalize probabilities

        if isPrimary == False:
            if len(eligible_indices) < stack_len:
                stack_len = len(eligible_indices)  # Adjust stack size if not enough players are available

        while len(selected_players) < stack_len and eligible_indices:
            current_eligible_indices = [i for i in eligible_indices if any(pos_matrix[i][roster_positions.index(pos)] > 0 and pos not in selected_positions for pos in stack_positions)]
            
            if not current_eligible_indices:
                #print("No more eligible players fitting the remaining positions.")
                break

            probabilities = ownership[current_eligible_indices]
            probabilities /= probabilities.sum()  # Normalize probabilities

            chosen_index = rng.choice(current_eligible_indices, p=probabilities)
            player_id = ids[chosen_index]

            # Determine eligible positions for the chosen player that have not been filled yet
            possible_positions = [pos for pos in stack_positions if pos_matrix[chosen_index][roster_positions.index(pos)] > 0 and pos not in selected_positions]
            
            if possible_positions:
                chosen_position = rng.choice(possible_positions)  # Randomly select one of the available positions
                selected_players.append(player_id)
                selected_positions.append(chosen_position)
                in_lineup[chosen_index] = 1  # Mark this player as selected in the lineup

                # Remove this index from future consideration
                eligible_indices.remove(chosen_index)
            else:
                #print(f"No valid positions available for player {player_id} in team {stack_team}. Retrying with remaining players...")
                eligible_indices.remove(chosen_index)

        return selected_players, selected_positions
    
    @staticmethod
    def verify_stack(lineup, team, min_stack_size):
        players_from_team = [player for pos, player in lineup.items() if player['Team'] == team]
        unique_positions = len(set(pos for pos, player in lineup.items() if player['Team'] == team))
        return len(players_from_team) >= min_stack_size and unique_positions >= min_stack_size

    
    @staticmethod
    def generate_lineups(params):
        rng = np.random.default_rng()
        (lu_num, ids, original_in_lineup, pos_matrix, ownership, initial_salary_floor, salary_ceiling, optimal_score, salaries,
        projections, max_pct_off_optimal, teams, opponents, team_stack, stack_len, overlap_limit,
        max_stack_len, secondary_stack, secondary_stack_len, max_hitters_per_team, site, roster_positions) = params

        max_retries = 1000  # Define maximum retry count
        salary_floor_decrement = initial_salary_floor * 0.01  # Decrease floor by 5% on each retry past halfway point
        #salary_floor_decrement = 0 # No salary floor decrement
        min_projection_decrement_factor = 0.05  # Decrease minimum projection factor by 5% each time
        #min_projection_decrement_factor = 0  # No projection decrement factor
        current_salary_floor = initial_salary_floor
        current_projection_factor = 1

        for attempt in range(max_retries):
            in_lineup = original_in_lineup.copy()
            lineup = {position: None for position in roster_positions}
            team_counts = {team: 0 for team in set(teams)} 
            total_salary = 0
            total_projection = 0
            hitter_vs_pitcher = 0
            num_players_remaining = len(roster_positions)
            isStack = bool(team_stack or secondary_stack)
            pitcher_opps = []
            #remove pitcher positions from roster positions to generate stack positions 
            pitcher_positions = ['P', 'P1', 'P2']
            stack_positions = [pos for pos in roster_positions if pos not in pitcher_positions]

            # Apply decrements every 100 retries
            if attempt % 100 == 0 and attempt != 0:
                current_salary_floor -= salary_floor_decrement
                current_projection_factor -= min_projection_decrement_factor

            # Implementing stack logic and filling positions
            if team_stack:
                primary_players, slotted_positions = MLB_GPP_Simulator.build_stack(ids, pos_matrix, teams, team_stack, ownership, stack_positions, rng, roster_positions, in_lineup, stack_len, isPrimary=True)
                if primary_players:
                    if len(primary_players) < stack_len:
                        #print(f'primary stack failed to get enough players for {team_stack} and {stack_len}')
                        continue
                    if len(set(slotted_positions)) < len(slotted_positions):
                        #print(f'primary stack failed to get unique positions for {team_stack} and {stack_len}')
                        continue
                    for player_id, pos in zip(primary_players, slotted_positions):
                        #print(f'player_id is {player_id} and pos is {pos}')
                        idx = np.where(ids == player_id)[0][0]
                        lineup[pos] = player_id
                        total_salary += salaries[idx]
                        total_projection += projections[idx]
                        in_lineup[idx] = 1
                        team_counts[teams[idx]] += 1
                        num_players_remaining -= 1

                # else:
                #     break  # If primary stack fails, retry
            #print(f'lineup is {lineup} with total salary {total_salary} and projection {total_projection} going to secondary stack')
            if secondary_stack:
                stack_positions = [pos for pos in roster_positions if pos not in ['P', 'P1', 'P2'] and lineup[pos] is None]
                #print(f'secondary stack is {secondary_stack} and secondary stack len is {secondary_stack_len} and positions to fill are {stack_positions}')
                try:
                    secondary_players, slotted_positions = MLB_GPP_Simulator.build_stack(ids, pos_matrix, teams, secondary_stack, ownership, stack_positions, rng, roster_positions, in_lineup, secondary_stack_len)
                except UnboundLocalError:
                    print(f'in lineup {lineup} and stack positions {stack_positions} and slotted positions {slotted_positions} length mismatch in stack len and eligible positions with stack team {secondary_stack} and secondary stack len {secondary_stack_len}')
                if secondary_players:
                    for player_id, pos in zip(secondary_players, slotted_positions):
                        #print(f'chosen secondary stack player_id is {player_id} and pos is {pos}')
                        idx = np.where(ids == player_id)[0][0]
                        lineup[pos] = player_id
                        total_salary += salaries[idx]
                        total_projection += projections[idx]
                        in_lineup[idx] = 1
                        team_counts[teams[idx]] += 1
                        num_players_remaining -= 1
                # if len(secondary_players) < secondary_stack_len:
                #     break #retry if we cant get enough players in the stack
                # else:
                #     break  # If secondary stack fails, retry
            #print(f'after sccondary stack lineup is {lineup}, salary is {total_salary}, projection is {total_projection}')
            # Fill other positions
            shuffled_positions = list(roster_positions)
            rng.shuffle(shuffled_positions)
            for position in shuffled_positions:
                if not lineup[position]:
                    #print(f'found empty lineup spot {position}')
                    result = MLB_GPP_Simulator.select_player(position, ids, in_lineup, pos_matrix, ownership, salaries,
                                                            projections, salary_ceiling - total_salary, current_salary_floor, rng,
                                                            roster_positions, team_counts, max_hitters_per_team, teams, overlap_limit,
                                                            opponents, pitcher_opps, salary_ceiling, num_players_remaining)
                    if result:
                        player_id, cost, proj = result
                        idx = np.where(ids == player_id)[0][0]
                        lineup[position] = player_id
                        #print(f'position is {position} and player_id is {player_id} and lineup is {lineup}')
                        total_salary += cost
                        total_projection += proj
                        #print(f'lineup is {lineup} new salary is {total_salary} and new projection is {total_projection}')
                        in_lineup[idx] = 1
                        
                        num_players_remaining -= 1
                        if position in ['P1', 'P2', 'P']:
                            pitcher_opps.append(opponents[idx])
                        else:  # Update pitcher_opps count for hitters
                            team_counts[teams[idx]] += 1

                    # else:
                    #     break  # No valid player found, trigger retry

            if all(value is not None for value in lineup.values()) and MLB_GPP_Simulator.is_valid_lineup(
                    lineup, total_salary, total_projection, current_salary_floor, salary_ceiling, optimal_score, current_projection_factor, bool(team_stack or secondary_stack)):
                if isStack:
                    lu_type = 'generated_stack'
                else:
                    lu_type = 'generated_nostack'
                return {
                    "Lineup": lineup,
                    "Wins": 0,
                    "Top1Percent": 0,
                    "ROI": 0,
                    "Cashes": 0,
                    "Type": lu_type,
                    "Count": 0,
                    "Salary": total_salary,
                    "Projection": total_projection
                }

        msg = f'Failed to generate lineup {lu_num} after {max_retries} attempts. lineup configs were {team_stack}, {stack_len}, {secondary_stack}, {secondary_stack_len} and {max_hitters_per_team} and {overlap_limit} and {current_salary_floor} and {current_projection_factor} and {salary_ceiling} and {optimal_score} and {site} and {roster_positions} and {team_counts} and {pitcher_opps} and final attempt wasa {lineup} with salary {total_salary} and projection {total_projection}'
        return msg

    def calculate_average_salaries(self):
        """Calculate average salaries for each team's hitters and pitchers."""
        team_salaries = {team: [] for team in self.stacks_dict.keys()}
        pitcher_salaries = []

        for player_id, info in self.player_dict.items():
            if 'P' in info['Position']:
                pitcher_salaries.append(info['Salary'])
            else:
                team_salaries[info['Team']].append(info['Salary'])

        team_average_salaries = {team: np.mean(salaries) if salaries else 0 for team, salaries in team_salaries.items()}
        average_pitcher_salary = np.mean(pitcher_salaries) if pitcher_salaries else 0
        average_hitter_salary = np.mean([sal for team, salaries in team_salaries.items() for sal in salaries])

        return team_average_salaries, average_pitcher_salary, average_hitter_salary

    def setup_stacks(self, diff):
        teams = list(self.stacks_dict.keys())
        probabilities = [self.stacks_dict[team] for team in teams]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        num_hitters_per_lineup = len([pos for pos in self.roster_construction if pos != 'P'])

        team_average_salaries, average_pitcher_salary, average_hitter_salary = self.calculate_average_salaries()
        primary_teams = []
        primary_lens = []
        secondary_teams = []
        secondary_lens = []

        for _ in range(diff):
            if np.random.rand() < self.pct_field_using_stacks:
                primary_team = np.random.choice(teams, p=probabilities)
                primary_teams.append(primary_team)
                primary_lens.append(np.random.choice([self.max_stack_len - 1, self.max_stack_len], p=[1 - self.pct_max_stack_len, self.pct_max_stack_len]))
            else:
                primary_teams.append('')
                primary_lens.append(0)

            if primary_teams[-1] and np.random.rand() < self.pct_field_using_secondary_stacks:
                #use primary_stack_len to set secondary_stack_len
                max_secondary_stack_len = num_hitters_per_lineup - primary_lens[-1]
                available_teams = [team for team in teams if team != primary_teams[-1]]
                secondary_stack_len = np.random.choice([max_secondary_stack_len-1,max_secondary_stack_len], p=[0.5,0.5])
                filtered_teams = []
                for team in available_teams:
                    # Check financial feasibility
                    num_pitchers = 2 if self.site == 'dk' else 1 
                    if primary_lens[-1] + secondary_stack_len == len(self.roster_construction) - num_pitchers:
                        total_estimated_cost = (team_average_salaries[primary_teams[-1]] * primary_lens[-1] +
                                                team_average_salaries[team] * 3 +  # Assume 3 players for secondary stack
                                                average_pitcher_salary * num_pitchers)  # Assume two pitchers
                    else:
                        total_estimated_cost = (team_average_salaries[primary_teams[-1]] * primary_lens[-1] +
                                                team_average_salaries[team] * 2 + average_hitter_salary + # Assume 3 players for secondary stack
                                                average_pitcher_salary * num_pitchers)  # Assume two pitchers                        
                    if self.min_lineup_salary *.9 <= total_estimated_cost <= self.salary:
                        filtered_teams.append(team)

                if filtered_teams:
                    probs = [self.stacks_dict[team] for team in filtered_teams]
                    probs = [p / sum(probs) for p in probs]
                    secondary_team = np.random.choice(filtered_teams, p=probs)
                    secondary_teams.append(secondary_team)
                    secondary_lens.append(secondary_stack_len)
                else:
                    secondary_teams.append('')
                    secondary_lens.append(0)
            else:
                secondary_teams.append('')
                secondary_lens.append(0)

        return {'primary': primary_teams, 'primary_len': primary_lens, 'secondary': secondary_teams, 'secondary_len': secondary_lens}
    
    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            teams = []
            opponents = []
            for k in self.player_dict.keys():
                if "Team" not in self.player_dict[k].keys():
                    print(
                        self.player_dict[k]["Name"],
                        " name mismatch between projections and player ids!",
                    )
                ids.append(self.player_dict[k]["ID"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                projections.append(self.player_dict[k]["Fpts"])
                teams.append(self.player_dict[k]["Team"])
                opponents.append(self.player_dict[k]["Opp"])
                pos_list = []
                for pos in self.roster_construction:
                    if pos in self.player_dict[k]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                positions.append(np.array(pos_list))
            in_lineup = np.zeros(shape=len(ids))
            ownership = np.array(ownership)
            salaries = np.array(salaries)
            projections = np.array(projections)
            pos_matrix = np.array(positions)
            ids = np.array(ids)
            optimal_score = self.optimal_score
            salary_floor = self.min_lineup_salary
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            stack_usage = self.pct_field_using_stacks
            teams = np.array(teams)
            opponents = np.array(opponents)
            overlap_limit = self.overlap_limit
            problems = []
            # Initialize stack setup
            stack_config = self.setup_stacks(diff)
            #check if stack config has any primary stacks below 1- max stack len
            # Construct lineup tuples for multiprocessing
            problems = [
                (
                    i,
                    ids,
                    np.zeros(len(ids)),  # fresh in_lineup for each problem
                    np.array(positions),
                    np.array(ownership),
                    self.min_lineup_salary,
                    self.salary,
                    self.optimal_score,
                    np.array(salaries),
                    np.array(projections),
                    self.max_pct_off_optimal,
                    np.array(teams),
                    np.array(opponents),
                    stack_config['primary'][i],
                    stack_config['primary_len'][i],
                    self.overlap_limit,
                    self.max_stack_len,
                    stack_config['secondary'][i],
                    stack_config['secondary_len'][i],
                    self.max_hitters_per_team,
                    self.site,
                    self.roster_positions,
                )
                for i in range(diff)
            ]
            # print(problems[0])
            # print(stacks)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(MLB_GPP_Simulator.generate_lineups, [(params,) for params in problems])
                print(
                    "number of running processes =",
                    pool.__dict__["_processes"]
                    if (pool.__dict__["_state"]).upper() == "RUN"
                    else None,
                )
                pool.close()
                pool.join()
                print("pool closed")
                self.update_field_lineups(output, diff)
                msg = str(diff) + " field lineups successfully generated. " + str(len(self.field_lineups.keys())) + " uniques."
                
                end_time = time.time()
                #self.simDoc.update({'jobProgressLog': ArrayUnion([msg])})
                print("lineups took " + str(end_time - start_time) + " seconds")
                # print(self.field_lineups)

        # print(self.field_lineups)
    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(range(max(self.field_lineups.keys()) + 1, max(self.field_lineups.keys()) + 1 + diff))
        nk = new_keys[0]
        for i, o in enumerate(output):
            if 'Lineup' not in o or not isinstance(o['Lineup'], dict):
                print(f"Error: Expected 'Lineup' in output but got: {o}")
            #print(o.values())
            #print names and positions and teams in lineup for double checking
            # for q,r in o['Lineup'].items():
            #     print(q, self.player_dict[r]['Name'], self.player_dict[r]['Position'], self.player_dict[r]['Team'])
            lineup_list = []
            for r in self.roster_positions:
                try:
                    x = o['Lineup'][r]
                    lineup_list.append(x)
                except:
                    f'Error in lineup generation, {o}'
                # if x is not None:
                #     for q in self.roster_construction:
                #         if q in self.player_dict[x]['Position']:
                    
                #             break
                    #print(r, self.player_dict[x]['Name'], self.player_dict[x]['Position'], self.player_dict[x]['Team'])
                #lineup_list.append(x)
            lineup_set = frozenset(sorted(lineup_list))
            #print(lineup_set)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1
                        
                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1
                
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    # Convert dict_values to a dictionary before assignment
                    lineup_data = dict(o)
                    lineup_data['Count'] += self.seen_lineups[lineup_set]
                    lineup_data['Lineup'] = lineup_list
                    # Now assign the dictionary to the field_lineups
                    self.field_lineups[nk] = lineup_data                 
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    @staticmethod
    def run_simulation_for_team(
        team_id, team, pitcher_samples_dict, num_iterations, opp_pitcher_ids, player_dict
    ):
        rng = np.random.Generator(np.random.PCG64())  # Create a new Generator instance
        existing_correlation_matrix = np.array([
        [1., 0.1855827, 0.17517424, 0.17516216, 0.17520923, 0.14935001, 0.15884211, 0.15799072, 0.16467186],
        [0.1855827, 1., 0.20226615, 0.15774969, 0.14370335, 0.14275675, 0.13338755, 0.13081362, 0.16291087],
        [0.17517424, 0.20226615, 1., 0.17394293, 0.16980602, 0.15196324, 0.14008804, 0.13916134, 0.14359494],
        [0.17516216, 0.15774969, 0.17394293, 1., 0.19942615, 0.17285594, 0.15883667, 0.12021929, 0.13140383],
        [0.17520923, 0.14370335, 0.16980602, 0.19942615, 1., 0.18785418, 0.1794043, 0.14219216, 0.12337999],
        [0.14935001, 0.14275675, 0.15196324, 0.17285594, 0.18785418, 1., 0.18722286, 0.14811003, 0.13624325],
        [0.15884211, 0.13338755, 0.14008804, 0.15883667, 0.1794043, 0.18722286, 1., 0.18284259, 0.15515191],
        [0.15799072, 0.13081362, 0.13916134, 0.12021929, 0.14219216, 0.14811003, 0.18284259, 1., 0.168358],
        [0.16467186, 0.16291087, 0.14359494, 0.13140383, 0.12337999, 0.13624325, 0.15515191, 0.168358, 1.]
        ])

        # Initialize an 11x11 matrix
        extended_correlation_matrix = np.zeros((11, 11))

        # Place the existing 9x9 matrix in the top-left corner
        extended_correlation_matrix[:9, :9] = existing_correlation_matrix

        # Adding the new rows for the starting and opposing pitchers
        extended_correlation_matrix[9, :10] = [-0.05, -0.03, -0.05, -0.02, -0.01, -0.02, -0.02, -0.03, -0.02, 1]
        extended_correlation_matrix[10, :10] = [-0.31, -0.3, -0.31, -0.29, -0.26, -0.25, -0.28, -0.27, -0.26, 0.05]

        # Add the symmetric column values (except the diagonal which is already 1 for the pitchers)
        extended_correlation_matrix[:10, 9] = extended_correlation_matrix[9, :10]
        extended_correlation_matrix[:10, 10] = extended_correlation_matrix[10, :10]

        # Set the last diagonal values to 1 (as the correlation of any item with itself is always 1)
        extended_correlation_matrix[9, 9] = 1
        extended_correlation_matrix[10, 10] = 1

        correlation_matrix = extended_correlation_matrix

        std_devs = [3] * 9 + [1] + [1]
        D = np.diag(std_devs)  # Create a diagonal matrix with the standard deviations
        covariance_matrix = np.dot(D, np.dot(correlation_matrix, D))

        #print(f'starting game simulation for team {team_id} and {opp_pitcher_ids.get(team_id, None)}')
        # print(covariance_matrix)
        try:
            team = sorted(
                team,
                key=lambda player: float("inf") if not isinstance(player["battingOrder"], int) else player["battingOrder"]
            )
        except Exception as e:
            print('Error:', str(e))
            print('unable to find batting order for team', team_id)
            for player in team:
                print(player)
        #print(team)
        hitters_tuple_keys = [
            player for player in team if "P" not in player["Position"] and player['battingOrder'] != 'NS' and player['battingOrder'] != '-'
        ]
        try:
            # Check if there are players assigned as pitchers in the team
            pitchers = [player for player in team if "P" in player["Position"]]
            if pitchers:
                pitcher_tuple_key = pitchers[0]  # Assuming the first pitcher is what we want
            else:
                # No pitchers found, use a dummy pitcher
                pitcher_tuple_key = create_dummy_pitcher()
                print(f"No valid pitcher found for team {team_id}. Using dummy pitcher: {pitcher_tuple_key['Name']}.")

            # Ensure the pitcher is in the samples dictionary
            if pitcher_tuple_key["ID"] not in pitcher_samples_dict:
                pitcher_samples = rng.normal(
                    loc=pitcher_tuple_key["Fpts"], scale=pitcher_tuple_key["StdDev"], size=num_iterations
                )
                pitcher_samples_dict[pitcher_tuple_key["ID"]] = pitcher_samples
            else:
                pitcher_samples = pitcher_samples_dict[pitcher_tuple_key["ID"]]
            pitcher_fpts = pitcher_tuple_key["Fpts"]
            pitcher_stddev = pitcher_tuple_key["StdDev"]
        except Exception as e:
            print(f"Error processing team {team_id}: {str(e)}")

        hitters_fpts = np.array([hitter["Fpts"] for hitter in hitters_tuple_keys])
        hitters_stddev = np.array([hitter["StdDev"] for hitter in hitters_tuple_keys])

        size = num_iterations

        # check if P has been simmed
        # if pitcher_tuple_key["ID"] not in pitcher_samples_dict:
        #     pitcher_samples = rng.normal(
        #         loc=pitcher_fpts, scale=pitcher_stddev, size=size
        #     )
        #     pitcher_samples_dict[pitcher_tuple_key["ID"]] = pitcher_samples
        # else:
        #     pitcher_samples = pitcher_samples_dict[pitcher_tuple_key["ID"]]

        # look up the Opp Pitcher ID in team_dict
            
        opposing_pitcher_id = opp_pitcher_ids.get(team_id, None)
        
        if opposing_pitcher_id and opposing_pitcher_id in player_dict:
            opposing_pitcher = player_dict[opposing_pitcher_id]
        else:
            # No valid opposing pitcher found, use a dummy pitcher
            opposing_pitcher = create_dummy_pitcher()
            opposing_pitcher_id = opposing_pitcher['ID']
            print(f"No valid opposing pitcher found for team {team_id}. Using dummy pitcher.")
        
        # Ensure the pitcher is in the samples dictionary
        if opposing_pitcher_id not in pitcher_samples_dict:
            opposing_pitcher_fpts = opposing_pitcher["Fpts"]
            opposing_pitcher_stddev = opposing_pitcher["StdDev"]
            opposing_pitcher_samples = np.random.normal(
                loc=opposing_pitcher_fpts, scale=opposing_pitcher_stddev, size=num_iterations
            )
            pitcher_samples_dict[opposing_pitcher_id] = opposing_pitcher_samples


        hitters_params = [
            (fpts, stddev) for fpts, stddev in zip(hitters_fpts, hitters_stddev)
        ]
        
        pitcher_params = (pitcher_fpts, pitcher_stddev)
        try:
            opposing_pitcher_params = (opposing_pitcher_fpts, opposing_pitcher_stddev)
        except:
            print(f'missing opposing pitcher data for team {team_id}')


        multi_normal = multivariate_normal(mean=[0] * 11, cov=covariance_matrix)

        samples = multi_normal.rvs(size=num_iterations)

        uniform_samples = norm.cdf(samples)

        cap = 60
        max_attempts = 1000

        hitters_samples = []
        for u, params in zip(uniform_samples.T, hitters_params):
            attempts = 0
            sample = gamma.ppf(u, *params)
            # If sample is an array, apply check and adjustment for each value
            if isinstance(sample, np.ndarray):
                for i, s in enumerate(sample):
                    while s > cap and attempts < max_attempts:
                        u = rng.uniform()  # Generate a new uniform random number
                        s = gamma.ppf(u, *params)
                        attempts += 1
                    if attempts == max_attempts:
                        s = params[0]  # If max attempts are reached, set sample to 0
                    sample[i] = s
            hitters_samples.append(sample)

        pitcher_samples = norm.ppf(uniform_samples.T[-2], *pitcher_params)

        opposing_pitcher_samples = norm.ppf(
            uniform_samples.T[-1], *opposing_pitcher_params
        )
        #print('samples completed for team', team_id)

        # fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
        # fig.tight_layout(pad=5.0)

        # for i, hitter in enumerate(hitters_tuple_keys):
        #     sns.kdeplot(hitters_samples[i], ax=ax1, label=hitter['Name'])

        # sns.kdeplot(pitcher_samples, ax=ax1, label=pitcher_tuple_key['Name'], linestyle='--')

        # sns.kdeplot(opposing_pitcher_samples, ax=ax1, label = opposing_pitcher['Name'] + " (Opp)", linestyle=':')

        # ax1.legend(loc='upper right', fontsize=14)
        # ax1.set_xlabel('Fpts', fontsize=14)
        # ax1.set_ylabel('Density', fontsize=14)
        # ax1.set_title(f'Team {team_id} Distributions', fontsize=14)
        # ax1.tick_params(axis='both', which='both', labelsize=14)

        # y_min, y_max = ax1.get_ylim()
        # ax1.set_ylim(y_min, y_max*1.1)

        # ax1.set_xlim(-5, 70)

        # # Sorting players and correlating their data
        # player_order = [player['Order'] if player['Order'] is not None else float('inf') for player in hitters_tuple_keys] + [1000, float('inf')]
        # player_names = [f"{player['Name']} ({player['Order']})" if player['Order'] is not None else f"{player['Name']} (P)" for player in hitters_tuple_keys] + [f"{pitcher_tuple_key['Name']} (P)", f"{opposing_pitcher['Name']} (Opp P)"]
        # samples_order = [hitters_samples[i] for i in range(len(hitters_samples))] + [pitcher_samples, opposing_pitcher_samples]
        # sorted_samples = [x for _, x in sorted(zip(player_order, samples_order))]

        # # Ensuring the data is correctly structured as a 2D array
        # sorted_samples_array = np.array(sorted_samples)
        # if sorted_samples_array.shape[0] < sorted_samples_array.shape[1]:
        #     sorted_samples_array = sorted_samples_array.T

        # correlation_matrix = pd.DataFrame(np.corrcoef(sorted_samples_array.T), columns=player_names, index=player_names)

        # sns.heatmap(correlation_matrix, annot=True, ax=ax2, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax2.set_title(f'Correlation Matrix for Team {team_id}', fontsize=14)

        # plt.savefig(f'output/simulation_plots/Team_{team_id}_Distributions_Correlation.png', bbox_inches='tight')
        # plt.close()

        temp_fpts_dict = {}
        for i, hitter in enumerate(hitters_tuple_keys):
            temp_fpts_dict[str(hitter["ID"])] = hitters_samples[i]

        if pitcher_tuple_key["ID"] == 'dummy':
            pass
        else:
            temp_fpts_dict[str(pitcher_tuple_key["ID"])] = pitcher_samples

        return temp_fpts_dict
    
    @staticmethod
    @nb.jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array
       
    def run_tournament_simulation(self):
        print("Running " + str(self.num_iterations) + " simulations")

        #reset player_dict to use player ids as keys
        

        start_time = time.time()
        temp_fpts_dict = {}
        pitcher_samples_dict = {}  # keep track of already simmed pitchers
        size = self.num_iterations

        with mp.Pool() as pool:
            team_simulation_params = [
                (team_id, team, pitcher_samples_dict, size, self.opp_pitcher_ids, self.player_dict)
                for team_id, team in self.teams_dict.items()
            ]
            results = pool.starmap(self.run_simulation_for_team, team_simulation_params)

        for res in results:
            temp_fpts_dict.update(res)

        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        #set self.player_dict to use player ids as keys
        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]
        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + "seconds. Outputting."
        )
        msg = str(self.num_iterations) + " Tournament simulations finished in " + str(round(diff, 2)) + " seconds. Outputting."

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            # print(x)
            salary = 0
            fpts_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            hitters_vs_pitcher = 0
            pitcher_opps = []
            for id in x["Lineup"]:
                v = self.player_dict[id]
                if "P" in v["Position"]:
                    pitcher_opps.append(v["Opp"])
            for id in x["Lineup"]:
                v = self.player_dict[id]
                salary += v["Salary"]
                fpts_p += v["Fpts"]
                ceil_p += v["Ceiling"]
                own_p.append(v["Ownership"] / 100)
                lu_names.append(v["Name"])
                if "P" not in v["Position"]:
                    lu_teams.append(v["Team"])
                    if v["Team"] in pitcher_opps:
                        hitters_vs_pitcher += 1
                continue
            counter = collections.Counter(lu_teams)
            stacks = counter.most_common(2)
            if len(stacks) == 1:
                print(f'only one stack found for lineup {x}, {stacks}, {counter}, {lu_teams}')
            own_p = np.prod(own_p)
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            Top1Percent_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            lu_type = x["Type"]
            simDupes = x['Count']
            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},${},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[9].replace("#", "-"),
                        x["Lineup"][9],
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        Top1Percent_p,
                        roi_p,
                        own_p,
                        roi_round,
                        str(stacks[0][0]) + " " + str(stacks[0][1]),
                        str(stacks[1][0]) + " " + str(stacks[1][1]),
                        hitters_vs_pitcher,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{}%,{},{},{},{},{}".format(
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[9].replace("#", "-"),
                        x["Lineup"][9],
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        Top1Percent_p,
                        own_p,
                        str(stacks[0][0]) + " " + str(stacks[0][1]),
                        str(stacks[1][0]) + " " + str(stacks[1][1]),
                        hitters_vs_pitcher,
                        lu_type,
                        simDupes
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        x["Lineup"][0],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[8].replace("#", "-"),
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        Top1Percent_p,
                        roi_p,
                        own_p,
                        roi_round,
                        str(stacks[0][0]) + " " + str(stacks[0][1]),
                        str(stacks[1][0]) + " " + str(stacks[1][1]),
                        hitters_vs_pitcher,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{},{},{},{},{},{}".format(
                        x["Lineup"][0],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[8].replace("#", "-"),
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        Top1Percent_p,
                        own_p,
                        str(stacks[0][0]) + " " + str(stacks[0][1]),
                        str(stacks[1][0]) + " " + str(stacks[1][1]),
                        hitters_vs_pitcher,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "P,P,C,1B,2B,3B,SS,OF,OF,OF,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type,Num Dupes\n"
                    )
                else:
                    f.write(
                        "P,P,C,1B,2B,3B,SS,OF,OF,OF,Fpts Proj,Ceiling,Salary,Win %,Top 10%, Proj. Own. Product,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type,Num Dupes\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type,Num Dupes\n"
                    )
                else:
                    f.write(
                        "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type,Num Dupes\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        # Initialize all player data
        unique_players = {player: {"Wins": 0, "Top1Percent": 0, "In": 0, "ROI": 0, "Cashes": 0} for player in self.player_dict}
        # Loop over all lineups and their outcomes once to aggregate player data
        for val in self.field_lineups.values():
            lineup_players = val["Lineup"]
            for player in lineup_players:
                unique_players[player]["In"] += val['Count']
                unique_players[player]["ROI"] += val["ROI"]
                unique_players[player]["Cashes"] += val["Cashes"]
                
                # Only increment Wins and Top1Percent if the lineup has them
                if val['Wins'] > 0:
                    unique_players[player]["Wins"] += val['Wins']  # Distribute the win among the players in the lineup
                if val['Top1Percent'] > 0:
                    unique_players[player]["Top1Percent"] += val['Top1Percent']   # Distribute the top 1% finish among the players in the lineup

        # Write the aggregated data to the output file
        with open(out_path, "w") as f:
            f.write("Player,Win%,Top1%,Cash%,Sim. Own%,Proj. Own%,Avg. Return\n")
            
            for player, data in unique_players.items():
                win_p = round(data["Wins"] / self.num_iterations * 100, 4)
                Top1Percent_p = round(data["Top1Percent"] / self.num_iterations * 100, 4)
                if data['In'] == 0:
                    cash_p = 0
                    roi_p = 0
                else:
                    cash_p = round(data["Cashes"] / data["In"] / self.num_iterations * 100, 4)
                    roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 4)
                field_p = round(data["In"] / self.field_size * 100, 4)
                proj_own = self.player_dict[player]["Ownership"]*100
                
                f.write(
                    "{},{}%,{}%,{}%,{}%,{}%,${}\n".format(
                        self.player_dict[player]['Name'].replace("#", "-"),
                        win_p,
                        Top1Percent_p,
                        cash_p,
                        field_p,
                        proj_own/100,
                        roi_p,
                    )
                )