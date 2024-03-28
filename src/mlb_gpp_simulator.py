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

@nb.jit(nopython=True)  # nopython mode ensures the function is fully optimized
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2
    


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
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    seen_lineups = {}
    seen_lineups_ix = {}
    game_info = {}
    matchups = {}
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
                for m in match:
                    m = m.strip()
                    if m == "WSH":
                        m = "WAS"
                    if m != team:
                        opp = m
                pos_str = str(position)
                print(player_name, opp, team, pos_str)

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
                if float(row["fpts"]) < self.projection_minimum:
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
                fpts = float(row["fpts"])
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
                    "Order": order,  # Handle blank orders
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
                self.teams_dict[team].append(
                    player_data
                )  # Add player data to their respective team

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
                    }
                    i += 1
        # print(self.field_lineups)

    @staticmethod
    def generate_lineups(
        lu_num,
        ids,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        teams,
        opponents,
        team_stack,
        stack_len,
        overlap_limit,
        max_stack_len,
        secondary_stack
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        total_players = pos_matrix.shape[1]
        if pos_matrix.shape[1] == 10:
            num_p_on_roster = 2
        else:
            num_p_on_roster = 1
        while reject:
            if team_stack == "":
                salary = 0
                proj = 0
                remaining_salary = salary_ceiling
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                lineup = []
                hitter_teams = []
                pitcher_opps = []
                hitters_opposing_pitcher = 0
                k = 0
                for pos in pos_matrix.T:
                    if k < num_p_on_roster:
                        # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                        valid_players = np.where((pos > 0) & (in_lineup == 0))
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = ownership[valid_players]
                        prob_list = prob_list / prob_list.sum()
                        choice = rng.choice(a=plyr_list, p=prob_list)
                        choice_idx = np.where(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += salaries[choice_idx]
                        remaining_salary -= salaries[choice_idx]
                        proj += projections[choice_idx]
                        pitcher_opps.append(opponents[choice_idx][0])
                    if k >= num_p_on_roster:
                        p1_opp = pitcher_opps[0]
                        if num_p_on_roster == 2:
                            p2_opp = pitcher_opps[1]
                        else:
                            p2_opp = "NOT_APPLICABLE"
                        if hitters_opposing_pitcher < overlap_limit:
                            if k == total_players-1:
                                valid_players = np.nonzero((pos > 0) & (in_lineup == 0) & (salaries <= remaining_salary) & (salary + salaries >= salary_floor))[0]
                            else:
                                valid_players = np.where((pos > 0) & (in_lineup == 0))
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            if k == total_players - 1:
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                            try:
                                if k == total_players - 1:
                                    choice = rng.choice(plyr_list, p=boosted_probabilities)
                                else:
                                    choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                                salary = 0
                                proj = 0
                                lineup = []
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                            choice_idx = np.where(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            remaining_salary -= salaries[choice_idx]
                            proj += projections[choice_idx]
                            hitter_teams.append(teams[choice_idx][0])
                            if teams[choice_idx][0] == p1_opp:
                                hitters_opposing_pitcher += 1
                            if teams[choice_idx][0] == p2_opp:
                                hitters_opposing_pitcher += 1
                        else:
                            if k == total_players-1:
                                valid_players = np.nonzero((pos > 0) & (in_lineup == 0) & (salaries <= remaining_salary) & (salary + salaries >= salary_floor) & (teams != p1_opp) & (teams != p2_opp))[0]
                            else:
                                valid_players = np.where(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (teams != p1_opp)
                                    & (teams != p2_opp)
                                )
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            if k == total_players - 1:
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                            try:
                                if k == total_players - 1:
                                    choice = rng.choice(plyr_list, p=boosted_probabilities)
                                else:
                                    choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                                salary = 0
                                proj = 0
                                lineup = []
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                            choice_idx = np.where(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            remaining_salary -= salaries[choice_idx]
                            proj += projections[choice_idx]
                            hitter_teams.append(teams[choice_idx][0])
                            if teams[choice_idx][0] == p1_opp:
                                hitters_opposing_pitcher += 1
                            if teams[choice_idx][0] == p2_opp:
                                hitters_opposing_pitcher += 1
                    k += 1
                # Must have a reasonable salary
                if salary >= salary_floor and salary <= salary_ceiling:
                    # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                    reasonable_projection = optimal_score - (
                        max_pct_off_optimal * optimal_score
                    )
                    if proj >= reasonable_projection:
                        mode = statistics.mode(hitter_teams)
                        if hitter_teams.count(mode) <= max_stack_len:
                            reject = False
                            lu = {
                                "Lineup": lineup,
                                "Wins": 0,
                                "Top1Percent": 0,
                                "ROI": 0,
                                "Cashes": 0,
                                "Type": "generated_nostack",
                                "Count": 0
                            }
            else:
                remaining_salary = salary_ceiling
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                hitter_teams = []
                pitcher_opps = []
                filled_pos = np.zeros(shape=pos_matrix.shape[1])
                team_stack_len = 0
                k = 0
                stack = True
                valid_team = np.where(teams == team_stack)[0]
                valid_players = np.unique(
                    valid_team[np.where(pos_matrix[valid_team, 2:] > 0)[0]]
                )
                hitters_opposing_pitcher = 0
                plyr_list = ids[valid_players]
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()
                stack_2 = True
                while stack:
                    choices = rng.choice(
                        a=plyr_list, p=prob_list, size=stack_len, replace=False
                    )
                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(str)
                    plyr_stack_indices = np.where(np.in1d(ids, choices))[0]
                    x = 0
                    for p in plyr_stack_indices:
                        if "0.0" in lineup[np.where(p > 0)[0]]:
                            for l in np.where(pos_matrix[p] > 0)[0]:
                                if lineup[l] == "0.0":
                                    lineup[l] = ids[p]
                                    x += 1
                                    break
                    if x == stack_len:
                        in_lineup[plyr_stack_indices] = 1
                        salary += sum(salaries[plyr_stack_indices])
                        # rint(salary)
                        proj += sum(projections[plyr_stack_indices])
                        # print(proj)
                        team_stack_len += stack_len
                        x = 0
                        stack = False
                if secondary_stack != "" and team_stack != "":
                    secondary_stack_len = 0
                    stack_2 = True
                    valid_team_2 = np.where(teams == secondary_stack)[0]
                    valid_players_2 = np.unique(
                        valid_team_2[np.where(pos_matrix[valid_team_2, 2:] > 0)[0]]
                    )
                    plyr_list_2 = ids[valid_players_2]
                    prob_list_2 = ownership[valid_players_2]
                    prob_list_2 = prob_list_2 / prob_list_2.sum()

                    while stack_2:
                        choices_2 = rng.choice(
                            a=plyr_list_2, p=prob_list_2, size=secondary_stack_len, replace=False
                        )
                        plyr_stack_indices_2 = np.where(np.in1d(ids, choices_2))[0]
                        x = 0
                        for p in plyr_stack_indices_2:
                            if "0.0" in lineup[np.where(p > 0)[0]]:
                                for l in np.where(pos_matrix[p] > 0)[0]:
                                    if lineup[l] == "0.0":
                                        lineup[l] = ids[p]
                                        x += 1
                                        break
                        if x == secondary_stack_len:
                            in_lineup[plyr_stack_indices_2] = 1
                            salary += sum(salaries[plyr_stack_indices_2])
                            proj += sum(projections[plyr_stack_indices_2])
                            secondary_stack_len += secondary_stack_len
                            x = 0
                            stack_2 = False

                for ix, (l, pos) in enumerate(zip(lineup, pos_matrix.T)):
                    # get pitchers irrespective of stack
                    #                    print(lu_num,ix, l, pos, k, lineup)
                    if l == "0.0":
                        if k < num_p_on_roster:
                            valid_players = np.where(
                                (pos > 0) & (in_lineup == 0) & (opponents != team_stack)
                            )
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being selected based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            # try:
                            if k == total_players - 1:
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                            try:
                                if k == total_players - 1:
                                    choice = rng.choice(plyr_list, p=boosted_probabilities)
                                else:
                                    choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                                salary = 0
                                proj = 0
                                lineup = []
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                            # except:
                            #    print(k, pos)
                            choice_idx = np.where(ids == choice)[0]
                            in_lineup[choice_idx] = 1
                            lineup[ix] = str(choice)
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            pitcher_opps.append(opponents[choice_idx][0])
                            k += 1
                        elif k >= num_p_on_roster:
                            p1_opp = pitcher_opps[0]
                            if num_p_on_roster == 2:
                                p2_opp = pitcher_opps[1]
                            else:
                                p2_opp = "NOT_APPLICABLE"
                            if hitters_opposing_pitcher < overlap_limit:                        
                                if k == total_players-1:
                                    valid_players = np.nonzero((pos > 0) & (in_lineup == 0) & (salaries <= remaining_salary) & (salary + salaries >= salary_floor))[0]
                                else:
                                    valid_players = np.nonzero((pos > 0) & (in_lineup == 0) & (salaries <= remaining_salary))[0]
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                if k == total_players - 1:
                                    boosted_salaries = np.array(
                                        [
                                            salary_boost(s, salary_ceiling)
                                            for s in salaries[valid_players]
                                        ]
                                    )
                                    boosted_probabilities = prob_list * boosted_salaries
                                    boosted_probabilities /= (
                                        boosted_probabilities.sum()
                                    )  # normalize to ensure it sums to 1
                                try:
                                    if k == total_players - 1:
                                        choice = rng.choice(plyr_list, p=boosted_probabilities)
                                    else:
                                        choice = rng.choice(plyr_list, p=prob_list)
                                except:
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                    salary = 0
                                    proj = 0
                                    lineup = []
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                choice_idx = np.where(ids == choice)[0]
                                lineup[ix] = str(choice)
                                in_lineup[choice_idx] = 1
                                salary += salaries[choice_idx]
                                proj += projections[choice_idx]
                                hitter_teams.append(teams[choice_idx][0])
                                if teams[choice_idx][0] == p1_opp:
                                    hitters_opposing_pitcher += 1
                                if teams[choice_idx][0] == p2_opp:
                                    hitters_opposing_pitcher += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                            else:
                                if k == total_players-1:
                                    try:
                                        valid_players = np.nonzero((pos > 0) & (in_lineup == 0) & (salaries <= remaining_salary) & (salary + salaries >= salary_floor) & (teams != p1_opp) & (teams != p2_opp) & (teams != team_stack))[0]
                                    except:
                                        print(f'Error: {pos} {in_lineup} {salaries} {remaining_salary} {salary} {salary_floor} {p1_opp} {p2_opp} {team_stack}')
                                else:
                                    valid_players = np.where(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (teams != p1_opp)
                                        & (teams != p2_opp)
                                        & (teams != team_stack)
                                    )
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                if k == total_players - 1:
                                    boosted_salaries = np.array(
                                        [
                                            salary_boost(s, salary_ceiling)
                                            for s in salaries[valid_players]
                                        ]
                                    )
                                    boosted_probabilities = prob_list * boosted_salaries
                                    boosted_probabilities /= (
                                        boosted_probabilities.sum()
                                    )  # normalize to ensure it sums to 1
                                try:
                                    if k == total_players - 1:
                                        choice = rng.choice(plyr_list, p=boosted_probabilities)
                                    else:
                                        choice = rng.choice(plyr_list, p=prob_list)
                                except:
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                    salary = 0
                                    proj = 0
                                    lineup = []
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                choice_idx = np.where(ids == choice)[0]
                                lineup[ix] = str(choice)
                                in_lineup[choice_idx] = 1
                                salary += salaries[choice_idx]
                                proj += projections[choice_idx]
                                hitter_teams.append(teams[choice_idx][0])
                                if teams[choice_idx][0] == p1_opp:
                                    hitters_opposing_pitcher += 1
                                if teams[choice_idx][0] == p2_opp:
                                    hitters_opposing_pitcher += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                            k += 1
                    else:
                        k += 1
                # Must have a reasonable salary    if team_stack_len >= stack_len and (secondary_stack_len == 0 or secondary_stack_len >= secondary_stack_len):

                if team_stack_len >= stack_len:
                    if salary >= salary_floor and salary <= salary_ceiling:
                        reasonable_projection = optimal_score - (
                            (max_pct_off_optimal * 1.25) * optimal_score
                        )
                        if proj >= reasonable_projection:
                            mode = statistics.mode(hitter_teams)
                            if hitter_teams.count(mode) <= max_stack_len:
                                reject = False
                                lu = {
                                    "Lineup": lineup,
                                    "Wins": 0,
                                    "Top1Percent": 0,
                                    "ROI": 0,
                                    "Cashes": 0,
                                    "Type": "generated_stack",
                                    "Count": 0
                                }
        return lu

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
            stacks = np.random.binomial(n=1, p=self.pct_field_using_stacks, size=diff)
            if self.site == "fd":
                stack_len = np.random.choice(
                    a=[3, 4],
                    p=[1 - self.pct_max_stack_len, self.pct_max_stack_len],
                    size=diff,
                )
                max_stack_len = 4
            else:
                max_stack_len = 5
                stack_len = np.random.choice(
                    a=[4, 5],
                    p=[1 - self.pct_max_stack_len, self.pct_max_stack_len],
                    size=diff,
                )
            a = list(self.stacks_dict.keys())
            p = np.array(list(self.stacks_dict.values()))
            probs = p / sum(p)
            stacks = stacks.astype(str)
            for i in range(len(stacks)):
                if stacks[i] == "1":
                    choice = random.choices(a, weights=probs, k=1)
                    stacks[i] = choice[0]
                else:
                    stacks[i] = ""
            secondary_stacks = np.random.binomial(n=1, p=self.pct_field_using_secondary_stacks, size=diff)
            if self.site == "fd":
                secondary_stack_len = np.random.choice(
                    a=[2, 3],  # Adjust the range as needed
                    size=diff,
                )
                max_secondary_stack_len = 3  # Adjust as needed
            else:
                max_secondary_stack_len = 3  # Adjust as needed
                secondary_stack_len = np.random.choice(
                    a=[2, 3],  # Adjust the range as needed
                    size=diff,
                )
            secondary_stacks = secondary_stacks.astype(str)

            for i in range(len(secondary_stacks)):
                if secondary_stacks[i] == "1":
                    secondary_choice = random.choices(a, weights=probs, k=1)
                    secondary_stacks[i] = secondary_choice[0]
                else:
                    secondary_stacks[i] = ""
            
            # creating tuples of the above np arrays plus which lineup number we are going to create
            # q = 0
            # for k in self.player_dict.keys():
            # if self.player_dict[k]['Team'] == stacks[0]:
            #    print(k, self.player_dict[k]['ID'])
            #    print(positions[q])
            # q += 1
            for i in range(diff):
                lu_tuple = (
                    i,
                    ids,
                    in_lineup,
                    pos_matrix,
                    ownership,
                    salary_floor,
                    salary_ceiling,
                    optimal_score,
                    salaries,
                    projections,
                    max_pct_off_optimal,
                    teams,
                    opponents,
                    stacks[i],
                    stack_len[i],
                    overlap_limit,
                    max_stack_len,
                    secondary_stacks[i]
                )
                problems.append(lu_tuple)
            # print(problems[0])
            # print(stacks)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
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
            #print(o.values())
            lineup_list = sorted(o['Lineup'])
            lineup_set = frozenset(lineup_list)
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

                    # Now assign the dictionary to the field_lineups
                    self.field_lineups[nk] = lineup_data                 
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    def run_simulation_for_team(
        self, team_id, team, pitcher_samples_dict, num_iterations
    ):
        correlation_matrix = np.array(
            [
                [
                    1.000000,
                    0.208567,
                    0.180173,
                    0.144913,
                    0.108717,
                    0.105452,
                    0.137026,
                    0.172705,
                    0.178171,
                    0.05,
                    -0.4,
                ],
                [
                    0.208567,
                    1.000000,
                    0.194440,
                    0.157801,
                    0.147979,
                    0.120411,
                    0.125511,
                    0.136052,
                    0.164456,
                    0.05,
                    -0.4,
                ],
                [
                    0.180173,
                    0.194440,
                    1.000000,
                    0.190412,
                    0.160093,
                    0.120162,
                    0.108959,
                    0.128614,
                    0.126364,
                    0.05,
                    -0.4,
                ],
                [
                    0.144913,
                    0.157801,
                    0.190412,
                    1.000000,
                    0.179935,
                    0.149753,
                    0.127822,
                    0.120928,
                    0.099442,
                    0.05,
                    -0.4,
                ],
                [
                    0.108717,
                    0.147979,
                    0.160093,
                    0.179935,
                    1.000000,
                    0.176625,
                    0.162855,
                    0.139522,
                    0.122343,
                    0.05,
                    -0.4,
                ],
                [
                    0.105452,
                    0.120411,
                    0.120162,
                    0.149753,
                    0.176625,
                    1.000000,
                    0.175045,
                    0.153736,
                    0.117608,
                    0.05,
                    -0.4,
                ],
                [
                    0.137026,
                    0.125511,
                    0.108959,
                    0.127822,
                    0.162855,
                    0.175045,
                    1.000000,
                    0.153188,
                    0.143971,
                    0.05,
                    -0.4,
                ],
                [
                    0.172705,
                    0.136052,
                    0.128614,
                    0.120928,
                    0.139522,
                    0.153736,
                    0.153188,
                    1.000000,
                    0.188197,
                    0.05,
                    -0.4,
                ],
                [
                    0.178171,
                    0.164456,
                    0.126364,
                    0.099442,
                    0.122343,
                    0.117608,
                    0.143971,
                    0.188197,
                    1.000000,
                    0.05,
                    -0.4,
                ],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1, -0.4],
                [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 1],
            ]
        )

        std_devs = [3] * 9 + [1] + [1]
        D = np.diag(std_devs)  # Create a diagonal matrix with the standard deviations
        covariance_matrix = np.dot(D, np.dot(correlation_matrix, D))

        # print(covariance_matrix)

        team = sorted(
            team,
            key=lambda player: float("inf")
            if player["Order"] is None
            else player["Order"],
        )
        # print(team)
        hitters_tuple_keys = [
            player for player in team if "P" not in player["Position"]
        ]
        pitcher_tuple_key = [player for player in team if "P" in player["Position"]][0]
        # print(pitcher_tuple_key)
        hitters_fpts = np.array([hitter["Fpts"] for hitter in hitters_tuple_keys])
        hitters_stddev = np.array([hitter["StdDev"] for hitter in hitters_tuple_keys])
        pitcher_fpts = pitcher_tuple_key["Fpts"]
        pitcher_stddev = pitcher_tuple_key["StdDev"]

        size = num_iterations

        # check if P has been simmed
        if pitcher_tuple_key["ID"] not in pitcher_samples_dict:
            pitcher_samples = np.random.normal(
                loc=pitcher_fpts, scale=pitcher_stddev, size=size
            )
            pitcher_samples_dict[pitcher_tuple_key["ID"]] = pitcher_samples
        else:
            pitcher_samples = pitcher_samples_dict[pitcher_tuple_key["ID"]]

        # find opp P
        opposing_pitcher_id = None
        for player in team:
            # print(player)
            if player["Opp"] in self.teams_dict:
                opposing_pitcher_id = next(
                    (
                        p["ID"]
                        for p in self.teams_dict[player["Opp"]]
                        if "P" in p["Position"]
                    ),
                    None,
                )
                break

        opposing_pitcher_samples = None

        # if opp P has not been simmed, sim it
        if opposing_pitcher_id is not None:
            opposing_pitcher = next(
                (p for p in self.teams_dict[player["Opp"]] if "P" in p["Position"]),
                None,
            )
            #  print(opposing_pitcher)
            if (
                opposing_pitcher is not None
                and opposing_pitcher_id not in pitcher_samples_dict
            ):
                opposing_pitcher_fpts = opposing_pitcher["Fpts"]
                # print(opposing_pitcher_fpts)
                opposing_pitcher_stddev = opposing_pitcher["StdDev"]
                opposing_pitcher_samples = np.random.normal(
                    loc=opposing_pitcher_fpts, scale=opposing_pitcher_stddev, size=size
                )
                pitcher_samples_dict[opposing_pitcher_id] = opposing_pitcher_samples

        hitters_params = [
            (fpts, stddev) for fpts, stddev in zip(hitters_fpts, hitters_stddev)
        ]
        pitcher_params = (pitcher_fpts, pitcher_stddev)
        opposing_pitcher_params = (opposing_pitcher_fpts, opposing_pitcher_stddev)

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
                        u = np.random.uniform()  # Generate a new uniform random number
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
            temp_fpts_dict[hitter["ID"]] = hitters_samples[i]

        temp_fpts_dict[pitcher_tuple_key["ID"]] = pitcher_samples

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

        start_time = time.time()
        temp_fpts_dict = {}
        pitcher_samples_dict = {}  # keep track of already simmed pitchers
        size = self.num_iterations

        with mp.Pool() as pool:
            team_simulation_params = [
                (team_id, team, pitcher_samples_dict, size)
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
        self.player_dict = {v["ID"]: v for v in self.player_dict.values()}

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
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
            own_p = np.prod(own_p)
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            Top1Percent_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            lu_type = x["Type"]
            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},${},{}%,{}%,{}%,{},${},{},{},{},{}".format(
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
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{}%,{},{},{},{}".format(
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
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{}%,{},${},{},{},{},{}".format(
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
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{},{},{},{},{}".format(
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
                        "P,P,C,1B,2B,3B,SS,OF,OF,OF,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type\n"
                    )
                else:
                    f.write(
                        "P,P,C,1B,2B,3B,SS,OF,OF,OF,Fpts Proj,Ceiling,Salary,Win %,Top 10%, Proj. Own. Product,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type\n"
                    )
                else:
                    f.write(
                        "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL,Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Stack1 Type,Stack2 Type,Num Opp Hitters,Lineup Type\n"
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