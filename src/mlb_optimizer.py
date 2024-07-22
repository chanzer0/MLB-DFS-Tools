from collections import Counter
import itertools
import json
import csv
import os
import datetime
import numpy as np
import pulp as plp


class MLB_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = 0
    randomness_amount = 0
    matchup_list = []

    primary_stack_min = 4
    primary_stack_max = 5
    secondary_stack_min = 2
    secondary_stack_max = 3
    primary_stack_teams = []
    secondary_stack_teams = []
    players_by_team = {}
    min_lineup_salary = 0
    max_batters_vs_pitchers = 0

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()

        self.problem = plp.LpProblem("MLB", plp.LpMaximize)

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        # need to load rules after loading projections to get teams
        self.load_rules()

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                player_name = row[name_key]
                matchup = row["game info"].split(" ")[0]

                # find the key in self.player_dict that matches the player_name
                player_tuple = None
                for key, attributes in self.player_dict.items():
                    if player_name == attributes["Name"]:
                        player_tuple = key
                        break

                if matchup not in self.matchup_list:
                    self.matchup_list.append(matchup)

                if player_tuple in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[player_tuple]["ID"] = int(row["id"])
                        self.player_dict[player_tuple]["Matchup"] = matchup
                    else:
                        self.player_dict[player_tuple]["ID"] = str(row["id"])
                        self.player_dict[player_tuple]["Matchup"] = matchup

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.primary_stack_min = int(self.config["primary_stack_min"])
        self.primary_stack_max = int(self.config["primary_stack_max"])
        self.secondary_stack_min = int(self.config["secondary_stack_min"])
        self.secondary_stack_max = int(self.config["secondary_stack_max"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_batters_vs_pitchers = int(self.config["num_hitters_vs_pitcher"])

        p_stack_teams = self.config["primary_stack_teams"]
        if p_stack_teams == "*":
            self.primary_stack_teams = self.team_list
        else:
            self.primary_stack_teams = self.config["primary_stack_teams"].split(",")

        s_stack_teams = self.config["secondary_stack_teams"]
        if s_stack_teams == "*":
            self.secondary_stack_teams = self.team_list
        else:
            self.secondary_stack_teams = self.config["secondary_stack_teams"].split(",")

        print(
            f"Primary stack teams: {self.primary_stack_teams} (Min: {self.primary_stack_min}, Max: {self.primary_stack_max})"
        )
        print(
            f"Secondary stack teams: {self.secondary_stack_teams} (Min: {self.secondary_stack_min}, Max: {self.secondary_stack_max})"
        )

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#")
                projection = float(row["fpts"])

                stddev = float(row["stddev"]) if "stddev" in row else projection * 0.4

                if projection < self.projection_minimum:
                    continue

                order = None
                try:
                    order = int(row["ord"])
                except ValueError:
                    order = 100

                player_name = row["name"].replace("-", "#")
                position = row["pos"].replace("SP", "P").replace("RP", "P")
                team = row["team"]

                if self.site == "fd":
                    if "C/1B" in position:
                        position = "C1B"
                    # add a util position for all non-pitchers
                    if "P" not in position:
                        position += "/UTIL"

                self.player_dict[(player_name, position, team)] = {
                    "Fpts": projection,
                    "ID": 0,
                    "Salary": int(row["salary"].replace(",", "")),
                    "Name": row["name"],
                    "Ownership": float(row["own%"]) if "own%" in row else 0,
                    "BattingOrder": order,
                    "StdDev": stddev,
                    "Team": team,
                    "Position": [pos for pos in position.split("/")],
                    "Matchup": "",
                    "Opponent": "",
                    "Tuple": (player_name, position, team),
                }

                if team not in self.team_list:
                    self.team_list.append(team)
                if team not in self.players_by_team:
                    self.players_by_team[team] = []
                    self.players_by_team[team].append((player_name, position, team))
                else:
                    self.players_by_team[team].append((player_name, position, team))

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {}
        for player_tuple, attributes in self.player_dict.items():
            player_id = attributes["ID"]
            for pos in attributes["Position"]:
                lp_variables[(player_tuple, pos, player_id)] = plp.LpVariable(
                    name=f"{player_tuple}_{pos}_{player_id}", cat=plp.LpBinary
                )

        # for player_tuple, items in self.player_dict.items():
        #     print(
        #         f"{player_tuple} - fpts:{items['Fpts']} - ${items['Salary']} - pos:{items['Position']} - team:{items['Team']} - id:{items['ID']}, matchup:{items['Matchup']}"
        #     )

        # set the objective - maximize fpts & set randomness amount from config
        self.problem += (
            plp.lpSum(
                np.maximum(
                    np.random.normal(
                        self.player_dict[player_tuple]["Fpts"],
                        (
                            self.player_dict[player_tuple]["StdDev"]
                            * self.randomness_amount
                            / 100
                        ),
                    ),
                    0,
                )
                * lp_variables[(player_tuple, pos, attributes["ID"])]
                for player_tuple, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            ),
            "Objective",
        )

        # Set the salary constraints
        max_salary = 50000 if self.site == "dk" else 35000
        self.problem += (
            plp.lpSum(
                self.player_dict[player_tuple]["Salary"]
                * lp_variables[(player_tuple, pos, attributes["ID"])]
                for player_tuple, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player_tuple]["Salary"]
                * lp_variables[(player_tuple, pos, attributes["ID"])]
                for player_tuple, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            >= self.min_lineup_salary,
            "Min Salary",
        )

        # For DK, Lineups will consist of 10 players and must include players scheduled to play in at least 2 different MLB games. Lineups must have no more than 5 hitters from any one team
        # For FD, Lineups will consist of 9 players and must include players scheduled to play in at least 2 different MLB games and 3 different teams. Lineups must have no more than 4 hitters from any one team
        matchup_limit = 9 if self.site == "dk" else 8
        for matchupIdent in self.matchup_list:
            self.problem += (
                plp.lpSum(
                    lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                    if attributes["Matchup"] in matchupIdent
                )
                <= matchup_limit - 1,
                f"Must not play all {matchup_limit} players from match {matchupIdent}",
            )

        hitter_limit = 5 if self.site == "dk" else 4
        for team_ident, player_tuples in self.players_by_team.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[
                        (
                            p_tuple,
                            pos,
                            self.player_dict[p_tuple]["ID"],
                        )
                    ]
                    for p_tuple in player_tuples
                    for pos in self.player_dict[p_tuple]["Position"]
                    if "P" not in pos
                )
                <= hitter_limit,
                f"Max {hitter_limit} hitters from {team_ident}",
            )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player_tuple, pos, attributes["ID"])]
                        for player_tuple, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Name"] in group
                    )
                    >= int(limit),
                    f"At least {limit} players {group}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player_tuple, pos, attributes["ID"])]
                        for player_tuple, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Name"] in group
                    )
                    <= int(limit),
                    f"At most {limit} players {group}",
                )

        # Matchup limits
        for matchup, limit in self.matchup_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                    if attributes["Matchup"] == matchup
                )
                <= int(limit),
                "At most {} players from {}".format(limit, matchup),
            )

        for matchup, limit in self.matchup_at_least.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                    if attributes["Matchup"] == matchup
                )
                >= int(limit),
                "At least {} players from {}".format(limit, matchup),
            )

        # Global team limit
        if self.global_team_limit is not None and (
            (self.global_team_limit < 5 and self.site == "dk")
            or (self.global_team_limit < 4 and self.site == "fd")
        ):
            for teamIdent in self.team_list:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player_tuple, pos, attributes["ID"])]
                        for player_tuple, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Team"] == teamIdent
                        and "P" not in attributes["Position"]
                    )
                    <= int(self.global_team_limit),
                    f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                )

        if self.site == "dk":
            # Constraints for specific positions
            # DK needs 2 P, 1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF
            for pos in ["P", "C", "1B", "2B", "3B", "SS", "OF"]:
                if pos == "P":
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player_tuple, pos, attributes["ID"])]
                            for player_tuple, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 2,
                        f"Must have 2 {pos}",
                    )
                elif pos == "OF":
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player_tuple, pos, attributes["ID"])]
                            for player_tuple, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 3,
                        f"Must have 3 {pos}",
                    )
                else:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player_tuple, pos, attributes["ID"])]
                            for player_tuple, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have 1 {pos}",
                    )
            # 10 players in total
            self.problem += (
                plp.lpSum(
                    lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                == 10,
                "Must have 10 players",
            )
        else:
            # Constraints for specific positions
            # FD needs 1 P, 1 C/1B, 1 2B, 1 3B, 1 SS, 3 OF, 1 UTIL
            for pos in ["P", "C1B", "2B", "3B", "SS", "OF", "UTIL"]:
                if pos == "OF":
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player_tuple, pos, attributes["ID"])]
                            for player_tuple, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 3,
                        f"Must have 3 {pos}",
                    )
                else:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player_tuple, pos, attributes["ID"])]
                            for player_tuple, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have 1 {pos}",
                    )
            # 9 players in total
            self.problem += (
                plp.lpSum(
                    lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                == 9,
                "Must have 9 players",
            )

        # Constraint to ensure each player is only selected once
        for player_tuple in self.player_dict:
            pos = self.player_dict[player_tuple]["Position"]
            player_id = self.player_dict[player_tuple]["ID"]
            self.problem += (
                plp.lpSum(lp_variables[(player_tuple, p, player_id)] for p in pos) <= 1,
                f"Can only select ({player_tuple, pos, player_id}) once",
            )

        # No hitters vs pitchers
        hit_stack = 5 if self.site == "dk" else 4
        for matchup in self.matchup_list:
            team_a = matchup.split("@")[0]
            team_b = matchup.split("@")[1]
            team_a_pitchers = [
                player_tuple
                for player_tuple, attributes in self.player_dict.items()
                if attributes["Team"] == team_a and "P" in attributes["Position"]
            ]
            team_b_pitchers = [
                player_tuple
                for player_tuple, attributes in self.player_dict.items()
                if attributes["Team"] == team_b and "P" in attributes["Position"]
            ]
            team_a_hitters = [
                (player_tuple, pos, self.player_dict[player_tuple]["ID"])
                for player_tuple, attributes in self.player_dict.items()
                for pos in attributes["Position"]
                if attributes["Team"] == team_a and "P" not in attributes["Position"]
            ]
            team_b_hitters = [
                (player_tuple, pos, self.player_dict[player_tuple]["ID"])
                for player_tuple, attributes in self.player_dict.items()
                for pos in attributes["Position"]
                if attributes["Team"] == team_b and "P" not in attributes["Position"]
            ]
            for pitcher in team_a_pitchers:
                self.problem += (
                    plp.lpSum(
                        (
                            hit_stack
                            * lp_variables[
                                (pitcher, "P", self.player_dict[pitcher]["ID"])
                            ]
                        )
                        + (
                            lp_variables[player_tuple]
                            for player_tuple in team_b_hitters
                        )
                    )
                    <= hit_stack + self.max_batters_vs_pitchers,
                    f"No {team_b} hitters vs {team_a} {pitcher}",
                )
            for pitcher in team_b_pitchers:
                self.problem += (
                    plp.lpSum(
                        (
                            hit_stack
                            * lp_variables[
                                (pitcher, "P", self.player_dict[pitcher]["ID"])
                            ]
                        )
                        + (
                            lp_variables[player_tuple]
                            for player_tuple in team_a_hitters
                        )
                    )
                    <= hit_stack + self.max_batters_vs_pitchers,
                    f"No {team_a} hitters vs {team_b} {pitcher}",
                )

        # primary and secondary team bat stacks
        # TODO - create plp.lpSums to enforce primary and secondary stack rules
        # For primary stack, we must include at least "self.primary_stack_min" players from each team in the primary stack and at most "self.primary_stack_max" players from each team in the primary stack
        # For secondary stack, we must include at least "self.secondary_stack_min" players from each team in the secondary stack and at most "self.secondary_stack_max" players from each team in the secondary stack
        # For primary stack, this must only include players from the teams in self.primary_stack_teams
        # For secondary stack, this must only include players from the teams in self.secondary_stack_teams
        # There must be 1 team in the primary stack and at least 1 team in the secondary stack
        # The same team cannot be in both the primary and secondary stack
        # The stack rules should only apply to teams when they are selected as a primary or secondary stack, otherwise they should not be enforced
        # Pitchers should not be included in the stack rules

        # Create binary variables to track if a team is selected as primary or secondary stack
        primary_stack_selected = {
            team: plp.LpVariable(f"primary_stack_{team}", cat=plp.LpBinary)
            for team in self.primary_stack_teams
        }
        secondary_stack_selected = {
            team: plp.LpVariable(f"secondary_stack_{team}", cat=plp.LpBinary)
            for team in self.secondary_stack_teams
        }

        # primary and secondary stacks
        for primary_team in self.primary_stack_teams:
            primary_team_players = [
                player_tuple
                for player_tuple, attributes in self.player_dict.items()
                if attributes["Team"] == primary_team
                and "P" not in attributes["Position"]
            ]

            # Count the number of players selected from this team
            team_player_count = plp.lpSum(
                lp_variables[(player_tuple, pos, self.player_dict[player_tuple]["ID"])]
                for player_tuple in primary_team_players
                for pos in self.player_dict[player_tuple]["Position"]
            )

            # If the team is selected as a primary stack, apply the primary stack rules
            self.problem += (
                team_player_count
                >= self.primary_stack_min * primary_stack_selected[primary_team],
                f"At least {self.primary_stack_min} players for {primary_team} in primary stack",
            )
            self.problem += (
                team_player_count
                <= self.primary_stack_max * primary_stack_selected[primary_team]
                + len(primary_team_players)
                * (1 - primary_stack_selected[primary_team]),
                f"At most {self.primary_stack_max} players for {primary_team} in primary stack",
            )

        for secondary_team in self.secondary_stack_teams:
            secondary_team_players = [
                player_tuple
                for player_tuple, attributes in self.player_dict.items()
                if attributes["Team"] == secondary_team
                and "P" not in attributes["Position"]
            ]

            # Count the number of players selected from this team
            team_player_count = plp.lpSum(
                lp_variables[(player_tuple, pos, self.player_dict[player_tuple]["ID"])]
                for player_tuple in secondary_team_players
                for pos in self.player_dict[player_tuple]["Position"]
            )

            # If the team is selected as a secondary stack, apply the secondary stack rules
            self.problem += (
                team_player_count
                >= self.secondary_stack_min * secondary_stack_selected[secondary_team],
                f"At least {self.secondary_stack_min} players for {secondary_team} in secondary stack",
            )
            self.problem += (
                team_player_count
                <= self.secondary_stack_max * secondary_stack_selected[secondary_team]
                + len(secondary_team_players)
                * (1 - secondary_stack_selected[secondary_team]),
                f"At most {self.secondary_stack_max} players for {secondary_team} in secondary stack",
            )

        # There must be 1 team in the primary stack and at least 1 team in the secondary stack
        self.problem += (
            plp.lpSum(primary_stack_selected.values()) == 1,
            "Must select one primary stack team",
        )
        self.problem += (
            plp.lpSum(secondary_stack_selected.values()) >= 1,
            "Must select at least one secondary stack team",
        )

        # The same team cannot be in both the primary and secondary stack
        for team in self.primary_stack_teams:
            if team in self.secondary_stack_teams:
                self.problem += (
                    primary_stack_selected[team] + secondary_stack_selected[team] <= 1,
                    f"Team {team} cannot be selected in both primary and secondary stacks",
                )

        # enforce adjacent batting order constraints
        for team in self.team_list:
            # Create a list of players in the team sorted by batting order
            sorted_players = sorted(
                [
                    player_tuple
                    for player_tuple in self.player_dict
                    if self.player_dict[player_tuple]["Team"] == team
                    and "P" not in self.player_dict[player_tuple]["Position"]
                    and self.player_dict[player_tuple]["BattingOrder"] < 10
                ],
                key=lambda x: self.player_dict[x]["BattingOrder"],
            )
            # Update the job doc with print statement of team and the batting order
            bat_order_str = (
                f"Team: {team}, {'🆗' if len(sorted_players) == 9 else '❌'} {len(sorted_players)} Batters: "
                + ", ".join(
                    [
                        f"({self.player_dict[player_tuple]['BattingOrder']}) {self.player_dict[player_tuple]['Name']}"
                        for player_tuple in sorted_players
                    ]
                )
            )
            print(bat_order_str)

            # Iterate over the sorted players and create constraints
            for i in range(len(sorted_players)):
                player_tuple = sorted_players[i]
                player_id = self.player_dict[player_tuple]["ID"]
                positions = self.player_dict[player_tuple]["Position"]

                # Create a list to store the adjacent players' variables
                adjacent_vars = []

                # Get the previous player index (considering cyclical order)
                prev_index = (i - 1) % len(sorted_players)
                prev_player_tuple = sorted_players[prev_index]
                prev_player_id = self.player_dict[prev_player_tuple]["ID"]
                prev_positions = self.player_dict[prev_player_tuple]["Position"]
                adjacent_vars.extend(
                    [
                        lp_variables[(prev_player_tuple, pos, prev_player_id)]
                        for pos in prev_positions
                    ]
                )

                # Get the next player index (considering cyclical order)
                next_index = (i + 1) % len(sorted_players)
                next_player_tuple = sorted_players[next_index]
                next_player_id = self.player_dict[next_player_tuple]["ID"]
                next_positions = self.player_dict[next_player_tuple]["Position"]
                adjacent_vars.extend(
                    [
                        lp_variables[(next_player_tuple, pos, next_player_id)]
                        for pos in next_positions
                    ]
                )

                # Create constraints for the current player
                for pos in positions:
                    # If the team is selected as a primary or secondary stack, enforce the batting order constraint
                    if team in self.primary_stack_teams:
                        self.problem += (
                            plp.lpSum(adjacent_vars)
                            - lp_variables[(player_tuple, pos, player_id)]
                            >= -1 + primary_stack_selected[team],
                            f"Primary stack batting order for {player_tuple}_{pos}_{player_id}",
                        )
                    elif team in self.secondary_stack_teams:
                        self.problem += (
                            plp.lpSum(adjacent_vars)
                            - lp_variables[(player_tuple, pos, player_id)]
                            >= -1 + secondary_stack_selected[team],
                            f"Secondary stack batting order for {player_tuple}_{pos}_{player_id}",
                        )
                    else:
                        self.problem += (
                            plp.lpSum(adjacent_vars)
                            - lp_variables[(player_tuple, pos, player_id)]
                            >= -1,
                            f"Non-stack Batting order for {player_tuple}_{pos}_{player_id}",
                        )

        # Crunch!
        self.problem.writeLP("problem.lp")
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError as e:
                print(f"Solver error. Continuing with export. {e}")
                i = self.num_lineups
                break

            # Check for infeasibility in problem
            if plp.LpStatus[self.problem.status] != "Optimal":
                print(
                    f"Non-optimal solution. Continuing with export. {plp.LpStatus[self.problem.status]}"
                )
                i = self.num_lineups
                break

            # Initial chunk size
            chunk_size = self.num_lineups / 5.0

            # Round to the nearest 5 or 10
            chunk_size = (
                round(chunk_size / 10.0) * 10
                if chunk_size > 10
                else round(chunk_size / 5.0) * 5
            )

            # Generate chunk sizes
            chunks = [0] + [chunk_size * i for i in range(1, 5)] + [self.num_lineups]

            if i in chunks:
                print(f"Successfully generated {i}/{self.num_lineups} lineups")
            if i == self.num_lineups - 1:
                print(
                    f"Successfully generated {self.num_lineups}/{self.num_lineups} lineups"
                )

            # Get the lineup and add it to our list
            selected_vars = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            self.lineups.append(selected_vars)

            # Ensure this lineup isn't picked again
            player_ids = [tpl[2] for tpl in selected_vars]
            player_keys_to_exlude = []
            for key, attr in self.player_dict.items():
                if attr["ID"] in player_ids:
                    for pos in attr["Position"]:
                        player_keys_to_exlude.append((key, pos, attr["ID"]))

            self.problem += (
                plp.lpSum(lp_variables[x] for x in player_keys_to_exlude)
                <= len(selected_vars) - self.num_uniques,
                f"Lineup {i}",
            )

            # Set a new random fpts projection within their distribution
            # set the objective - maximize fpts & set randomness amount from config
            self.problem += (
                plp.lpSum(
                    np.maximum(
                        np.random.normal(
                            self.player_dict[player_tuple]["Fpts"],
                            (
                                self.player_dict[player_tuple]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        ),
                        0,
                    )
                    * lp_variables[(player_tuple, pos, attributes["ID"])]
                    for player_tuple, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                ),
                "Objective",
            )

    def sort_lineup(self, lineup):
        if self.site == "dk":
            order = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
            sorted_lineup = [None] * 10
        else:
            order = ["P", "C1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
            sorted_lineup = [None] * 9

        for player in lineup:
            player_tuple, pos, pid = player
            order_idx = order.index(pos)

            while sorted_lineup[order_idx] is not None:
                order_idx += 1
                if order_idx >= len(sorted_lineup):
                    break

            if order_idx < len(sorted_lineup):
                sorted_lineup[order_idx] = player_tuple

        return sorted_lineup

    def construct_stack_str(self, lineup):
        # Extract team identifiers from the list
        teams = [
            self.player_dict[player]["Team"]
            for player in lineup
            if "P" not in self.player_dict[player]["Position"]
        ]

        # Count the occurrences of each team
        team_counts = {}
        for team in teams:
            if team in team_counts:
                team_counts[team] += 1
            else:
                team_counts[team] = 1

        # Find the maximum count (stack5)
        max_count = max(team_counts.values())

        # Find all teams with the maximum count (stack5)
        max_teams = [team for team, count in team_counts.items() if count == max_count]

        # Find the second maximum count (stack2)
        second_max_count = 0
        for count in team_counts.values():
            if count < max_count and count > second_max_count:
                second_max_count = count

        # Find all teams with the second maximum count (stack2)
        second_max_teams = [
            team for team, count in team_counts.items() if count == second_max_count
        ]

        # Construct the stack string for stack5
        primary_stack_strs = [f"{team} {max_count}" for team in max_teams]
        primary_stack_str = " - ".join(primary_stack_strs)

        # Construct the stack string for stack2
        secondary_stack_strs = [
            f"{team} {second_max_count}" for team in second_max_teams
        ]
        secondary_stack_str = " - ".join(secondary_stack_strs)

        # Combine the stack5 and stack2 strings
        if secondary_stack_str:
            combined_string = f"{primary_stack_str};{secondary_stack_str}"
        else:
            combined_string = primary_stack_str

        return combined_string

    def output(self):
        print("Optimization complete. Starting output.")

        sorted_lineups = []
        for lineup in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineups.append(sorted_lineup)

        total_lineups = len(sorted_lineups)
        team_stack_counts = {}
        csv_str = (
            "P,P,C,1B,2B,3B,SS,OF,OF,OF,Salary,Fpts Proj,Own. Product,Own. Sum,Stack,StdDev\n"  # 16
            if self.site == "dk"
            else "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL,Salary,Fpts Proj,Own. Product,Own. Sum,Stack,StdDev\n"  # 15
        )
        lineup_output = []

        for x in sorted_lineups:
            # Calculate various stats
            salary = sum(self.player_dict[player]["Salary"] for player in x)
            fpts_p = sum(self.player_dict[player]["Fpts"] for player in x)
            own_p = np.prod(
                [self.player_dict[player]["Ownership"] / 100.0 for player in x]
            )
            own_s = sum([self.player_dict[player]["Ownership"] for player in x])
            std_dev = sum(self.player_dict[player]["StdDev"] for player in x)
            stack_key = self.construct_stack_str(x)

            if "-" in stack_key:
                for stack in stack_key.split(" - "):
                    team = stack.split(" ")[0]
                    if stack not in team_stack_counts:
                        team_stack_counts[stack] = Counter()
                    team_stack_counts[stack][team] += 1
            else:
                team = stack_key.split(" ")[0]
                if stack_key not in team_stack_counts:
                    team_stack_counts[stack_key] = Counter()
                team_stack_counts[stack_key][team] += 1

            format_string = (
                "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}\n"
                if self.site == "dk"
                else "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}\n"
            )
            if self.site == "dk":
                csv_str += format_string.format(
                    self.player_dict[x[0]]["Name"],
                    self.player_dict[x[0]]["ID"],
                    self.player_dict[x[1]]["Name"],
                    self.player_dict[x[1]]["ID"],
                    self.player_dict[x[2]]["Name"],
                    self.player_dict[x[2]]["ID"],
                    self.player_dict[x[3]]["Name"],
                    self.player_dict[x[3]]["ID"],
                    self.player_dict[x[4]]["Name"],
                    self.player_dict[x[4]]["ID"],
                    self.player_dict[x[5]]["Name"],
                    self.player_dict[x[5]]["ID"],
                    self.player_dict[x[6]]["Name"],
                    self.player_dict[x[6]]["ID"],
                    self.player_dict[x[7]]["Name"],
                    self.player_dict[x[7]]["ID"],
                    self.player_dict[x[8]]["Name"],
                    self.player_dict[x[8]]["ID"],
                    self.player_dict[x[9]]["Name"],
                    self.player_dict[x[9]]["ID"],
                    salary,
                    round(fpts_p, 2),
                    own_p,
                    own_s,
                    stack_key,
                    std_dev,
                )
            else:
                csv_str += format_string.format(
                    self.player_dict[x[0]]["ID"].replace("#", "-"),
                    self.player_dict[x[0]]["Name"],
                    self.player_dict[x[1]]["ID"].replace("#", "-"),
                    self.player_dict[x[1]]["Name"],
                    self.player_dict[x[2]]["ID"].replace("#", "-"),
                    self.player_dict[x[2]]["Name"],
                    self.player_dict[x[3]]["ID"].replace("#", "-"),
                    self.player_dict[x[3]]["Name"],
                    self.player_dict[x[4]]["ID"].replace("#", "-"),
                    self.player_dict[x[4]]["Name"],
                    self.player_dict[x[5]]["ID"].replace("#", "-"),
                    self.player_dict[x[5]]["Name"],
                    self.player_dict[x[6]]["ID"].replace("#", "-"),
                    self.player_dict[x[6]]["Name"],
                    self.player_dict[x[7]]["ID"].replace("#", "-"),
                    self.player_dict[x[7]]["Name"],
                    self.player_dict[x[8]]["ID"].replace("#", "-"),
                    self.player_dict[x[8]]["Name"],
                    salary,
                    round(fpts_p, 2),
                    own_p,
                    own_s,
                    stack_key,
                    std_dev,
                )

        # stacks_export = {"stackExposures": {}}
        # for stack_type, teams in team_stack_counts.items():
        #     stacks_export["stackExposures"][stack_type] = []
        #     for team, count in teams.items():
        #         stacks_export["stackExposures"][stack_type].append(
        #             {
        #                 "team": team,
        #                 "percentage": float(
        #                     "{:.2f}".format(count / total_lineups * 100)
        #                 ),
        #                 "count": count,
        #             }
        #         )

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_optimal_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        )
        with open(out_path, "w") as f:
            f.write(csv_str)

        print(f"Output complete. Lineups saved to {out_path}")
