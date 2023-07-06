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
#import fuzzywuzzy
import itertools
import collections
import re
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GammaUnivariate, GaussianUnivariate
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning)



class MLB_Subset:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    id_name_dict = {}
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None
    use_lineup_input = None
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4

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

        ownership_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["ownership_path"]),
        )
        self.load_ownership(ownership_path)

        boom_bust_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        )
        self.load_boom_bust(boom_bust_path)
        
        stacks_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["team_stacks_path"]),
        )        
        self.load_team_stacks(stacks_path)

        top_batters_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["top_batters_path"]),
        )
        self.load_top_batters(top_batters_path)

        top_pitchers_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["top_pitchers_path"]),
        )
        self.load_top_pitchers(top_pitchers_path)




        
 #       batting_order_path = os.path.join(
 #           os.path.dirname(__file__),
#            "../{}_data/{}".format(site, self.config["batting_order_path"]),
#        )                
#        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.roster_construction = ["P", "P","C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
            self.salary = 50000
        
        elif site == "fd":
            self.roster_construction = ["P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
            self.salary = 60000

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
    
    #make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["builder_min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["builder_max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config['builder_pct_field_using_stacks'])
        self.default_hitter_var = float(self.config['default_hitter_var'])
        self.default_pitcher_var = float(self.config['default_pitcher_var'])
        self.pct_5man_stacks = float(self.config['builder_pct_5man_stacks'])
        self.overlap_limit = float(self.config['builder_num_hitters_vs_pitcher'])

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        for p,s in self.player_dict.items():
            if s["ID"]==0:
                print(s["Name"])
        problem = plp.LpProblem('MLB', plp.LpMaximize)
        lp_variables = {self.player_dict[(player, pos_str, team)]['ID']: plp.LpVariable(
            str(self.player_dict[(player, pos_str, team)]['ID']), cat='Binary') for (player, pos_str, team) in self.player_dict}

        # set the objective - maximize fpts
        problem += plp.lpSum(self.player_dict[(player, pos_str, team)]['Fpts'] * lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                             for (player, pos_str, team) in self.player_dict), 'Objective'

        # Set the salary constraints
        problem += plp.lpSum(self.player_dict[(player, pos_str, team)]['Salary'] * lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                             for (player, pos_str, team) in self.player_dict) <= self.salary

        if self.site == 'dk':
            # Need 2 pitchers
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'P' in self.player_dict[(player, pos_str, team)]['Position']) == 2
            # Need 1 catcher
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'C' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 1 first baseman
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if '1B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict  if '2B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict  if '3B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'SS' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 3 outfielders
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'OF' in self.player_dict[(player, pos_str, team)]['Position'])  == 3
            # Can only roster 8 total players
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict) == 10
            
                        # Max 5 hitters per team
            for team in self.team_list:
                problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if (self.player_dict[(player, pos_str, team)]['Team'] == team & self.player_dict[(player, pos_str, team)]['Position']!='P')) <= 5
                
        else:
            # Need 2 pitchers
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'P' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 1 catcher or first baseman
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if 'C' in self.player_dict[(player, pos_str, team)]['Position']) == 1

            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if '1B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 1 second baseman 
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if '2B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 1 third baseman
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict if '3B' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict  if 'SS' in self.player_dict[(player, pos_str, team)]['Position']) == 1
            # Need 3 outfielders
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict  if 'OF' in self.player_dict[(player, pos_str, team)]
                                 ['Position'])  == 3

            # Need 1 UTIL 
            
            # Can only roster 8 total players
            problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict ) == 10
            
                        # Max 5 hitters per team
            for team in self.team_list:
                problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]['ID']]
                                 for (player,pos_str,team) in self.player_dict  if (self.player_dict[(player, pos_str, team)]['Team'] == team & self.player_dict[(player, pos_str, team)]['Position']!='P')) <= 5
                
       # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                len(self.num_lineups), self.num_lineups))

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
                if 'P' in row['position']:
                    row['position'] = 'P'
                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                position = [pos for pos in row['position'].split('/')]
                if row['teamabbrev'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['teamabbrev']
                match =  re.search(pattern='(\w{2,4}@\w{2,4})', string=row['game info'])
                opp = ''
                match = match.groups()[0].split('@')
                for m in match:
                    m = m.strip()
                    if m == 'WSH':
                        m = 'WAS'
                    if m != team:
                        opp = m
                  # adjust "opponent" to match your CSV column
                pos_str = str(position)
                if (player_name,pos_str, team) in self.player_dict:
                    self.player_dict[(player_name,pos_str, team)]["ID"] = str(row["id"])
                    self.player_dict[(player_name,pos_str, team)]["Team"] =  row["teamabbrev"]
                    self.player_dict[(player_name,pos_str, team)]["Opp"] = opp
                #else:
                #    print(row[name_key] + ' not found in projections!')
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
                if 'P' in row['pos']:
                    row['pos'] = 'P'
                position = [pos for pos in row['pos'].split('/')]
                if row['team'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['team']
                pos_str = str(position)
                if row['ord'] == '-':
                    order = None
                else:
                    order = int(row["ord"])
                player_data = {
                    "Fpts": float(row["fpts"]),
                    "Position": position,
                    "Name" : player_name,
                    "Team" : team,
                    "Opp" : '',
                    "ID": '',
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": 0,
                    "Ceiling": 0,
                    "Ownership": 0.1,
                    "Order": order,  # Handle blank orders
                    "In Lineup": False,
                    "TopFp%": 0,
                    "TopVal%": 0
                }
                self.player_dict[(player_name, pos_str,team)] = player_data
                self.teams_dict[team].append(player_data)  # Add player data to their respective team



                                 
    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()                
                position = [pos for pos in row['pos'].split('/')]
                if row['team'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['team']
                pos_str = str(position)
                if (player_name,pos_str, team) in self.player_dict:
                    self.player_dict[(player_name,pos_str, team)]["Ownership"] = float(row["own%"])

    # Load standard deviations
    def load_boom_bust(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                #print(row)
                player_name = row["name"].replace("-", "#").lower()                
                position = [pos for pos in row['pos'].split('/')]
                pos_str = str(position)
                if row['team'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['team']
                if (player_name,pos_str, team) in self.player_dict:
                    self.player_dict[(player_name,pos_str, team)]["StdDev"] = float(row["stddev"])
                    self.player_dict[(player_name,pos_str, team)]["Ceiling"] = float(row["ceiling"])

    def load_top_batters(self, path):
        player_position_dict = {}  # This will track unique players and their positions

        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()
                position = row['pos']
                team = 'WAS' if row['team'] == 'WSH' else row['team']
                
                # We use a tuple of (player_name, team) as the key for our dictionary
                player_key = (player_name, team)
                
                # Check if the player is already in our dictionary
                if player_key in player_position_dict:
                    # If the player is already in our dictionary, append the new position
                    # But only if it's not already in their position list
                    if position not in player_position_dict[player_key]:
                        player_position_dict[player_key].append(position)
                else:
                    # If the player is not in our dictionary, add them
                    player_position_dict[player_key] = [position]
                    
                # Now we use the player_position_dict for the position part in our player_dict
                pos_list = player_position_dict[player_key]
                pos_str = str(pos_list)  # Convert the list to a string
                if (player_name, pos_str, team) in self.player_dict:

                    self.player_dict[(player_name, pos_str, team)]["TopFp%"] = float(row["topfp%"])
                    self.player_dict[(player_name, pos_str, team)]["TopVal%"] = float(row["topval%"])






    def load_top_pitchers(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()
                # print(row)
                pos_str = str(['P'])   # Create a list with 'P' and convert it to a string         
                if row['team'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['team']
                if (player_name, pos_str, team) in self.player_dict:
                    self.player_dict[(player_name, pos_str, team)]["TopFp%"] = float(row["top2%"])
                    self.player_dict[(player_name, pos_str, team)]["TopVal%"] = float(row["top2val%"])



    
    def adjust_default_stdev(self):
        for (player_name,pos, team) in self.player_dict.keys():
            if self.player_dict[(player_name,pos,team)]['StdDev'] == 0:
                if self.player_dict[(player_name,pos,team)]["Position"]== ["P"]:
                    print(player_name + ' has no stddev, defaulting to ' + str(self.default_pitcher_var) + '*projection')
                    self.player_dict[(player_name,pos,team)]["StdDev"] = self.player_dict[(player_name,pos,team)]["Fpts"]*self.default_pitcher_var
                else:
                    print(player_name + ' has no stddev, defaulting to ' + str(self.default_hitter_var) + '*projection')
                    self.player_dict[(player_name,pos,team)]["StdDev"] = self.player_dict[(player_name,pos,team)]["Fpts"]*self.default_hitter_var           
                    
    def load_team_stacks(self,path):
        with open(path) as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if row['team'] == 'WSH':
                    team = 'WAS'
                else:
                    team = row['team']
                # this changes the stack selection weighting processing the generate_field_lineups
                self.stacks_dict[team] = (float(row["top stack %"]) + float(row["top value %"])) / 2

                
    
                    
    def remap(self, fieldnames):
        return ["P","C/1B","2B","3B","SS","OF","OF","OF","UTIL"]

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
                        str(row[0].split("(")[1].replace(")","")),
                        str(row[1].split("(")[1].replace(")","")),
                        str(row[2].split("(")[1].replace(")","")),
                        str(row[3].split("(")[1].replace(")","")),
                        str(row[4].split("(")[1].replace(")","")),
                        str(row[5].split("(")[1].replace(")","")),
                        str(row[6].split("(")[1].replace(")","")),
                        str(row[7].split("(")[1].replace(")","")),
                        str(row[8].split("(")[1].replace(")","")), 
                        str(row[9].split("(")[1].replace(")",""))
                    ]
                    # storing if this lineup was made by an optimizer or with the generation process in this script
                    self.field_lineups[i] = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top10": 0,
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
                        str(row[0].split("(")[1].replace(")","")),
                        str(row[1].split("(")[1].replace(")","")),
                        str(row[2].split("(")[1].replace(")","")),
                        str(row[3].split("(")[1].replace(")","")),
                        str(row[4].split("(")[1].replace(")","")),
                        str(row[5].split("(")[1].replace(")","")),
                        str(row[6].split("(")[1].replace(")","")),
                        str(row[7].split("(")[1].replace(")","")),
                        str(row[8].split("(")[1].replace(")","")) 
                    ]
                    # storing if this lineup was made by an optimizer or with the generation process in this script
                    self.field_lineups[i] = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top10": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "opto",
                    }
                    i += 1                
        #print(self.field_lineups)

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
        topfpts,
        topval
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        np.random.seed(lu_num)
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        if pos_matrix.shape[1] == 10:
            num_p_on_roster = 2
        else:
            num_p_on_roster = 1
        while reject:
            if team_stack == '':
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                lineup = []
                hitter_teams = []
                pitcher_opps = []
                hitters_opposing_pitcher = 0
                k=0
                for pos in pos_matrix.T:
                    if k <num_p_on_roster:
                    # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                        valid_players = np.where((pos > 0) & (in_lineup == 0))
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                        prob_list = prob_list / prob_list.sum()
                        choice = np.random.choice(a=plyr_list, p=prob_list)
                        choice_idx = np.where(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += salaries[choice_idx]
                        proj += projections[choice_idx]
                        pitcher_opps.append(opponents[choice_idx][0])
                    if k >=num_p_on_roster:
                        p1_opp = pitcher_opps[0]
                        if num_p_on_roster == 2:
                            p2_opp = pitcher_opps[1]
                        else:
                            p2_opp = 'NOT_APPLICABLE'
                        if hitters_opposing_pitcher < overlap_limit:
                            valid_players = np.where((pos > 0) & (in_lineup == 0))
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                            prob_list = prob_list / prob_list.sum()
                            choice = np.random.choice(a=plyr_list, p=prob_list)
                            choice_idx = np.where(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            hitter_teams.append(teams[choice_idx][0])
                            if teams[choice_idx][0] == p1_opp:
                                hitters_opposing_pitcher += 1
                            if teams[choice_idx][0] == p2_opp:
                                hitters_opposing_pitcher += 1
                        else:
                            valid_players = np.where((pos > 0) & (in_lineup == 0)& (teams!=p1_opp)& (teams!=p2_opp))   
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                            prob_list = prob_list / prob_list.sum()
                            choice = np.random.choice(a=plyr_list, p=prob_list)
                            choice_idx = np.where(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            hitter_teams.append(teams[choice_idx][0]) 
                            if teams[choice_idx][0] == p1_opp:
                                hitters_opposing_pitcher += 1
                            if teams[choice_idx][0] == p2_opp:
                                hitters_opposing_pitcher += 1      
                    k +=1 
                # Must have a reasonable salary
                if salary >= salary_floor and salary <= salary_ceiling:
                    # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                    reasonable_projection = optimal_score - (
                        max_pct_off_optimal * optimal_score
                    )
                    if proj >= reasonable_projection:
                        mode = statistics.mode(hitter_teams)
                        if hitter_teams.count(mode) <= 5:                 
                            reject = False
                            lus[lu_num] = {
                                "Lineup": lineup,
                                "Wins": 0,
                                "Top10": 0,
                                "ROI": 0,
                                "Cashes": 0,
                                "Type": "generated_nostack",
                            }
            else:
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                hitter_teams = []
                pitcher_opps = []
                filled_pos = np.zeros(shape=pos_matrix.shape[1])
                team_stack_len = 0
                k=0
                stack = True
                valid_team = np.where(teams == team_stack)[0]
                valid_players = np.unique(valid_team[np.where(pos_matrix[valid_team,2:]>0)[0]])
                hitters_opposing_pitcher = 0
                plyr_list = ids[valid_players]
                prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                prob_list = prob_list / prob_list.sum()
                while stack: 
                    choices = np.random.choice(a=plyr_list, p=prob_list, size=stack_len, replace=False)
                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(str)
                    plyr_stack_indices = np.where(np.in1d(ids, choices))[0]
                    x=0
                    for p in plyr_stack_indices:
                        if '0.0' in lineup[np.where(p>0)[0]]:
                            for l in np.where(pos_matrix[p]>0)[0]:
                                if lineup[l] == '0.0':
                                    lineup[l] = ids[p]
                                    x+=1
                                    break
                    if x==stack_len:
                        in_lineup[plyr_stack_indices] =1
                        salary += sum(salaries[plyr_stack_indices])
                        #rint(salary)
                        proj += sum(projections[plyr_stack_indices])
                        #print(proj)
                        team_stack_len += stack_len
                        x=0
                        stack = False
                for ix, (l,pos) in enumerate(zip(lineup,pos_matrix.T)):
                    # get pitchers irrespective of stack
#                    print(lu_num,ix, l, pos, k, lineup)
                    if l == '0.0':
                        if k <num_p_on_roster:
                            valid_players = np.where((pos > 0) & (in_lineup == 0) & (opponents!=team_stack))
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being selected based on ownership and who is eligible at the position
                            prob_list = (topfpts[valid_players] + topval[valid_players]) / 2
                            # print(prob_list) 
                            prob_list = prob_list / prob_list.sum()
                            #try:
                            choice = np.random.choice(a=plyr_list, p=prob_list)
                            #except:
                            #    print(k, pos)
                            choice_idx = np.where(ids == choice)[0]
                            in_lineup[choice_idx] = 1
                            lineup[ix] = str(choice)
                            salary += salaries[choice_idx]
                            proj += projections[choice_idx]
                            pitcher_opps.append(opponents[choice_idx][0])
                            k +=1                         
                        elif k >=num_p_on_roster:
                            p1_opp = pitcher_opps[0]
                            if num_p_on_roster == 2:
                                p2_opp = pitcher_opps[1]
                            else:
                                p2_opp = 'NOT_APPLICABLE'
                            if hitters_opposing_pitcher < overlap_limit:
                                valid_players = np.where((pos > 0) & (in_lineup == 0)& (teams!=team_stack))
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                                prob_list = prob_list / prob_list.sum()
                                choice = np.random.choice(a=plyr_list, p=prob_list)
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
                                valid_players = np.where((pos > 0) & (in_lineup == 0)& (teams!=p1_opp)& (teams!=p2_opp)& (teams!=team_stack))   
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = (topfpts[valid_players] + topval[valid_players]) / 2 
                                prob_list = prob_list / prob_list.sum()
                                choice = np.random.choice(a=plyr_list, p=prob_list)
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
                            k +=1 
                    else:
                        k+=1
                # Must have a reasonable salary
                if team_stack_len >=stack_len:
                    if salary >= salary_floor and salary <= salary_ceiling:
                    # loosening reasonable projection constraint for team stacks
                        reasonable_projection = optimal_score - (
                            (max_pct_off_optimal*1.25) * optimal_score
                        )
                        if proj >= reasonable_projection:
                            mode = statistics.mode(hitter_teams)
                            if hitter_teams.count(mode) <= 5:                 
                                reject = False
                                lus[lu_num] = {
                                    "Lineup": lineup,
                                    "Wins": 0,
                                    "Top10": 0,
                                    "ROI": 0,
                                    "Cashes": 0,
                                    "Type": "generated_stack",
                                }                
        

        
        return lus
    


    def generate_field_lineups(self):
        # print(self.player_dict)
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print('Generating ' + str(diff) + ' lineups.')
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            teams = []
            opponents = []
            topfpts = []
            topval = []
            for k in self.player_dict.keys():
                if 'Team' not in self.player_dict[k].keys():
                    print(self.player_dict[k]['Name'], ' name mismatch between projections and player ids!')
                ids.append(self.player_dict[k]['ID'])
                ownership.append(self.player_dict[k]['Ownership'])
                salaries.append(self.player_dict[k]['Salary'])
                projections.append(self.player_dict[k]['Fpts'])
                teams.append(self.player_dict[k]['Team'])
                opponents.append(self.player_dict[k]['Opp'])
                topfpts.append(self.player_dict[k]['TopFp%'])
                topval.append(self.player_dict[k]['TopVal%'])
                pos_list = []
                for pos in self.roster_construction:
                    if pos in self.player_dict[k]['Position']:
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
            topfpts = np.array(topfpts)
            topval = np.array(topval)
            overlap_limit = self.overlap_limit
            problems = []
            stacks = np.random.binomial(n=1,p=self.pct_field_using_stacks,size=diff)
            stack_len = np.random.choice(a=[4,5],p=[1-self.pct_5man_stacks, self.pct_5man_stacks],size=diff)
            a = list(self.stacks_dict.keys())
            p = np.array(list(self.stacks_dict.values()))
            probs = p/sum(p)
            stacks = stacks.astype(str)
            for i in range(len(stacks)):
                if stacks[i] == '1':
                    choice = random.choices(a,weights=probs,k=1)
                    stacks[i] = choice[0]
                else:
                    stacks[i] = ''
            # creating tuples of the above np arrays plus which lineup number we are going to create
            #q = 0
            #for k in self.player_dict.keys():
                #if self.player_dict[k]['Team'] == stacks[0]:
                #    print(k, self.player_dict[k]['ID'])
                #    print(positions[q])
                #q += 1
            for i in range(diff):
                lu_tuple = (i, ids, in_lineup, pos_matrix,ownership, salary_floor, salary_ceiling, optimal_score, salaries, projections,max_pct_off_optimal, teams, opponents, stacks[i], stack_len[i], overlap_limit, topfpts, topval)
                problems.append(lu_tuple)
            #print(problems[0])
            #print(stacks)
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



            if len(self.field_lineups) == 0:
                new_keys = list(range(0, self.field_size))
            else:
                new_keys = list(
                    range(max(self.field_lineups.keys()) + 1, self.field_size)
                )
            nk = new_keys[0]
            for i, o in enumerate(output):
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                self.field_lineups[nk] = o[i]
                nk += 1
            end_time = time.time()

            print("lineups took " + str(end_time - start_time) + " seconds")
            print(str(diff) + " field lineups successfully generated")

                # Create DataFrame from lineups
            # Create DataFrame from lineups

            lineups_data = []
            for index, x in self.field_lineups.items():
                lineup_row = {}
                salary = 0
                fpts_p = 0
                own_p = []
                lu_names = []
                lu_teams = []
                hitters_vs_pitcher = 0
                pitcher_opps = []
                for id in x["Lineup"]:
                    for k,v in self.player_dict.items():
                        if v["ID"] == id:  
                            if 'P' in v["Position"]:
                                pitcher_opps.append(v['Opp'])         
                for id in x["Lineup"]:
                    for k,v in self.player_dict.items():
                        if v["ID"] == id:
                            salary += v["Salary"]
                            fpts_p += v["Fpts"]
                            own_p.append(v["Ownership"]/100)
                            lu_names.append(v["Name"])
                            if 'P' not in v["Position"]:
                                lu_teams.append(v['Team'])
                                if v['Team'] in pitcher_opps:
                                    hitters_vs_pitcher += 1
                            continue
                counter = collections.Counter(lu_teams)
                stacks = counter.most_common(2)
                own_p = np.prod(own_p)
                lu_type = x["Type"]

                # creating the lineup string
                lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({})".format(
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
                    x["Lineup"][9]
                )

                
                # continue adding relevant features to the lineup_row dictionary
                lineup_row['salary'] = salary
                lineup_row['fpts_p'] = fpts_p
                lineup_row['own_p'] = own_p
                lineup_row['stacks'] = str(stacks[0][0]) + ' ' + str(stacks[0][1]), str(stacks[1][0]) + ' ' + str(stacks[1][1])
                lineup_row['hitters_vs_pitcher'] = hitters_vs_pitcher
                lineup_row['lu_type'] = lu_type
                lineup_row['lineup_str'] = lineup_str
                lineups_data.append(lineup_row)

            # Convert list of dictionaries to DataFrame
            df_lineups = pd.DataFrame(lineups_data)

            # split lineup_str into separate columns and assign your custom column names
            lineup_df = df_lineups['lineup_str'].str.split(',', expand=True)
            lineup_df.columns = ['P','P','C','1B','2B','3B','SS','OF','OF','OF']

            # Convert stacks column to DataFrame
            stacks_df = df_lineups['stacks'].apply(pd.Series)
            # Rename columns
            stacks_df.columns = ['main', 'sub']

            # drop the original lineup_str and stacks columns from df_lineups and concat lineup_df and stacks_df
            df_lineups = df_lineups.drop(['lineup_str', 'stacks'], axis=1)
            df_lineups = pd.concat([lineup_df, stacks_df, df_lineups], axis=1)

            # Write DataFrame to CSV
            df_lineups.to_csv('/Users/jack/GitHub/MLB-DFS-Tools/dk_data/tournament_lineups.csv', index=False)
