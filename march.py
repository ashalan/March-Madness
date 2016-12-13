from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
import pickle

# load the model from disk
input_features = pickle.load(open('pkl/input_features.pkl', 'rb'))
n = pickle.load(open('pkl/n.pkl', 'rb'))
outputs = pickle.load(open('pkl/outputs.pkl', 'rb'))
train_outputs = pickle.load(open('pkl/train_outputs.pkl', 'rb'))
inputs = pickle.load(open('pkl/inputs.pkl', 'rb'))
normalizer = pickle.load(open('pkl/normalizer.pkl', 'rb'))
train_inputs = pickle.load(open('pkl/train_inputs.pkl', 'rb'))
games_rnd = pickle.load(open('pkl/games_rnd.pkl', 'rb'))
team_stats = pickle.load(open('pkl/team_stats.pkl', 'rb'))

def get_last_games_avg_both(team1, team2, season):
    team_1 = games_rnd[(games_rnd['team1'] == team1)&(games_rnd['team2'] == team2)&(games_rnd['Season'] == season)]
    team_2 = games_rnd[(games_rnd['team1'] == team2)&(games_rnd['team2'] == team1)&(games_rnd['Season'] == season)]
#     team_2 = games_rnd[(games_rnd['team2'] == team)&(games_rnd['Season'] == season)][[x for x in games_rnd.columns.values if '2' in x]+['Daynum', 'Season']]
    
    team_1_first = team_1[(team_1['team1'] == team1)&(team_1['Season'] == season)][[x for x in team_1.columns.values if '1' in x]+['Daynum', 'Season']]
    team_2_first = team_1[(team_1['team2'] == team2)&(team_1['Season'] == season)][[x for x in team_1.columns.values if '2' in x]+['Daynum', 'Season']]
    
    team_1_second = team_2[(team_2['team1'] == team1)&(team_2['Season'] == season)][[x for x in team_2.columns.values if '1' in x]+['Daynum', 'Season']]
    team_2_second = team_2[(team_2['team2'] == team2)&(team_2['Season'] == season)][[x for x in team_2.columns.values if '2' in x]+['Daynum', 'Season']]
    
    team_1_first.columns = team_1_first.columns.map(lambda x: x.strip('1'))
    team_2_first.columns = team_2_first.columns.map(lambda x: x.strip('2'))
    
    team_1_second.columns = team_1_second.columns.map(lambda x: x.strip('1'))
    team_2_second.columns = team_2_second.columns.map(lambda x: x.strip('2'))

    games = pd.concat([team_1_first, team_2_first, team_1_second, team_2_second])
    games = games.sort_values('Daynum')
    del games['Daynum']
    return games.groupby(['Season','team'])['ast', 'blk', 'dr', 'fga', 'fga3', 'fgm', 'fgm3', 'fta', 'ftm', 'or', 'pf', 'score', 'stl', 'to'].mean().reset_index()

def get_last_n_games_avg(n, team, season):
    team_1 = games_rnd[(games_rnd['team1'] == team)&(games_rnd['Season'] == season)][[x for x in games_rnd.columns.values if '1' in x]+['Daynum', 'Season']]
    team_2 = games_rnd[(games_rnd['team2'] == team)&(games_rnd['Season'] == season)][[x for x in games_rnd.columns.values if '2' in x]+['Daynum', 'Season']]

    team_1.columns = team_1.columns.map(lambda x: x.strip('1'))
    team_2.columns = team_2.columns.map(lambda x: x.strip('2'))

    games = pd.concat([team_1, team_2])
    games = games.sort_values('Daynum')
    del games['Daynum']
    return games.iloc[-n:].groupby(['Season','team'])['ast', 'blk', 'dr', 'fga', 'fga3', 'fgm', 'fgm3', 'fta', 'ftm', 'or', 'pf', 'score', 'stl', 'to'].mean().reset_index()

def neural_result(input):
    """Call the neural network, and translates its output to a match result."""
    n_output = n.activate(input[0]) 
    if n_output >= 0.5:
        return 2
    else:
        return 1

def output(year, team1, team2):
    inputs = []
    diff_year_1 = ''
    diff_year_2 = ''
    n_games = 20
    if year in range(2003, 2017):
        if len(get_last_games_avg_both(team1, team2, year)) != 0:
            team_stats = get_last_games_avg_both(team1, team2, year)
        else:
            team_stats = pd.concat([get_last_n_games_avg(n_games, team1, year), get_last_n_games_avg(n_games, team2, year)])
    elif year < 2003:
        if len(get_last_games_avg_both(team1, team2, 2003)) != 0:
            team_stats = get_last_games_avg_both(team1, team2, year)
        else:
            team_stats = pd.concat([get_last_n_games_avg(n_games, team1, 2003), get_last_n_games_avg(n_games, team2, 2003)])
    else:
        if len(get_last_games_avg_both(team1, team2, 2016)) != 0:
            team_stats = get_last_games_avg_both(team1, team2, year)
        else:
            team_stats = pd.concat([get_last_n_games_avg(n_games, team1, 2016), get_last_n_games_avg(n_games, team2, 2016)])
    for feature in input_features:
        from_team_2 = '2' in feature
        feature = feature.replace('2', '')
        feature = feature.replace('1', '')
        if feature in [x for x in team_stats.columns.values if x != 'Season']:
            team = team2 if from_team_2 else team1
            try:
                value = team_stats[(team_stats.team == team)&(team_stats.Season == year)].iloc[[0]][feature].values[0]
            except:
                if from_team_2:
                    diff_year_2 = team
                else:
                    diff_year_1 = team
                value = team_stats[team_stats['team'] == team].iloc[[-1]][feature].values[0]
        elif feature == 'Season':
            value = year
        else:
            raise ValueError("Don't know where to get feature: " + feature)
        inputs.append(value)

    inputs = normalizer.transform(np.array(inputs).reshape((1, -1)))
    result = neural_result(inputs)
    results = {}
    if diff_year_1 != '':
        year_used = team_stats[team_stats['team'] == team1].iloc[[-1]]['Season'].values[0]
        results['warning1'] = "Couldn't find data from "+str(year)+" for "+diff_year_1+", used "+str(int(year_used))+" instead."
    if diff_year_2 != '':
        year_used = team_stats[team_stats['team'] == team2].iloc[[-1]]['Season'].values[0]
        results['warning2'] = "Couldn't find data from "+str(year)+" for "+diff_year_2+", used "+str(int(year_used))+" instead."

    if result == 1:
        results['winner'] = team1
    elif result == 2:
        results['winner'] = team2
    else:
        results['winner'] = 'Unknown result: ' + str(result)

    return results

#-------- ROUTES GO HERE -----------#
@app.route('/prediction')
def prediction():
    year = request.args.get('year', 0, type=int)
    team1 = request.args.get('team1', 0, type=str)
    team2 = request.args.get('team2', 0, type=str)
    return jsonify(output(year, team1, team2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/json/<path:path>')
def send_json(path):
    return send_from_directory('json', path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# 74, 49, 0, 23, 7, 25